"""Unit tests for deterministic grading logic using handcrafted episode logs."""

from __future__ import annotations

from grader.grader import Grader
from models.contamination_spec import ContaminationSpec


def _srm_spec() -> ContaminationSpec:
    """Construct an SRM contamination specification for tests."""
    return ContaminationSpec(
        contamination_type="srm",
        true_effect_size=0.0,
        visible_effect_size=0.062,
        optimal_investigation_steps=3,
        required_queries=["run_srm_check"],
        ground_truth_evidence={
            "actual_split": 0.438,
            "srm_detected": True,
        },
    )


def _sutva_spec() -> ContaminationSpec:
    """Construct a SUTVA contamination specification for tests."""
    return ContaminationSpec(
        contamination_type="sutva_violation",
        true_effect_size=0.012,
        visible_effect_size=0.083,
        optimal_investigation_steps=8,
        required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
        ground_truth_evidence={
            "control_users_in_pricing_treatment": 0.71,
            "control_users_exposed_to_treatment_behavior": 0.23,
        },
    )


def _clean_spec() -> ContaminationSpec:
    """Construct a clean-experiment specification for tests."""
    return ContaminationSpec(
        contamination_type="clean",
        true_effect_size=0.038,
        visible_effect_size=0.038,
        optimal_investigation_steps=4,
        required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"],
        ground_truth_evidence={"srm_detected": False},
    )


def test_grader_scores_correct_srm_invalidation_high() -> None:
    """Correct SRM verdict with matching evidence should score strongly."""
    episode_log = [
        {
            "action_type": "run_srm_check",
            "parameters": {},
            "reasoning": "Counts look asymmetric; running formal SRM check now.",
            "confidence": 0.65,
        },
        {
            "action_type": "flag_contamination",
            "parameters": {
                "contamination_type": "srm",
                "evidence_facts": [
                    "actual split is 43.8% treatment",
                    "srm detected true",
                ],
            },
            "reasoning": "Randomization was broken, so the result is invalid.",
            "confidence": 0.9,
        },
    ]

    result = Grader.grade_episode(episode_log, _srm_spec())

    assert result["verdict_action"] == "flag_contamination"
    assert result["breakdown"]["verdict"] == 1.0
    assert result["breakdown"]["type_id"] == 1.0
    assert result["breakdown"]["investigation"] == 1.0
    assert result["breakdown"]["evidence"] > 0.0
    assert result["final_score"] >= 0.75


def test_grader_penalizes_approving_contaminated() -> None:
    """Approving a contaminated result should have low final score."""
    episode_log = [
        {
            "action_type": "approve_result",
            "parameters": {},
            "reasoning": "Significant result; approving immediately.",
            "confidence": 0.95,
        }
    ]

    result = Grader.grade_episode(episode_log, _srm_spec())

    assert result["verdict_action"] == "approve_result"
    assert result["breakdown"]["verdict"] == 0.0
    assert result["breakdown"]["type_id"] == 0.0
    assert 0.0 <= result["final_score"] <= 0.20


def test_grader_applies_partial_type_credit_matrix() -> None:
    """Related contamination labels should receive partial type-id credit."""
    episode_log = [
        {
            "action_type": "query_assignment_overlap",
            "parameters": {},
            "reasoning": "Need to validate overlap with concurrent pricing experiments.",
            "confidence": 0.6,
        },
        {
            "action_type": "flag_contamination",
            "parameters": {
                "contamination_type": "network_spillover",
                "evidence_facts": [
                    "control users exposed to treatment behavior is 23%",
                ],
            },
            "reasoning": "Interference signal found, but mechanism may overlap with SUTVA.",
            "confidence": 0.7,
        },
    ]

    result = Grader.grade_episode(episode_log, _sutva_spec())

    assert result["breakdown"]["verdict"] == 1.0
    assert result["breakdown"]["type_id"] == 0.5


def test_grader_supports_event_style_logs() -> None:
    """Nested event logs from state manager should be parsed correctly."""
    episode_log = [
        {
            "timestamp": "2026-03-30T12:00:00Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "run_srm_check",
                    "parameters": {},
                    "reasoning": "Checking assignment balance before making any verdict.",
                    "confidence": 0.7,
                }
            },
        },
        {
            "timestamp": "2026-03-30T12:00:02Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "query_temporal",
                    "parameters": {},
                    "reasoning": "Temporal trend check to avoid novelty confounds.",
                    "confidence": 0.75,
                }
            },
        },
        {
            "timestamp": "2026-03-30T12:00:05Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "query_assignment_overlap",
                    "parameters": {},
                    "reasoning": "Overlap checked and randomization units are non-interfering.",
                    "confidence": 0.82,
                }
            },
        },
        {
            "timestamp": "2026-03-30T12:00:08Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "approve_result",
                    "parameters": {},
                    "reasoning": "Core validity checks passed; approving clean experiment.",
                    "confidence": 0.85,
                }
            },
        },
    ]

    result = Grader.grade_episode(episode_log, _clean_spec())

    assert result["verdict_action"] == "approve_result"
    assert result["breakdown"]["verdict"] == 1.0
    assert result["breakdown"]["investigation"] == 1.0
    assert result["final_score"] >= 0.55
