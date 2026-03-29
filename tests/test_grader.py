"""Unit tests for deterministic episode grading logic."""

from __future__ import annotations

from grader.grader import Grader
from models.contamination_spec import ContaminationSpec


def _srm_spec() -> ContaminationSpec:
    """Build a representative SRM contamination spec for grader tests."""
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


def test_grade_episode_rewards_correct_srm_verdict_with_evidence() -> None:
    """Correct verdict/type/investigation/evidence should produce a strong score."""
    spec = _srm_spec()
    episode_log = [
        {
            "action_type": "run_srm_check",
            "parameters": {},
            "reasoning": "Count imbalance suggests SRM; running formal test.",
            "confidence": 0.7,
        },
        {
            "action_type": "flag_contamination",
            "parameters": {
                "contamination_type": "srm",
                "evidence_facts": [
                    "actual split is 43.8% treatment (vs 50% intended)",
                    "srm detected true",
                ],
                "recommended_action": "rerun",
            },
            "reasoning": "SRM invalidates randomization integrity.",
            "confidence": 0.9,
        },
    ]

    result = Grader.grade_episode(episode_log, spec)

    assert result["verdict_action"] == "flag_contamination"
    assert result["breakdown"]["verdict"] == 1.0
    assert result["breakdown"]["type_id"] == 1.0
    assert result["breakdown"]["investigation"] == 1.0
    assert result["breakdown"]["evidence"] > 0.0
    assert result["final_score"] >= 0.75


def test_grade_episode_penalizes_approving_contaminated_experiment() -> None:
    """Approving a contaminated experiment should score poorly."""
    spec = _srm_spec()
    episode_log = [
        {
            "action_type": "approve_result",
            "parameters": {},
            "reasoning": "Looks significant; approving.",
            "confidence": 0.95,
        }
    ]

    result = Grader.grade_episode(episode_log, spec)

    assert result["verdict_action"] == "approve_result"
    assert result["breakdown"]["verdict"] == 0.0
    assert result["breakdown"]["type_id"] == 0.0
    assert 0.0 <= result["final_score"] <= 0.20


def test_grade_episode_handles_event_style_logs_from_state_manager() -> None:
    """Grader should parse logs where action is nested in event payload."""
    spec = _srm_spec()
    episode_log = [
        {
            "timestamp": "2026-03-30T10:00:00Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "run_srm_check",
                    "parameters": {},
                    "reasoning": "Formal SRM test.",
                    "confidence": 0.8,
                },
                "reward": -0.01,
                "observation_delta": {"newly_revealed_fields": ["randomization_check"]},
            },
        },
        {
            "timestamp": "2026-03-30T10:00:01Z",
            "event_type": "step",
            "payload": {
                "action": {
                    "action_type": "flag_contamination",
                    "parameters": {
                        "contamination_type": "srm",
                        "evidence_facts": ["actual split 43.8%", "srm detected true"],
                        "recommended_action": "rerun",
                    },
                    "reasoning": "SRM confirmed.",
                    "confidence": 0.9,
                },
                "reward": 0.5,
                "observation_delta": {"newly_revealed_fields": []},
            },
        },
    ]

    result = Grader.grade_episode(episode_log, spec)

    assert result["verdict_action"] == "flag_contamination"
    assert result["breakdown"]["verdict"] == 1.0
    assert result["breakdown"]["investigation"] == 1.0
    assert result["final_score"] >= 0.70
