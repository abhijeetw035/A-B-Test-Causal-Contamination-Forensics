"""Deterministic episode grader for contamination forensics tasks."""

from __future__ import annotations

import re
from typing import Any

from models.contamination_spec import ContaminationSpec


TYPE_MATCH_MATRIX: dict[tuple[str, str], float] = {
    ("srm", "srm"): 1.0,
    ("srm", "underpowered_overclaim"): 0.3,
    ("underpowered_overclaim", "srm"): 0.3,
    ("sutva_violation", "network_spillover"): 0.5,
    ("network_spillover", "sutva_violation"): 0.5,
    ("simpsons_paradox", "multiple_testing"): 0.2,
    ("novelty_effect", "simpsons_paradox"): 0.1,
}

TERMINAL_ACTIONS: set[str] = {"flag_contamination", "approve_result", "request_rerun"}
STRICT_SCORE_MIN = 0.01
STRICT_SCORE_MAX = 0.99

RELEVANT_QUERY_MAP: dict[str, set[str]] = {
    "clean": {"run_srm_check", "query_temporal", "query_assignment_overlap"},
    "srm": {"run_srm_check", "inspect_randomization"},
    "sutva_violation": {"query_assignment_overlap", "check_network_exposure", "compute_mde"},
    "novelty_effect": {"query_temporal", "query_secondary_metrics"},
    "simpsons_paradox": {"query_temporal", "query_subgroup", "query_secondary_metrics"},
    "network_spillover": {"query_assignment_overlap", "check_network_exposure"},
    "multiple_testing": {"query_secondary_metrics", "compute_mde"},
    "underpowered_overclaim": {"compute_mde", "query_temporal"},
}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to an inclusive range."""
    return max(low, min(high, value))


def _clamp_task_score_open_unit_interval(value: float) -> float:
    """Clamp final task score to strict open interval (0,1).

    The benchmark validator rejects exact boundary values 0.0 and 1.0.
    """
    return _clamp(value, STRICT_SCORE_MIN, STRICT_SCORE_MAX)


def _extract_actions(episode_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract action objects from flat or event-style episode logs.

    Args:
        episode_log: Episode records where actions may be top-level or nested.

    Returns:
        Ordered list of action dictionaries.
    """
    actions: list[dict[str, Any]] = []
    for entry in episode_log:
        if "action_type" in entry:
            actions.append(entry)
            continue

        payload = entry.get("payload")
        if isinstance(payload, dict):
            action = payload.get("action")
            if isinstance(action, dict) and "action_type" in action:
                actions.append(action)
    return actions


def _extract_final_action(episode_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the final action, preferring terminal actions when present.

    Args:
        episode_log: Episode records.

    Returns:
        The last terminal action dictionary, or the last action if no terminal action exists.
    """
    actions = _extract_actions(episode_log)
    if not actions:
        return {}

    for action in reversed(actions):
        if action.get("action_type") in TERMINAL_ACTIONS:
            return action
    return actions[-1]


def _compute_evidence_strength(queries_executed: list[str], spec: ContaminationSpec) -> float:
    """Compute evidence strength from required and relevant query coverage.

    Args:
        queries_executed: Action types executed up to a step.
        spec: Hidden contamination specification.

    Returns:
        Evidence strength score in [0.0, 1.0].
    """
    executed = {q for q in queries_executed if q not in TERMINAL_ACTIONS}
    required = set(spec.required_queries or [])
    relevant = RELEVANT_QUERY_MAP.get(spec.contamination_type, set())

    required_coverage = 1.0 if not required else len(executed & required) / len(required)
    relevant_coverage = 0.0 if not relevant else len(executed & relevant) / len(relevant)
    return _clamp((0.7 * required_coverage) + (0.3 * relevant_coverage), 0.0, 1.0)


def _evidence_strength_at_step(queries_executed: list[str], spec: ContaminationSpec) -> float:
    """Backward-compatible alias for step-wise evidence strength computation.

    Args:
        queries_executed: Action types executed up to a step.
        spec: Hidden contamination specification.

    Returns:
        Evidence strength score in [0.0, 1.0].
    """
    return _compute_evidence_strength(queries_executed, spec)


def _verify_evidence_facts(claimed_facts: list[str], spec: ContaminationSpec) -> float:
    """Verify evidence facts against ground truth with tolerant numeric matching.

    Args:
        claimed_facts: Facts provided in terminal contamination report.
        spec: Hidden contamination specification with ground_truth_evidence.

    Returns:
        Fraction of checkable facts that are correct in [0.0, 1.0].
    """
    if not claimed_facts:
        return 0.0

    ground_truth = spec.ground_truth_evidence or {}
    if not ground_truth:
        return 0.0

    checked = 0
    matched = 0

    for fact in claimed_facts:
        lower_fact = fact.lower()
        for key, gt_value in ground_truth.items():
            key_phrase = key.replace("_", " ")
            if key_phrase not in lower_fact:
                continue

            checked += 1

            if isinstance(gt_value, str):
                if gt_value.lower() in lower_fact:
                    matched += 1
                continue

            if isinstance(gt_value, bool):
                expected = "true" if gt_value else "false"
                if expected in lower_fact:
                    matched += 1
                continue

            if isinstance(gt_value, (int, float)):
                numbers = re.findall(r"[-+]?\d*\.?\d+", fact)
                if not numbers:
                    continue
                candidate = float(numbers[-1])
                if "%" in fact:
                    candidate /= 100.0

                tolerance = 0.10 * max(abs(float(gt_value)), 1e-6)
                if abs(candidate - float(gt_value)) <= tolerance:
                    matched += 1

    return 0.0 if checked == 0 else matched / checked


class Grader:
    """Deterministic grader with fixed weighted scoring dimensions."""

    @staticmethod
    def grade_episode(episode_log: list[dict[str, Any]], spec: ContaminationSpec) -> dict[str, Any]:
        """Grade an episode and return weighted score breakdown.

        Args:
            episode_log: Full episode actions/events.
            spec: Hidden contamination specification for ground truth.

        Returns:
            A dictionary with `final_score`, `breakdown`, and metadata.
        """
        actions = _extract_actions(episode_log)
        final_action = _extract_final_action(episode_log)
        verdict = str(final_action.get("action_type", ""))

        if spec.contamination_type == "clean":
            verdict_score = 1.0 if verdict == "approve_result" else 0.0
        else:
            verdict_score = 1.0 if verdict == "flag_contamination" else 0.0

        type_score = 0.0
        if verdict == "flag_contamination":
            claimed_type = str(final_action.get("parameters", {}).get("contamination_type", ""))
            type_score = TYPE_MATCH_MATRIX.get(
                (claimed_type, spec.contamination_type),
                1.0 if claimed_type == spec.contamination_type else 0.0,
            )

        required = set(spec.required_queries or [])
        executed = {str(a.get("action_type", "")) for a in actions}
        investigation_score = 1.0 if not required else len(required & executed) / len(required)

        evidence_score = 0.0
        if verdict == "flag_contamination":
            evidence_facts = final_action.get("parameters", {}).get("evidence_facts", [])
            if isinstance(evidence_facts, list):
                evidence_score = _verify_evidence_facts(evidence_facts, spec)

        confidence_actions = [a for a in actions if isinstance(a.get("confidence"), (int, float))]
        if len(confidence_actions) >= 2:
            calibration_errors: list[float] = []
            seen_actions: list[str] = []

            for action in actions:
                action_type = str(action.get("action_type", ""))
                seen_actions.append(action_type)
                confidence = action.get("confidence")
                if isinstance(confidence, (int, float)):
                    expected = _compute_evidence_strength(seen_actions, spec)
                    calibration_errors.append(abs(float(confidence) - expected))

            avg_error = sum(calibration_errors) / max(len(calibration_errors), 1)
            calibration_score = max(0.0, 1.0 - (avg_error * 2.0))
        else:
            calibration_score = 0.5

        breakdown = {
            "verdict": round(verdict_score, 4),
            "type_id": round(type_score, 4),
            "investigation": round(investigation_score, 4),
            "evidence": round(evidence_score, 4),
            "calibration": round(calibration_score, 4),
        }

        weights = {
            "verdict": 0.35,
            "type_id": 0.25,
            "investigation": 0.20,
            "evidence": 0.12,
            "calibration": 0.08,
        }

        final_score = sum(breakdown[k] * weights[k] for k in weights)
        final_score = _clamp_task_score_open_unit_interval(final_score)

        return {
            "final_score": round(final_score, 4),
            "breakdown": breakdown,
            "weights": weights,
            "verdict_action": verdict,
            "ground_truth_type": spec.contamination_type,
        }


__all__ = [
    "Grader",
    "TYPE_MATCH_MATRIX",
    "_verify_evidence_facts",
    "_compute_evidence_strength",
    "_evidence_strength_at_step",
]
