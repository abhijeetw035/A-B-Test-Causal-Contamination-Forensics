"""Deterministic grading stubs."""

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


RELEVANT_QUERY_MAP: dict[str, set[str]] = {
    "clean": {"run_srm_check", "query_temporal", "query_assignment_overlap"},
    "srm": {"run_srm_check", "inspect_randomization"},
    "sutva_violation": {"query_assignment_overlap", "check_network_exposure", "compute_mde"},
    "novelty_effect": {"query_temporal", "query_secondary_metrics"},
    "simpsons_paradox": {"query_temporal", "query_subgroup", "query_secondary_metrics"},
    "network_spillover": {"check_network_exposure", "query_assignment_overlap", "query_subgroup"},
    "multiple_testing": {"query_secondary_metrics", "compute_mde"},
    "underpowered_overclaim": {"compute_mde", "query_temporal"},
}


TERMINAL_ACTIONS: set[str] = {"flag_contamination", "approve_result", "request_rerun"}


def _extract_actions(episode_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract action entries from either flat or event-based episode logs.

    Args:
        episode_log: Raw episode records.

    Returns:
        Normalized action dictionaries with action_type/parameters/confidence.
    """
    actions: list[dict[str, Any]] = []
    for entry in episode_log:
        if "action_type" in entry:
            actions.append(entry)
            continue

        payload = entry.get("payload", {})
        action = payload.get("action")
        if isinstance(action, dict) and "action_type" in action:
            actions.append(action)
    return actions


def _extract_final_action(episode_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the final terminal action if present, else latest action.

    Args:
        episode_log: Raw episode records.

    Returns:
        Best-effort terminal/final action payload dictionary.
    """
    actions = _extract_actions(episode_log)
    if not actions:
        return {}

    for action in reversed(actions):
        if action.get("action_type") in TERMINAL_ACTIONS:
            return action
    return actions[-1]


def _query_is_relevant(action_type: str, spec: ContaminationSpec) -> bool:
    """Check whether an investigative query is relevant for the contamination type.

    Args:
        action_type: Query action type.
        spec: Hidden contamination specification.

    Returns:
        True when the query is informative for this contamination type.
    """
    return action_type in RELEVANT_QUERY_MAP.get(spec.contamination_type, set())


def _compute_evidence_strength(queries_executed: list[str], spec: ContaminationSpec) -> float:
    """Estimate evidence strength from required and relevant query coverage.

    Args:
        queries_executed: Ordered list of executed action types.
        spec: Hidden contamination specification.

    Returns:
        Continuous evidence score in [0.0, 1.0].
    """
    executed = {q for q in queries_executed if q not in TERMINAL_ACTIONS}
    required = set(spec.required_queries or [])
    relevant = RELEVANT_QUERY_MAP.get(spec.contamination_type, set())

    required_coverage = 1.0 if not required else len(executed & required) / len(required)
    relevant_coverage = 0.0 if not relevant else len(executed & relevant) / len(relevant)
    strength = 0.7 * required_coverage + 0.3 * relevant_coverage
    return max(0.0, min(1.0, strength))


def _verify_evidence_facts(claimed_facts: list[str], spec: ContaminationSpec) -> float:
    """Verify factual claims against ground-truth evidence with numeric tolerance.

    Args:
        claimed_facts: List of agent-provided factual strings.
        spec: Hidden contamination specification containing ground truth evidence.

    Returns:
        Fraction of checkable claims that match the ground truth.
    """
    if not claimed_facts:
        return 0.0

    gt = spec.ground_truth_evidence or {}
    if not gt:
        return 0.0

    checked = 0
    matched = 0

    for fact in claimed_facts:
        lower_fact = fact.lower()
        for key, value in gt.items():
            key_phrase = key.replace("_", " ")
            if key_phrase not in lower_fact:
                continue

            checked += 1

            if isinstance(value, bool):
                expected = "true" if value else "false"
                if expected in lower_fact:
                    matched += 1
                continue

            if isinstance(value, str):
                if value.lower() in lower_fact:
                    matched += 1
                continue

            if isinstance(value, (int, float)):
                nums = re.findall(r"[-+]?\d*\.?\d+", fact)
                if not nums:
                    continue
                candidate = float(nums[-1])
                if "%" in fact:
                    candidate /= 100.0

                tolerance = 0.10 * max(abs(float(value)), 1e-6)
                if abs(candidate - float(value)) <= tolerance:
                    matched += 1
                continue

    return 0.0 if checked == 0 else matched / checked


class Grader:
    """Grades completed episode logs against hidden contamination specs."""

    @staticmethod
    def grade_episode(
        episode_log: list[dict[str, Any]],
        spec: ContaminationSpec,
    ) -> dict[str, Any]:
        """Grade a full episode log and return a structured score payload.

        Args:
            episode_log: Ordered list of action and reward records for an episode.
            spec: Hidden contamination specification for ground-truth evaluation.

        Returns:
            A score dictionary containing final score and breakdown fields.
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
            action_types_so_far: list[str] = []
            for action in actions:
                action_type = str(action.get("action_type", ""))
                action_types_so_far.append(action_type)
                confidence = action.get("confidence")
                if isinstance(confidence, (int, float)):
                    expected = _compute_evidence_strength(action_types_so_far, spec)
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

        final_score = sum(breakdown[key] * weights[key] for key in weights)
        final_score = max(0.0, min(1.0, final_score))

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
    "_compute_evidence_strength",
    "_query_is_relevant",
    "_verify_evidence_facts",
]
