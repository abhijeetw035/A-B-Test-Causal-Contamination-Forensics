"""Reward computation for A/B test contamination forensics episodes."""

from __future__ import annotations

from typing import Any

from models.action import AuditAction
from models.contamination_spec import ContaminationSpec
from models.reward import StepReward


TYPE_MATCH_MATRIX: dict[tuple[str, str], float] = {
    ("srm", "srm"): 1.0,
    ("srm", "underpowered_overclaim"): 0.3,
    ("underpowered_overclaim", "srm"): 0.3,
    ("sutva_violation", "network_spillover"): 0.5,
    ("network_spillover", "sutva_violation"): 0.5,
    ("simpsons_paradox", "multiple_testing"): 0.2,
    ("novelty_effect", "simpsons_paradox"): 0.1,
}

INVESTIGATIVE_ACTIONS: set[str] = {
    "query_subgroup",
    "query_temporal",
    "run_srm_check",
    "query_assignment_overlap",
    "check_network_exposure",
    "inspect_randomization",
    "query_secondary_metrics",
    "compute_mde",
}

TERMINAL_ACTIONS: set[str] = {"flag_contamination", "approve_result", "request_rerun"}

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


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to an inclusive numeric range."""
    return max(low, min(high, value))


def _query_is_relevant(action: AuditAction, spec: ContaminationSpec) -> bool:
    """Return whether an investigative action is relevant to the contamination type.

    Args:
        action: Current action payload.
        spec: Hidden contamination specification.

    Returns:
        True if the action is in the relevant query set for this contamination type.
    """
    relevant = RELEVANT_QUERY_MAP.get(spec.contamination_type, set())
    return action.action_type in relevant


def _query_reveals_signal(action: AuditAction, spec: ContaminationSpec) -> bool:
    """Estimate whether the action is expected to reveal contamination evidence.

    Args:
        action: Current action payload.
        spec: Hidden contamination specification.

    Returns:
        True when the query should produce a clear signal for the current spec.
    """
    signal_map: dict[str, set[str]] = {
        "srm": {"run_srm_check", "inspect_randomization"},
        "sutva_violation": {"query_assignment_overlap", "check_network_exposure", "compute_mde"},
        "novelty_effect": {"query_temporal"},
        "simpsons_paradox": {"query_subgroup", "query_temporal", "query_secondary_metrics"},
        "network_spillover": {"check_network_exposure"},
        "multiple_testing": {"query_secondary_metrics"},
        "underpowered_overclaim": {"compute_mde"},
    }
    return action.action_type in signal_map.get(spec.contamination_type, set())


def _compute_evidence_strength(executed_queries: list[str], spec: ContaminationSpec) -> float:
    """Compute evidence strength from required and relevant query coverage.

    Args:
        executed_queries: Query types executed so far.
        spec: Hidden contamination specification.

    Returns:
        Evidence strength in [0.0, 1.0].
    """
    executed = {query for query in executed_queries if query in INVESTIGATIVE_ACTIONS}
    required = set(spec.required_queries or [])
    relevant = RELEVANT_QUERY_MAP.get(spec.contamination_type, set())

    required_coverage = 1.0 if not required else len(executed & required) / len(required)
    relevant_coverage = 0.0 if not relevant else len(executed & relevant) / len(relevant)
    return _clamp((0.7 * required_coverage) + (0.3 * relevant_coverage), 0.0, 1.0)


class RewardEngine:
    """Deterministic reward engine for step-level and terminal scoring."""

    @staticmethod
    def compute(action: AuditAction, state: Any, spec: ContaminationSpec) -> StepReward:
        """Compute reward for one step with component-level attribution.

        Args:
            action: Current action payload.
            state: Current episode state with executed query history.
            spec: Hidden contamination specification.

        Returns:
            Structured StepReward with step reward, component breakdown, and cumulative reward.
        """
        components: dict[str, float] = {
            "investigation": 0.0,
            "duplicate_penalty": 0.0,
            "calibration": 0.0,
            "terminal": 0.0,
            "efficiency": 0.0,
        }

        executed_queries = list(getattr(state, "executed_queries", []))

        if action.action_type in INVESTIGATIVE_ACTIONS:
            if _query_is_relevant(action, spec):
                components["investigation"] += 0.08
                if _query_reveals_signal(action, spec):
                    components["investigation"] += 0.06
            else:
                components["investigation"] -= 0.03

            if action.action_type in executed_queries:
                components["duplicate_penalty"] -= 0.03

        evidence_strength = _compute_evidence_strength(executed_queries, spec)
        calibration_error = abs(float(action.confidence) - evidence_strength)
        calibration_score = max(0.0, 1.0 - calibration_error * 2.0)
        components["calibration"] += 0.05 * calibration_score

        if action.action_type == "flag_contamination":
            claimed_type = str(action.parameters.get("contamination_type", ""))
            if spec.contamination_type == "clean":
                components["terminal"] -= 0.40
            else:
                type_score = TYPE_MATCH_MATRIX.get(
                    (claimed_type, spec.contamination_type),
                    1.0 if claimed_type == spec.contamination_type else 0.0,
                )
                components["terminal"] += 0.30 * type_score

                evidence_facts = action.parameters.get("evidence_facts", [])
                evidence_count = len(evidence_facts) if isinstance(evidence_facts, list) else 0
                required_count = len(spec.ground_truth_evidence or {})
                evidence_score = 0.0 if required_count == 0 else min(evidence_count / required_count, 1.0)
                components["terminal"] += 0.20 * evidence_score

                if "estimated_true_effect" in action.parameters and spec.true_effect_size != spec.visible_effect_size:
                    estimate = float(action.parameters["estimated_true_effect"])
                    range_size = abs(spec.visible_effect_size - spec.true_effect_size) + 1e-6
                    estimate_accuracy = max(0.0, 1.0 - abs(estimate - spec.true_effect_size) / range_size)
                    components["terminal"] += 0.15 * estimate_accuracy

                terminal_calibration = 1.0 - abs(float(action.confidence) - evidence_strength)
                components["terminal"] += 0.10 * _clamp(terminal_calibration, 0.0, 1.0)

        elif action.action_type == "approve_result":
            if spec.contamination_type == "clean":
                components["terminal"] += 0.35
                terminal_calibration = 1.0 - abs(float(action.confidence) - 0.9)
                components["terminal"] += 0.10 * _clamp(terminal_calibration, 0.0, 1.0)
            else:
                components["terminal"] -= 0.40

        elif action.action_type == "request_rerun":
            if spec.contamination_type != "clean":
                components["terminal"] += 0.10
            else:
                components["terminal"] -= 0.10

        if getattr(state, "episode_done", False):
            step_count = int(getattr(state, "step_count", 0))
            steps_over_optimal = max(0, step_count - (spec.optimal_investigation_steps * 2))
            components["efficiency"] -= 0.03 * steps_over_optimal

        step_reward = round(sum(components.values()), 4)
        cumulative_base = float(getattr(state, "cumulative_reward", 0.0))
        cumulative_reward = round(cumulative_base + step_reward, 4)

        return StepReward(
            step_reward=step_reward,
            components={k: round(v, 4) for k, v in components.items()},
            cumulative_reward=cumulative_reward,
            reasoning="Deterministic reward from investigation relevance, calibration, terminal outcome, and efficiency.",
        )
