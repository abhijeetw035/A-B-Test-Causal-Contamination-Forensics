"""Action validation and execution against per-session episode state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.action import AuditAction

from env.state_manager import EpisodeState


INVESTIGATIVE_ACTIONS: tuple[str, ...] = (
    "query_subgroup",
    "query_temporal",
    "run_srm_check",
    "query_assignment_overlap",
    "check_network_exposure",
    "inspect_randomization",
    "query_secondary_metrics",
    "compute_mde",
    "simulate_counterfactual",
    "request_expert_review",
)

TERMINAL_ACTIONS: tuple[str, ...] = (
    "flag_contamination",
    "approve_result",
    "request_rerun",
)

ALLOWED_SUBGROUP_DIMENSIONS: tuple[str, ...] = (
    "device_type",
    "country",
    "enrollment_cohort",
    "user_segment",
    "platform_version",
)

ACTION_COSTS: dict[str, float] = {
    "run_srm_check": 50.0,
    "query_temporal": 100.0,
    "compute_mde": 50.0,
    "query_subgroup": 300.0,
    "query_secondary_metrics": 400.0,
    "query_assignment_overlap": 1000.0,
    "check_network_exposure": 2000.0,
    "inspect_randomization": 1500.0,
    "simulate_counterfactual": 2500.0,
    "request_expert_review": 3000.0,
    "flag_contamination": 0.0,
    "approve_result": 0.0,
    "request_rerun": 0.0,
}


@dataclass(slots=True)
class ActionExecutionResult:
    """Structured action execution outcome consumed by route logic."""

    accepted: bool
    is_duplicate: bool
    is_terminal: bool
    error: str | None
    revealed_key: str | None
    revealed_value: Any
    reward_delta: float
    cost: float


class ActionExecutor:
    """Validates and executes one `AuditAction` against episode state."""

    @staticmethod
    def execute(action: AuditAction, state: EpisodeState) -> ActionExecutionResult:
        """Validate preconditions and resolve reveal payload for the given action.

        Args:
            action: Parsed agent action model.
            state: Mutable session state.

        Returns:
            `ActionExecutionResult` describing validity, payload, and reward delta.
        """
        if state.episode_done:
            return ActionExecutionResult(
                accepted=False,
                is_duplicate=False,
                is_terminal=False,
                error="Episode already terminated.",
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=0.0,
            )

        cost = ACTION_COSTS.get(action.action_type, 0.0)
        if state.budget_used + cost > state.budget:
            return ActionExecutionResult(
                accepted=False,
                is_duplicate=False,
                is_terminal=False,
                error=f"Budget exhausted. Required: ${cost}, Available: ${state.budget - state.budget_used}",
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=0.0,
            )

        if state.step_count >= state.max_steps:
            return ActionExecutionResult(
                accepted=False,
                is_duplicate=False,
                is_terminal=False,
                error="Action budget exhausted.",
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=0.0,
            )

        if action.action_type in INVESTIGATIVE_ACTIONS and action.action_type in state.executed_queries:
            return ActionExecutionResult(
                accepted=True,
                is_duplicate=True,
                is_terminal=False,
                error=None,
                revealed_key=action.action_type,
                revealed_value=state.revealed_data.get(action.action_type),
                reward_delta=-0.03,
                cost=0.0,
            )

        validation_error = ActionExecutor._validate_action_parameters(action)
        if validation_error is not None:
            return ActionExecutionResult(
                accepted=False,
                is_duplicate=False,
                is_terminal=False,
                error=validation_error,
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=0.0,
            )

        if action.action_type in TERMINAL_ACTIONS:
            return ActionExecutionResult(
                accepted=True,
                is_duplicate=False,
                is_terminal=True,
                error=None,
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=cost,
            )

        query_payloads = state.data.get("query_payloads", {})
        if action.action_type == "query_subgroup":
            dimension = action.parameters["dimension"]
            subgroup_map: dict[str, Any] = query_payloads.get("query_subgroup", {})
            return ActionExecutionResult(
                accepted=True,
                is_duplicate=False,
                is_terminal=False,
                error=None,
                revealed_key="query_subgroup",
                revealed_value={dimension: subgroup_map.get(dimension, [])},
                reward_delta=-0.01,
                cost=cost,
            )

        payload = query_payloads.get(action.action_type)
        if payload is None:
            return ActionExecutionResult(
                accepted=False,
                is_duplicate=False,
                is_terminal=False,
                error=f"No payload configured for action '{action.action_type}'.",
                revealed_key=None,
                revealed_value=None,
                reward_delta=0.0,
                cost=0.0,
            )

        return ActionExecutionResult(
            accepted=True,
            is_duplicate=False,
            is_terminal=False,
            error=None,
            revealed_key=action.action_type,
            revealed_value=payload,
            reward_delta=-0.01,
            cost=cost,
        )

    @staticmethod
    def _validate_action_parameters(action: AuditAction) -> str | None:
        """Validate action-type specific parameter constraints.

        Args:
            action: Action model to validate.

        Returns:
            Error message when invalid; otherwise None.
        """
        params = action.parameters

        if action.action_type == "query_subgroup":
            dimension = params.get("dimension")
            if not isinstance(dimension, str):
                return "query_subgroup requires parameters.dimension as a string."
            if dimension not in ALLOWED_SUBGROUP_DIMENSIONS:
                return f"Unsupported subgroup dimension '{dimension}'."

        if action.action_type == "flag_contamination":
            required = ("contamination_type", "evidence_facts", "recommended_action")
            missing = [field for field in required if field not in params]
            if missing:
                return f"flag_contamination missing required fields: {', '.join(missing)}"
            if not isinstance(params.get("evidence_facts"), list):
                return "flag_contamination.parameters.evidence_facts must be a list of strings."

        if action.action_type == "request_rerun":
            if "reason" not in params or "recommended_changes" not in params:
                return "request_rerun requires parameters.reason and parameters.recommended_changes."
            if not isinstance(params.get("recommended_changes"), list):
                return "request_rerun.parameters.recommended_changes must be a list of strings."

        return None
