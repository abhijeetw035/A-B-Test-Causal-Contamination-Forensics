"""API routes wired to environment core modules with progressive reveal behavior."""

from copy import deepcopy
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, ValidationError

from env.action_executor import ActionExecutor
from env.data_generator import DataGenerator
from env.observation_builder import ObservationBuilder
from env.reward_engine import RewardEngine
from env.state_manager import StateManager
from env.synthetic_fixtures import CLEAN_EXPERIMENT_FIXTURE, CONTAMINATED_EXPERIMENT_FIXTURE
from models.action import AuditAction
from models.observation import ExperimentObservation
from tasks.task_generator import TaskGenerator


router = APIRouter()


class ResetRequest(BaseModel):
    """Request body for reset endpoint."""

    task_id: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=42)


class StepResult(BaseModel):
    """Response shape for a step call."""

    observation: ExperimentObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class SessionStateResponse(BaseModel):
    """Simplified session state for phase-1 stub."""

    session_id: str
    step_count: int
    steps_remaining: int
    executed_queries: list[str]
    episode_done: bool
    cumulative_reward: float


class MetadataResponse(BaseModel):
    """OpenEnv metadata endpoint response payload."""

    name: str
    description: str
    version: str
    mode: str
    endpoints: Dict[str, str]


def _build_mock_data_bundle(task_id: int, seed: int) -> dict[str, Any]:
    """Build a DataGenerator-compatible payload using static fixtures.

    This keeps ObservationBuilder/ActionExecutor unblocked while real generation
    logic is developed independently.
    """
    base_fixture = CLEAN_EXPERIMENT_FIXTURE if task_id == 4 else CONTAMINATED_EXPERIMENT_FIXTURE
    fixture = deepcopy(base_fixture)

    experiment_id = f"{fixture['experiment_id']}_t{task_id}_s{seed}"
    aggregate = fixture["aggregate_results"]
    control_count = int(aggregate["control_count"])
    treatment_count = int(aggregate["treatment_count"])

    query_payloads: dict[str, Any] = {
        "run_srm_check": {
            "expected_split": 0.5,
            "actual_split": round(treatment_count / max(control_count + treatment_count, 1), 6),
            "chi_square_statistic": 12.31 if task_id != 4 else 0.18,
            "p_value": 0.00045 if task_id != 4 else 0.671,
            "srm_detected": task_id != 4,
            "severity": "mild" if task_id != 4 else "none",
        },
        "query_temporal": [
            {
                "date": fixture["experiment_metadata"]["start_date"],
                "control_mean": aggregate["control_mean"],
                "treatment_mean": aggregate["treatment_mean"],
                "relative_lift": aggregate["relative_lift"],
                "control_count": control_count // 2,
                "treatment_count": treatment_count // 2,
            },
            {
                "date": fixture["experiment_metadata"]["end_date"],
                "control_mean": aggregate["control_mean"],
                "treatment_mean": aggregate["treatment_mean"],
                "relative_lift": aggregate["relative_lift"],
                "control_count": control_count - (control_count // 2),
                "treatment_count": treatment_count - (treatment_count // 2),
            },
        ],
        "query_subgroup": {
            "device_type": [
                {
                    "dimension": "device_type",
                    "value": "ios",
                    "control_mean": aggregate["control_mean"],
                    "treatment_mean": aggregate["treatment_mean"],
                    "relative_lift": aggregate["relative_lift"],
                    "control_count": max(control_count // 3, 1),
                    "treatment_count": max(treatment_count // 3, 1),
                }
            ],
            "country": [
                {
                    "dimension": "country",
                    "value": "us",
                    "control_mean": aggregate["control_mean"],
                    "treatment_mean": aggregate["treatment_mean"],
                    "relative_lift": aggregate["relative_lift"],
                    "control_count": max(control_count // 2, 1),
                    "treatment_count": max(treatment_count // 2, 1),
                }
            ],
            "enrollment_cohort": [
                {
                    "dimension": "enrollment_cohort",
                    "value": "days_1_3",
                    "control_mean": aggregate["control_mean"],
                    "treatment_mean": aggregate["treatment_mean"],
                    "relative_lift": aggregate["relative_lift"],
                    "control_count": max(control_count // 2, 1),
                    "treatment_count": max(treatment_count // 2, 1),
                }
            ],
            "user_segment": [
                {
                    "dimension": "user_segment",
                    "value": "returning",
                    "control_mean": aggregate["control_mean"],
                    "treatment_mean": aggregate["treatment_mean"],
                    "relative_lift": aggregate["relative_lift"],
                    "control_count": max(control_count // 2, 1),
                    "treatment_count": max(treatment_count // 2, 1),
                }
            ],
            "platform_version": [
                {
                    "dimension": "platform_version",
                    "value": "v2",
                    "control_mean": aggregate["control_mean"],
                    "treatment_mean": aggregate["treatment_mean"],
                    "relative_lift": aggregate["relative_lift"],
                    "control_count": max(control_count // 2, 1),
                    "treatment_count": max(treatment_count // 2, 1),
                }
            ],
        },
        "query_assignment_overlap": {
            "experiment_ids": [experiment_id, "exp_2024_pricing_011"],
            "overlap_fractions": {
                experiment_id: {
                    "control": 0.71 if task_id == 3 else 0.09,
                    "treatment": 0.28 if task_id == 3 else 0.11,
                }
            },
        },
        "check_network_exposure": {
            "control_all": 0.23 if task_id == 3 else 0.03,
            "control_high_degree": 0.34 if task_id == 3 else 0.05,
            "control_low_degree": 0.14 if task_id == 3 else 0.02,
        },
        "inspect_randomization": {
            "algorithm": "hash_mod_user_id",
            "seed": seed,
            "assignment_log_complete": True,
            "notes": "Fixture mode: replaying mock assignment traces.",
        },
        "query_secondary_metrics": {
            "session_length": {
                "control_mean": round(max(aggregate["control_mean"] - 0.03, 0.0001), 6),
                "treatment_mean": round(max(aggregate["treatment_mean"] - 0.03, 0.0001), 6),
                "relative_lift": aggregate["relative_lift"],
                "absolute_lift": aggregate["absolute_lift"],
                "p_value": aggregate["p_value"],
                "control_count": control_count,
                "treatment_count": treatment_count,
                "confidence_interval_lower": aggregate["confidence_interval_lower"],
                "confidence_interval_upper": aggregate["confidence_interval_upper"],
            },
            "revenue_per_user": {
                "control_mean": round(max(aggregate["control_mean"] - 0.08, 0.0001), 6),
                "treatment_mean": round(max(aggregate["treatment_mean"] - 0.08, 0.0001), 6),
                "relative_lift": aggregate["relative_lift"],
                "absolute_lift": aggregate["absolute_lift"],
                "p_value": aggregate["p_value"],
                "control_count": control_count,
                "treatment_count": treatment_count,
                "confidence_interval_lower": aggregate["confidence_interval_lower"],
                "confidence_interval_upper": aggregate["confidence_interval_upper"],
            },
        },
        "compute_mde": {
            "observed_effect_size": aggregate["absolute_lift"],
            "required_sample_per_arm": int(max(control_count, treatment_count) * (2 if task_id != 4 else 1)),
            "actual_sample_per_arm": int(min(control_count, treatment_count)),
            "achieved_power": 0.79 if task_id != 4 else 0.92,
            "underpowered": task_id != 4,
        },
        "peer_experiment_list": [
            {
                "experiment_id": "exp_2024_pricing_011",
                "randomization_unit": "user_id",
                "time_overlap": True,
                "owner": "monetization_team",
            }
        ],
        "simulate_counterfactual": {
            "unconfounded_ate_estimate": aggregate["absolute_lift"] - 0.005,
            "confounding_robustness_value": 0.81,
            "methodology": "Double Machine Learning (Causal Forest)",
        },
        "request_expert_review": {
            "hint": "The Staff Data Scientist says: 'Look closely at the randomization constraints and network interference. SUTVA might be violated.'",
            "expert": "Dr. Sarah",
        },
    }

    return {
        "experiment_id": experiment_id,
        "primary_metric": fixture["primary_metric"],
        "aggregate_results": fixture["aggregate_results"],
        "experiment_metadata": fixture["experiment_metadata"],
        "available_queries": fixture["available_queries"],
        "query_payloads": query_payloads,
    }


def _compute_observation_delta(previous: ExperimentObservation, current: ExperimentObservation) -> dict[str, Any]:
    """Compute a compact observation delta for logging.

    Args:
        previous: Observation prior to handling an action.
        current: Observation after handling an action.

    Returns:
        Dictionary with newly revealed fields and step counter changes.
    """
    prev_dump = previous.model_dump()
    curr_dump = current.model_dump()

    newly_revealed = [
        key
        for key, value in curr_dump.items()
        if value is not None and prev_dump.get(key) is None
    ]

    return {
        "newly_revealed_fields": newly_revealed,
        "steps_taken_before": prev_dump.get("steps_taken", 0),
        "steps_taken_after": curr_dump.get("steps_taken", 0),
        "steps_remaining_before": prev_dump.get("steps_remaining", 0),
        "steps_remaining_after": curr_dump.get("steps_remaining", 0),
    }

@router.get("/health")
def health() -> Dict[str, str]:
    """Health probe endpoint for container/platform checks."""

    return {"status": "healthy", "legacy_status": "ok"}


@router.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return lightweight environment metadata for OpenEnv runtime validation."""

    return MetadataResponse(
        name="ab-test-contamination-forensics",
        description="Environment for auditing A/B test contamination and experiment validity.",
        version="0.1.0",
        mode="simulation",
        endpoints={
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "mcp": "POST /mcp",
        },
    )


@router.get("/schema")
def schema() -> Dict[str, Any]:
    """Expose action/observation/state schemas for OpenEnv runtime validation."""

    return {
        "action": AuditAction.model_json_schema(),
        "observation": ExperimentObservation.model_json_schema(),
        "state": SessionStateResponse.model_json_schema(),
    }


@router.post("/mcp")
def mcp_rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal JSON-RPC endpoint stub for OpenEnv runtime validation probes."""

    rpc_id = payload.get("id")
    method = payload.get("method")

    if method == "health":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"status": "healthy"},
        }

    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "status": "ok",
            "message": "Minimal MCP stub endpoint",
        },
    }


@router.post("/reset", response_model=ExperimentObservation)
def reset(payload: ResetRequest = Body(default={"task_id": 1, "seed": 42})) -> ExperimentObservation:
    """Reset session using task sampling, synthetic data generation, and state init."""

    if isinstance(payload, dict):
        payload = ResetRequest(**payload)

    spec = TaskGenerator.sample(task_id=payload.task_id, seed=payload.seed)

    data_mode = "real_generator"
    try:
        data = DataGenerator.generate(spec=spec, seed=payload.seed)
    except Exception:  # pragma: no cover - fallback safety net
        data = _build_mock_data_bundle(task_id=payload.task_id, seed=payload.seed)
        data_mode = "mock_fixture_fallback"

    state = StateManager.init(task_id=payload.task_id, seed=payload.seed, spec=spec, data=data, max_steps=15)
    StateManager.log_event(state, event_type="data_mode_selected", payload={"mode": data_mode})
    return ObservationBuilder.build_initial(state)


@router.post("/step", response_model=StepResult)
def step(action: AuditAction, session_id: str) -> StepResult:
    """Execute one environment step using validation/execution core components."""

    state = StateManager.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    previous_observation = ObservationBuilder.build_updated(state)

    try:
        validated_action = AuditAction(**action.model_dump())
    except ValidationError as exc:
        observation = ObservationBuilder.build_updated(state)
        observation_delta = _compute_observation_delta(previous_observation, observation)
        StateManager.mark_invalid_action(
            state,
            str(exc),
            action=action.model_dump(),
            reward=0.0,
            observation_delta=observation_delta,
        )
        return StepResult(
            observation=observation,
            reward=0.0,
            done=state.episode_done,
            info={"error": str(exc), "termination_reason": state.termination_reason},
        )

    if state.episode_done:
        return StepResult(
            observation=ObservationBuilder.build_updated(state),
            reward=0.0,
            done=True,
            info={"message": "episode already terminated"},
        )

    execution = ActionExecutor.execute(validated_action, state)
    if not execution.accepted:
        observation = ObservationBuilder.build_updated(state)
        observation_delta = _compute_observation_delta(previous_observation, observation)
        StateManager.mark_invalid_action(
            state,
            execution.error or "Invalid action",
            action=validated_action.model_dump(),
            reward=0.0,
            observation_delta=observation_delta,
        )
        return StepResult(
            observation=observation,
            reward=0.0,
            done=state.episode_done,
            info={"error": execution.error, "termination_reason": state.termination_reason},
        )

    if execution.is_duplicate:
        reward = execution.reward_delta
        state.cumulative_reward += reward
        observation = ObservationBuilder.build_updated(state)
        observation_delta = _compute_observation_delta(previous_observation, observation)
        StateManager.log_event(
            state,
            event_type="duplicate_action",
            payload={
                "action": validated_action.model_dump(),
                "reward": reward,
                "done": state.episode_done,
                "termination_reason": state.termination_reason,
                "observation_delta": observation_delta,
            },
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={"cached": True, "cumulative_reward": round(state.cumulative_reward, 4)},
        )

    state.invalid_action_count = 0

    if execution.revealed_key is not None:
        state.revealed_data[execution.revealed_key] = execution.revealed_value
        if execution.revealed_key not in state.executed_queries:
            state.executed_queries.append(execution.revealed_key)

    reward_result = RewardEngine.compute(validated_action, state, state.spec)

    StateManager.consume_step(state)

    if execution.is_terminal:
        state.episode_done = True
        state.termination_reason = "agent_verdict"

    state.budget_used += execution.cost

    if state.budget_used >= state.budget and not execution.is_terminal:
        state.episode_done = True
        state.termination_reason = "budget_exhausted"

    reward = reward_result.step_reward
    state.cumulative_reward += reward

    observation = ObservationBuilder.build_updated(state)
    observation_delta = _compute_observation_delta(previous_observation, observation)
    StateManager.log_event(
        state,
        event_type="step",
        payload={
            "action": validated_action.model_dump(),
            "reward": reward,
            "done": state.episode_done,
            "termination_reason": state.termination_reason,
            "steps_taken": state.step_count,
            "observation_delta": observation_delta,
            "reward_components": reward_result.components,
        },
    )
    return StepResult(
        observation=observation,
        reward=reward,
        done=state.episode_done,
        info={
            "termination_reason": state.termination_reason,
            "cumulative_reward": round(state.cumulative_reward, 4),
            "reward_components": reward_result.components,
        },
    )


@router.get("/state", response_model=SessionStateResponse)
def state(session_id: str) -> SessionStateResponse:
    """Return phase-1 in-memory session state."""

    session = StateManager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return SessionStateResponse(**StateManager.public_state(session))
