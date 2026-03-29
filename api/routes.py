"""API routes wired to environment core modules with progressive reveal behavior."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from env.action_executor import ActionExecutor
from env.data_generator import DataGenerator
from env.observation_builder import ObservationBuilder
from env.state_manager import StateManager
from models.action import AuditAction
from models.observation import ExperimentObservation
from tasks.task_generator import TaskGenerator


router = APIRouter()


class ResetRequest(BaseModel):
    """Request body for reset endpoint."""

    task_id: int = Field(default=1, ge=1, le=4)
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

@router.get("/health")
def health() -> Dict[str, str]:
    """Health probe endpoint for container/platform checks."""

    return {"status": "ok"}


@router.post("/reset", response_model=ExperimentObservation)
def reset(payload: ResetRequest) -> ExperimentObservation:
    """Reset session using task sampling, synthetic data generation, and state init."""

    spec = TaskGenerator.sample(task_id=payload.task_id, seed=payload.seed)
    data = DataGenerator.generate(spec=spec, seed=payload.seed)
    state = StateManager.init(task_id=payload.task_id, seed=payload.seed, spec=spec, data=data, max_steps=15)
    return ObservationBuilder.build_initial(state)


@router.post("/step", response_model=StepResult)
def step(action: AuditAction, session_id: str) -> StepResult:
    """Execute one environment step using validation/execution core components."""

    state = StateManager.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    try:
        validated_action = AuditAction(**action.model_dump())
    except ValidationError as exc:
        StateManager.mark_invalid_action(state, str(exc))
        observation = ObservationBuilder.build_updated(state)
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
        StateManager.mark_invalid_action(state, execution.error or "Invalid action")
        observation = ObservationBuilder.build_updated(state)
        return StepResult(
            observation=observation,
            reward=0.0,
            done=state.episode_done,
            info={"error": execution.error, "termination_reason": state.termination_reason},
        )

    if execution.is_duplicate:
        state.cumulative_reward += execution.reward_delta
        StateManager.log_event(
            state,
            event_type="duplicate_action",
            payload={
                "action": validated_action.model_dump(),
                "reward": execution.reward_delta,
            },
        )
        return StepResult(
            observation=ObservationBuilder.build_updated(state),
            reward=execution.reward_delta,
            done=False,
            info={"cached": True, "cumulative_reward": round(state.cumulative_reward, 4)},
        )

    state.invalid_action_count = 0

    if execution.revealed_key is not None:
        state.revealed_data[execution.revealed_key] = execution.revealed_value
        if execution.revealed_key not in state.executed_queries:
            state.executed_queries.append(execution.revealed_key)

    StateManager.consume_step(state)

    if execution.is_terminal:
        state.episode_done = True
        state.termination_reason = "agent_verdict"

    reward = execution.reward_delta
    state.cumulative_reward += reward

    observation = ObservationBuilder.build_updated(state)
    StateManager.log_event(
        state,
        event_type="step",
        payload={
            "action": validated_action.model_dump(),
            "reward": reward,
            "done": state.episode_done,
            "termination_reason": state.termination_reason,
            "steps_taken": state.step_count,
        },
    )
    return StepResult(
        observation=observation,
        reward=reward,
        done=state.episode_done,
        info={
            "termination_reason": state.termination_reason,
            "cumulative_reward": round(state.cumulative_reward, 4),
        },
    )


@router.get("/state", response_model=SessionStateResponse)
def state(session_id: str) -> SessionStateResponse:
    """Return phase-1 in-memory session state."""

    session = StateManager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return SessionStateResponse(**StateManager.public_state(session))
