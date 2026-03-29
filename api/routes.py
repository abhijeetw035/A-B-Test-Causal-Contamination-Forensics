"""Barebones API routes with hardcoded environment responses."""

from datetime import date
from typing import Any, Dict, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.action import AuditAction
from models.observation import AggregateResult, ExperimentMeta, ExperimentObservation


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


class SessionState(BaseModel):
    """Simplified session state for phase-1 stub."""

    session_id: str
    step_count: int
    steps_remaining: int
    executed_queries: list[str]
    episode_done: bool
    cumulative_reward: float


SESSIONS: Dict[str, SessionState] = {}


def _available_queries() -> list[str]:
    """Return complete query/action catalog as required by the blueprint."""

    return [
        "query_subgroup",
        "query_temporal",
        "run_srm_check",
        "query_assignment_overlap",
        "check_network_exposure",
        "inspect_randomization",
        "query_secondary_metrics",
        "compute_mde",
        "flag_contamination",
        "approve_result",
        "request_rerun",
    ]


def _hardcoded_observation(session_id: str, steps_taken: int = 0) -> ExperimentObservation:
    """Build a deterministic hardcoded observation for scaffold/testing."""

    max_steps = 15
    return ExperimentObservation(
        session_id=session_id,
        experiment_id="exp_2024_growth_007",
        primary_metric="D7 retention rate",
        aggregate_results=AggregateResult(
            control_mean=0.412,
            treatment_mean=0.433,
            relative_lift=0.051,
            absolute_lift=0.021,
            p_value=0.03,
            control_count=61847,
            treatment_count=48203,
            confidence_interval_lower=0.005,
            confidence_interval_upper=0.038,
        ),
        experiment_metadata=ExperimentMeta(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 2, 15),
            targeting_rule="all_users_18+ in US",
            intended_split=0.50,
            randomization_unit="user_id",
            platform="mobile_ios",
            experiment_owner="growth_team",
            hypothesis="Improved onboarding increases D7 retention",
        ),
        available_queries=_available_queries(),
        steps_taken=steps_taken,
        steps_remaining=max_steps - steps_taken,
    )


@router.get("/health")
def health() -> Dict[str, str]:
    """Health probe endpoint for container/platform checks."""

    return {"status": "ok"}


@router.post("/reset", response_model=ExperimentObservation)
def reset(payload: ResetRequest) -> ExperimentObservation:
    """Reset environment state and return a hardcoded initial observation."""

    session_id = f"session_{uuid4().hex[:12]}"
    SESSIONS[session_id] = SessionState(
        session_id=session_id,
        step_count=0,
        steps_remaining=15,
        executed_queries=[],
        episode_done=False,
        cumulative_reward=0.0,
    )
    _ = payload  # kept for API compatibility with future implementation
    return _hardcoded_observation(session_id=session_id, steps_taken=0)


@router.post("/step", response_model=StepResult)
def step(action: AuditAction, session_id: str) -> StepResult:
    """Advance one step using dummy logic and return hardcoded result."""

    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    if state.episode_done:
        return StepResult(
            observation=_hardcoded_observation(session_id=session_id, steps_taken=state.step_count),
            reward=0.0,
            done=True,
            info={"message": "episode already terminated"},
        )

    state.step_count += 1
    state.steps_remaining = max(0, 15 - state.step_count)

    terminal_action: tuple[str, ...] = ("flag_contamination", "approve_result", "request_rerun")
    if action.action_type not in terminal_action:
        state.executed_queries.append(action.action_type)

    done = action.action_type in terminal_action or state.step_count >= 15
    reward = -0.01 if not done else 0.0

    if done:
        state.episode_done = True

    state.cumulative_reward += reward

    observation = _hardcoded_observation(session_id=session_id, steps_taken=state.step_count)
    return StepResult(
        observation=observation,
        reward=reward,
        done=done,
        info={
            "termination_reason": "agent_verdict" if action.action_type in terminal_action else None,
            "cumulative_reward": round(state.cumulative_reward, 4),
        },
    )


@router.get("/state", response_model=SessionState)
def state(session_id: str) -> SessionState:
    """Return phase-1 in-memory session state."""

    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return session
