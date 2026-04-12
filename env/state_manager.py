"""Episode state management with session isolation and lightweight logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from models.contamination_spec import ContaminationSpec


@dataclass(slots=True)
class EpisodeState:
    """Mutable per-session episode state used by the environment runtime.

    Attributes:
        session_id: Unique id for one active episode session.
        experiment_id: Experiment identifier visible to the agent.
        task_id: Task bucket id selected at reset.
        seed: Seed used to generate deterministic synthetic data.
        step_count: Number of accepted actions consumed from budget.
        max_steps: Maximum step budget per episode.
        executed_queries: Investigative action types executed so far.
        revealed_data: Append-only map of action_type to revealed payload.
        invalid_action_count: Consecutive invalid action counter.
        episode_done: True after terminal verdict, budget exhaustion, or invalid limit.
        termination_reason: Categorical termination reason, when complete.
        episode_log: Full action/event log entries with timestamps.
        cumulative_reward: Running reward total for the session.
        spec: Hidden contamination specification.
        data: Synthetic data bundle returned by DataGenerator.
    """

    session_id: str
    experiment_id: str
    task_id: int
    seed: int
    step_count: int
    max_steps: int
    executed_queries: list[str] = field(default_factory=list)
    revealed_data: dict[str, Any] = field(default_factory=dict)
    invalid_action_count: int = 0
    episode_done: bool = False
    termination_reason: str | None = None
    episode_log: list[dict[str, Any]] = field(default_factory=list)
    cumulative_reward: float = 0.0
    spec: ContaminationSpec | None = None
    data: dict[str, Any] = field(default_factory=dict)
    budget: float = 10000.0
    budget_used: float = 0.0


class StateManager:
    """In-memory session state manager with append-only episode logging."""

    _sessions: dict[str, EpisodeState] = {}
    _log_dir = Path("logs/episodes")

    @classmethod
    def init(
        cls,
        *,
        task_id: int,
        seed: int,
        spec: ContaminationSpec,
        data: dict[str, Any],
        max_steps: int = 15,
    ) -> EpisodeState:
        """Create and store a new session state.

        Args:
            task_id: Selected task id.
            seed: Episode seed used for deterministic generation.
            spec: Hidden contamination spec.
            data: Full synthetic data payload from DataGenerator.
            max_steps: Step budget for the episode.

        Returns:
            The initialized episode state object.
        """
        session_id = f"session_{uuid4().hex[:12]}"
        state = EpisodeState(
            session_id=session_id,
            experiment_id=data["experiment_id"],
            task_id=task_id,
            seed=seed,
            step_count=0,
            max_steps=max_steps,
            spec=spec,
            data=data,
            budget=10000.0,
            budget_used=0.0,
        )
        cls._sessions[session_id] = state
        cls.log_event(
            state,
            event_type="episode_start",
            payload={"task_id": task_id, "seed": seed, "experiment_id": state.experiment_id},
        )
        return state

    @classmethod
    def get(cls, session_id: str) -> EpisodeState | None:
        """Fetch a session state by id.

        Args:
            session_id: Session identifier.

        Returns:
            EpisodeState if present, else None.
        """
        return cls._sessions.get(session_id)

    @classmethod
    def mark_invalid_action(
        cls,
        state: EpisodeState,
        error_message: str,
        *,
        action: dict[str, Any] | None = None,
        reward: float = 0.0,
        observation_delta: dict[str, Any] | None = None,
    ) -> None:
        """Increment invalid action counter and terminate when threshold is reached.

        Args:
            state: Current episode state.
            error_message: Validation error details for logging.
            action: Action payload that caused invalidation, if available.
            reward: Step reward associated with this invalid action.
            observation_delta: Observation changes resulting from this action.
        """
        state.invalid_action_count += 1
        if state.invalid_action_count >= 3:
            state.episode_done = True
            state.termination_reason = "invalid_action_limit"

        cls.log_event(
            state,
            event_type="invalid_action",
            payload={
                "action": action,
                "error": error_message,
                "reward": reward,
                "invalid_action_count": state.invalid_action_count,
                "termination_reason": state.termination_reason,
                "observation_delta": observation_delta or {},
            },
        )

    @classmethod
    def consume_step(cls, state: EpisodeState) -> None:
        """Consume one step from action budget and enforce budget-based termination.

        Args:
            state: Current episode state.
        """
        state.step_count += 1
        if state.step_count >= state.max_steps:
            state.episode_done = True
            if state.termination_reason is None:
                state.termination_reason = "budget_exhausted"

    @classmethod
    def log_event(cls, state: EpisodeState, event_type: str, payload: dict[str, Any]) -> None:
        """Append an event to memory log and persist a JSON-line log record.

        Args:
            state: Session state receiving the log event.
            event_type: Categorical event name.
            payload: Event payload dictionary.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        state.episode_log.append(event)
        cls._write_log_line(state.session_id, event)

    @classmethod
    def public_state(cls, state: EpisodeState) -> dict[str, Any]:
        """Return sanitized state representation for `/state` response.

        Args:
            state: Session state to serialize.

        Returns:
            Dictionary without hidden contamination spec fields.
        """
        return {
            "session_id": state.session_id,
            "step_count": state.step_count,
            "steps_remaining": max(state.max_steps - state.step_count, 0),
            "executed_queries": list(state.executed_queries),
            "episode_done": state.episode_done,
            "cumulative_reward": round(state.cumulative_reward, 4),
            "investigation_budget": state.budget,
            "budget_spent": round(state.budget_used, 2),
        }

    @classmethod
    def _write_log_line(cls, session_id: str, event: dict[str, Any]) -> None:
        """Persist one event JSON line to `logs/episodes/<session_id>.jsonl`.

        Args:
            session_id: Session id used for log file naming.
            event: Event payload to persist.
        """
        import json

        cls._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = cls._log_dir / f"{session_id}.jsonl"
        with log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(event, ensure_ascii=False) + "\n")

    @classmethod
    def get_episode_log(cls, session_id: str) -> list[dict[str, Any]]:
        """Return in-memory episode log for a given session.

        Args:
            session_id: Session identifier.

        Returns:
            Ordered in-memory event list for the session, or empty list.
        """
        state = cls.get(session_id)
        if state is None:
            return []
        return list(state.episode_log)

    @classmethod
    def read_persisted_episode_log(cls, session_id: str) -> list[dict[str, Any]]:
        """Read persisted JSONL episode log records for grading/replay.

        Args:
            session_id: Session identifier.

        Returns:
            List of decoded event dictionaries from disk.
        """
        import json

        log_path = cls._log_dir / f"{session_id}.jsonl"
        if not log_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with log_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events
