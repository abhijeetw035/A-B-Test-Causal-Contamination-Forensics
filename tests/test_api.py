"""API behavior tests for session isolation, duplicate handling, and termination conditions."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure project root is importable when tests are run from different cwd contexts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app
from env.state_manager import StateManager


client = TestClient(app)


def _valid_action(action_type: str, parameters: dict | None = None) -> dict:
    """Build a valid action payload for API tests.

    Args:
        action_type: Supported action name.
        parameters: Optional action parameters.

    Returns:
        Dictionary payload accepted by `/step`.
    """
    return {
        "action_type": action_type,
        "parameters": parameters or {},
        "reasoning": "Running structured audit step for diagnosis.",
        "confidence": 0.8,
    }


def _create_session(task_id: int = 1, seed: int = 42) -> str:
    """Create a new session via `/reset` and return its session id."""
    response = client.post("/reset", json={"task_id": task_id, "seed": seed})
    assert response.status_code == 200
    payload = response.json()
    return payload["session_id"]


def setup_function() -> None:
    """Reset global in-memory session store between tests."""
    StateManager._sessions.clear()


def test_concurrent_sessions_do_not_bleed_state() -> None:
    """Ensure actions on one session never mutate the other session state."""
    session_a = _create_session(task_id=1, seed=101)
    session_b = _create_session(task_id=4, seed=202)

    step_resp = client.post(f"/step?session_id={session_a}", json=_valid_action("run_srm_check"))
    assert step_resp.status_code == 200
    assert step_resp.json()["done"] is False

    state_a = client.get(f"/state?session_id={session_a}")
    state_b = client.get(f"/state?session_id={session_b}")
    assert state_a.status_code == 200
    assert state_b.status_code == 200

    payload_a = state_a.json()
    payload_b = state_b.json()

    assert payload_a["step_count"] == 1
    assert payload_a["executed_queries"] == ["run_srm_check"]
    assert payload_b["step_count"] == 0
    assert payload_b["executed_queries"] == []


def test_duplicate_query_returns_cached_data_without_consuming_step() -> None:
    """Second call to same investigative action should be cached and penalized lightly."""
    session_id = _create_session(task_id=1, seed=303)

    first = client.post(f"/step?session_id={session_id}", json=_valid_action("run_srm_check"))
    second = client.post(f"/step?session_id={session_id}", json=_valid_action("run_srm_check"))

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["reward"] == -0.01
    assert second_payload["reward"] == -0.03
    assert second_payload["info"]["cached"] is True

    state = client.get(f"/state?session_id={session_id}")
    assert state.status_code == 200
    state_payload = state.json()
    assert state_payload["step_count"] == 1
    assert state_payload["executed_queries"] == ["run_srm_check"]


def test_invalid_action_limit_terminates_after_three_failures() -> None:
    """Three invalid actions should end the episode with invalid_action_limit."""
    session_id = _create_session(task_id=2, seed=404)

    invalid_payload = _valid_action(
        "flag_contamination",
        parameters={
            # intentionally missing required keys: evidence_facts, recommended_action
            "contamination_type": "srm"
        },
    )

    r1 = client.post(f"/step?session_id={session_id}", json=invalid_payload)
    r2 = client.post(f"/step?session_id={session_id}", json=invalid_payload)
    r3 = client.post(f"/step?session_id={session_id}", json=invalid_payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200

    assert r1.json()["done"] is False
    assert r2.json()["done"] is False
    assert r3.json()["done"] is True
    assert r3.json()["info"]["termination_reason"] == "invalid_action_limit"


def test_terminal_action_sets_agent_verdict_done() -> None:
    """Terminal actions should stop the episode immediately with agent_verdict."""
    session_id = _create_session(task_id=4, seed=505)

    terminal = client.post(f"/step?session_id={session_id}", json=_valid_action("approve_result"))
    assert terminal.status_code == 200

    payload = terminal.json()
    assert payload["done"] is True
    assert payload["info"]["termination_reason"] == "agent_verdict"


def test_budget_exhaustion_terminates_when_step_budget_reached() -> None:
    """Episode should terminate with budget_exhausted once step_count reaches max_steps."""
    session_id = _create_session(task_id=1, seed=606)

    state = StateManager.get(session_id)
    assert state is not None
    state.max_steps = 1

    step = client.post(f"/step?session_id={session_id}", json=_valid_action("run_srm_check"))
    assert step.status_code == 200

    payload = step.json()
    assert payload["done"] is True
    assert payload["info"]["termination_reason"] == "budget_exhausted"

    final_state = client.get(f"/state?session_id={session_id}")
    assert final_state.status_code == 200
    assert final_state.json()["episode_done"] is True


def test_session_logs_are_accessible_for_grading() -> None:
    """Session logs should be readable from memory and persisted JSONL."""
    session_id = _create_session(task_id=1, seed=707)

    step = client.post(f"/step?session_id={session_id}", json=_valid_action("run_srm_check"))
    assert step.status_code == 200

    memory_log = StateManager.get_episode_log(session_id)
    disk_log = StateManager.read_persisted_episode_log(session_id)

    assert len(memory_log) >= 2
    assert len(disk_log) >= 2
    assert any(event.get("event_type") == "step" for event in memory_log)
    assert any(event.get("event_type") == "step" for event in disk_log)
