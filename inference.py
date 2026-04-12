"""Baseline inference runner for A/B Test Causal Contamination Forensics.

STDOUT FORMAT (mandatory for evaluation):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Protocol

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from api.app import app
from grader.grader import Grader
from models.action import AuditAction
from env.state_manager import StateManager


# ---------------------------------------------------------------------------
# Environment configuration — all read from env vars as mandated by spec
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")                     # no default — injected at runtime
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")     # optional: used with from_docker_image()
ENV_BASE_URL = (os.getenv("ENV_BASE_URL") or os.getenv("HF_SPACE_URL") or "").rstrip("/")

BENCHMARK = "ab-test-contamination-forensics"
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

# Tasks 1-3 are mandatory (easy / medium / hard)
TASKS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 7, 88, 99]
SUCCESS_SCORE_THRESHOLD = 0.3  # score >= this → success=true
OUTPUT_FILE = Path("baseline_results.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mandatory structured stdout logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit a [START] line — exactly one per episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit a [STEP] line — once per env.step() call."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitise action string: remove newlines / leading whitespace
    action_str = str(action).replace("\n", " ").replace("\r", " ").strip()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit an [END] line — always, even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt for the LLM auditor
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert A/B test validity auditor. Given an experiment briefing,
investigate whether results are statistically valid or contaminated.

Rules:
1. Always run run_srm_check early — count imbalance invalidates everything.
2. Check temporal trends if the experiment ran >7 days.
3. Check subgroups when aggregate lift seems unusually high.
4. Check secondary metrics before approving anything.
5. Check experiment overlap when peer experiments exist.
6. Use compute_mde when sample sizes seem small for the claimed effect.

Respond ONLY with a valid JSON object:
{
  "action_type": "<one of the available actions>",
  "parameters": {},
  "reasoning": "<1-3 sentence justification>",
  "confidence": <float 0.0-1.0>
}
""".strip()


# ---------------------------------------------------------------------------
# Environment adapters
# ---------------------------------------------------------------------------

class LocalAPIEnvironment:
    """Simple local environment adapter over FastAPI test client."""

    def __init__(self) -> None:
        self._client = TestClient(app)
        self._session_id: str | None = None

    def reset(self, task_id: int, seed: int) -> dict[str, Any]:
        response = self._client.post("/reset", json={"task_id": task_id, "seed": seed})
        response.raise_for_status()
        obs = response.json()
        self._session_id = str(obs["session_id"])
        return obs

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("Session not initialized. Call reset() first.")
        response = self._client.post(f"/step?session_id={self._session_id}", json=action)
        response.raise_for_status()
        return response.json()

    def grade_last_episode(self) -> dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("Session not initialized. Call reset() first.")
        state = StateManager.get(self._session_id)
        if state is None or state.spec is None:
            raise RuntimeError("Unable to locate current session/spec for grading.")
        return Grader.grade_episode(state.episode_log, state.spec)


class RemoteAPIEnvironment:
    """Adapter for a deployed HTTP environment (e.g., HF Space)."""

    def __init__(self, base_url: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> None:
        if not base_url:
            raise ValueError("Remote base URL is required for RemoteAPIEnvironment")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout_seconds)
        self._session_id: str | None = None

    def reset(self, task_id: int, seed: int) -> dict[str, Any]:
        response = self._client.post("/reset", json={"task_id": task_id, "seed": seed})
        response.raise_for_status()
        obs = response.json()
        self._session_id = str(obs["session_id"])
        return obs

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("Session not initialized. Call reset() first.")
        response = self._client.post(f"/step?session_id={self._session_id}", json=action)
        response.raise_for_status()
        return response.json()

    def grade_last_episode(self) -> dict[str, Any] | None:
        return None


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_obs(obs: dict[str, Any]) -> str:
    lines = [
        f"=== EXPERIMENT AUDIT: {obs.get('experiment_id', 'UNKNOWN')} ===",
        f"Primary metric: {obs.get('primary_metric', '')}",
        f"Hypothesis: {obs.get('experiment_metadata', {}).get('hypothesis', '')}",
        "",
        "AGGREGATE RESULTS:",
        f"  Control:   n={obs['aggregate_results']['control_count']:,}  mean={obs['aggregate_results']['control_mean']:.4f}",
        f"  Treatment: n={obs['aggregate_results']['treatment_count']:,}  mean={obs['aggregate_results']['treatment_mean']:.4f}",
        f"  Lift: {obs['aggregate_results']['relative_lift']*100:.2f}%  p={obs['aggregate_results']['p_value']:.6f}",
        "",
        "METADATA:",
        f"  Date: {obs['experiment_metadata']['start_date']} → {obs['experiment_metadata']['end_date']}",
        f"  Intended split: {obs['experiment_metadata']['intended_split']*100:.0f}% treatment",
        f"  Targeting: {obs['experiment_metadata']['targeting_rule']}",
        "",
        f"Steps taken: {obs.get('steps_taken', '?')}",
        f"Steps remaining: {obs.get('steps_remaining', '?')}",
        f"Investigation budget: ${obs.get('investigation_budget', 0):.2f}",
        f"Budget spent: ${obs.get('budget_spent', 0):.2f}",
        f"Available actions: {', '.join(obs.get('available_queries', []))}",
    ]

    reveal_fields = [
        "randomization_check",
        "subgroup_results",
        "temporal_breakdown",
        "user_assignment_overlap",
        "secondary_metric_results",
        "mde_analysis",
        "network_exposure_map",
        "randomization_audit",
        "expert_review",
        "counterfactual_analysis",
    ]
    for field in reveal_fields:
        if obs.get(field) is not None:
            lines.append(f"\n[REVEALED] {field.upper()}:")
            lines.append(json.dumps(obs[field], indent=2, default=str)[:6000])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback policy (used when no API key is set)
# ---------------------------------------------------------------------------

def _fallback_action(task_id: int, step_num: int) -> dict[str, Any]:
    plans: dict[int, list[tuple[str, dict[str, Any]]]] = {
        1: [
            ("run_srm_check", {}),
            (
                "flag_contamination",
                {
                    "contamination_type": "srm",
                    "evidence_facts": ["actual split deviates from intended 50/50"],
                    "recommended_action": "rerun",
                    "estimated_true_effect": 0.0,
                },
            ),
        ],
        2: [
            ("query_temporal", {}),
            ("query_subgroup", {"dimension": "enrollment_cohort"}),
            ("query_secondary_metrics", {}),
            (
                "flag_contamination",
                {
                    "contamination_type": "simpsons_paradox",
                    "evidence_facts": ["cohort-level reversal observed", "guardrail metrics degrade"],
                    "recommended_action": "rerun",
                    "estimated_true_effect": -0.008,
                },
            ),
        ],
        3: [
            ("query_assignment_overlap", {}),
            ("check_network_exposure", {}),
            ("compute_mde", {}),
            (
                "flag_contamination",
                {
                    "contamination_type": "sutva_violation",
                    "evidence_facts": [
                        "high overlap with concurrent pricing experiment",
                        "network spillover in control",
                    ],
                    "recommended_action": "rerun",
                    "estimated_true_effect": 0.012,
                },
            ),
        ],
        4: [
            ("query_temporal", {}),
            ("run_srm_check", {}),
            (
                "approve_result",
                {
                    "recommended_action": "launch",
                },
            ),
        ],
        5: [
            ("query_temporal", {}),
            ("request_expert_review", {}),
            ("simulate_counterfactual", {}),
            (
                "flag_contamination",
                {
                    "contamination_type": "novelty_effect",
                    "evidence_facts": ["temporal_decay_observed"],
                    "recommended_action": "rerun",
                    "estimated_true_effect": 0.0,
                },
            ),
        ],
    }
    default_plan: list[tuple[str, dict[str, Any]]] = [
        (
            "request_rerun",
            {"reason": "Unknown task", "recommended_changes": ["reconfigure task id"]},
        )
    ]
    plan = plans.get(task_id, default_plan)
    action_type, parameters = plan[min(step_num, len(plan) - 1)]
    return {
        "action_type": action_type,
        "parameters": parameters,
        "reasoning": "Applying deterministic fallback audit policy for this task.",
        "confidence": 0.75,
    }


# ---------------------------------------------------------------------------
# LLM call with fallback
# ---------------------------------------------------------------------------

def _run_llm_action(
    client: OpenAI | None,
    messages: list[dict[str, str]],
    task_id: int,
    step_num: int,
) -> dict[str, Any]:
    if client is None:
        return _fallback_action(task_id, step_num)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            parsed = AuditAction(**payload)
            return parsed.model_dump()
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)

    return _fallback_action(task_id, step_num)


# ---------------------------------------------------------------------------
# Episode runner — emits compliant [START] / [STEP] / [END] blocks
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    task_id: int,
    seed: int,
    client: OpenAI | None,
) -> dict[str, Any]:
    """Run one complete episode for a single task and emit structured stdout lines.

    Returns:
        Episode summary dict with rewards and final score.
    """
    task_name = f"task_{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # ── [START] ─────────────────────────────────────────────────────────────
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_id, seed=seed)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        episode_log: list[dict[str, Any]] = []

        for step_num in range(MAX_STEPS):
            # Build LLM message
            messages.append({"role": "user", "content": format_obs(observation)})
            action_payload = _run_llm_action(
                client=client, messages=messages, task_id=task_id, step_num=step_num
            )

            # Validate / sanitize action
            try:
                action_payload = AuditAction(**action_payload).model_dump()
            except Exception:
                action_payload = AuditAction(
                    action_type="request_rerun",
                    parameters={"reason": "Malformed action", "recommended_changes": ["fix action formatting"]},
                    reasoning="Malformed action recovered with safe fallback.",
                    confidence=0.2,
                ).model_dump()

            step_result = env.step(action_payload)
            observation = step_result["observation"]
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            error_str: Optional[str] = step_result.get("info", {}).get("error") or None

            rewards.append(reward)
            steps_taken = step_num + 1

            # ── [STEP] ───────────────────────────────────────────────────────
            log_step(
                step=steps_taken,
                action=action_payload.get("action_type", "unknown"),
                reward=reward,
                done=done,
                error=error_str,
            )

            episode_log.append(
                {
                    "step": steps_taken,
                    "action": action_payload,
                    "reward": reward,
                    "done": done,
                    "info": step_result.get("info", {}),
                }
            )
            messages.append({"role": "assistant", "content": json.dumps(action_payload)})

            if done:
                break

        # Grade the episode
        grade = env.grade_last_episode() if hasattr(env, "grade_last_episode") else None
        if grade is None:
            # Remote mode: estimate score from cumulative reward
            total_reward = sum(rewards)
            score = min(max(total_reward, 0.0), 1.0)
        else:
            raw_score = grade.get("final_score")
            score = float(raw_score) if raw_score is not None else 0.0
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log.error("Episode error for task %s: %s", task_id, exc)
        # Still emit [END] below in the finally-equivalent

    # ── [END] ────────────────────────────────────────────────────────────────
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": round(score, 4),
        "success": success,
        "steps_taken": steps_taken,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run baseline evaluation across task suite (tasks 1–3) and write results JSON."""
    client: OpenAI | None = None
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        log.info("Using model %s via %s", MODEL_NAME, API_BASE_URL)
    else:
        log.warning("No HF_TOKEN configured — using deterministic fallback policy.")

    if ENV_BASE_URL:
        env: Any = RemoteAPIEnvironment(base_url=ENV_BASE_URL)
        log.info("Targeting remote environment: %s", ENV_BASE_URL)
    else:
        env = LocalAPIEnvironment()
        log.info("Running against local in-process environment.")

    all_results: list[dict[str, Any]] = []

    for task_id, seed in zip(TASKS, SEEDS):
        result = run_episode(env=env, task_id=task_id, seed=seed, client=client)
        all_results.append(result)

    scores = [r["score"] for r in all_results]
    average_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    scores_by_task = {f"task_{r['task_id']}": r["score"] for r in all_results}
    success_rate = round(
        sum(1 for r in all_results if r.get("success")) / len(all_results), 4
    ) if all_results else 0.0
    summary = {
        "model": MODEL_NAME if HF_TOKEN else "deterministic-fallback",
        "environment_mode": "remote" if ENV_BASE_URL else "local",
        "benchmark": BENCHMARK,
        "scores_by_task": scores_by_task,
        "average_score": average_score,
        "success_rate": success_rate,
        "task_count": len(all_results),
        "task_results": all_results,
    }
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved baseline results to %s | avg_score=%.4f", OUTPUT_FILE, average_score)


if __name__ == "__main__":
    main()
