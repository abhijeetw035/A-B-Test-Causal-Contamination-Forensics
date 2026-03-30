---
title: A/B Test Causal Contamination Forensics
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# A/B Test Causal Contamination Forensics

Production-grade OpenEnv environment where agents audit experiment validity and detect hidden contamination (SRM, Simpson's paradox, SUTVA interference, spillover, and power failures).

## What this project does

- Exposes an environment API with progressive disclosure (`/reset`, `/step`, `/state`, `/health`)
- Generates deterministic synthetic experiment scenarios across 4 tasks
- Scores agent behavior with a deterministic grader (no LLM-as-judge)
- Includes a baseline `inference.py` runner for local and deployed environments

## Repository structure

- `api/` — FastAPI app and route handlers
- `env/` — state manager, action executor, observation builder, reward engine, data generation
- `tasks/` — deterministic contamination task variants and dispatcher
- `grader/` — deterministic rubric and evidence verification
- `models/` — Pydantic/dataclass models for observations, actions, and contamination specs
- `tests/` — API + grader coverage tests
- `openenv.yaml` — OpenEnv metadata/config
- `inference.py` — baseline evaluation runner

## Action space and observation space

### Actions

Investigative actions:

- `query_subgroup`
- `query_temporal`
- `run_srm_check`
- `query_assignment_overlap`
- `check_network_exposure`
- `inspect_randomization`
- `query_secondary_metrics`
- `compute_mde`

Terminal actions:

- `flag_contamination`
- `approve_result`
- `request_rerun`

### Observations

Initial observation includes aggregate results + metadata. Additional fields are unlocked progressively as investigative actions are executed (e.g., `randomization_check`, `temporal_breakdown`, `subgroup_results`, `mde_analysis`).

## Tasks

1. **Task 1 — SRM (Easy):** Detect sample-ratio mismatch invalidation.
2. **Task 2 — Simpson's + guardrails (Medium):** Detect subgroup reversal and metric conflict.
3. **Task 3 — Multi-layer contamination (Hard):** Detect overlap/interference/spillover + underpowered overclaim risk.
4. **Task 4 — Clean with red herrings (Expert):** Avoid false positives and approve valid result.

## Local setup

### Requirements

- Python 3.11+
- Optional: Docker

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860
```

### Run tests

```bash
pytest -q tests/test_api.py tests/test_grader.py
```

## Inference

`inference.py` supports **two modes**:

1. **Local mode (default):** uses in-process FastAPI test client.
2. **Remote mode:** set `ENV_BASE_URL` (or `HF_SPACE_URL`) to run against a deployed URL.

### LLM env vars

- `API_BASE_URL` (default `https://router.huggingface.co/v1`)
- `MODEL_NAME`
- `HF_TOKEN` (or `API_KEY`)

### Optional runtime env vars

- `ENV_BASE_URL` or `HF_SPACE_URL` (for live endpoint mode)
- `MAX_STEPS`
- `TEMPERATURE`
- `MAX_TOKENS`
- `REQUEST_TIMEOUT_SECONDS`

### Run baseline

```bash
python inference.py
```

Results are written to `baseline_results.json`.

## OpenEnv compliance

Local metadata file: `openenv.yaml`

Expected checks:

- `openenv validate openenv.yaml`
- `POST /reset` returns valid observation payload
- `POST /step` returns valid step payload with bounded reward
- `GET /state` excludes hidden contamination spec

## Hugging Face Space deployment

1. Create a Docker Space.
2. Add secrets (`HF_TOKEN`, optional `API_BASE_URL`, `MODEL_NAME`).
3. Add Space remote and push:

```bash
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

4. Verify live health:

```bash
curl -s https://<your-space>.hf.space/health
```

5. Run live validation:

```bash
openenv validate https://<your-space>.hf.space
```

6. Run live inference:

```bash
ENV_BASE_URL=https://<your-space>.hf.space python inference.py
```

## Baseline status (current workspace)

- Focused tests: `11 passed` (API + grader)
- Deterministic local fallback run: stable `baseline_results.json` across repeated executions
- Local structural validation on `openenv.yaml`: required top-level keys present

## Troubleshooting

- If `openenv` CLI is unavailable, install in the active env and confirm entrypoint availability.
- If inference fails with auth errors, verify `HF_TOKEN`/`API_KEY` and model access permissions.
- If remote inference fails, verify `ENV_BASE_URL` points to a live Space and `/health` returns 200.
