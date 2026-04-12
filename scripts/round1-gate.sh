#!/usr/bin/env bash
set -euo pipefail

SPACE_URL="${1:-https://abhijeetw035-ab-test-contamination-env.hf.space}"
REPO_DIR="${2:-.}"
LOG_FILE="${3:-/tmp/inference_round1.log}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "FAIL: repo_dir not found: ${2:-.}"
  exit 1
fi

cd "$REPO_DIR"

echo "[1/6] Required env vars..."
for var in API_BASE_URL MODEL_NAME HF_TOKEN; do
  if [ -z "$(printenv "$var")" ]; then
    echo "FAIL: required env var is missing -> $var"
    exit 1
  fi
done
echo "PASS: all required env vars are set"

echo "[2/6] Pre-submission validator (Space ping + docker build + openenv validate)..."
./scripts/validate-submission.sh "$SPACE_URL" "$REPO_DIR"

echo "[3/6] Run inference.py and capture stdout..."
python3 inference.py > "$LOG_FILE"
echo "PASS: inference.py completed"

echo "[4/6] Validate strict stdout contract..."
python3 scripts/check_inference_stdout.py "$LOG_FILE"

echo "[5/6] Validate baseline_results.json integrity..."
python3 - <<'PY'
import json
from pathlib import Path

p = Path("baseline_results.json")
if not p.exists():
    raise SystemExit("FAIL: baseline_results.json missing")

data = json.loads(p.read_text(encoding="utf-8"))
scores_by_task = data.get("scores_by_task")
if not isinstance(scores_by_task, dict):
    raise SystemExit("FAIL: baseline_results.json missing 'scores_by_task' object")

required = {"task_1", "task_2", "task_3"}
missing = required - set(scores_by_task.keys())
if missing:
    raise SystemExit(f"FAIL: missing required tasks in scores_by_task: {sorted(missing)}")

for task, score in scores_by_task.items():
    value = float(score)
  if not (0.0 < value < 1.0):
    raise SystemExit(f"FAIL: score for {task} must be strictly between 0 and 1: {value}")

print("PASS: baseline_results.json task scores are present and bounded [0,1]")
PY

echo "[6/6] Determinism evidence artifact..."
python3 scripts/generate_determinism_report.py --runs 3 --output artifacts/determinism_report.json

echo "✅ ROUND-1 GATE PASSED"
echo "Artifacts:"
echo "  - stdout log: $LOG_FILE"
echo "  - baseline results: $REPO_DIR/baseline_results.json"
echo "  - determinism report: $REPO_DIR/artifacts/determinism_report.json"
