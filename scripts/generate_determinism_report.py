#!/usr/bin/env python3
"""Generate a determinism evidence report by repeatedly running inference.py.

The script executes inference multiple times, compares per-task scores,
and writes a machine-readable artifact useful for submission evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"baseline results not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scores_map(results: dict[str, Any]) -> dict[str, float]:
    task_results = results.get("task_results", [])
    scores: dict[str, float] = {}
    for entry in task_results:
        task_id = entry.get("task_id")
        key = f"task_{task_id}"
        scores[key] = float(entry.get("score", 0.0))
    return scores


def _run_once(repo_dir: Path, force_fallback: bool) -> tuple[float, dict[str, float]]:
    env = os.environ.copy()
    if force_fallback:
        env.pop("HF_TOKEN", None)
        env.pop("API_KEY", None)

    start = time.time()
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.time() - start, 3)

    if proc.returncode != 0:
        raise RuntimeError(
            "inference.py failed during determinism generation\n"
            f"returncode={proc.returncode}\n"
            f"stdout_tail={proc.stdout[-1500:]}\n"
            f"stderr_tail={proc.stderr[-1500:]}"
        )

    data = _load_results(repo_dir / "baseline_results.json")
    return elapsed, _scores_map(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create determinism evidence report")
    parser.add_argument("--runs", type=int, default=3, help="How many repeated runs to execute (default: 3)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/determinism_report.json"),
        help="Output report path",
    )
    parser.add_argument(
        "--no-force-fallback",
        action="store_true",
        help="Do not strip HF_TOKEN/API_KEY; use current env model calls as-is",
    )
    args = parser.parse_args()

    repo_dir = Path.cwd()
    if args.runs < 2:
        print("FAIL: runs must be >= 2")
        return 2

    run_records: list[dict[str, Any]] = []
    task_series: dict[str, list[float]] = {}

    for idx in range(1, args.runs + 1):
        runtime_s, scores = _run_once(repo_dir, force_fallback=not args.no_force_fallback)
        run_records.append({"run": idx, "runtime_seconds": runtime_s, "scores_by_task": scores})
        for task, score in scores.items():
            task_series.setdefault(task, []).append(score)

    deterministic = all(len(set(values)) == 1 for values in task_series.values()) if task_series else False
    out_of_range = {
        task: [v for v in values if not (0.0 <= v <= 1.0)]
        for task, values in task_series.items()
        if any(not (0.0 <= v <= 1.0) for v in values)
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs,
        "mode": "forced_fallback" if not args.no_force_fallback else "current_environment",
        "deterministic": deterministic,
        "tasks_checked": sorted(task_series.keys()),
        "out_of_range_scores": out_of_range,
        "run_records": run_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"PASS: determinism report written to {args.output}")
    print(f"deterministic={deterministic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
