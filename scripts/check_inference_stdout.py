#!/usr/bin/env python3
"""Validate strict structured stdout contract for inference runs.

Expected line types (single-line only):
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

START_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(\d+) action=(.*) reward=(-?\d+\.\d{2}) done=(true|false) error=(null|.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) score=(-?\d+(?:\.\d+)?) rewards=([-?\d\.,]*)$"
)


def _is_rewards_field_valid(rewards_text: str) -> bool:
    if rewards_text == "":
        return True
    parts = rewards_text.split(",")
    return all(re.fullmatch(r"-?\d+\.\d{2}", p or "") is not None for p in parts)


def validate_lines(lines: list[str]) -> list[str]:
    errors: list[str] = []

    structured = [ln for ln in lines if ln.startswith("[START]") or ln.startswith("[STEP]") or ln.startswith("[END]")]
    if not structured:
        return ["No structured lines found. Expected [START]/[STEP]/[END]."]

    state = "expect_start"
    step_count = 0

    for idx, line in enumerate(structured, start=1):
        if line.startswith("[START]"):
            if state != "expect_start":
                errors.append(f"Line {idx}: unexpected [START] while state={state}: {line}")
            if START_RE.fullmatch(line) is None:
                errors.append(f"Line {idx}: invalid [START] format: {line}")
            state = "expect_step_or_end"
            step_count = 0
            continue

        if line.startswith("[STEP]"):
            if state not in {"expect_step_or_end", "expect_more_steps_or_end"}:
                errors.append(f"Line {idx}: unexpected [STEP] while state={state}: {line}")
            match = STEP_RE.fullmatch(line)
            if match is None:
                errors.append(f"Line {idx}: invalid [STEP] format: {line}")
            else:
                step_num = int(match.group(1))
                step_count += 1
                if step_num != step_count:
                    errors.append(
                        f"Line {idx}: step index not sequential within episode (expected {step_count}, got {step_num})"
                    )
            state = "expect_more_steps_or_end"
            continue

        if line.startswith("[END]"):
            if state not in {"expect_step_or_end", "expect_more_steps_or_end"}:
                errors.append(f"Line {idx}: unexpected [END] while state={state}: {line}")
            match = END_RE.fullmatch(line)
            if match is None:
                errors.append(f"Line {idx}: invalid [END] format: {line}")
            else:
                steps = int(match.group(2))
                rewards = match.group(4)
                if steps != step_count:
                    errors.append(f"Line {idx}: [END] steps={steps} does not match number of [STEP] lines={step_count}")
                if not _is_rewards_field_valid(rewards):
                    errors.append(f"Line {idx}: rewards list must be comma-separated values with 2 decimals: {rewards}")
            state = "expect_start"
            continue

    if state != "expect_start":
        errors.append("Output ended mid-episode: final [END] line missing.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate structured stdout logs from inference.py")
    parser.add_argument("log_file", type=Path, help="Path to captured stdout log")
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"FAIL: log file not found: {args.log_file}")
        return 2

    lines = args.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    errors = validate_lines(lines)

    if errors:
        print("FAIL: Structured stdout contract violations found:")
        for issue in errors[:50]:
            print(f" - {issue}")
        if len(errors) > 50:
            print(f" - ... and {len(errors) - 50} more")
        return 1

    print("PASS: Structured stdout contract is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
