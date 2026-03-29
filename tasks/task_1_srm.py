"""Task 1 hidden spec variants: Sample Ratio Mismatch (SRM)."""

from __future__ import annotations

from models.contamination_spec import ContaminationSpec


def get_task_specs() -> list[ContaminationSpec]:
    """Return deterministic SRM contamination variants for Task 1.

    Returns:
        A list of SRM-focused hidden contamination specifications.
    """
    return [
        ContaminationSpec(
            contamination_type="srm",
            true_effect_size=0.0,
            visible_effect_size=0.062,
            optimal_investigation_steps=3,
            srm_actual_split=0.438,
            required_queries=["run_srm_check"],
            ground_truth_evidence={
                "actual_split": 0.438,
                "intended_split": 0.50,
                "srm_detected": True,
            },
        ),
        ContaminationSpec(
            contamination_type="srm",
            true_effect_size=0.0,
            visible_effect_size=0.051,
            optimal_investigation_steps=3,
            srm_actual_split=0.445,
            required_queries=["run_srm_check"],
            ground_truth_evidence={
                "actual_split": 0.445,
                "intended_split": 0.50,
                "srm_detected": True,
            },
        ),
        ContaminationSpec(
            contamination_type="srm",
            true_effect_size=0.0,
            visible_effect_size=0.074,
            optimal_investigation_steps=3,
            srm_actual_split=0.421,
            required_queries=["run_srm_check"],
            ground_truth_evidence={
                "actual_split": 0.421,
                "intended_split": 0.50,
                "srm_detected": True,
            },
        ),
    ]
