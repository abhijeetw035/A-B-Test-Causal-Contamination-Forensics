"""Task 4 hidden spec variants: clean experiment with red herrings."""

from __future__ import annotations

from models.contamination_spec import ContaminationSpec


def get_task_specs() -> list[ContaminationSpec]:
    """Return deterministic clean variants for false-positive resistance testing.

    Returns:
        A list of clean hidden contamination specifications.
    """
    return [
        ContaminationSpec(
            contamination_type="clean",
            true_effect_size=0.038,
            visible_effect_size=0.038,
            optimal_investigation_steps=4,
            required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"],
            ground_truth_evidence={
                "srm_detected": False,
                "outage_day": 4,
                "overlap_is_non_interfering": True,
            },
        ),
        ContaminationSpec(
            contamination_type="clean",
            true_effect_size=0.034,
            visible_effect_size=0.034,
            optimal_investigation_steps=4,
            required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"],
            ground_truth_evidence={
                "srm_detected": False,
                "outage_day": 5,
                "overlap_is_non_interfering": True,
            },
        ),
        ContaminationSpec(
            contamination_type="clean",
            true_effect_size=0.041,
            visible_effect_size=0.041,
            optimal_investigation_steps=4,
            required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"],
            ground_truth_evidence={
                "srm_detected": False,
                "outage_day": 3,
                "overlap_is_non_interfering": True,
            },
        ),
    ]
