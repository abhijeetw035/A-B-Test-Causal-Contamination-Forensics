"""Task 2 hidden spec variants: Simpson's paradox scenarios."""

from __future__ import annotations

from models.contamination_spec import ContaminationSpec


def get_task_specs() -> list[ContaminationSpec]:
    """Return deterministic Simpson's paradox contamination variants for Task 2.

    Returns:
        A list of Simpson's paradox hidden contamination specifications.
    """
    return [
        ContaminationSpec(
            contamination_type="simpsons_paradox",
            true_effect_size=-0.008,
            visible_effect_size=0.051,
            optimal_investigation_steps=5,
            contaminated_subgroup="enrollment_cohort",
            required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
            ground_truth_evidence={
                "contaminated_subgroup": "enrollment_cohort",
                "early_cohort_treatment_fraction": 0.67,
                "session_length_lift": -0.041,
            },
        ),
        ContaminationSpec(
            contamination_type="simpsons_paradox",
            true_effect_size=-0.011,
            visible_effect_size=0.043,
            optimal_investigation_steps=5,
            contaminated_subgroup="enrollment_cohort",
            required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
            ground_truth_evidence={
                "contaminated_subgroup": "enrollment_cohort",
                "early_cohort_treatment_fraction": 0.64,
                "session_length_lift": -0.038,
            },
        ),
        ContaminationSpec(
            contamination_type="simpsons_paradox",
            true_effect_size=-0.006,
            visible_effect_size=0.058,
            optimal_investigation_steps=5,
            contaminated_subgroup="enrollment_cohort",
            required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
            ground_truth_evidence={
                "contaminated_subgroup": "enrollment_cohort",
                "early_cohort_treatment_fraction": 0.69,
                "session_length_lift": -0.042,
            },
        ),
    ]
