"""Task 3 hidden spec variants: SUTVA + spillover + power confounds."""

from __future__ import annotations

from models.contamination_spec import ContaminationSpec


def get_task_specs() -> list[ContaminationSpec]:
    """Return deterministic multi-layer contamination variants for Task 3.

    Returns:
        A list of SUTVA-led hidden contamination specifications.
    """
    return [
        ContaminationSpec(
            contamination_type="sutva_violation",
            true_effect_size=0.012,
            visible_effect_size=0.083,
            optimal_investigation_steps=8,
            interference_experiment_id="exp_2024_pricing_011",
            network_spillover_fraction=0.23,
            required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
            ground_truth_evidence={
                "control_users_in_pricing_treatment": 0.71,
                "treatment_users_in_pricing_treatment": 0.28,
                "control_users_exposed_to_treatment_behavior": 0.23,
                "achieved_power": 0.21,
            },
        ),
        ContaminationSpec(
            contamination_type="sutva_violation",
            true_effect_size=0.010,
            visible_effect_size=0.076,
            optimal_investigation_steps=8,
            interference_experiment_id="exp_2024_pricing_014",
            network_spillover_fraction=0.19,
            required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
            ground_truth_evidence={
                "control_users_in_pricing_treatment": 0.68,
                "treatment_users_in_pricing_treatment": 0.31,
                "control_users_exposed_to_treatment_behavior": 0.19,
                "achieved_power": 0.24,
            },
        ),
        ContaminationSpec(
            contamination_type="sutva_violation",
            true_effect_size=0.014,
            visible_effect_size=0.089,
            optimal_investigation_steps=8,
            interference_experiment_id="exp_2024_pricing_020",
            network_spillover_fraction=0.26,
            required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
            ground_truth_evidence={
                "control_users_in_pricing_treatment": 0.73,
                "treatment_users_in_pricing_treatment": 0.27,
                "control_users_exposed_to_treatment_behavior": 0.26,
                "achieved_power": 0.18,
            },
        ),
    ]
