"""Task specification sampling stubs."""

from __future__ import annotations

import random

from models.contamination_spec import ContaminationSpec


class TaskGenerator:
    """Samples hidden contamination specifications by task and seed."""

    @staticmethod
    def sample(task_id: int, seed: int) -> ContaminationSpec:
        """Sample a deterministic contamination spec for the requested task.

        Args:
            task_id: OpenEnv task id for scenario selection.
            seed: Random seed used to choose a deterministic variant.

        Returns:
            A contamination specification used for episode generation.
        """
        rng = random.Random(task_id * 10_000 + seed)

        if task_id == 1:
            variants = [
                ContaminationSpec(
                    contamination_type="srm",
                    true_effect_size=0.0,
                    visible_effect_size=0.062,
                    optimal_investigation_steps=3,
                    srm_actual_split=0.438,
                    required_queries=["run_srm_check"],
                ),
                ContaminationSpec(
                    contamination_type="srm",
                    true_effect_size=0.0,
                    visible_effect_size=0.055,
                    optimal_investigation_steps=3,
                    srm_actual_split=0.451,
                    required_queries=["run_srm_check"],
                ),
            ]
            return rng.choice(variants)

        if task_id == 2:
            variants = [
                ContaminationSpec(
                    contamination_type="simpsons_paradox",
                    true_effect_size=-0.008,
                    visible_effect_size=0.051,
                    optimal_investigation_steps=5,
                    contaminated_subgroup="enrollment_cohort",
                    required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
                ),
                ContaminationSpec(
                    contamination_type="simpsons_paradox",
                    true_effect_size=-0.005,
                    visible_effect_size=0.046,
                    optimal_investigation_steps=5,
                    contaminated_subgroup="enrollment_cohort",
                    required_queries=["query_temporal", "query_subgroup", "query_secondary_metrics"],
                ),
            ]
            return rng.choice(variants)

        if task_id == 3:
            variants = [
                ContaminationSpec(
                    contamination_type="sutva_violation",
                    true_effect_size=0.012,
                    visible_effect_size=0.083,
                    optimal_investigation_steps=8,
                    interference_experiment_id="exp_2024_pricing_011",
                    network_spillover_fraction=0.23,
                    required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
                ),
                ContaminationSpec(
                    contamination_type="sutva_violation",
                    true_effect_size=0.010,
                    visible_effect_size=0.072,
                    optimal_investigation_steps=8,
                    interference_experiment_id="exp_2024_pricing_011",
                    network_spillover_fraction=0.19,
                    required_queries=["query_assignment_overlap", "check_network_exposure", "compute_mde"],
                ),
            ]
            return rng.choice(variants)

        if task_id == 4:
            return ContaminationSpec(
                contamination_type="clean",
                true_effect_size=0.038,
                visible_effect_size=0.038,
                optimal_investigation_steps=4,
                required_queries=["run_srm_check", "query_temporal", "query_assignment_overlap"],
            )

        raise ValueError(f"Unsupported task_id={task_id}. Expected one of [1, 2, 3, 4].")

