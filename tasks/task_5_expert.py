"""Task 5 hidden spec variants: Novelty Effect & Bot Traffic (Expert)."""

from __future__ import annotations

from models.contamination_spec import ContaminationSpec


def get_task_specs() -> list[ContaminationSpec]:
    """Return deterministic expert contamination variants for Task 5.

    Returns:
        A list of Novelty-led hidden contamination specifications.
    """
    return [
        ContaminationSpec(
            contamination_type="novelty_effect",
            true_effect_size=0.0,
            visible_effect_size=0.045,
            optimal_investigation_steps=6,
            novelty_half_life_days=3,
            required_queries=["query_temporal", "query_subgroup"],
            ground_truth_evidence={
                "temporal_decay_observed": True,
                "novelty_effect_size": 0.045,
            },
        ),
        ContaminationSpec(
            contamination_type="novelty_effect",
            true_effect_size=0.005,
            visible_effect_size=0.065,
            optimal_investigation_steps=6,
            novelty_half_life_days=2,
            required_queries=["query_temporal", "query_subgroup"],
            ground_truth_evidence={
                "temporal_decay_observed": True,
                "novelty_effect_size": 0.060,
            },
        ),
        ContaminationSpec(
            contamination_type="novelty_effect",
            true_effect_size=-0.010,
            visible_effect_size=0.082,
            optimal_investigation_steps=6,
            novelty_half_life_days=2,
            required_queries=["query_temporal", "query_subgroup"],
            ground_truth_evidence={
                "temporal_decay_observed": True,
                "novelty_effect_size": 0.092,
            },
        ),
    ]
