"""Hidden contamination specification used to generate and grade episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ContaminationType = Literal[
    "clean",
    "srm",
    "sutva_violation",
    "novelty_effect",
    "simpsons_paradox",
    "network_spillover",
    "multiple_testing",
    "underpowered_overclaim",
]


@dataclass(slots=True)
class ContaminationSpec:
    """Hidden ground-truth contamination metadata for one episode.

    Attributes:
        contamination_type: True contamination category for this episode.
        true_effect_size: True causal effect size.
        visible_effect_size: Effect size exposed in aggregate results.
        optimal_investigation_steps: Target number of steps for efficient auditing.
        contaminated_subgroup: Subgroup dimension/value for Simpson's paradox cases.
        interference_experiment_id: Related experiment id for SUTVA cases.
        novelty_half_life_days: Half-life for novelty decay episodes.
        srm_actual_split: Actual treatment split ratio for SRM episodes.
        network_spillover_fraction: Fraction of control exposed via network effects.
        required_queries: Queries expected for full investigation coverage.
        ground_truth_evidence: Canonical evidence facts used by the grader.
    """

    contamination_type: ContaminationType
    true_effect_size: float
    visible_effect_size: float
    optimal_investigation_steps: int
    contaminated_subgroup: str | None = None
    interference_experiment_id: str | None = None
    novelty_half_life_days: int | None = None
    srm_actual_split: float | None = None
    network_spillover_fraction: float | None = None
    required_queries: list[str] | None = None
    ground_truth_evidence: dict[str, Any] | None = None
