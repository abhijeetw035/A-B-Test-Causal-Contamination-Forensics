"""Observation models for A/B test contamination forensics environment."""

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class AggregateResult(BaseModel):
    """Aggregate experiment summary statistics."""

    control_mean: float
    treatment_mean: float
    relative_lift: float
    absolute_lift: float
    p_value: float
    control_count: int
    treatment_count: int
    confidence_interval_lower: float
    confidence_interval_upper: float


class ExperimentMeta(BaseModel):
    """Experiment metadata visible to the auditing agent."""

    start_date: date
    end_date: date
    targeting_rule: str
    intended_split: float
    randomization_unit: str
    platform: str
    experiment_owner: str
    hypothesis: str


class DailyResult(BaseModel):
    """Daily metric breakdown row."""

    date: date
    control_mean: float
    treatment_mean: float
    relative_lift: float
    control_count: int
    treatment_count: int


class SubgroupResult(BaseModel):
    """Metric breakdown for one subgroup value."""

    dimension: str
    value: str
    control_mean: float
    treatment_mean: float
    relative_lift: float
    control_count: int
    treatment_count: int


class SRMResult(BaseModel):
    """Sample ratio mismatch check output."""

    expected_split: float
    actual_split: float
    chi_square_statistic: float
    p_value: float
    srm_detected: bool
    severity: Literal["none", "mild", "severe"]


class OverlapMatrix(BaseModel):
    """Cross-experiment user overlap matrix."""

    experiment_ids: List[str]
    overlap_fractions: Dict[str, Dict[str, float]]


class MDEAnalysis(BaseModel):
    """Minimum detectable effect and power analysis output."""

    observed_effect_size: float
    required_sample_per_arm: int
    actual_sample_per_arm: int
    achieved_power: float
    underpowered: bool


class ExperimentObservation(BaseModel):
    """Full environment observation returned to the agent."""

    session_id: str
    experiment_id: str
    primary_metric: str
    aggregate_results: AggregateResult
    experiment_metadata: ExperimentMeta
    available_queries: List[str]
    steps_taken: int
    steps_remaining: int

    subgroup_results: Optional[Dict[str, List[SubgroupResult]]] = None
    temporal_breakdown: Optional[List[DailyResult]] = None
    user_assignment_overlap: Optional[OverlapMatrix] = None
    randomization_check: Optional[SRMResult] = None
    network_exposure_map: Optional[Dict[str, float]] = None
    secondary_metric_results: Optional[Dict[str, AggregateResult]] = None
    mde_analysis: Optional[MDEAnalysis] = None
    randomization_audit: Optional[Dict[str, Any]] = None
    peer_experiment_list: Optional[List[Dict[str, Any]]] = None
