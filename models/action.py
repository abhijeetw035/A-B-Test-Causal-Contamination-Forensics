"""Action models for A/B test contamination forensics environment."""

from typing import Dict, Literal

from pydantic import BaseModel, Field


class AuditAction(BaseModel):
    """Agent action payload for investigative and terminal actions."""

    action_type: Literal[
        "query_subgroup",
        "query_temporal",
        "run_srm_check",
        "query_assignment_overlap",
        "check_network_exposure",
        "inspect_randomization",
        "query_secondary_metrics",
        "compute_mde",
        "flag_contamination",
        "approve_result",
        "request_rerun",
    ]
    parameters: Dict = Field(default_factory=dict)
    reasoning: str = Field(..., min_length=10, max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
