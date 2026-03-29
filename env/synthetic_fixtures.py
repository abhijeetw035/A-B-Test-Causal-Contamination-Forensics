"""Hardcoded synthetic experiment fixtures for API smoke testing."""

from __future__ import annotations

from typing import Any


CLEAN_EXPERIMENT_FIXTURE: dict[str, Any] = {
    "session_id": "session_clean_001",
    "experiment_id": "exp_2024_search_055",
    "primary_metric": "click_through_rate",
    "aggregate_results": {
        "control_mean": 0.2140,
        "treatment_mean": 0.2221,
        "relative_lift": 0.0380,
        "absolute_lift": 0.0081,
        "p_value": 0.0002,
        "control_count": 250000,
        "treatment_count": 250000,
        "confidence_interval_lower": 0.0042,
        "confidence_interval_upper": 0.0120,
    },
    "experiment_metadata": {
        "start_date": "2024-04-01",
        "end_date": "2024-04-28",
        "targeting_rule": "all_search_users_us",
        "intended_split": 0.50,
        "randomization_unit": "user_id",
        "platform": "web",
        "experiment_owner": "search_team",
        "hypothesis": "New ranking improves click-through rate.",
    },
    "available_queries": [
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
    ],
    "steps_taken": 0,
    "steps_remaining": 15,
}


CONTAMINATED_EXPERIMENT_FIXTURE: dict[str, Any] = {
    "session_id": "session_contaminated_001",
    "experiment_id": "exp_2024_growth_007",
    "primary_metric": "d7_retention_rate",
    "aggregate_results": {
        "control_mean": 0.4120,
        "treatment_mean": 0.4375,
        "relative_lift": 0.0620,
        "absolute_lift": 0.0255,
        "p_value": 0.0300,
        "control_count": 61847,
        "treatment_count": 48203,
        "confidence_interval_lower": 0.0040,
        "confidence_interval_upper": 0.0470,
    },
    "experiment_metadata": {
        "start_date": "2024-01-15",
        "end_date": "2024-02-15",
        "targeting_rule": "all_users_18_plus_us",
        "intended_split": 0.50,
        "randomization_unit": "user_id",
        "platform": "mobile_ios",
        "experiment_owner": "growth_team",
        "hypothesis": "Improved onboarding increases D7 retention.",
    },
    "available_queries": [
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
    ],
    "steps_taken": 0,
    "steps_remaining": 15,
}


def get_api_test_fixtures() -> dict[str, dict[str, Any]]:
    """Return exactly one clean and one contaminated fixture for API tests.

    Returns:
        A mapping with keys 'clean' and 'contaminated'.
    """
    return {
        "clean": CLEAN_EXPERIMENT_FIXTURE,
        "contaminated": CONTAMINATED_EXPERIMENT_FIXTURE,
    }
