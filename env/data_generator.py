"""Synthetic experiment data generation from hidden contamination specs."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from models.contamination_spec import ContaminationSpec


AVAILABLE_QUERIES: list[str] = [
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


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float into an inclusive range."""
    return max(lower, min(upper, value))


def _safe_relative_lift(control_mean: float, treatment_mean: float) -> float:
    """Compute relative lift while guarding against division by zero."""
    if control_mean <= 0:
        return 0.0
    return (treatment_mean - control_mean) / control_mean


def _two_proportion_stats(
    control_success: int,
    control_count: int,
    treatment_success: int,
    treatment_count: int,
) -> dict[str, float]:
    """Compute p-value and confidence interval for two-proportion comparison."""
    p1 = control_success / control_count
    p2 = treatment_success / treatment_count
    diff = p2 - p1

    pooled = (control_success + treatment_success) / (control_count + treatment_count)
    se_pooled = np.sqrt(max(pooled * (1.0 - pooled) * (1.0 / control_count + 1.0 / treatment_count), 1e-16))
    z_stat = diff / se_pooled
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    se_unpooled = np.sqrt(
        max(
            p1 * (1.0 - p1) / control_count + p2 * (1.0 - p2) / treatment_count,
            1e-16,
        )
    )
    margin = 1.96 * se_unpooled

    return {
        "control_mean": round(p1, 6),
        "treatment_mean": round(p2, 6),
        "absolute_lift": round(diff, 6),
        "relative_lift": round(_safe_relative_lift(p1, p2), 6),
        "p_value": round(float(_clamp(p_value, 1e-12, 1.0)), 12),
        "confidence_interval_lower": round(diff - margin, 6),
        "confidence_interval_upper": round(diff + margin, 6),
    }


def _aggregate_result_from_rates(
    control_rate: float,
    treatment_rate: float,
    control_count: int,
    treatment_count: int,
) -> dict[str, Any]:
    """Build aggregate statistics from rates and arm counts."""
    control_success = int(round(_clamp(control_rate, 1e-6, 1 - 1e-6) * control_count))
    treatment_success = int(round(_clamp(treatment_rate, 1e-6, 1 - 1e-6) * treatment_count))
    stats = _two_proportion_stats(
        control_success=control_success,
        control_count=control_count,
        treatment_success=treatment_success,
        treatment_count=treatment_count,
    )
    stats["control_count"] = int(control_count)
    stats["treatment_count"] = int(treatment_count)
    return stats


def _srm_payload(control_count: int, treatment_count: int, expected_split: float) -> dict[str, Any]:
    """Compute SRM test payload from observed arm counts."""
    total = control_count + treatment_count
    exp_treatment = total * expected_split
    exp_control = total * (1.0 - expected_split)
    chi_square = ((control_count - exp_control) ** 2) / max(exp_control, 1e-9)
    chi_square += ((treatment_count - exp_treatment) ** 2) / max(exp_treatment, 1e-9)
    p_value = float(1.0 - chi2.cdf(chi_square, df=1))

    severity = "none"
    if p_value < 1e-6:
        severity = "severe"
    elif p_value < 1e-3:
        severity = "mild"

    return {
        "expected_split": round(expected_split, 6),
        "actual_split": round(treatment_count / max(total, 1), 6),
        "chi_square_statistic": round(float(chi_square), 6),
        "p_value": round(_clamp(p_value, 0.0, 1.0), 12),
        "srm_detected": p_value < 1e-3,
        "severity": severity,
    }


def _generate_temporal_breakdown(
    rng: np.random.Generator,
    start_date: date,
    end_date: date,
    control_rate: float,
    treatment_rate: float,
    control_count: int,
    treatment_count: int,
    spec: ContaminationSpec,
) -> list[dict[str, Any]]:
    """Generate per-day aggregate rows with contamination-specific temporal patterns."""
    n_days = max((end_date - start_date).days + 1, 1)
    day_prob = rng.dirichlet(np.ones(n_days))
    control_daily_counts = rng.multinomial(control_count, day_prob)
    treatment_daily_counts = rng.multinomial(treatment_count, day_prob)

    rows: list[dict[str, Any]] = []
    half_life = spec.novelty_half_life_days or 4
    outage_day = None
    if spec.contamination_type == "clean":
        outage_day = int((spec.ground_truth_evidence or {}).get("outage_day", 4))

    for idx in range(n_days):
        day = start_date + timedelta(days=idx)
        c_n = int(max(control_daily_counts[idx], 1))
        t_n = int(max(treatment_daily_counts[idx], 1))

        c_mean = _clamp(control_rate + rng.normal(0.0, 0.004), 1e-4, 1 - 1e-4)
        t_mean = treatment_rate

        if spec.contamination_type == "novelty_effect":
            decay = 0.5 ** (idx / max(half_life, 1))
            novelty_lift = (spec.visible_effect_size * 2.5) * decay
            t_mean = _clamp(c_mean * (1.0 + novelty_lift), 1e-4, 1 - 1e-4)
        elif spec.contamination_type == "simpsons_paradox":
            if idx <= 2:
                # Early spike from cohort composition confound.
                t_mean = _clamp(c_mean * 1.22, 1e-4, 1 - 1e-4)
            else:
                # Later periods flatten/reverse to expose paradox.
                t_mean = _clamp(c_mean * (0.99 + rng.normal(0.0, 0.005)), 1e-4, 1 - 1e-4)
        elif spec.contamination_type == "clean" and outage_day is not None and (idx + 1) == outage_day:
            # Red herring: one-day symmetric outage dip that should not imply contamination.
            c_mean = _clamp(c_mean * 0.88, 1e-4, 1 - 1e-4)
            t_mean = _clamp(c_mean * (1.0 + rng.normal(0.0, 0.002)), 1e-4, 1 - 1e-4)
        else:
            t_mean = _clamp(treatment_rate + rng.normal(0.0, 0.004), 1e-4, 1 - 1e-4)

        rows.append(
            {
                "date": day.isoformat(),
                "control_mean": round(c_mean, 6),
                "treatment_mean": round(t_mean, 6),
                "relative_lift": round(_safe_relative_lift(c_mean, t_mean), 6),
                "control_count": c_n,
                "treatment_count": t_n,
            }
        )

    return rows


def _generic_subgroup_rows(
    rng: np.random.Generator,
    dimension: str,
    values: list[str],
    control_rate: float,
    treatment_rate: float,
    total_control: int,
    total_treatment: int,
) -> list[dict[str, Any]]:
    """Generate plausible subgroup rows around aggregate rates."""
    control_parts = rng.dirichlet(np.ones(len(values)))
    treatment_parts = rng.dirichlet(np.ones(len(values)))

    control_counts = np.maximum(rng.multinomial(total_control, control_parts), 1)
    treatment_counts = np.maximum(rng.multinomial(total_treatment, treatment_parts), 1)

    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(values):
        c_mean = _clamp(control_rate + rng.normal(0.0, 0.007), 1e-4, 1 - 1e-4)
        t_mean = _clamp(treatment_rate + rng.normal(0.0, 0.007), 1e-4, 1 - 1e-4)
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "control_mean": round(c_mean, 6),
                "treatment_mean": round(t_mean, 6),
                "relative_lift": round(_safe_relative_lift(c_mean, t_mean), 6),
                "control_count": int(control_counts[idx]),
                "treatment_count": int(treatment_counts[idx]),
            }
        )

    return rows


def _simpsons_enrollment_rows(total_control: int, total_treatment: int) -> list[dict[str, Any]]:
    """Generate enrollment cohort rows that produce Simpson-style reversal."""
    early_control = int(round(total_control * 0.35))
    late_control = max(total_control - early_control, 1)
    early_treatment = int(round(total_treatment * 0.67))
    late_treatment = max(total_treatment - early_treatment, 1)

    return [
        {
            "dimension": "enrollment_cohort",
            "value": "days_1_3",
            "control_mean": 0.68,
            "treatment_mean": 0.71,
            "relative_lift": round(_safe_relative_lift(0.68, 0.71), 6),
            "control_count": early_control,
            "treatment_count": early_treatment,
        },
        {
            "dimension": "enrollment_cohort",
            "value": "days_4_21",
            "control_mean": 0.39,
            "treatment_mean": 0.38,
            "relative_lift": round(_safe_relative_lift(0.39, 0.38), 6),
            "control_count": late_control,
            "treatment_count": late_treatment,
        },
    ]


def _mde_payload(
    control_rate: float,
    treatment_rate: float,
    actual_sample_per_arm: int,
    contamination_type: str,
) -> dict[str, Any]:
    """Compute MDE/power analysis payload for the observed effect size."""
    alpha = 0.05
    target_power = 0.80
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(target_power)

    effect = max(abs(treatment_rate - control_rate), 1e-6)
    p1 = _clamp(control_rate, 1e-6, 1 - 1e-6)
    p2 = _clamp(treatment_rate, 1e-6, 1 - 1e-6)
    p_bar = (p1 + p2) / 2.0

    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    required = int(np.ceil(numerator / (effect**2)))

    # Ensure explicitly underpowered behavior for the dedicated contamination type.
    if contamination_type in {"underpowered_overclaim", "sutva_violation"}:
        required = max(required, actual_sample_per_arm * 3)
    elif contamination_type == "clean":
        required = min(required, max(1000, int(actual_sample_per_arm * 0.85)))

    se_h1 = np.sqrt(max(p1 * (1.0 - p1) / actual_sample_per_arm + p2 * (1.0 - p2) / actual_sample_per_arm, 1e-16))
    z_effect = effect / se_h1
    achieved_power = float(norm.cdf(z_effect - z_alpha))

    return {
        "observed_effect_size": round(treatment_rate - control_rate, 6),
        "required_sample_per_arm": int(required),
        "actual_sample_per_arm": int(actual_sample_per_arm),
        "achieved_power": round(_clamp(achieved_power, 0.0, 1.0), 6),
        "underpowered": actual_sample_per_arm < required,
    }


def _secondary_metrics(
    control_count: int,
    treatment_count: int,
    contamination_type: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """Generate guardrail metric aggregates."""
    session_length_control = _clamp(rng.uniform(0.32, 0.46), 1e-4, 1 - 1e-4)
    if contamination_type == "simpsons_paradox":
        session_length_treatment = _clamp(session_length_control * (1.0 - 0.041), 1e-4, 1 - 1e-4)
    else:
        session_length_treatment = _clamp(session_length_control * (1.0 + rng.normal(0.004, 0.01)), 1e-4, 1 - 1e-4)

    revenue_control = _clamp(rng.uniform(0.08, 0.18), 1e-4, 1 - 1e-4)
    revenue_treatment = _clamp(revenue_control * (1.0 + rng.normal(0.01, 0.015)), 1e-4, 1 - 1e-4)

    return {
        "session_length": _aggregate_result_from_rates(
            control_rate=session_length_control,
            treatment_rate=session_length_treatment,
            control_count=control_count,
            treatment_count=treatment_count,
        ),
        "revenue_per_user": _aggregate_result_from_rates(
            control_rate=revenue_control,
            treatment_rate=revenue_treatment,
            control_count=control_count,
            treatment_count=treatment_count,
        ),
    }


class DataGenerator:
    """Generates deterministic synthetic experiment payloads from contamination specs."""

    @staticmethod
    def generate(spec: ContaminationSpec, seed: int) -> dict[str, Any]:
        """Generate synthetic experiment data with deterministic seed control.

        Args:
            spec: Hidden contamination specification for the episode.
            seed: Random seed controlling all stochastic generation.

        Returns:
            Dictionary containing initial observation fields and all revealable query payloads.
        """
        seed_offsets = {
            "clean": 11,
            "srm": 23,
            "sutva_violation": 37,
            "novelty_effect": 41,
            "simpsons_paradox": 53,
            "network_spillover": 67,
            "multiple_testing": 79,
            "underpowered_overclaim": 97,
        }
        rng = np.random.default_rng(seed + seed_offsets[spec.contamination_type])

        total_count = int(rng.integers(10_000, 500_001))
        intended_split = 0.50

        if spec.contamination_type == "srm":
            treatment_split = spec.srm_actual_split if spec.srm_actual_split is not None else float(rng.uniform(0.41, 0.46))
        else:
            treatment_split = 0.50

        treatment_count = int(round(total_count * treatment_split))
        control_count = max(total_count - treatment_count, 1)
        treatment_count = max(treatment_count, 1)

        control_rate = float(rng.uniform(0.12, 0.58))
        treatment_rate = _clamp(control_rate * (1.0 + spec.visible_effect_size), 1e-4, 1 - 1e-4)

        aggregate_results = _aggregate_result_from_rates(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            control_count=control_count,
            treatment_count=treatment_count,
        )

        start_date = date(2024, int(rng.integers(1, 11)), int(rng.integers(1, 22)))
        duration_days = int(rng.integers(21, 36))
        end_date = start_date + timedelta(days=duration_days)

        experiment_id = f"exp_2024_{spec.contamination_type}_{seed:05d}"
        hypothesis = "Treatment improves primary metric without harming guardrails."
        primary_metric = "d7_retention_rate"
        platform = str(rng.choice(["mobile_ios", "mobile_android", "web"]))
        targeting_rule = "all_users_18_plus_us"

        if spec.contamination_type == "sutva_violation":
            primary_metric = "purchase_conversion_rate"
            hypothesis = "New checkout flow increases purchase conversion."
            targeting_rule = "us_users_purchased_last_90_days"
        elif spec.contamination_type == "clean":
            primary_metric = "click_through_rate"
            hypothesis = "New ranking improves click-through rate."
            targeting_rule = "all_search_users_us"
        elif spec.contamination_type == "simpsons_paradox":
            primary_metric = "d7_retention_rate"
            hypothesis = "Redesigned feed increases D7 retention for mobile users."
            targeting_rule = "all_mobile_users"

        experiment_metadata = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "targeting_rule": targeting_rule,
            "intended_split": intended_split,
            "randomization_unit": "user_id",
            "platform": platform,
            "experiment_owner": str(rng.choice(["growth_team", "product_team", "search_team", "commerce_team"])),
            "hypothesis": hypothesis,
        }

        temporal_breakdown = _generate_temporal_breakdown(
            rng=rng,
            start_date=start_date,
            end_date=end_date,
            control_rate=aggregate_results["control_mean"],
            treatment_rate=aggregate_results["treatment_mean"],
            control_count=control_count,
            treatment_count=treatment_count,
            spec=spec,
        )

        subgroup_payload: dict[str, list[dict[str, Any]]] = {
            "device_type": _generic_subgroup_rows(
                rng=rng,
                dimension="device_type",
                values=["ios", "android", "web"],
                control_rate=aggregate_results["control_mean"],
                treatment_rate=aggregate_results["treatment_mean"],
                total_control=control_count,
                total_treatment=treatment_count,
            ),
            "country": _generic_subgroup_rows(
                rng=rng,
                dimension="country",
                values=["us", "ca", "uk"],
                control_rate=aggregate_results["control_mean"],
                treatment_rate=aggregate_results["treatment_mean"],
                total_control=control_count,
                total_treatment=treatment_count,
            ),
            "user_segment": _generic_subgroup_rows(
                rng=rng,
                dimension="user_segment",
                values=["new", "returning", "high_value"],
                control_rate=aggregate_results["control_mean"],
                treatment_rate=aggregate_results["treatment_mean"],
                total_control=control_count,
                total_treatment=treatment_count,
            ),
            "platform_version": _generic_subgroup_rows(
                rng=rng,
                dimension="platform_version",
                values=["v1", "v2", "v3"],
                control_rate=aggregate_results["control_mean"],
                treatment_rate=aggregate_results["treatment_mean"],
                total_control=control_count,
                total_treatment=treatment_count,
            ),
        }

        subgroup_payload["enrollment_cohort"] = (
            _simpsons_enrollment_rows(control_count, treatment_count)
            if spec.contamination_type == "simpsons_paradox"
            else _generic_subgroup_rows(
                rng=rng,
                dimension="enrollment_cohort",
                values=["days_1_3", "days_4_21"],
                control_rate=aggregate_results["control_mean"],
                treatment_rate=aggregate_results["treatment_mean"],
                total_control=control_count,
                total_treatment=treatment_count,
            )
        )

        peer_experiment_id = spec.interference_experiment_id or "exp_2024_pricing_011"
        peer_experiment_list: list[dict[str, Any]] = [
            {
                "experiment_id": peer_experiment_id,
                "randomization_unit": "user_id" if spec.contamination_type == "sutva_violation" else "session_id",
                "time_overlap": True,
                "owner": "monetization_team",
            }
        ]

        overlap_fractions = {
            experiment_id: {
                "control": 0.71 if spec.contamination_type == "sutva_violation" else round(float(rng.uniform(0.02, 0.14)), 4),
                "treatment": 0.28 if spec.contamination_type == "sutva_violation" else round(float(rng.uniform(0.02, 0.14)), 4),
            }
        }

        network_fraction = (
            spec.network_spillover_fraction
            if spec.network_spillover_fraction is not None
            else (0.23 if spec.contamination_type == "network_spillover" else float(rng.uniform(0.01, 0.06)))
        )
        network_exposure_map = {
            "control_all": round(float(network_fraction), 6),
            "control_high_degree": round(float(min(1.0, network_fraction * 1.5)), 6),
            "control_low_degree": round(float(max(0.0, network_fraction * 0.6)), 6),
        }

        randomization_audit = {
            "algorithm": "hash_mod_user_id",
            "seed": seed,
            "assignment_log_complete": True,
            "notes": (
                "Detected stale bucketing cache during rollout."
                if spec.contamination_type == "srm"
                else (
                    f"Platform outage observed on day {(spec.ground_truth_evidence or {}).get('outage_day', 4)}; affected both arms symmetrically."
                    if spec.contamination_type == "clean"
                    else "No assignment anomalies detected."
                )
            ),
        }

        secondary_metrics = _secondary_metrics(
            control_count=control_count,
            treatment_count=treatment_count,
            contamination_type=spec.contamination_type,
            rng=rng,
        )

        mde_analysis = _mde_payload(
            control_rate=aggregate_results["control_mean"],
            treatment_rate=aggregate_results["treatment_mean"],
            actual_sample_per_arm=min(control_count, treatment_count),
            contamination_type=spec.contamination_type,
        )

        simulate_counterfactual = {
            "unconfounded_ate_estimate": round(spec.true_effect_size, 6),
            "confounding_robustness_value": round(float(rng.uniform(0.75, 0.95)), 4),
            "methodology": "Double Machine Learning (Causal Forest)",
        }

        expert_hint = "No obvious issues detected by the expert. Looks mostly clean."
        if spec.contamination_type == "srm":
            expert_hint = "Have you looked at the sample sizes? Sometimes the hashing bucket is skewed."
        elif spec.contamination_type == "simpsons_paradox":
            expert_hint = "I've seen something like this before. Try breaking it down by cohort or device type. Aggregates hide things."
        elif spec.contamination_type == "sutva_violation":
            expert_hint = "Wait, are there other experiments running at the same time? Or maybe users are interacting with each other?"
        elif spec.contamination_type == "novelty_effect":
            expert_hint = "Users always click more on new features on day 1. Check the daily time-series breakdown."
        elif spec.contamination_type == "network_spillover":
            expert_hint = "If control users interact with treatment users, the control mean gets contaminated."
        elif spec.contamination_type == "underpowered_overclaim":
            expert_hint = "Are we sure we have enough data to detect an effect this small? Might want to run an MDE check."
        elif spec.contamination_type == "multiple_testing":
            expert_hint = "If you look at 50 secondary metrics, one of them will be significant by chance. Check guardrails."

        request_expert_review = {
            "hint": f"The Principal Data Scientist says: '{expert_hint}'",
            "expert": "Dr. Sarah",
        }

        return {
            "experiment_id": experiment_id,
            "primary_metric": primary_metric,
            "aggregate_results": aggregate_results,
            "experiment_metadata": experiment_metadata,
            "available_queries": AVAILABLE_QUERIES,
            "query_payloads": {
                "run_srm_check": _srm_payload(
                    control_count=control_count,
                    treatment_count=treatment_count,
                    expected_split=intended_split,
                ),
                "query_temporal": temporal_breakdown,
                "query_subgroup": subgroup_payload,
                "query_assignment_overlap": {
                    "experiment_ids": [experiment_id, peer_experiment_id],
                    "overlap_fractions": overlap_fractions,
                },
                "check_network_exposure": network_exposure_map,
                "inspect_randomization": randomization_audit,
                "query_secondary_metrics": secondary_metrics,
                "compute_mde": mde_analysis,
                "peer_experiment_list": peer_experiment_list,
                "simulate_counterfactual": simulate_counterfactual,
                "request_expert_review": request_expert_review,
            },
        }
