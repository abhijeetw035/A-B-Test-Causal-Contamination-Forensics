"""Observation composition with progressive reveal logic."""

from __future__ import annotations

from models.observation import ExperimentObservation

from env.state_manager import EpisodeState


class ObservationBuilder:
    """Build initial and updated observations from mutable episode state."""

    @staticmethod
    def build_initial(state: EpisodeState) -> ExperimentObservation:
        """Build initial observation with only baseline fields.

        Args:
            state: Session state containing generated synthetic data.

        Returns:
            `ExperimentObservation` with no optional reveal fields populated.
        """
        data = state.data
        return ExperimentObservation(
            session_id=state.session_id,
            experiment_id=data["experiment_id"],
            primary_metric=data["primary_metric"],
            aggregate_results=data["aggregate_results"],
            experiment_metadata=data["experiment_metadata"],
            available_queries=data["available_queries"],
            steps_taken=state.step_count,
            steps_remaining=max(state.max_steps - state.step_count, 0),
            investigation_budget=state.budget,
            budget_spent=state.budget_used,
        )

    @staticmethod
    def build_updated(state: EpisodeState) -> ExperimentObservation:
        """Build cumulative observation with progressive unlock behavior.

        Args:
            state: Session state with `revealed_data` populated by executed actions.

        Returns:
            `ExperimentObservation` including all fields revealed so far.
        """
        observation = ObservationBuilder.build_initial(state).model_dump()

        if "run_srm_check" in state.revealed_data:
            observation["randomization_check"] = state.revealed_data["run_srm_check"]

        if "query_subgroup" in state.revealed_data:
            observation["subgroup_results"] = state.revealed_data["query_subgroup"]

        if "query_temporal" in state.revealed_data:
            observation["temporal_breakdown"] = state.revealed_data["query_temporal"]

        if "query_assignment_overlap" in state.revealed_data:
            observation["user_assignment_overlap"] = state.revealed_data["query_assignment_overlap"]
            observation["peer_experiment_list"] = state.data.get("query_payloads", {}).get("peer_experiment_list")

        if "check_network_exposure" in state.revealed_data:
            observation["network_exposure_map"] = state.revealed_data["check_network_exposure"]

        if "query_secondary_metrics" in state.revealed_data:
            observation["secondary_metric_results"] = state.revealed_data["query_secondary_metrics"]

        if "compute_mde" in state.revealed_data:
            observation["mde_analysis"] = state.revealed_data["compute_mde"]

        if "inspect_randomization" in state.revealed_data:
            observation["randomization_audit"] = state.revealed_data["inspect_randomization"]

        if "simulate_counterfactual" in state.revealed_data:
            observation["counterfactual_analysis"] = state.revealed_data["simulate_counterfactual"]

        if "request_expert_review" in state.revealed_data:
            observation["expert_review"] = state.revealed_data["request_expert_review"]

        observation["steps_taken"] = state.step_count
        observation["steps_remaining"] = max(state.max_steps - state.step_count, 0)
        observation["investigation_budget"] = state.budget
        observation["budget_spent"] = state.budget_used
        return ExperimentObservation(**observation)
