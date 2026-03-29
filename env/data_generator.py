"""Synthetic experiment data generation stubs."""

from __future__ import annotations

from typing import Any

from models.contamination_spec import ContaminationSpec


class DataGenerator:
    """Generates synthetic experiment payloads from hidden contamination specs."""

    @staticmethod
    def generate(spec: ContaminationSpec, seed: int) -> dict[str, Any]:
        """Generate a deterministic synthetic experiment payload for one episode.

        Args:
            spec: Hidden contamination specification for the episode.
            seed: Random seed used for deterministic data generation.

        Returns:
            A dictionary containing raw experiment data for observation building.
        """
        pass
