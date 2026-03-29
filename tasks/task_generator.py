"""Task specification sampling stubs."""

from __future__ import annotations

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
        pass
