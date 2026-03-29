"""Task specification sampling stubs."""

from __future__ import annotations

import random

from models.contamination_spec import ContaminationSpec
from tasks.task_1_srm import get_task_specs as get_task_1_specs
from tasks.task_2_simpsons import get_task_specs as get_task_2_specs
from tasks.task_3_multilayer import get_task_specs as get_task_3_specs
from tasks.task_4_clean import get_task_specs as get_task_4_specs


TASK_SPEC_BUILDERS: dict[int, callable] = {
    1: get_task_1_specs,
    2: get_task_2_specs,
    3: get_task_3_specs,
    4: get_task_4_specs,
}


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
        if task_id not in TASK_SPEC_BUILDERS:
            raise ValueError(f"Unsupported task_id={task_id}. Expected one of: {sorted(TASK_SPEC_BUILDERS)}")

        specs = TASK_SPEC_BUILDERS[task_id]()
        if not specs:
            raise ValueError(f"No specs configured for task_id={task_id}")

        rng = random.Random((task_id * 1_000_003) + seed)
        return specs[rng.randrange(len(specs))]
