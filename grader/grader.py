"""Deterministic grading stubs."""

from __future__ import annotations

from typing import Any

from models.contamination_spec import ContaminationSpec


class Grader:
    """Grades completed episode logs against hidden contamination specs."""

    @staticmethod
    def grade_episode(
        episode_log: list[dict[str, Any]],
        spec: ContaminationSpec,
    ) -> dict[str, Any]:
        """Grade a full episode log and return a structured score payload.

        Args:
            episode_log: Ordered list of action and reward records for an episode.
            spec: Hidden contamination specification for ground-truth evaluation.

        Returns:
            A score dictionary containing final score and breakdown fields.
        """
        pass
