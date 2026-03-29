"""Reward models for environment step and episode scoring."""

from typing import Dict, List

from pydantic import BaseModel


class StepReward(BaseModel):
    """Reward information for one environment step."""

    step_reward: float
    components: Dict[str, float]
    cumulative_reward: float
    reasoning: str


class EpisodeReward(BaseModel):
    """Aggregated reward information for one episode."""

    total_reward: float
    step_rewards: List[StepReward]
    terminal_reward: float
    efficiency_penalty: float
    calibration_reward: float
