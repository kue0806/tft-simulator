"""
TFT RL Training Module.

Contains training utilities, agent pool management, and self-play training.
"""

from .agent_pool import AgentPool, AgentVersion
from .self_play_trainer import (
    SelfPlayTrainer,
    SelfPlayConfig,
    CurriculumConfig,
    SelfPlayCallback,
)

__all__ = [
    "AgentPool",
    "AgentVersion",
    "SelfPlayTrainer",
    "SelfPlayConfig",
    "CurriculumConfig",
    "SelfPlayCallback",
]

