"""
RL Models for TFT Agent.

Provides multiple model architectures for comparison:
1. MaskablePPO - PPO with sb3-contrib action masking
2. CustomMaskedPPO - PPO with custom masking implementation
3. DuelingDQN - Dueling DQN architecture
4. TransformerPPO - Transformer-based policy network with PPO
"""

from .base import BaseRLModel, ModelConfig
from .maskable_ppo import MaskablePPOModel
from .custom_masked_ppo import CustomMaskedPPO
from .dueling_dqn import DuelingDQNModel
from .transformer_ppo import TransformerPPO

__all__ = [
    "BaseRLModel",
    "ModelConfig",
    "MaskablePPOModel",
    "CustomMaskedPPO",
    "DuelingDQNModel",
    "TransformerPPO",
]
