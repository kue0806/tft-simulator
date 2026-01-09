"""
MaskablePPO Model.

Uses sb3-contrib's MaskablePPO for proper action masking during training.
This is the recommended approach for environments with invalid actions.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym

    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    torch = None
    nn = None
    MaskablePPO = None
    BaseFeaturesExtractor = object
    BaseCallback = object
    gym = None

from .base import BaseRLModel, ModelConfig, TrainingMetrics


class TFTFeaturesExtractor(BaseFeaturesExtractor if SB3_CONTRIB_AVAILABLE else object):
    """
    Feature extractor for TFT state.

    Uses a deep MLP with layer normalization for stable training.
    """

    def __init__(
        self,
        observation_space: "gym.spaces.Box",
        features_dim: int = 256,
        hidden_dims: tuple = (512, 256),
    ):
        if not SB3_CONTRIB_AVAILABLE:
            return

        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim, features_dim),
            nn.ReLU(),
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, observations: "torch.Tensor") -> "torch.Tensor":
        return self.network(observations)


class MaskablePPOCallback(BaseCallback if SB3_CONTRIB_AVAILABLE else object):
    """Callback for tracking training progress."""

    def __init__(self, metrics: TrainingMetrics, verbose: int = 0):
        if SB3_CONTRIB_AVAILABLE:
            super().__init__(verbose)
        self.metrics = metrics
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Track episode metrics
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[idx]
                placement = info.get("placement", 8)
                episode_reward = info.get("episode", {}).get("r", 0)
                episode_length = info.get("episode", {}).get("l", 0)

                self.metrics.add_episode(
                    reward=episode_reward,
                    length=episode_length,
                    placement=placement,
                )

        return True

    def _on_rollout_end(self) -> None:
        stats = self.metrics.get_recent_stats(100)

        if stats:
            self.logger.record("tft/avg_placement", stats.get("avg_placement", 0))
            self.logger.record("tft/top4_rate", stats.get("top4_rate", 0))
            self.logger.record("tft/win_rate", stats.get("win_rate", 0))
            self.logger.record("tft/avg_reward", stats.get("avg_reward", 0))


class MaskablePPOModel(BaseRLModel):
    """
    MaskablePPO from sb3-contrib.

    Key features:
    - Proper action masking during policy gradient computation
    - Masked actions get -inf logits, ensuring 0 probability
    - Most theoretically correct approach for action masking
    """

    def __init__(self, env, config: Optional[ModelConfig] = None):
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "sb3-contrib required for MaskablePPO. "
                "Install with: pip install sb3-contrib"
            )
        super().__init__(env, config)

    def _setup_model(self):
        """Initialize MaskablePPO model."""
        policy_kwargs = dict(
            features_extractor_class=TFTFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=self.config.features_dim,
                hidden_dims=tuple(self.config.hidden_dims),
            ),
            net_arch=dict(
                pi=self.config.hidden_dims,
                vf=self.config.hidden_dims,
            ),
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=self.config.seed,
        )

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Select action with masking."""
        action, states = self.model.predict(
            obs,
            action_masks=action_mask,
            deterministic=deterministic,
        )
        return int(action), {"states": states}

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ) -> "MaskablePPOModel":
        """Train the model."""
        # Create callback for metrics tracking
        metrics_callback = MaskablePPOCallback(self.metrics, verbose=self.config.verbose)

        callbacks = [metrics_callback]
        if callback is not None:
            callbacks.append(callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )

        return self

    def save(self, path: str):
        """Save model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))

    def load(self, path: str) -> "MaskablePPOModel":
        """Load model."""
        self.model = MaskablePPO.load(path, env=self.env)
        return self

    @property
    def name(self) -> str:
        return "MaskablePPO"
