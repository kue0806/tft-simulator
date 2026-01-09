"""
TFT Agent Training Pipeline.

Provides PPO training with invalid action masking for TFT environment.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    torch = None
    nn = None
    PPO = None
    BaseCallback = object
    DummyVecEnv = None
    BaseFeaturesExtractor = object
    gym = None

from src.rl.env.tft_env import TFTEnv


if SB3_AVAILABLE:

    class TFTFeaturesExtractor(BaseFeaturesExtractor):
        """
        Custom feature extractor for TFT state.

        Uses structured feature extraction to leverage TFT state structure.
        """

        def __init__(
            self, observation_space: gym.spaces.Box, features_dim: int = 256
        ):
            super().__init__(observation_space, features_dim)

            input_dim = observation_space.shape[0]

            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.network(observations)

else:

    class TFTFeaturesExtractor:
        """Placeholder when SB3 not available."""

        def __init__(self, observation_space=None, features_dim: int = 256):
            pass


class MaskablePPO:
    """
    PPO with Invalid Action Masking.

    Applies large negative logits to invalid actions.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        device: str = "auto",
        verbose: int = 1,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 and torch required. "
                "Install with: pip install stable-baselines3 torch"
            )

        policy_kwargs = dict(
            features_extractor_class=TFTFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
        )

    def predict(
        self, obs: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> int:
        """Prediction with masking."""
        action, _ = self.model.predict(obs, deterministic=False)

        if mask is not None:
            # If invalid action, select random valid one
            if mask[action] == 0:
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0  # PASS

        return int(action)

    def learn(
        self, total_timesteps: int, callback: Optional[BaseCallback] = None
    ):
        """Train agent."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path: str):
        """Save model."""
        self.model.save(path)

    def load(self, path: str):
        """Load model."""
        self.model = PPO.load(path)


class TrainingCallback(BaseCallback):
    """Training callback for logging."""

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.placements: List[int] = []

    def _on_step(self) -> bool:
        # Log episode end
        for info in self.locals.get("infos", []):
            if "placement" in info and info["placement"] is not None:
                self.placements.append(info["placement"])

        return True

    def _on_rollout_end(self) -> None:
        # Rollout end statistics
        if self.placements:
            avg_placement = np.mean(self.placements[-100:])
            self.logger.record("tft/avg_placement", avg_placement)

            recent = self.placements[-100:]
            top4_rate = sum(1 for p in recent if p <= 4) / len(recent)
            self.logger.record("tft/top4_rate", top4_rate)


def train_agent(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    save_dir: str = "models/tft_agent",
    log_dir: str = "logs/tft",
) -> Optional[MaskablePPO]:
    """
    Train TFT agent.

    Args:
        total_timesteps: Total training timesteps.
        n_envs: Number of parallel environments.
        learning_rate: Learning rate.
        save_dir: Model save directory.
        log_dir: Log directory.

    Returns:
        Trained agent or None if SB3 not available.
    """
    if not SB3_AVAILABLE:
        print("stable-baselines3 not available. Skipping training.")
        return None

    # Create environments
    def make_env():
        return TFTEnv(render_mode=None)

    env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Create agent
    agent = MaskablePPO(
        env=env,
        learning_rate=learning_rate,
        verbose=1,
    )

    # Callback
    callback = TrainingCallback(log_dir)

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, callback=callback)

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_path / f"tft_ppo_{timestamp}"
    agent.save(str(model_path))

    print(f"Model saved to {model_path}")

    return agent


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate agent.

    Args:
        model_path: Model path.
        n_episodes: Number of evaluation episodes.

    Returns:
        Evaluation results.
    """
    if not SB3_AVAILABLE:
        print("stable-baselines3 not available. Skipping evaluation.")
        return {}

    env = TFTEnv(render_mode=None)

    # Create agent and load
    dummy_env = DummyVecEnv([lambda: TFTEnv(render_mode=None)])
    agent = MaskablePPO(env=dummy_env)
    agent.load(model_path)

    placements: List[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            mask = info.get("valid_action_mask")
            action = agent.predict(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        placement = info.get("placement", 8)
        placements.append(placement)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}: Placement {placement}")

    # Results
    avg_placement = np.mean(placements)
    top4_rate = sum(1 for p in placements if p <= 4) / len(placements)
    win_rate = sum(1 for p in placements if p == 1) / len(placements)

    print(f"\n=== Evaluation Results ({n_episodes} episodes) ===")
    print(f"Average Placement: {avg_placement:.2f}")
    print(f"Top 4 Rate: {top4_rate * 100:.1f}%")
    print(f"Win Rate: {win_rate * 100:.1f}%")

    return {
        "avg_placement": avg_placement,
        "top4_rate": top4_rate,
        "win_rate": win_rate,
        "placements": placements,
    }


if __name__ == "__main__":
    # Training run
    agent = train_agent(
        total_timesteps=500_000,
        n_envs=4,
    )

    # Evaluation
    if agent:
        evaluate_agent("models/tft_agent/tft_ppo_latest", n_episodes=50)
