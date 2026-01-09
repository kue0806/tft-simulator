#!/usr/bin/env python
"""
Training script for full TFT environment.
8 players, 50 rounds, 1M timesteps.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn

from src.rl.env.tft_env import TFTEnv


class TFTFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for TFT state."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
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


class TrainingCallback(BaseCallback):
    """Training callback for logging TFT-specific metrics."""

    def __init__(self, log_dir: str, num_players: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_players = num_players

        self.placements = []
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode end
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if idx < len(infos):
                    info = infos[idx]
                    if "placement" in info and info["placement"] is not None:
                        self.placements.append(info["placement"])

        return True

    def _on_rollout_end(self) -> None:
        if self.placements:
            recent = self.placements[-100:]
            avg_placement = np.mean(recent)
            # Top 4 in 8-player game
            top4_rate = sum(1 for p in recent if p <= 4) / len(recent)
            win_rate = sum(1 for p in recent if p == 1) / len(recent)

            self.logger.record("tft/avg_placement", avg_placement)
            self.logger.record("tft/top4_rate", top4_rate)
            self.logger.record("tft/win_rate", win_rate)
            self.logger.record("tft/episodes", len(self.placements))

            if len(self.placements) % 100 == 0:
                print(f"  Episodes: {len(self.placements)}, Avg Placement: {avg_placement:.2f}, "
                      f"Top 4: {top4_rate*100:.1f}%, Win: {win_rate*100:.1f}%")


def train():
    """Train TFT agent on full 8-player environment."""

    print("=" * 60)
    print("TFT RL Training - Full 8-Player Environment")
    print("=" * 60)
    print(f"Players: 8")
    print(f"Max rounds: 50")
    print(f"Timesteps: 1,000,000")
    print(f"Parallel envs: 8")
    print(f"Learning rate: 1e-4")
    print("=" * 60)

    # Create environments
    def make_env():
        return TFTEnv(num_players=8, max_rounds=50, render_mode=None)

    # Use DummyVecEnv for stability (SubprocVecEnv can have issues)
    env = DummyVecEnv([make_env for _ in range(8)])

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=TFTFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Can increase to 0.05 if needed
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/tft_full",
        device="cpu",  # Use CPU for MLP policy
    )

    # Callback
    callback = TrainingCallback("logs/tft_full", num_players=8)

    # Train
    print("\nStarting training...")
    print("This will take a while (~30-60 minutes for 1M steps)")
    model.learn(total_timesteps=1_000_000, callback=callback)

    # Save
    save_dir = Path("models/tft_full")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_dir / f"tft_ppo_{timestamp}"
    model.save(str(model_path))

    print(f"\nModel saved to: {model_path}")

    # Also save as 'latest'
    latest_path = save_dir / "tft_ppo_latest"
    model.save(str(latest_path))
    print(f"Also saved as: {latest_path}")

    return model, callback.placements


def evaluate(model_path: str, n_episodes: int = 100):
    """Evaluate trained model."""

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    env = TFTEnv(num_players=8, max_rounds=50, render_mode=None)
    model = PPO.load(model_path)

    placements = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=False)

            # Apply action masking
            mask = info.get("valid_action_mask")
            if mask is not None and mask[action] == 0:
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        placement = info.get("placement", 8)
        placements.append(placement)

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}: Placement {placement}")

    # Results
    avg_placement = np.mean(placements)
    top4_rate = sum(1 for p in placements if p <= 4) / len(placements)
    win_rate = sum(1 for p in placements if p == 1) / len(placements)

    print("\n" + "=" * 60)
    print(f"Evaluation Results ({n_episodes} episodes)")
    print("=" * 60)
    print(f"Average Placement: {avg_placement:.2f}")
    print(f"Top 4 Rate: {top4_rate * 100:.1f}%")
    print(f"Win Rate: {win_rate * 100:.1f}%")
    print()
    print("Placement distribution:")
    for rank in range(1, 9):
        count = sum(1 for p in placements if p == rank)
        pct = count / len(placements) * 100
        bar = "#" * int(pct / 2)
        print(f"  {rank}: {count:3d} ({pct:5.1f}%) {bar}")

    return {
        "avg_placement": avg_placement,
        "top4_rate": top4_rate,
        "win_rate": win_rate,
        "placements": placements,
    }


if __name__ == "__main__":
    # Train
    model, training_placements = train()

    # Evaluate
    results = evaluate("models/tft_full/tft_ppo_latest", n_episodes=100)

    # Check success criteria
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    if results["top4_rate"] >= 0.50:
        print("SUCCESS: Top 4 rate >= 50%")
    else:
        print(f"Target not reached: Top 4 rate = {results['top4_rate']*100:.1f}% (need 50%)")
        print("\nSuggestions for improvement:")
        print("  1. Increase entropy coefficient to 0.05")
        print("  2. Increase placement reward weights")
        print("  3. Increase training steps to 2M")
