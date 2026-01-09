#!/usr/bin/env python
"""
Quick Training Test Script.

Tests the RL training pipeline with optimized settings.
Run with: python scripts/quick_train.py
"""

import time
import sys
from pathlib import Path

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Check dependencies
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable-baselines3 and torch required!")
    print("Install with: pip install stable-baselines3 torch")
    sys.exit(1)

from src.rl.env.tft_env import TFTEnv
from src.rl.env.state_encoder import EncoderConfig
from src.rl.env.reward_calculator import RewardConfig


class TrainingMetrics(BaseCallback):
    """Callback for tracking training metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.placements = []
        self.start_time = None
        self.episodes_completed = 0

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("Training Started")
        print("="*60)

    def _on_step(self) -> bool:
        # Check for episode end in infos
        for info in self.locals.get("infos", []):
            if "placement" in info and info["placement"] is not None:
                self.placements.append(info["placement"])
                self.episodes_completed += 1

                # Print progress every 10 episodes
                if self.episodes_completed % 10 == 0:
                    recent = self.placements[-100:]
                    avg_placement = np.mean(recent)
                    top4_rate = sum(1 for p in recent if p <= 4) / len(recent) * 100
                    win_rate = sum(1 for p in recent if p == 1) / len(recent) * 100

                    elapsed = time.time() - self.start_time
                    eps_per_sec = self.episodes_completed / elapsed

                    print(f"Episode {self.episodes_completed:4d} | "
                          f"Avg Place: {avg_placement:.2f} | "
                          f"Top4: {top4_rate:5.1f}% | "
                          f"Win: {win_rate:4.1f}% | "
                          f"{eps_per_sec:.1f} ep/s")

        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("Training Complete")
        print("="*60)
        print(f"Total episodes: {self.episodes_completed}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Episodes/sec: {self.episodes_completed/elapsed:.2f}")

        if self.placements:
            print(f"\nFinal Stats (last 100 episodes):")
            recent = self.placements[-100:]
            print(f"  Average Placement: {np.mean(recent):.2f}")
            print(f"  Top 4 Rate: {sum(1 for p in recent if p <= 4)/len(recent)*100:.1f}%")
            print(f"  Win Rate: {sum(1 for p in recent if p == 1)/len(recent)*100:.1f}%")
            print(f"  Placement Distribution:")
            for i in range(1, 9):
                count = sum(1 for p in recent if p == i)
                print(f"    {i}st: {count:3d} ({count/len(recent)*100:5.1f}%)")


def make_env(env_id: int = 0):
    """Create TFT environment with optimized config."""
    def _init():
        # Use optimized encoder config (no full opponent units)
        encoder_config = EncoderConfig(
            encode_opponent_units=False,  # Use summary instead
        )

        # Use dense reward config
        reward_config = RewardConfig()

        return TFTEnv(
            num_players=8,
            agent_player_idx=0,
            max_rounds=50,
            render_mode=None,
            encoder_config=encoder_config,
            reward_config=reward_config,
        )
    return _init


def benchmark_env():
    """Benchmark environment step speed."""
    print("\n" + "="*60)
    print("Environment Benchmark")
    print("="*60)

    env = make_env()()

    # Warmup
    obs, info = env.reset()
    for _ in range(100):
        mask = info.get("valid_action_mask", None)
        if mask is not None:
            valid_actions = np.where(mask > 0)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    # Benchmark
    num_steps = 1000
    start = time.perf_counter()

    obs, info = env.reset()
    for _ in range(num_steps):
        mask = info.get("valid_action_mask", None)
        if mask is not None:
            valid_actions = np.where(mask > 0)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    elapsed = time.perf_counter() - start
    steps_per_sec = num_steps / elapsed

    print(f"Steps: {num_steps}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Steps/sec: {steps_per_sec:.1f}")
    print(f"State dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.n}")

    env.close()
    return steps_per_sec


def quick_train(
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
):
    """Quick training run."""
    print("\n" + "="*60)
    print("Quick Training Configuration")
    print("="*60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")

    # Create parallel environments
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Get state/action dims
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")

    # Create PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=512,  # Reduced for faster updates
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
    )

    # Training callback
    callback = TrainingMetrics(verbose=1)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    env.close()
    return model, callback


def evaluate_random_agent(n_episodes: int = 50):
    """Evaluate random agent baseline."""
    print("\n" + "="*60)
    print("Random Agent Baseline")
    print("="*60)

    env = make_env()()
    placements = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            mask = info.get("valid_action_mask", None)
            if mask is not None:
                valid_actions = np.where(mask > 0)[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            else:
                action = env.action_space.sample()

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

        placement = info.get("placement", 8)
        placements.append(placement)

    print(f"Episodes: {n_episodes}")
    print(f"Average Placement: {np.mean(placements):.2f}")
    print(f"Top 4 Rate: {sum(1 for p in placements if p <= 4)/len(placements)*100:.1f}%")
    print(f"Win Rate: {sum(1 for p in placements if p == 1)/len(placements)*100:.1f}%")

    env.close()
    return placements


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="TFT RL Quick Training")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmark")
    parser.add_argument("--random-baseline", action="store_true", help="Only run random baseline")
    args = parser.parse_args()

    print("="*60)
    print("TFT RL Training Test")
    print("="*60)

    # Always run benchmark first
    benchmark_env()

    if args.benchmark_only:
        return

    # Random baseline
    if args.random_baseline:
        evaluate_random_agent(50)
        return

    # Run random baseline for comparison
    print("\nEvaluating random baseline first...")
    evaluate_random_agent(20)

    # Quick training
    model, callback = quick_train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
