"""
Self-Play Trainer for TFT RL.

Implements self-play training with:
- Curriculum learning (gradual transition from random bots to self-play)
- Agent pool management (past versions)
- ELO tracking and evaluation
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch.nn as nn

from src.rl.env.self_play_env import SelfPlayEnv
from src.rl.training.agent_pool import AgentPool, AgentVersion


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning schedule."""
    # Phase 1: 0 to phase1_end steps
    phase1_end: int = 200_000
    phase1_self_play_ratio: float = 0.5  # 50% self-play, 50% random

    # Phase 2: phase1_end to phase2_end steps
    phase2_end: int = 500_000
    phase2_self_play_ratio: float = 0.75  # 75% self-play, 25% random

    # Phase 3: phase2_end+ steps
    phase3_self_play_ratio: float = 1.0  # 100% self-play


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # Training
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    learning_rate: float = 1e-4
    ent_coef: float = 0.02  # Higher entropy for exploration

    # Agent pool
    pool_save_interval: int = 50_000
    pool_max_size: int = 20
    pool_min_size: int = 10

    # Evaluation
    eval_interval: int = 100_000
    eval_episodes: int = 100

    # Curriculum
    curriculum: CurriculumConfig = None

    # Stabilization
    min_win_rate_threshold: float = 0.4
    lr_reduction_factor: float = 0.5
    ent_increase_factor: float = 1.5

    def __post_init__(self):
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()


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


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training.

    Handles:
    - Agent pool management (saving versions)
    - Curriculum schedule updates
    - Periodic evaluation
    - Learning stabilization
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        agent_pool: AgentPool,
        envs: List[SelfPlayEnv],
        log_dir: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config = config
        self.agent_pool = agent_pool
        self.envs = envs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.placements = []
        self.episode_count = 0
        self.last_save_step = 0
        self.last_eval_step = 0

        # ELO history
        self.elo_history: List[Dict[str, Any]] = []

        # Evaluation results
        self.eval_results: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        # Track episode completions
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if idx < len(infos):
                    info = infos[idx]
                    if "placement" in info and info["placement"] is not None:
                        self.placements.append(info["placement"])
                        self.episode_count += 1

        timestep = self.num_timesteps

        # Update curriculum schedule
        self._update_curriculum(timestep)

        # Save to agent pool
        if timestep - self.last_save_step >= self.config.pool_save_interval:
            self._save_to_pool(timestep)
            self.last_save_step = timestep

        # Periodic evaluation
        if timestep - self.last_eval_step >= self.config.eval_interval:
            self._run_evaluation(timestep)
            self.last_eval_step = timestep

        return True

    def _on_rollout_end(self) -> None:
        if self.placements:
            recent = self.placements[-100:]
            avg_placement = np.mean(recent)
            top4_rate = sum(1 for p in recent if p <= 4) / len(recent)
            win_rate = sum(1 for p in recent if p == 1) / len(recent)

            self.logger.record("tft/avg_placement", avg_placement)
            self.logger.record("tft/top4_rate", top4_rate)
            self.logger.record("tft/win_rate", win_rate)
            self.logger.record("tft/episodes", len(self.placements))

            # Log agent pool stats
            pool_stats = self.agent_pool.get_pool_stats()
            self.logger.record("pool/size", pool_stats["size"])
            self.logger.record("pool/avg_elo", pool_stats["avg_elo"])

            # Log curriculum phase
            timestep = self.num_timesteps
            curriculum = self.config.curriculum
            if timestep < curriculum.phase1_end:
                phase = 1
                ratio = curriculum.phase1_self_play_ratio
            elif timestep < curriculum.phase2_end:
                phase = 2
                ratio = curriculum.phase2_self_play_ratio
            else:
                phase = 3
                ratio = curriculum.phase3_self_play_ratio

            self.logger.record("curriculum/phase", phase)
            self.logger.record("curriculum/self_play_ratio", ratio)

            if self.verbose > 0 and len(self.placements) % 100 == 0:
                print(
                    f"  Episodes: {len(self.placements)}, "
                    f"Avg Placement: {avg_placement:.2f}, "
                    f"Top 4: {top4_rate*100:.1f}%, "
                    f"Pool Size: {pool_stats['size']}, "
                    f"Phase: {phase}"
                )

    def _update_curriculum(self, timestep: int):
        """Update self-play ratio based on curriculum schedule."""
        curriculum = self.config.curriculum

        if timestep < curriculum.phase1_end:
            ratio = curriculum.phase1_self_play_ratio
        elif timestep < curriculum.phase2_end:
            ratio = curriculum.phase2_self_play_ratio
        else:
            ratio = curriculum.phase3_self_play_ratio

        # Update all environments
        for env in self.envs:
            if hasattr(env, "set_self_play_ratio"):
                env.set_self_play_ratio(ratio)

    def _save_to_pool(self, timestep: int):
        """Save current model to agent pool."""
        if self.verbose > 0:
            print(f"\n[Step {timestep}] Saving agent to pool...")

        # Evaluate vs random bots quickly
        win_rate_vs_random = None
        avg_placement_vs_random = None

        if self.placements:
            recent = self.placements[-50:]
            avg_placement_vs_random = np.mean(recent)
            win_rate_vs_random = sum(1 for p in recent if p == 1) / len(recent)

        # Save to pool
        version = self.agent_pool.save_agent(
            self.model,
            timestep,
            win_rate_vs_random=win_rate_vs_random,
            avg_placement_vs_random=avg_placement_vs_random,
        )

        # Update opponent policy in environments
        self._update_opponent_policy()

        if self.verbose > 0:
            print(
                f"  Saved version {version.version_id} "
                f"(Pool size: {len(self.agent_pool)})"
            )

    def _update_opponent_policy(self):
        """Update opponent policy in all environments."""
        if len(self.agent_pool) == 0:
            return

        def make_policy(model):
            def policy_fn(obs):
                action, _ = model.predict(obs, deterministic=False)
                return int(action)
            return policy_fn

        # Sample opponent from pool
        opponent_version = self.agent_pool.sample_opponent(exclude_latest=True)
        if opponent_version is None:
            opponent_version = self.agent_pool.get_latest()

        if opponent_version is not None:
            opponent_model = self.agent_pool.load_agent(opponent_version, PPO)
            policy_fn = make_policy(opponent_model)

            for env in self.envs:
                if hasattr(env, "set_opponent_policy"):
                    env.set_opponent_policy(policy_fn)

    def _run_evaluation(self, timestep: int):
        """Run evaluation against random bots and past versions."""
        if self.verbose > 0:
            print(f"\n[Step {timestep}] Running evaluation...")

        results = {
            "timestep": timestep,
            "vs_random": self._evaluate_vs_random(),
            "vs_pool": self._evaluate_vs_pool(),
        }

        self.eval_results.append(results)

        # Log results
        self.logger.record("eval/random_top4_rate", results["vs_random"]["top4_rate"])
        self.logger.record("eval/random_avg_placement", results["vs_random"]["avg_placement"])

        if results["vs_pool"]:
            self.logger.record("eval/pool_win_rate", results["vs_pool"]["win_rate"])
            self.logger.record("eval/pool_avg_placement", results["vs_pool"]["avg_placement"])

        # Check for stabilization
        self._check_stabilization(results)

        if self.verbose > 0:
            print(f"  vs Random: Top4={results['vs_random']['top4_rate']*100:.1f}%")
            if results["vs_pool"]:
                print(f"  vs Pool: WinRate={results['vs_pool']['win_rate']*100:.1f}%")

        return results

    def _evaluate_vs_random(self, n_episodes: int = None) -> Dict[str, float]:
        """Evaluate current model against random bots."""
        if n_episodes is None:
            n_episodes = min(self.config.eval_episodes, 20)  # Quick eval

        env = SelfPlayEnv(num_players=8, max_rounds=50)
        env.set_self_play_ratio(0.0)  # All random bots

        placements = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)

                # Action masking
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

        return {
            "avg_placement": np.mean(placements),
            "top4_rate": sum(1 for p in placements if p <= 4) / len(placements),
            "win_rate": sum(1 for p in placements if p == 1) / len(placements),
            "placements": placements,
        }

    def _evaluate_vs_pool(self, n_episodes: int = None) -> Optional[Dict[str, float]]:
        """Evaluate current model against agents from the pool."""
        if len(self.agent_pool) < 2:
            return None

        if n_episodes is None:
            n_episodes = min(self.config.eval_episodes, 20)

        # Create environment with self-play opponents
        env = SelfPlayEnv(num_players=8, max_rounds=50)
        env.set_self_play_ratio(1.0)  # All self-play

        # Use pool opponent
        opponent_version = self.agent_pool.sample_opponent(exclude_latest=True)
        if opponent_version is None:
            return None

        opponent_model = self.agent_pool.load_agent(opponent_version, PPO)

        def policy_fn(obs):
            action, _ = opponent_model.predict(obs, deterministic=False)
            return int(action)

        env.set_opponent_policy(policy_fn)

        placements = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)

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

        # Update ELO
        current_version = self.agent_pool.get_latest()
        if current_version:
            wins = sum(1 for p in placements if p == 1)
            losses = n_episodes - wins

            for _ in range(wins):
                self.agent_pool.update_elo(current_version, opponent_version)
            for _ in range(losses):
                self.agent_pool.update_elo(opponent_version, current_version)

        return {
            "avg_placement": np.mean(placements),
            "top4_rate": sum(1 for p in placements if p <= 4) / len(placements),
            "win_rate": sum(1 for p in placements if p == 1) / len(placements),
            "opponent_version": opponent_version.version_id,
            "placements": placements,
        }

    def _check_stabilization(self, results: Dict):
        """Check if training needs stabilization."""
        # Check win rate vs pool
        if results["vs_pool"]:
            win_rate = results["vs_pool"]["win_rate"]
            if win_rate < self.config.min_win_rate_threshold:
                if self.verbose > 0:
                    print(
                        f"  Warning: Low win rate ({win_rate*100:.1f}%), "
                        "considering stabilization..."
                    )
                # Could reduce learning rate or increase entropy here
                # For now, just log the warning

    def get_elo_history(self) -> List[Dict[str, Any]]:
        """Get ELO rating history for all versions."""
        history = []
        for version in self.agent_pool.versions:
            history.append({
                "version_id": version.version_id,
                "timestep": version.timestep,
                "elo_rating": version.elo_rating,
                "games_played": version.games_played,
            })
        return history


class SelfPlayTrainer:
    """
    Self-Play Trainer for TFT.

    Usage:
        trainer = SelfPlayTrainer(config)
        trainer.train()
        results = trainer.get_results()
    """

    def __init__(
        self,
        config: Optional[SelfPlayConfig] = None,
        save_dir: str = "models/self_play",
        log_dir: str = "logs/self_play",
    ):
        self.config = config or SelfPlayConfig()
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.agent_pool = AgentPool(
            save_dir=str(self.save_dir / "agent_pool"),
            max_size=self.config.pool_max_size,
            min_size=self.config.pool_min_size,
            save_interval=self.config.pool_save_interval,
        )

        self.model = None
        self.envs = None
        self.callback = None

    def _create_envs(self) -> DummyVecEnv:
        """Create vectorized environments."""
        self.raw_envs = []

        def make_env():
            env = SelfPlayEnv(
                num_players=8,
                max_rounds=50,
                self_play_ratio=self.config.curriculum.phase1_self_play_ratio,
                place_units_for_bots=True,
            )
            self.raw_envs.append(env)
            return env

        vec_env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
        return vec_env

    def _create_model(self, env: DummyVecEnv) -> PPO:
        """Create PPO model."""
        policy_kwargs = dict(
            features_extractor_class=TFTFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=self.config.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            device="cpu",
        )

        return model

    def train(self, resume_path: Optional[str] = None):
        """
        Run self-play training.

        Args:
            resume_path: Optional path to resume from
        """
        print("=" * 60)
        print("TFT Self-Play Training")
        print("=" * 60)
        print(f"Total steps: {self.config.total_timesteps:,}")
        print(f"Parallel envs: {self.config.n_envs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Entropy coef: {self.config.ent_coef}")
        print(f"Pool save interval: {self.config.pool_save_interval:,}")
        print(f"Eval interval: {self.config.eval_interval:,}")
        print()
        print("Curriculum Schedule:")
        c = self.config.curriculum
        print(f"  Phase 1 (0-{c.phase1_end:,}): {c.phase1_self_play_ratio*100:.0f}% self-play")
        print(f"  Phase 2 ({c.phase1_end:,}-{c.phase2_end:,}): {c.phase2_self_play_ratio*100:.0f}% self-play")
        print(f"  Phase 3 ({c.phase2_end:,}+): {c.phase3_self_play_ratio*100:.0f}% self-play")
        print("=" * 60)

        # Create environments
        self.envs = self._create_envs()

        # Create or load model
        if resume_path:
            print(f"Resuming from: {resume_path}")
            self.model = PPO.load(resume_path, env=self.envs)
        else:
            self.model = self._create_model(self.envs)

        # Create callback
        self.callback = SelfPlayCallback(
            config=self.config,
            agent_pool=self.agent_pool,
            envs=self.raw_envs,
            log_dir=str(self.log_dir),
            verbose=1,
        )

        # Train
        print("\nStarting training...")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=self.callback,
        )

        # Save final model
        final_path = self.save_dir / "final_model"
        self.model.save(str(final_path))
        print(f"\nFinal model saved to: {final_path}")

        # Run final evaluation
        print("\nRunning final evaluation...")
        final_results = self._final_evaluation()

        return final_results

    def _final_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive final evaluation."""
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        # Evaluate vs random bots (100 episodes)
        vs_random = self._evaluate_vs_random_full()
        print(f"\nVs Random Bots (100 episodes):")
        print(f"  Average Placement: {vs_random['avg_placement']:.2f}")
        print(f"  Top 4 Rate: {vs_random['top4_rate']*100:.1f}%")
        print(f"  Win Rate: {vs_random['win_rate']*100:.1f}%")

        # Evaluate vs past versions
        vs_past = self._evaluate_vs_past_versions()
        if vs_past:
            print(f"\nVs Past Versions (10 gen ago):")
            print(f"  Average Placement: {vs_past['avg_placement']:.2f}")
            print(f"  Win Rate: {vs_past['win_rate']*100:.1f}%")

        # ELO summary
        elo_history = self.callback.get_elo_history() if self.callback else []
        print(f"\nAgent Pool Stats:")
        pool_stats = self.agent_pool.get_pool_stats()
        print(f"  Pool Size: {pool_stats['size']}")
        print(f"  Average ELO: {pool_stats['avg_elo']:.1f}")
        print(f"  Max ELO: {pool_stats['max_elo']:.1f}")
        print(f"  Min ELO: {pool_stats['min_elo']:.1f}")

        # Check success criteria
        print("\n" + "=" * 60)
        print("Success Criteria Check")
        print("=" * 60)

        success = True

        # Criteria 1: Random bot Top 4 rate >= 90%
        if vs_random["top4_rate"] >= 0.9:
            print(f"✓ Random bot Top 4 rate >= 90%: {vs_random['top4_rate']*100:.1f}%")
        else:
            print(f"✗ Random bot Top 4 rate >= 90%: {vs_random['top4_rate']*100:.1f}%")
            success = False

        # Criteria 2: Win rate vs 10-gen-ago version >= 60%
        if vs_past and vs_past["win_rate"] >= 0.6:
            print(f"✓ Win rate vs past version >= 60%: {vs_past['win_rate']*100:.1f}%")
        elif vs_past:
            print(f"✗ Win rate vs past version >= 60%: {vs_past['win_rate']*100:.1f}%")
            success = False
        else:
            print("  (Could not evaluate vs past version - pool too small)")

        print("=" * 60)
        if success:
            print("SUCCESS: All criteria met!")
        else:
            print("Some criteria not met. Consider longer training or parameter tuning.")

        return {
            "vs_random": vs_random,
            "vs_past": vs_past,
            "pool_stats": pool_stats,
            "elo_history": elo_history,
            "success": success,
        }

    def _evaluate_vs_random_full(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Full evaluation against random bots."""
        env = SelfPlayEnv(num_players=8, max_rounds=50)
        env.set_self_play_ratio(0.0)

        placements = []
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)

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
                print(f"  Episode {ep + 1}/{n_episodes}: Placement {placement}")

        return {
            "avg_placement": np.mean(placements),
            "top4_rate": sum(1 for p in placements if p <= 4) / len(placements),
            "win_rate": sum(1 for p in placements if p == 1) / len(placements),
            "placements": placements,
        }

    def _evaluate_vs_past_versions(self, generations_ago: int = 10) -> Optional[Dict[str, Any]]:
        """Evaluate against a version from N generations ago."""
        if len(self.agent_pool) < generations_ago + 1:
            return None

        # Get version from N generations ago
        past_version = self.agent_pool.versions[-(generations_ago + 1)]
        past_model = self.agent_pool.load_agent(past_version, PPO)

        env = SelfPlayEnv(num_players=8, max_rounds=50)
        env.set_self_play_ratio(1.0)

        def policy_fn(obs):
            action, _ = past_model.predict(obs, deterministic=False)
            return int(action)

        env.set_opponent_policy(policy_fn)

        placements = []
        n_episodes = 100

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)

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

        return {
            "avg_placement": np.mean(placements),
            "top4_rate": sum(1 for p in placements if p <= 4) / len(placements),
            "win_rate": sum(1 for p in placements if p == 1) / len(placements),
            "opponent_version": past_version.version_id,
            "generations_ago": generations_ago,
            "placements": placements,
        }

    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            "pool_stats": self.agent_pool.get_pool_stats(),
            "eval_results": self.callback.eval_results if self.callback else [],
            "elo_history": self.callback.get_elo_history() if self.callback else [],
        }


def main():
    """Run self-play training."""
    config = SelfPlayConfig(
        total_timesteps=2_000_000,
        n_envs=8,
        learning_rate=1e-4,
        ent_coef=0.02,
        pool_save_interval=50_000,
        eval_interval=100_000,
    )

    trainer = SelfPlayTrainer(
        config=config,
        save_dir="models/self_play",
        log_dir="logs/self_play",
    )

    results = trainer.train()
    return results


if __name__ == "__main__":
    main()
