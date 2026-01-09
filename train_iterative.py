#!/usr/bin/env python3
"""
TFT RL Iterative Self-Play Training.

Continuously trains models against the best versions of other models.
Each iteration:
1. Load best models from previous iteration
2. Train each model against the others
3. Save improved models
4. Repeat until convergence or max iterations

This creates an arms race where models continuously improve.
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from src.rl.env.self_play_env import SelfPlayEnv, create_league_env
from src.rl.env.tft_env import TFTEnv
from src.rl.models.base import ModelConfig, TrainingMetrics
from src.rl.models.custom_masked_ppo import CustomMaskedPPO
from src.rl.models.dueling_dqn import DuelingDQNModel
from src.rl.models.transformer_ppo import TransformerPPO


@dataclass
class IterativeConfig:
    """Iterative training configuration."""
    timesteps_per_iteration: int = 50000  # Timesteps per model per iteration
    max_iterations: int = 20  # Maximum training iterations
    min_improvement: float = 0.05  # Minimum improvement to continue (placement)
    patience: int = 3  # Iterations without improvement before stopping
    eval_episodes: int = 100  # Episodes for evaluation
    save_every: int = 1  # Save models every N iterations


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_models_from_dir(model_dir: Path, device: str) -> Dict[str, Any]:
    """Load all models from a directory."""
    models = {}
    model_classes = {
        "custommaskedppo": CustomMaskedPPO,
        "duelingdqn": DuelingDQNModel,
        "duelingdqnmodel": DuelingDQNModel,
        "transformerppo": TransformerPPO,
    }

    # Track loaded model types to avoid duplicates
    loaded_types = set()

    # Try to load best models first, then final models
    for pattern in ["*_best.pt", "*_final.pt"]:
        for model_file in model_dir.glob(pattern):
            model_name = model_file.stem.replace("_best", "").replace("_final", "").replace("_league", "").lower()

            # Find matching model class
            model_class = None
            for key, cls in model_classes.items():
                if key in model_name:
                    model_class = cls
                    break

            if model_class is None:
                continue

            # Skip if already loaded this model type
            if model_class.__name__ in loaded_types:
                continue

            try:
                # Load checkpoint to get config
                checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
                saved_config = checkpoint.get("config", None)

                if saved_config is not None:
                    config = saved_config
                    config.device = device
                else:
                    config = ModelConfig(device=device)

                # Create dummy env
                dummy_env = TFTEnv(num_players=8, max_rounds=50)

                # Create and load model
                model = model_class(dummy_env, config)
                model.load(str(model_file).replace(".pt", ""))

                clean_name = model_class.__name__
                models[clean_name] = model
                loaded_types.add(clean_name)
                print(f"  Loaded: {clean_name}")

            except Exception as e:
                print(f"  Error loading {model_file.name}: {e}")

    return models


def evaluate_against_opponents(
    model,
    opponent_models: Dict[str, Any],
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate a model against opponent models."""
    # Create distribution
    num_opponents = 7
    num_models = len(opponent_models)

    if num_models == 0:
        # Fall back to random bot evaluation
        env = TFTEnv(num_players=8, max_rounds=50)
        placements = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action_mask = info.get("valid_action_mask")
                action, _ = model.predict(obs, action_mask=action_mask, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            placement = info.get("placement", 8)
            placements.append(placement)
        return {
            "avg_placement": np.mean(placements),
            "std_placement": np.std(placements),
            "top4_rate": np.mean([p <= 4 for p in placements]),
            "win_rate": np.mean([p == 1 for p in placements]),
            "avg_reward": 0.0,
        }

    slots_per_model = num_opponents // num_models
    remainder = num_opponents % num_models

    distribution = {}
    for i, name in enumerate(opponent_models.keys()):
        count = slots_per_model + (1 if i < remainder else 0)
        distribution[name] = count

    # Create environment with opponent models
    env = create_league_env(
        trained_models=opponent_models,
        model_distribution=distribution,
        num_players=8,
        max_rounds=50,
    )

    placements = []
    rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action_mask = info.get("valid_action_mask")
            action, _ = model.predict(obs, action_mask=action_mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        placement = info.get("placement", 8)
        placements.append(placement)
        rewards.append(episode_reward)

    return {
        "avg_placement": np.mean(placements),
        "std_placement": np.std(placements),
        "top4_rate": np.mean([p <= 4 for p in placements]),
        "win_rate": np.mean([p == 1 for p in placements]),
        "avg_reward": np.mean(rewards),
    }


def train_model_iteration(
    model_name: str,
    model,
    opponent_models: Dict[str, Any],
    timesteps: int,
    device: str,
) -> Tuple[Any, Dict[str, float]]:
    """Train a model for one iteration against opponents."""
    # Create opponent distribution (exclude self)
    other_models = {k: v for k, v in opponent_models.items() if k != model_name}

    if not other_models:
        print(f"  No opponents for {model_name}")
        return model, {}

    num_opponents = 7
    num_models = len(other_models)
    slots_per_model = num_opponents // num_models
    remainder = num_opponents % num_models

    distribution = {}
    for i, name in enumerate(other_models.keys()):
        count = slots_per_model + (1 if i < remainder else 0)
        distribution[name] = count

    # Create self-play environment
    env = create_league_env(
        trained_models=other_models,
        model_distribution=distribution,
        num_players=8,
        max_rounds=50,
    )

    # Update model's environment
    model.env = env

    # Train
    print(f"  Training {model_name} for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        log_interval=100,
        progress_bar=True,
    )

    # Evaluate
    eval_stats = evaluate_against_opponents(model, other_models, n_episodes=50)

    return model, eval_stats


def find_best_model_dir() -> Optional[Path]:
    """Find the directory with the best trained models."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    # Check for iterative training results first
    iterative_dirs = sorted(models_dir.glob("iterative_*"), reverse=True)
    for d in iterative_dirs:
        best_dir = d / "best"
        if best_dir.exists() and list(best_dir.glob("*.pt")):
            return best_dir

    # Then check phase2
    phase2_dirs = sorted([
        d for d in models_dir.glob("league_phase2_*")
    ], reverse=True)
    if phase2_dirs:
        return phase2_dirs[0]

    # Then phase1
    league_dirs = sorted([
        d for d in models_dir.glob("league_*")
        if "phase2" not in d.name
    ], reverse=True)
    if league_dirs:
        return league_dirs[0]

    return None


def main():
    """Main iterative training function."""
    print("=" * 60)
    print("TFT RL Iterative Self-Play Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = get_device()
    config = IterativeConfig(
        timesteps_per_iteration=50000,
        max_iterations=20,
        min_improvement=0.01,
        patience=5,
        eval_episodes=100,
    )

    # Find starting models
    start_dir = find_best_model_dir()
    if start_dir is None:
        print("\nNo trained models found!")
        print("Run 'python train_league.py' first.")
        return

    print(f"\nLoading models from: {start_dir}")
    models = load_models_from_dir(start_dir, device)

    if len(models) < 2:
        print(f"Need at least 2 models. Found: {list(models.keys())}")
        return

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/iterative_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    log_dir = Path(f"logs/iterative_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "iterations": [],
        "config": {
            "timesteps_per_iteration": config.timesteps_per_iteration,
            "max_iterations": config.max_iterations,
            "min_improvement": config.min_improvement,
            "patience": config.patience,
        }
    }

    # Track best performance
    best_placements = {name: 8.0 for name in models.keys()}
    iterations_without_improvement = 0

    # Initial evaluation (against random bots to establish baseline)
    print("\n" + "=" * 60)
    print("Initial Evaluation (vs Random Bots)")
    print("=" * 60)

    for model_name, model in models.items():
        # Evaluate against random bots (empty dict = random bots only)
        stats = evaluate_against_opponents(model, {}, n_episodes=config.eval_episodes)
        print(f"{model_name}: Placement={stats['avg_placement']:.2f}, "
              f"Top4={stats['top4_rate']:.1%}, Win={stats['win_rate']:.1%}")
        best_placements[model_name] = stats["avg_placement"]

    # Iterative training loop
    for iteration in range(1, config.max_iterations + 1):
        print("\n" + "=" * 60)
        print(f"Iteration {iteration}/{config.max_iterations}")
        print("=" * 60)

        iteration_start = time.time()
        iteration_results = {}
        any_improvement = False

        # Train each model
        for model_name in list(models.keys()):
            model = models[model_name]

            # Train
            model, train_stats = train_model_iteration(
                model_name=model_name,
                model=model,
                opponent_models=models,
                timesteps=config.timesteps_per_iteration,
                device=device,
            )

            # Evaluate against random bots (more stable than against learning opponents)
            eval_stats = evaluate_against_opponents(
                model, {}, n_episodes=config.eval_episodes
            )

            # Track improvement
            prev_best = best_placements[model_name]
            curr_placement = eval_stats["avg_placement"]
            improvement = prev_best - curr_placement

            if improvement > config.min_improvement:
                best_placements[model_name] = curr_placement
                any_improvement = True
                status = "IMPROVED"
            else:
                status = "no change"

            print(f"  {model_name}: Placement={curr_placement:.2f} ({status}), "
                  f"Top4={eval_stats['top4_rate']:.1%}, Win={eval_stats['win_rate']:.1%}")

            iteration_results[model_name] = {
                "placement": curr_placement,
                "top4_rate": eval_stats["top4_rate"],
                "win_rate": eval_stats["win_rate"],
                "improvement": improvement,
            }

            # Update model in dict
            models[model_name] = model

            # Clear GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()

        # Save iteration results
        iteration_time = time.time() - iteration_start
        history["iterations"].append({
            "iteration": iteration,
            "time_seconds": iteration_time,
            "results": iteration_results,
            "any_improvement": any_improvement,
        })

        # Save models periodically
        if iteration % config.save_every == 0:
            iter_dir = output_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(exist_ok=True)
            for model_name, model in models.items():
                save_path = iter_dir / f"{model_name.lower()}"
                model.save(str(save_path))

            # Also save to best directory
            for model_name, model in models.items():
                save_path = best_dir / f"{model_name.lower()}_best"
                model.save(str(save_path))

        # Check for convergence
        if any_improvement:
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
            print(f"\nNo improvement for {iterations_without_improvement} iteration(s)")

        if iterations_without_improvement >= config.patience:
            print(f"\nConverged after {iteration} iterations (no improvement for {config.patience} iterations)")
            break

        print(f"\nIteration {iteration} completed in {iteration_time/60:.1f} minutes")

    # Final evaluation (against random bots for consistency)
    print("\n" + "=" * 60)
    print("Final Evaluation (vs Random Bots)")
    print("=" * 60)

    final_results = {}
    for model_name, model in models.items():
        # Evaluate against random bots
        stats = evaluate_against_opponents(model, {}, n_episodes=config.eval_episodes)
        print(f"{model_name}: Placement={stats['avg_placement']:.2f}±{stats['std_placement']:.2f}, "
              f"Top4={stats['top4_rate']:.1%}, Win={stats['win_rate']:.1%}")
        final_results[model_name] = stats

        # Save final model
        save_path = best_dir / f"{model_name.lower()}_best"
        model.save(str(save_path))

    # Save history
    history["final_results"] = final_results
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Iterative Training Complete!")
    print(f"  Models saved to: {best_dir}")
    print(f"  History saved to: {history_path}")
    print("=" * 60)

    return models, history


if __name__ == "__main__":
    result = main()
    if result is not None:
        models, history = result
