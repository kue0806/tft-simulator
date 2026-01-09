#!/usr/bin/env python3
"""
TFT RL League Training - Phase 2: Self-Play

Uses trained models from Phase 1 as opponents.
Each model trains against a mix of other trained models.

Example distribution for 8 players:
- Player 0: Training model (current)
- Players 1-2: CustomMaskedPPO
- Players 3-4: DuelingDQN
- Players 5-6: TransformerPPO
- Player 7: Previous version of training model (or random)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from src.rl.env.self_play_env import SelfPlayEnv, create_league_env
from src.rl.models.base import ModelConfig, TrainingMetrics
from src.rl.models.custom_masked_ppo import CustomMaskedPPO
from src.rl.models.dueling_dqn import DuelingDQNModel
from src.rl.models.transformer_ppo import TransformerPPO


@dataclass
class LeagueConfig:
    """League training configuration."""
    timesteps_per_round: int = 20000  # Timesteps per training round
    num_rounds: int = 10  # Total training rounds
    target_placement: float = 2.0  # Target avg placement to beat
    target_top4_rate: float = 0.80  # Target top4 rate
    patience: int = 3  # Rounds without improvement before stopping


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        print("GPU not available, using CPU")
    return device


def load_trained_models(model_dir: Path) -> Dict[str, Any]:
    """Load trained models from Phase 1."""
    models = {}
    model_classes = {
        "custommaskableppo": CustomMaskedPPO,
        "custommaskerppo": CustomMaskedPPO,
        "custommaskedppo": CustomMaskedPPO,
        "duelingdqn": DuelingDQNModel,
        "transformerppo": TransformerPPO,
    }

    print(f"\nLoading models from: {model_dir}")

    # Find model files
    for model_file in model_dir.glob("*.pt"):
        model_name = model_file.stem.lower().replace("_", "")

        # Find matching model class
        model_class = None
        for key, cls in model_classes.items():
            if key in model_name:
                model_class = cls
                break

        if model_class is None:
            print(f"  Skipping unknown model: {model_file.name}")
            continue

        try:
            # Create dummy env for model initialization
            from src.rl.env.tft_env import TFTEnv
            dummy_env = TFTEnv(num_players=8, max_rounds=50)

            # Load checkpoint first to get the saved config
            checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
            saved_config = checkpoint.get("config", None)

            # Use saved config if available, otherwise create default
            if saved_config is not None:
                config = saved_config
                # Update device to current device
                config.device = get_device()
            else:
                config = ModelConfig(device=get_device())

            # Create and load model with the correct config
            model = model_class(dummy_env, config)
            model.load(str(model_file).replace(".pt", ""))

            # Store with clean name
            clean_name = model_class.__name__
            models[clean_name] = model
            print(f"  Loaded: {clean_name} from {model_file.name}")

        except Exception as e:
            print(f"  Error loading {model_file.name}: {e}")
            import traceback
            traceback.print_exc()

    return models


def create_model_distribution(
    trained_models: Dict[str, Any],
    training_model_name: str,
) -> Dict[str, int]:
    """
    Create opponent distribution for 7 opponent slots.

    Example:
        If training CustomMaskedPPO with DuelingDQN and TransformerPPO available:
        - DuelingDQN: 3 slots
        - TransformerPPO: 3 slots
        - (remaining slot filled by random or previous version)
    """
    other_models = {
        name: model for name, model in trained_models.items()
        if name != training_model_name
    }

    num_opponents = 7
    num_other_models = len(other_models)

    if num_other_models == 0:
        return {}

    # Distribute evenly
    slots_per_model = num_opponents // num_other_models
    remainder = num_opponents % num_other_models

    distribution = {}
    for i, name in enumerate(other_models.keys()):
        count = slots_per_model + (1 if i < remainder else 0)
        distribution[name] = count

    return distribution


def train_model_in_league(
    model_class,
    model_name: str,
    trained_models: Dict[str, Any],
    config: ModelConfig,
    league_config: LeagueConfig,
    save_dir: Path,
) -> Tuple[Dict, Any]:
    """Train a model against other trained models."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} in League")
    print(f"{'='*60}")

    # Create opponent distribution (exclude self)
    distribution = create_model_distribution(trained_models, model_name)

    if not distribution:
        print(f"No opponent models available for {model_name}")
        return {"error": "No opponents"}, None

    print(f"Opponent distribution: {distribution}")

    # Create self-play environment
    opponent_models = {
        name: trained_models[name]
        for name in distribution.keys()
    }

    env = create_league_env(
        trained_models=opponent_models,
        model_distribution=distribution,
        num_players=8,
        max_rounds=50,
    )

    # Initialize model (fresh or load from Phase 1)
    if model_name in trained_models:
        model = trained_models[model_name]
        print(f"Using pre-trained {model_name}")
    else:
        model = model_class(env, config)
        print(f"Training {model_name} from scratch")

    # Training loop
    start_time = time.time()
    best_placement = 8.0
    rounds_without_improvement = 0

    for round_num in range(league_config.num_rounds):
        print(f"\n--- League Round {round_num + 1}/{league_config.num_rounds} ---")

        # Train for timesteps_per_round
        model.learn(
            total_timesteps=league_config.timesteps_per_round,
            log_interval=50,
            progress_bar=True,
        )

        # Evaluate
        eval_stats = model.evaluate(n_episodes=30, deterministic=True)
        avg_placement = eval_stats.get("avg_placement", 8.0)
        top4_rate = eval_stats.get("top4_rate", 0.0)
        win_rate = eval_stats.get("win_rate", 0.0)

        print(f"Eval: Placement={avg_placement:.2f}, Top4={top4_rate:.1%}, Win={win_rate:.1%}")

        # Track improvement
        if avg_placement < best_placement:
            best_placement = avg_placement
            rounds_without_improvement = 0

            # Save best model
            save_path = save_dir / f"{model_name.lower()}_league_best"
            model.save(str(save_path))
            print(f"New best! Saved to {save_path}")
        else:
            rounds_without_improvement += 1

        # Check early stopping
        if (avg_placement <= league_config.target_placement and
            top4_rate >= league_config.target_top4_rate):
            print(f"\nTarget reached! Stopping early.")
            break

        if rounds_without_improvement >= league_config.patience:
            print(f"\nNo improvement for {league_config.patience} rounds. Stopping.")
            break

        # Print opponent stats
        opp_stats = env.get_opponent_stats()
        if opp_stats:
            print("Opponent performance:")
            for opp_name, stats in opp_stats.items():
                print(f"  {opp_name}: Placement={stats['avg_placement']:.2f}, "
                      f"Top4={stats['top4_rate']:.1%}")

    training_time = time.time() - start_time

    # Final save
    save_path = save_dir / f"{model_name.lower()}_league_final"
    model.save(str(save_path))

    # Final evaluation
    final_stats = model.evaluate(n_episodes=50, deterministic=True)

    result = {
        "model_name": model_name,
        "training_time_seconds": training_time,
        "rounds_completed": round_num + 1,
        "final_stats": final_stats,
        "best_placement": best_placement,
        "opponent_distribution": distribution,
    }

    print(f"\n{model_name} League Training Complete!")
    print(f"  Time: {training_time/60:.1f} minutes")
    print(f"  Placement: {final_stats.get('avg_placement', 0):.2f}")
    print(f"  Top4: {final_stats.get('top4_rate', 0):.1%}")
    print(f"  Win: {final_stats.get('win_rate', 0):.1%}")

    return result, model


def find_latest_phase1_dir() -> Optional[Path]:
    """Find the most recent Phase 1 model directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    # Find league directories but exclude phase2 directories
    league_dirs = [
        d for d in sorted(models_dir.glob("league_*"), reverse=True)
        if "phase2" not in d.name
    ]
    if league_dirs:
        return league_dirs[0]

    training_dirs = sorted(models_dir.glob("training_*"), reverse=True)
    if training_dirs:
        return training_dirs[0]

    return None


def main():
    """Main Phase 2 training function."""
    print("="*60)
    print("TFT RL League Training - Phase 2: Self-Play")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = get_device()

    # Find Phase 1 models
    phase1_dir = find_latest_phase1_dir()
    if phase1_dir is None:
        print("\nNo Phase 1 models found!")
        print("Run 'python train_league.py' first to train models vs bots.")
        return

    print(f"\nUsing Phase 1 models from: {phase1_dir}")

    # Load trained models
    trained_models = load_trained_models(phase1_dir)

    if len(trained_models) < 2:
        print(f"\nNeed at least 2 trained models for league training.")
        print(f"Found: {list(trained_models.keys())}")
        return

    print(f"\nLoaded {len(trained_models)} models: {list(trained_models.keys())}")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/league_phase2_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"logs/league_phase2_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # League config
    league_config = LeagueConfig(
        timesteps_per_round=20000,
        num_rounds=10,
        target_placement=2.0,
        target_top4_rate=0.80,
        patience=3,
    )

    # Model classes
    model_classes = {
        "CustomMaskedPPO": CustomMaskedPPO,
        "DuelingDQN": DuelingDQNModel,
        "DuelingDQNModel": DuelingDQNModel,  # Handle both naming conventions
        "TransformerPPO": TransformerPPO,
    }

    # Train each model in the league
    all_results = {}

    for model_name in trained_models.keys():
        if model_name not in model_classes:
            print(f"Skipping {model_name} - no model class")
            continue

        config = ModelConfig(device=device)
        result, model = train_model_in_league(
            model_class=model_classes[model_name],
            model_name=model_name,
            trained_models=trained_models,
            config=config,
            league_config=league_config,
            save_dir=save_dir,
        )

        all_results[model_name] = result

        # Update trained models with new version
        if model is not None:
            trained_models[model_name] = model

        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Placement':>10} {'Top4%':>10} {'Win%':>10}")
    print("-"*60)

    for model_name, result in all_results.items():
        if "error" not in result:
            stats = result.get("final_stats", {})
            print(f"{model_name:<20} "
                  f"{stats.get('avg_placement', 0):>10.2f} "
                  f"{stats.get('top4_rate', 0)*100:>9.1f}% "
                  f"{stats.get('win_rate', 0)*100:>9.1f}%")
        else:
            print(f"{model_name:<20} {'ERROR':>10}")

    # Save summary
    summary = {
        "training_timestamp": timestamp,
        "phase1_models_dir": str(phase1_dir),
        "results": all_results,
        "league_config": {
            "timesteps_per_round": league_config.timesteps_per_round,
            "num_rounds": league_config.num_rounds,
            "target_placement": league_config.target_placement,
            "target_top4_rate": league_config.target_top4_rate,
        },
    }

    summary_path = log_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("="*60)
    print(f"Phase 2 Complete!")
    print(f"  Models saved to: {save_dir}")
    print(f"  Summary saved to: {summary_path}")

    return all_results, trained_models


if __name__ == "__main__":
    results, models = main()
