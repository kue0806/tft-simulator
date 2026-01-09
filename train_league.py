#!/usr/bin/env python3
"""
TFT RL League Training Script

Phase 1: Train each model against bots until they dominate (early stopping)
Phase 2: Self-play league training with trained models as opponents
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

from src.rl.env.tft_env import TFTEnv
from src.rl.models.base import ModelConfig, TrainingMetrics
from src.rl.models.custom_masked_ppo import CustomMaskedPPO
from src.rl.models.dueling_dqn import DuelingDQNModel
from src.rl.models.transformer_ppo import TransformerPPO


@dataclass
class EarlyStopConfig:
    """Early stopping configuration."""
    target_placement: float = 1.5  # Stop when avg placement <= this
    target_top4_rate: float = 0.90  # Stop when top4 rate >= this
    min_episodes: int = 50  # Minimum episodes before checking
    check_window: int = 100  # Window size for calculating stats
    patience: int = 3  # Number of consecutive checks meeting criteria


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        print("⚠️ GPU not available, using CPU")
    return device


def create_config(device: str, model_name: str) -> ModelConfig:
    """Create model-specific configuration."""
    base_kwargs = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "device": device,
        "verbose": 1,
    }

    if model_name == "CustomMaskedPPO":
        return ModelConfig(
            **base_kwargs,
            n_steps=1024,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            hidden_dims=[256, 128],
            features_dim=256,
        )
    elif model_name == "DuelingDQN":
        return ModelConfig(
            learning_rate=1e-4,
            gamma=0.99,
            batch_size=64,
            buffer_size=50000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=500,
            tau=0.005,
            hidden_dims=[256, 128],
            features_dim=256,
            max_grad_norm=10.0,
            device=device,
            verbose=1,
        )
    elif model_name == "TransformerPPO":
        return ModelConfig(
            learning_rate=1e-4,
            gamma=0.99,
            batch_size=32,
            n_steps=512,
            n_epochs=5,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=1.0,
            hidden_dims=[128],
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            device=device,
            verbose=1,
        )
    else:
        return ModelConfig(**base_kwargs, hidden_dims=[256, 128], features_dim=256)


def check_early_stop(metrics: TrainingMetrics, config: EarlyStopConfig) -> Tuple[bool, Dict]:
    """Check if early stopping criteria are met."""
    if len(metrics.placements) < config.min_episodes:
        return False, {}

    stats = metrics.get_recent_stats(config.check_window)
    avg_placement = stats.get("avg_placement", 8.0)
    top4_rate = stats.get("top4_rate", 0.0)

    criteria_met = (
        avg_placement <= config.target_placement and
        top4_rate >= config.target_top4_rate
    )

    return criteria_met, stats


def train_model_with_early_stop(
    model_class,
    config: ModelConfig,
    env: TFTEnv,
    model_name: str,
    save_dir: Path,
    early_stop_config: EarlyStopConfig,
    max_timesteps: int = 100000,
) -> Tuple[Dict, Any]:
    """Train a model with early stopping."""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
    print(f"Max Timesteps: {max_timesteps:,}")
    print(f"Early Stop: placement <= {early_stop_config.target_placement}, top4 >= {early_stop_config.target_top4_rate:.0%}")
    print(f"Device: {config.device}")

    start_time = time.time()
    consecutive_success = 0

    try:
        model = model_class(env, config)

        # Training loop with early stopping checks
        timesteps_done = 0
        check_interval = 2000  # Check every N timesteps

        while timesteps_done < max_timesteps:
            # Train for check_interval steps
            steps_to_train = min(check_interval, max_timesteps - timesteps_done)
            model.learn(
                total_timesteps=steps_to_train,
                log_interval=100,
                progress_bar=True,
            )
            timesteps_done += steps_to_train

            # Check early stopping
            should_stop, stats = check_early_stop(model.metrics, early_stop_config)

            if stats:
                print(f"\n📊 [{timesteps_done:,}/{max_timesteps:,}] "
                      f"Placement: {stats.get('avg_placement', 0):.2f}, "
                      f"Top4: {stats.get('top4_rate', 0):.1%}, "
                      f"Win: {stats.get('win_rate', 0):.1%}")

            if should_stop:
                consecutive_success += 1
                print(f"   ✅ Criteria met ({consecutive_success}/{early_stop_config.patience})")

                if consecutive_success >= early_stop_config.patience:
                    print(f"\n🎯 Early stopping triggered at {timesteps_done:,} timesteps!")
                    break
            else:
                consecutive_success = 0

        training_time = time.time() - start_time

        # Get final stats
        final_stats = model.metrics.get_recent_stats(100)

        # Save model
        save_path = save_dir / model_name.lower().replace(" ", "_")
        model.save(str(save_path))
        print(f"💾 Model saved to: {save_path}")

        # Evaluate
        print(f"\n📊 Evaluating {model_name}...")
        eval_stats = model.evaluate(n_episodes=50, deterministic=True)

        result = {
            "model_name": model_name,
            "timesteps_trained": timesteps_done,
            "training_time_seconds": training_time,
            "training_stats": final_stats,
            "eval_stats": eval_stats,
            "episodes_completed": len(model.metrics.placements),
            "early_stopped": timesteps_done < max_timesteps,
        }

        print(f"\n✅ {model_name} Training Complete!")
        print(f"   Timesteps: {timesteps_done:,}")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Avg Placement: {eval_stats.get('avg_placement', 0):.2f}")
        print(f"   Top 4 Rate: {eval_stats.get('top4_rate', 0):.1%}")
        print(f"   Win Rate: {eval_stats.get('win_rate', 0):.1%}")

        return result, model

    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"model_name": model_name, "error": str(e)}, None


def phase1_train_vs_bots(device: str, save_dir: Path) -> Dict[str, Any]:
    """Phase 1: Train all models against bots until they dominate."""
    print("\n" + "="*60)
    print("📋 PHASE 1: Training Models vs Bots")
    print("="*60)

    model_classes = {
        "CustomMaskedPPO": CustomMaskedPPO,
        "DuelingDQN": DuelingDQNModel,
        "TransformerPPO": TransformerPPO,
    }

    early_stop_config = EarlyStopConfig(
        target_placement=1.5,
        target_top4_rate=0.90,
        min_episodes=30,
        check_window=50,
        patience=2,
    )

    results = {}
    trained_models = {}

    for model_name, model_class in model_classes.items():
        env = TFTEnv(num_players=8, max_rounds=50)
        config = create_config(device, model_name)

        result, model = train_model_with_early_stop(
            model_class=model_class,
            config=config,
            env=env,
            model_name=model_name,
            save_dir=save_dir,
            early_stop_config=early_stop_config,
            max_timesteps=50000,
        )

        results[model_name] = result
        if model is not None:
            trained_models[model_name] = model

        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

    return results, trained_models


def main():
    """Main training function."""
    print("="*60)
    print("🎮 TFT RL League Training")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = get_device()

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/league_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"logs/league_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train vs Bots
    phase1_results, trained_models = phase1_train_vs_bots(device, save_dir)

    # Print Phase 1 Summary
    print("\n" + "="*60)
    print("📊 PHASE 1 SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Timesteps':>10} {'Placement':>10} {'Top4%':>10} {'Win%':>10}")
    print("-"*60)

    for model_name, result in phase1_results.items():
        if "error" not in result:
            eval_stats = result.get("eval_stats", {})
            print(f"{model_name:<20} "
                  f"{result.get('timesteps_trained', 0):>10,} "
                  f"{eval_stats.get('avg_placement', 0):>10.2f} "
                  f"{eval_stats.get('top4_rate', 0)*100:>9.1f}% "
                  f"{eval_stats.get('win_rate', 0)*100:>9.1f}%")
        else:
            print(f"{model_name:<20} {'ERROR':>10}")

    # Save summary
    summary = {
        "training_timestamp": timestamp,
        "phase1_results": phase1_results,
        "trained_models": list(trained_models.keys()),
    }

    summary_path = log_dir / "league_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("="*60)
    print(f"✅ Phase 1 Complete! Models saved to: {save_dir}")
    print(f"   Summary saved to: {summary_path}")

    # Check if we have enough trained models for Phase 2
    if len(trained_models) >= 2:
        print(f"\n🎯 {len(trained_models)} models ready for Phase 2 (Self-Play League)")
        print("   Run 'python train_league_phase2.py' to continue training")

    return phase1_results, trained_models


if __name__ == "__main__":
    results, models = main()
