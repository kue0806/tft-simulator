#!/usr/bin/env python3
"""
TFT RL Model Training Script - All 4 Models

Trains all 4 RL models sequentially with GPU support and monitoring.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from src.rl.env.tft_env import TFTEnv
from src.rl.models.base import ModelConfig, TrainingMetrics
from src.rl.models.maskable_ppo import MaskablePPOModel
from src.rl.models.custom_masked_ppo import CustomMaskedPPO
from src.rl.models.dueling_dqn import DuelingDQNModel
from src.rl.models.transformer_ppo import TransformerPPO

# Import ActionMasker for MaskablePPO
try:
    from sb3_contrib.common.wrappers import ActionMasker
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False


def mask_fn(env):
    """Return valid action mask for MaskablePPO."""
    return env.get_valid_action_mask()


def wrap_env_for_maskable_ppo(env):
    """Wrap environment with ActionMasker for MaskablePPO compatibility."""
    if SB3_CONTRIB_AVAILABLE:
        return ActionMasker(env, mask_fn)
    return env


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


def create_configs(device: str):
    """Create model-specific configurations."""

    # Base config with GPU
    base_config = ModelConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,  # Conservative for 2GB GPU
        device=device,
        verbose=1,
    )

    # MaskablePPO config
    maskable_ppo_config = ModelConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=1024,  # Reduced for faster iterations
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        hidden_dims=[256, 128],
        features_dim=256,
        device=device,
        verbose=1,
    )

    # CustomMaskedPPO config
    custom_ppo_config = ModelConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=1024,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        hidden_dims=[256, 128],
        features_dim=256,
        device=device,
        verbose=1,
    )

    # DuelingDQN config
    dqn_config = ModelConfig(
        learning_rate=1e-4,  # Lower for DQN stability
        gamma=0.99,
        batch_size=64,
        buffer_size=50000,  # Reduced for memory
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        tau=0.005,  # Soft update
        hidden_dims=[256, 128],
        features_dim=256,
        max_grad_norm=10.0,
        device=device,
        verbose=1,
    )

    # TransformerPPO config
    transformer_config = ModelConfig(
        learning_rate=1e-4,  # Lower for transformer
        gamma=0.99,
        batch_size=32,  # Smaller for transformer memory
        n_steps=512,
        n_epochs=5,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy for exploration
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

    return {
        "MaskablePPO": maskable_ppo_config,
        "CustomMaskedPPO": custom_ppo_config,
        "DuelingDQN": dqn_config,
        "TransformerPPO": transformer_config,
    }


def train_model(model_class, config, env, total_timesteps, model_name, save_dir):
    """Train a single model and save results."""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")

    start_time = time.time()

    try:
        # Create model
        model = model_class(env, config)

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=50,
            progress_bar=True,
        )

        training_time = time.time() - start_time

        # Get final stats
        metrics = model.get_metrics()
        final_stats = metrics.get_recent_stats(100)

        # Save model
        save_path = save_dir / model_name.lower().replace(" ", "_")
        model.save(str(save_path))
        print(f"💾 Model saved to: {save_path}")

        # Evaluate
        print(f"\n📊 Evaluating {model_name}...")
        eval_stats = model.evaluate(n_episodes=50, deterministic=True)

        result = {
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "training_time_seconds": training_time,
            "training_stats": final_stats,
            "eval_stats": eval_stats,
            "episodes_completed": len(metrics.placements),
            "config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "gamma": config.gamma,
                "device": config.device,
            }
        }

        print(f"\n✅ {model_name} Training Complete!")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Episodes: {len(metrics.placements)}")
        print(f"   Avg Placement: {eval_stats.get('avg_placement', 0):.2f}")
        print(f"   Top 4 Rate: {eval_stats.get('top4_rate', 0):.1%}")
        print(f"   Win Rate: {eval_stats.get('win_rate', 0):.1%}")

        return result, model

    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"model_name": model_name, "error": str(e)}, None


def main():
    """Main training function."""
    print("="*60)
    print("🎮 TFT RL Model Training - All 4 Models")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = get_device()
    configs = create_configs(device)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/training_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"logs/training_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training timesteps per model (adjust based on time available)
    # Start with 50k for initial validation, increase to 200k+ for full training
    TIMESTEPS_PER_MODEL = 50000

    # Model classes (Skip MaskablePPO for now - requires ActionMasker fix)
    model_classes = {
        # "MaskablePPO": MaskablePPOModel,  # TODO: Fix ActionMasker integration
        "CustomMaskedPPO": CustomMaskedPPO,
        "DuelingDQN": DuelingDQNModel,
        "TransformerPPO": TransformerPPO,
    }

    # Results storage
    all_results = []
    trained_models = {}

    # Train each model
    for model_name, model_class in model_classes.items():
        # Create fresh environment for each model
        env = TFTEnv(num_players=8, max_rounds=50)

        # Wrap env with ActionMasker for MaskablePPO
        if model_name == "MaskablePPO":
            env = wrap_env_for_maskable_ppo(env)

        config = configs[model_name]

        result, model = train_model(
            model_class=model_class,
            config=config,
            env=env,
            total_timesteps=TIMESTEPS_PER_MODEL,
            model_name=model_name,
            save_dir=save_dir,
        )

        all_results.append(result)
        if model is not None:
            trained_models[model_name] = model

        # Clear GPU memory between models
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save summary
    summary = {
        "training_timestamp": timestamp,
        "total_models": len(model_classes),
        "timesteps_per_model": TIMESTEPS_PER_MODEL,
        "device": device,
        "results": all_results,
    }

    summary_path = log_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final comparison
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Placement':>10} {'Top4%':>10} {'Win%':>10} {'Time':>10}")
    print("-"*60)

    for result in all_results:
        if "error" not in result:
            eval_stats = result.get("eval_stats", {})
            print(f"{result['model_name']:<20} "
                  f"{eval_stats.get('avg_placement', 0):>10.2f} "
                  f"{eval_stats.get('top4_rate', 0)*100:>9.1f}% "
                  f"{eval_stats.get('win_rate', 0)*100:>9.1f}% "
                  f"{result.get('training_time_seconds', 0)/60:>9.1f}m")
        else:
            print(f"{result['model_name']:<20} {'ERROR':>10}")

    print("="*60)
    print(f"✅ All models trained! Results saved to: {log_dir}")
    print(f"   Models saved to: {save_dir}")

    return all_results, trained_models


if __name__ == "__main__":
    results, models = main()
