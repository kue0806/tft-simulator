#!/usr/bin/env python3
"""
Single Model Training with Checkpointing.

Trains a single RL model with:
- Regular checkpoint saving
- Early stopping when target placement reached
- Clear model naming convention

Usage:
    python train_single_model.py --model CustomPPO --timesteps 500000
    python train_single_model.py --model DuelingDQN --target-placement 5.0
    python train_single_model.py --model DuelingDQN --resume models/trained/DuelingDQN/model_best
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class EarlyStoppingCallback:
    """Callback for early stopping and checkpointing."""

    def __init__(
        self,
        model,
        model_name: str,
        save_dir: Path,
        target_placement: float = 4.0,
        checkpoint_interval: int = 50,  # episodes
        patience: int = 200,  # episodes without improvement
    ):
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        self.target_placement = target_placement
        self.checkpoint_interval = checkpoint_interval
        self.patience = patience

        self.best_placement = 8.0
        self.episodes_without_improvement = 0
        self.last_checkpoint_episode = 0
        self.start_time = time.time()

    def on_episode_end(self, episode: int, placement: float, avg_placement: float):
        """Called after each episode."""
        # Check for improvement
        if avg_placement < self.best_placement:
            self.best_placement = avg_placement
            self.episodes_without_improvement = 0

            # Save best model
            self._save_checkpoint("best", avg_placement)
        else:
            self.episodes_without_improvement += 1

        # Regular checkpoint
        if episode - self.last_checkpoint_episode >= self.checkpoint_interval:
            self._save_checkpoint(f"ep{episode}", avg_placement)
            self.last_checkpoint_episode = episode

        # Early stopping disabled - always continue training
        return False

    def _save_checkpoint(self, suffix: str, avg_placement: float):
        """Save model checkpoint."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)

        # Format: ModelName_placement_X.XX_timestamp_suffix
        filename = f"{self.model_name}_placement_{avg_placement:.2f}_{datetime.now().strftime('%m%d_%H%M')}_{suffix}"
        filepath = self.save_dir / filename

        self.model.save(str(filepath))
        print(f"💾 Saved: {filename} (training time: {hours}h {mins}m)")


def train_model(
    model_name: str,
    total_timesteps: int,
    target_placement: float,
    checkpoint_interval: int,
    patience: int,
    save_dir: str,
    seed: int,
    resume_path: str = None,
):
    """Train a single model."""
    from src.rl.env.tft_env import TFTEnv
    from src.rl.models.base import ModelConfig

    # Import model class
    model_classes = {}

    try:
        from src.rl.models.custom_masked_ppo import CustomMaskedPPO
        model_classes["CustomPPO"] = CustomMaskedPPO
    except ImportError:
        pass

    try:
        from src.rl.models.dueling_dqn import DuelingDQNModel
        model_classes["DuelingDQN"] = DuelingDQNModel
    except ImportError:
        pass

    try:
        from src.rl.models.transformer_ppo import TransformerPPO
        model_classes["TransformerPPO"] = TransformerPPO
    except ImportError:
        pass

    try:
        from src.rl.models.maskable_ppo import MaskablePPOModel
        model_classes["MaskablePPO"] = MaskablePPOModel
    except ImportError:
        pass

    if model_name not in model_classes:
        print(f"Model '{model_name}' not available.")
        print(f"Available models: {list(model_classes.keys())}")
        return None

    # Setup
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)
    print(f"Max timesteps: {total_timesteps:,}")
    print(f"Target placement: {target_placement}")
    print(f"Checkpoint interval: {checkpoint_interval} episodes")
    print(f"Patience: {patience} episodes")
    print(f"Save directory: {save_path}")
    print(f"Seed: {seed}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    print("=" * 60)

    # Create environment and model
    env = TFTEnv(render_mode=None)

    config = ModelConfig(
        seed=seed,
        learning_rate=3e-4,
        hidden_dims=[256, 128],
        features_dim=256,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
    )

    model_class = model_classes[model_name]

    # Create model first
    model = model_class(env, config)

    # Load weights if resuming
    if resume_path:
        print(f"\nLoading model from {resume_path}...")
        model.load(resume_path)
        print("Model loaded successfully!")

    # Create callback
    callback = EarlyStoppingCallback(
        model=model,
        model_name=model_name,
        save_dir=save_path,
        target_placement=target_placement,
        checkpoint_interval=checkpoint_interval,
        patience=patience,
    )

    # Custom training loop with callback
    print("\nStarting training...")
    start_time = time.time()

    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    recent_placements = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_timesteps, desc=f"Training {model_name}")
    except ImportError:
        pbar = None

    timestep = 0
    should_stop = False

    while timestep < total_timesteps and not should_stop:
        # Get action mask
        action_mask = info.get("valid_action_mask")
        if action_mask is None:
            action_mask = np.ones(env.action_space.n, dtype=bool)

        # Predict action
        action, _ = model.predict(obs, action_mask, deterministic=False)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # For PPO models, we need to store in buffer
        if hasattr(model, 'buffer'):
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(model.device)
                _, log_prob, _, value = model.network.get_action_and_value(
                    obs_tensor, mask_tensor
                )
                log_prob = log_prob.item()
                value = value.item()

            model.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask,
            )
        elif hasattr(model, 'replay_buffer'):
            # For DQN models
            from src.rl.models.dueling_dqn import Transition
            next_action_mask = info.get("valid_action_mask")
            if next_action_mask is None:
                next_action_mask = np.ones(env.action_space.n, dtype=bool)

            transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )
            model.replay_buffer.add(transition)

            # Update epsilon
            model.epsilon = max(
                model.config.exploration_final_eps,
                model.epsilon - model.epsilon_decay,
            )

        episode_reward += reward
        episode_length += 1
        timestep += 1
        model.total_timesteps = timestep

        if pbar:
            pbar.update(1)

        # Episode end
        if done:
            placement = info.get("placement", 8)
            recent_placements.append(placement)
            if len(recent_placements) > 100:
                recent_placements.pop(0)

            avg_placement = np.mean(recent_placements) if recent_placements else 8.0
            episode_count += 1

            model.metrics.add_episode(episode_reward, episode_length, placement)

            # Update progress bar
            if pbar and episode_count % 10 == 0:
                top4 = sum(1 for p in recent_placements if p <= 4) / len(recent_placements) if recent_placements else 0
                pbar.set_postfix({
                    "ep": episode_count,
                    "placement": f"{avg_placement:.2f}",
                    "top4": f"{top4:.1%}",
                    "best": f"{callback.best_placement:.2f}",
                })

            # Callback
            should_stop = callback.on_episode_end(episode_count, placement, avg_placement)

            # Reset
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Update model (for PPO: after n_steps, for DQN: every step)
        if hasattr(model, 'buffer') and len(model.buffer) >= config.n_steps:
            model._update()
        elif hasattr(model, 'replay_buffer') and len(model.replay_buffer) >= config.batch_size:
            model._update()

    if pbar:
        pbar.close()

    # Final save
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    final_avg = np.mean(recent_placements) if recent_placements else 8.0
    final_name = f"{model_name}_placement_{final_avg:.2f}_{datetime.now().strftime('%m%d_%H%M')}_final"
    model.save(str(save_path / final_name))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total timesteps: {timestep:,}")
    print(f"Training time: {hours}h {mins}m")
    print(f"Final avg placement: {final_avg:.2f}")
    print(f"Best avg placement: {callback.best_placement:.2f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train single TFT RL model")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["CustomPPO", "DuelingDQN", "TransformerPPO", "MaskablePPO"],
        help="Model to train"
    )

    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Maximum training timesteps (default: 1000000)"
    )

    parser.add_argument(
        "--target-placement",
        type=float,
        default=4.0,
        help="Target average placement to stop training (default: 4.0)"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=500,
        help="Episodes without improvement before early stop (default: 500)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/trained",
        help="Directory to save models"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from"
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        total_timesteps=args.timesteps,
        target_placement=args.target_placement,
        checkpoint_interval=args.checkpoint_interval,
        patience=args.patience,
        save_dir=args.save_dir,
        seed=args.seed,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
