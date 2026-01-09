#!/usr/bin/env python
"""
Self-Play Training Script for TFT.

Trains an agent using self-play with curriculum learning:
- Phase 1 (0-200k): 50% random bots, 50% self-play
- Phase 2 (200k-500k): 25% random bots, 75% self-play
- Phase 3 (500k+): 100% self-play

Total: 2M timesteps, 8 parallel environments
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.rl.training.self_play_trainer import (
    SelfPlayTrainer,
    SelfPlayConfig,
    CurriculumConfig,
)


def main():
    parser = argparse.ArgumentParser(description="TFT Self-Play Training")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2M)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="Entropy coefficient (default: 0.02)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50_000,
        help="Agent pool save interval (default: 50k)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        help="Evaluation interval (default: 100k)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume training from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/self_play",
        help="Directory to save models",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/self_play",
        help="Directory for logs",
    )

    args = parser.parse_args()

    # Create config
    curriculum_config = CurriculumConfig(
        phase1_end=200_000,
        phase1_self_play_ratio=0.5,
        phase2_end=500_000,
        phase2_self_play_ratio=0.75,
        phase3_self_play_ratio=1.0,
    )

    config = SelfPlayConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        pool_save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        curriculum=curriculum_config,
    )

    # Create trainer and run
    trainer = SelfPlayTrainer(
        config=config,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    results = trainer.train(resume_path=args.resume)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
