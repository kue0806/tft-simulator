#!/usr/bin/env python
"""
TFT RL Agent Training Script.

Usage:
    # Train agent
    python train_tft.py --train --timesteps 1000000

    # Evaluate model
    python train_tft.py --eval models/tft_agent/tft_ppo_latest --episodes 100

    # Both train and eval
    python train_tft.py --train --timesteps 500000 --eval-after
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rl.training.trainer import train_agent, evaluate_agent


def main():
    parser = argparse.ArgumentParser(
        description="TFT RL Agent Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train for 1M timesteps:
        python train_tft.py --train --timesteps 1000000

    Evaluate saved model:
        python train_tft.py --eval models/tft_agent/tft_ppo_20240101_120000

    Train with 8 parallel envs:
        python train_tft.py --train --n-envs 8
        """,
    )

    # Training args
    parser.add_argument(
        "--train", action="store_true", help="Train agent"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/tft_agent",
        help="Model save directory (default: models/tft_agent)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/tft",
        help="Log directory (default: logs/tft)",
    )

    # Evaluation args
    parser.add_argument(
        "--eval",
        type=str,
        help="Evaluate model at path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--eval-after",
        action="store_true",
        help="Evaluate after training",
    )

    args = parser.parse_args()

    # Check for action
    if not args.train and not args.eval:
        parser.print_help()
        print("\nError: Specify --train or --eval")
        sys.exit(1)

    model_path = None

    # Train
    if args.train:
        print("=" * 60)
        print("TFT Agent Training")
        print("=" * 60)
        print(f"Timesteps: {args.timesteps:,}")
        print(f"Environments: {args.n_envs}")
        print(f"Learning rate: {args.lr}")
        print(f"Save dir: {args.save_dir}")
        print("=" * 60)

        agent = train_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
        )

        if agent:
            # Get latest model path
            save_path = Path(args.save_dir)
            models = sorted(save_path.glob("tft_ppo_*"))
            if models:
                model_path = str(models[-1])

    # Evaluate
    if args.eval:
        model_path = args.eval
    elif args.eval_after and model_path:
        pass  # Use model from training
    elif args.eval_after:
        print("No model to evaluate (training may have failed)")
        sys.exit(1)
    else:
        model_path = None

    if model_path:
        print()
        print("=" * 60)
        print("TFT Agent Evaluation")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Episodes: {args.episodes}")
        print("=" * 60)

        results = evaluate_agent(
            model_path=model_path,
            n_episodes=args.episodes,
        )

        if results:
            print()
            print("Placement distribution:")
            placements = results.get("placements", [])
            for rank in range(1, 9):
                count = sum(1 for p in placements if p == rank)
                pct = count / len(placements) * 100 if placements else 0
                bar = "#" * int(pct / 2)
                print(f"  {rank}: {count:3d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
