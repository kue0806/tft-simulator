#!/usr/bin/env python3
"""
TFT RL Model Comparison Training Script.

Trains and compares multiple RL models on the TFT environment.

Usage:
    python train_models.py --timesteps 100000 --eval-episodes 100
    python train_models.py --models CustomPPO DuelingDQN --timesteps 50000
    python train_models.py --quick  # Quick test run
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Train and compare TFT RL models")

    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100_000,
        help="Total training timesteps per model (default: 100000)"
    )

    parser.add_argument(
        "--eval-episodes", "-e",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)"
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["MaskablePPO", "CustomPPO", "DuelingDQN", "TransformerPPO"],
        default=None,
        help="Models to train (default: all available)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/comparison",
        help="Directory to save models"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/comparison",
        help="Directory to save logs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (10k timesteps, 20 eval episodes)"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot after training"
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.timesteps = 10_000
        args.eval_episodes = 20

    print("=" * 60)
    print("TFT RL Model Comparison")
    print("=" * 60)
    print(f"Timesteps per model: {args.timesteps:,}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Save directory: {args.save_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Import modules
    try:
        from src.rl.env.tft_env import TFTEnv
        from src.rl.models.base import ModelConfig
        from src.rl.models.model_comparison import ModelComparisonRunner
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install torch numpy gymnasium")
        sys.exit(1)

    # Import model classes
    model_classes = {}

    try:
        from src.rl.models.maskable_ppo import MaskablePPOModel
        model_classes["MaskablePPO"] = MaskablePPOModel
        print("✓ MaskablePPO available")
    except ImportError as e:
        print(f"✗ MaskablePPO not available: {e}")

    try:
        from src.rl.models.custom_masked_ppo import CustomMaskedPPO
        model_classes["CustomPPO"] = CustomMaskedPPO
        print("✓ CustomPPO available")
    except ImportError as e:
        print(f"✗ CustomPPO not available: {e}")

    try:
        from src.rl.models.dueling_dqn import DuelingDQNModel
        model_classes["DuelingDQN"] = DuelingDQNModel
        print("✓ DuelingDQN available")
    except ImportError as e:
        print(f"✗ DuelingDQN not available: {e}")

    try:
        from src.rl.models.transformer_ppo import TransformerPPO
        model_classes["TransformerPPO"] = TransformerPPO
        print("✓ TransformerPPO available")
    except ImportError as e:
        print(f"✗ TransformerPPO not available: {e}")

    if not model_classes:
        print("\nNo models available. Please install PyTorch:")
        print("  pip install torch")
        sys.exit(1)

    # Filter models if specified
    if args.models:
        model_classes = {
            name: cls
            for name, cls in model_classes.items()
            if name in args.models
        }

    print(f"\nModels to train: {list(model_classes.keys())}")

    # Create environment factory
    def make_env():
        return TFTEnv(render_mode=None)

    # Create runner
    runner = ModelComparisonRunner(
        env_factory=make_env,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )

    # Add models
    config = ModelConfig(
        seed=args.seed,
        learning_rate=3e-4,
        hidden_dims=[256, 128],
        features_dim=256,
        n_steps=1024,  # Reduce for faster iterations
        batch_size=64,
    )

    for name, model_class in model_classes.items():
        runner.add_model(name, model_class, config)

    # Run comparison
    try:
        results = runner.run(
            total_timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            log_interval=50,
            progress_bar=not args.no_progress,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results:")

    # Print summary
    runner.print_summary()

    # Generate plot if requested
    if args.plot:
        try:
            plot_path = Path(args.log_dir) / "comparison_plot.png"
            runner.plot_comparison(save_path=str(plot_path))
        except Exception as e:
            print(f"Could not generate plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
