"""
Model Comparison Runner.

Trains and evaluates multiple RL models for comparison.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from .base import BaseRLModel, ModelConfig, TrainingMetrics


@dataclass
class ModelResult:
    """Results from training and evaluating a model."""

    model_name: str
    training_time: float  # seconds
    total_timesteps: int

    # Final evaluation metrics
    avg_placement: float
    std_placement: float
    top4_rate: float
    win_rate: float
    avg_reward: float

    # Training progression (sampled every N episodes)
    placement_history: List[float]
    reward_history: List[float]

    # Loss history
    policy_loss_history: List[float]
    value_loss_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelComparisonRunner:
    """
    Runs comparison experiments between different RL models.

    Usage:
        runner = ModelComparisonRunner(env)
        runner.add_model("MaskablePPO", MaskablePPOModel)
        runner.add_model("DuelingDQN", DuelingDQNModel)
        results = runner.run(total_timesteps=100_000)
        runner.print_summary()
    """

    def __init__(
        self,
        env_factory,
        save_dir: str = "models/comparison",
        log_dir: str = "logs/comparison",
        seed: Optional[int] = 42,
    ):
        """
        Initialize comparison runner.

        Args:
            env_factory: Callable that creates a new environment instance.
            save_dir: Directory to save trained models.
            log_dir: Directory to save logs and results.
            seed: Random seed for reproducibility.
        """
        self.env_factory = env_factory
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.seed = seed

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Type[BaseRLModel]] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.results: Dict[str, ModelResult] = {}

    def add_model(
        self,
        name: str,
        model_class: Type[BaseRLModel],
        config: Optional[ModelConfig] = None,
    ):
        """
        Add a model to the comparison.

        Args:
            name: Display name for the model.
            model_class: Model class (subclass of BaseRLModel).
            config: Optional custom configuration.
        """
        self.models[name] = model_class
        self.configs[name] = config or ModelConfig(seed=self.seed)

    def run(
        self,
        total_timesteps: int = 100_000,
        eval_episodes: int = 100,
        log_interval: int = 50,
        progress_bar: bool = True,
    ) -> Dict[str, ModelResult]:
        """
        Run comparison experiment.

        Args:
            total_timesteps: Training timesteps per model.
            eval_episodes: Evaluation episodes after training.
            log_interval: Logging frequency (episodes).
            progress_bar: Show training progress bar.

        Returns:
            Dictionary of model results.
        """
        print("=" * 60)
        print("Starting Model Comparison Experiment")
        print(f"Models: {list(self.models.keys())}")
        print(f"Total timesteps per model: {total_timesteps:,}")
        print(f"Evaluation episodes: {eval_episodes}")
        print("=" * 60)

        for name, model_class in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print("=" * 60)

            # Create fresh environment for this model
            env = self.env_factory()

            # Create model
            config = self.configs[name]
            try:
                model = model_class(env, config)
            except Exception as e:
                print(f"Failed to create {name}: {e}")
                continue

            # Train
            start_time = time.time()
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    log_interval=log_interval,
                    progress_bar=progress_bar,
                )
            except Exception as e:
                print(f"Training failed for {name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            training_time = time.time() - start_time

            # Save model
            model_path = self.save_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save(str(model_path))

            # Evaluate
            print(f"\nEvaluating {name}...")
            eval_results = model.evaluate(n_episodes=eval_episodes)

            # Get training history
            metrics = model.get_metrics()
            placement_history = self._sample_history(metrics.placements, 100)
            reward_history = self._sample_history(metrics.episode_rewards, 100)

            # Store results
            result = ModelResult(
                model_name=name,
                training_time=training_time,
                total_timesteps=total_timesteps,
                avg_placement=eval_results["avg_placement"],
                std_placement=eval_results["std_placement"],
                top4_rate=eval_results["top4_rate"],
                win_rate=eval_results["win_rate"],
                avg_reward=eval_results["avg_reward"],
                placement_history=placement_history,
                reward_history=reward_history,
                policy_loss_history=self._sample_history(metrics.policy_losses, 100),
                value_loss_history=self._sample_history(metrics.value_losses, 100),
            )

            self.results[name] = result

            # Print individual result
            print(f"\n{name} Results:")
            print(f"  Training time: {training_time:.1f}s")
            print(f"  Avg placement: {result.avg_placement:.2f} (±{result.std_placement:.2f})")
            print(f"  Top 4 rate: {result.top4_rate:.1%}")
            print(f"  Win rate: {result.win_rate:.1%}")

        # Save all results
        self._save_results()

        return self.results

    def _sample_history(self, history: List[float], n_samples: int) -> List[float]:
        """Sample N points from history for plotting."""
        if not history:
            return []
        if len(history) <= n_samples:
            return list(history)

        indices = np.linspace(0, len(history) - 1, n_samples, dtype=int)
        return [history[i] for i in indices]

    def _save_results(self):
        """Save results to JSON file."""
        results_dict = {
            name: result.to_dict()
            for name, result in self.results.items()
        }

        results_path = self.log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def print_summary(self):
        """Print comparison summary table."""
        if not self.results:
            print("No results to display. Run experiment first.")
            return

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        # Sort by avg_placement (lower is better)
        sorted_results = sorted(
            self.results.values(),
            key=lambda r: r.avg_placement,
        )

        print(f"\n{'Model':<20} {'Placement':>12} {'Top 4':>10} {'Win Rate':>10} {'Time':>10}")
        print("-" * 62)

        for result in sorted_results:
            print(
                f"{result.model_name:<20} "
                f"{result.avg_placement:>8.2f}±{result.std_placement:<3.2f} "
                f"{result.top4_rate:>9.1%} "
                f"{result.win_rate:>9.1%} "
                f"{result.training_time:>8.1f}s"
            )

        print("-" * 62)

        # Best model
        best = sorted_results[0]
        print(f"\n🏆 Best Model: {best.model_name}")
        print(f"   Average Placement: {best.avg_placement:.2f}")
        print(f"   Top 4 Rate: {best.top4_rate:.1%}")

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison charts.

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return

        if not self.results:
            print("No results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Placement bar chart
        ax1 = axes[0, 0]
        names = list(self.results.keys())
        placements = [r.avg_placement for r in self.results.values()]
        stds = [r.std_placement for r in self.results.values()]

        bars = ax1.bar(names, placements, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax1.set_ylabel("Average Placement")
        ax1.set_title("Model Comparison: Placement")
        ax1.set_ylim(0, 8)

        # Add value labels
        for bar, val in zip(bars, placements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', va='bottom')

        # 2. Top 4 and Win rate
        ax2 = axes[0, 1]
        x = np.arange(len(names))
        width = 0.35

        top4_rates = [r.top4_rate * 100 for r in self.results.values()]
        win_rates = [r.win_rate * 100 for r in self.results.values()]

        bars1 = ax2.bar(x - width/2, top4_rates, width, label='Top 4 Rate', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, win_rates, width, label='Win Rate', color='gold', alpha=0.7)

        ax2.set_ylabel("Rate (%)")
        ax2.set_title("Top 4 and Win Rates")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.legend()
        ax2.set_ylim(0, 100)

        # 3. Placement history (training curve)
        ax3 = axes[1, 0]
        for name, result in self.results.items():
            if result.placement_history:
                ax3.plot(result.placement_history, label=name, alpha=0.8)

        ax3.set_xlabel("Training Progress")
        ax3.set_ylabel("Average Placement (moving)")
        ax3.set_title("Training Curves")
        ax3.legend()
        ax3.set_ylim(0, 8)

        # 4. Training time comparison
        ax4 = axes[1, 1]
        times = [r.training_time for r in self.results.values()]

        bars = ax4.bar(names, times, color='coral', alpha=0.8)
        ax4.set_ylabel("Time (seconds)")
        ax4.set_title("Training Time")

        for bar, val in zip(bars, times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}s', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def run_quick_comparison(
    total_timesteps: int = 50_000,
    eval_episodes: int = 50,
):
    """
    Run a quick comparison of all available models.

    This is a convenience function for testing.
    """
    from src.rl.env.tft_env import TFTEnv

    # Import models
    from .maskable_ppo import MaskablePPOModel
    from .custom_masked_ppo import CustomMaskedPPO
    from .dueling_dqn import DuelingDQNModel
    from .transformer_ppo import TransformerPPO

    # Environment factory
    def make_env():
        return TFTEnv(render_mode=None)

    # Create runner
    runner = ModelComparisonRunner(
        env_factory=make_env,
        save_dir="models/quick_comparison",
        log_dir="logs/quick_comparison",
        seed=42,
    )

    # Add models (skip MaskablePPO if sb3-contrib not available)
    try:
        runner.add_model("MaskablePPO", MaskablePPOModel)
    except Exception as e:
        print(f"Skipping MaskablePPO: {e}")

    runner.add_model("CustomPPO", CustomMaskedPPO)
    runner.add_model("DuelingDQN", DuelingDQNModel)
    runner.add_model("TransformerPPO", TransformerPPO)

    # Run comparison
    results = runner.run(
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
        log_interval=20,
    )

    # Print summary
    runner.print_summary()

    # Try to plot
    try:
        runner.plot_comparison(save_path="logs/quick_comparison/comparison_plot.png")
    except Exception as e:
        print(f"Could not create plot: {e}")

    return runner
