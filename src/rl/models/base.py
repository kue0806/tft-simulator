"""
Base RL Model Interface.

Defines abstract base class for all TFT RL models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RunningMeanStd:
    """
    Running mean and standard deviation for observation normalization.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def get_state(self) -> Dict[str, Any]:
        """Get state for saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from saved data."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


@dataclass
class ModelConfig:
    """Configuration for RL models."""

    # Common hyperparameters
    learning_rate: float = 3e-4
    learning_rate_end: float = 1e-5  # For LR scheduling
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000

    # PPO-specific
    n_steps: int = 2048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # DQN-specific
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    target_update_interval: int = 1000
    tau: float = 0.005  # For soft update (smaller = more stable)

    # Advanced DQN features (2019-2021)
    use_noisy_networks: bool = True  # Noisy Networks for exploration (2017)
    noisy_std_init: float = 0.5  # Initial noise standard deviation
    use_munchausen: bool = True  # Munchausen RL (2020)
    munchausen_alpha: float = 0.9  # Munchausen scaling factor
    munchausen_tau: float = 0.03  # Munchausen entropy temperature
    munchausen_clip: float = -1.0  # Munchausen log-policy clipping
    n_step_returns: int = 3  # N-step returns (Rainbow)

    # Network architecture (expanded for TFT complexity)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    features_dim: int = 256
    dropout: float = 0.1  # Dropout rate for regularization
    use_orthogonal_init: bool = True  # Orthogonal weight initialization

    # Transformer-specific
    n_heads: int = 4
    n_layers: int = 2
    d_model: int = 128

    # Observation normalization
    normalize_observations: bool = True
    normalize_rewards: bool = False  # Reward normalization (optional)

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "linear" or "cosine"
    warmup_ratio: float = 0.05  # Warmup ratio for cosine schedule

    # Training
    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 1


@dataclass
class TrainingMetrics:
    """Training metrics container."""

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    placements: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    # Additional metrics
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)

    def add_episode(self, reward: float, length: int, placement: int):
        """Add episode result."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.placements.append(placement)

    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """Get statistics for recent episodes."""
        recent_placements = self.placements[-n:] if self.placements else []
        recent_rewards = self.episode_rewards[-n:] if self.episode_rewards else []

        stats = {}

        if recent_placements:
            stats["avg_placement"] = np.mean(recent_placements)
            stats["top4_rate"] = sum(1 for p in recent_placements if p <= 4) / len(recent_placements)
            stats["win_rate"] = sum(1 for p in recent_placements if p == 1) / len(recent_placements)

        if recent_rewards:
            stats["avg_reward"] = np.mean(recent_rewards)
            stats["std_reward"] = np.std(recent_rewards)

        return stats


class BaseRLModel(ABC):
    """
    Abstract base class for TFT RL models.

    All models must implement:
    - predict(): Select action given observation
    - learn(): Train for specified timesteps
    - save()/load(): Model persistence
    """

    def __init__(self, env, config: Optional[ModelConfig] = None):
        """
        Initialize model.

        Args:
            env: Gymnasium environment or vectorized environment.
            config: Model configuration.
        """
        self.env = env
        self.config = config or ModelConfig()
        self.metrics = TrainingMetrics()
        self._setup_model()

    @abstractmethod
    def _setup_model(self):
        """Initialize model components (networks, optimizers, etc.)."""
        pass

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Select action given observation.

        Args:
            obs: Observation array.
            action_mask: Boolean mask for valid actions (True = valid).
            deterministic: If True, select best action; else sample.

        Returns:
            Tuple of (action, optional info dict).
        """
        pass

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ) -> "BaseRLModel":
        """
        Train model.

        Args:
            total_timesteps: Total training timesteps.
            callback: Optional callback for custom logging.
            log_interval: Logging frequency (episodes).
            progress_bar: Show progress bar.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Save path (without extension).
        """
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseRLModel":
        """
        Load model from disk.

        Args:
            path: Model path.

        Returns:
            Self for chaining.
        """
        pass

    def get_metrics(self) -> TrainingMetrics:
        """Get training metrics."""
        return self.metrics

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        pass

    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            n_episodes: Number of evaluation episodes.
            deterministic: Use deterministic actions.

        Returns:
            Evaluation statistics.
        """
        placements = []
        total_rewards = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                mask = info.get("valid_action_mask")
                action, _ = self.predict(obs, mask, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            placement = info.get("placement", 8)
            placements.append(placement)
            total_rewards.append(episode_reward)

        return {
            "avg_placement": np.mean(placements),
            "std_placement": np.std(placements),
            "top4_rate": sum(1 for p in placements if p <= 4) / len(placements),
            "win_rate": sum(1 for p in placements if p == 1) / len(placements),
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "n_episodes": n_episodes,
        }
