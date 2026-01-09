"""
Custom Masked PPO Implementation.

Implements PPO with action masking from scratch using PyTorch.
Provides more control over the masking mechanism.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .base import BaseRLModel, ModelConfig, TrainingMetrics, RunningMeanStd


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout experiences."""

    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    action_masks: List[np.ndarray]

    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def __len__(self):
        return len(self.observations)


def orthogonal_init(layer, gain: float = np.sqrt(2)):
    """Apply orthogonal initialization to a layer."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with action masking support.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        features_dim: int = 256,
        dropout: float = 0.1,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.dropout_rate = dropout

        # Shared feature extractor with dropout
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        # Actor head (policy) with dropout
        self.actor = nn.Sequential(
            nn.Linear(features_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], action_dim),
        )

        # Critic head (value function) with dropout
        self.critic = nn.Sequential(
            nn.Linear(features_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1),
        )

        # Apply orthogonal initialization
        if use_orthogonal_init:
            self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
            # Use smaller gain for output layers
            orthogonal_init(self.actor[-1], gain=0.01)  # Policy head
            orthogonal_init(self.critic[-1], gain=1.0)  # Value head

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim]
            action_mask: Boolean mask [batch, action_dim], True = valid

        Returns:
            Tuple of (action_logits, value)
        """
        features = self.features(obs)
        logits = self.actor(features)
        value = self.critic(features)

        # Apply action masking
        if action_mask is not None:
            # Ensure at least one action is valid to prevent NaN
            if not action_mask.any(dim=-1).all():
                # If any batch has no valid actions, make first action valid
                action_mask = action_mask.clone()
                action_mask[:, 0] = True
            # Set invalid action logits to very negative value
            logits = logits.masked_fill(~action_mask, -1e8)  # Use finite value instead of -inf

        return logits, value.squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Observation tensor
            action_mask: Boolean mask for valid actions
            action: Optional pre-selected action (for training)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(obs, action_mask)

        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class CustomMaskedPPO(BaseRLModel):
    """
    Custom PPO implementation with action masking.

    Key features:
    - Full control over training loop
    - Custom masking in log probability computation
    - Supports various learning rate schedules
    """

    def __init__(self, env, config: Optional[ModelConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        super().__init__(env, config)

    def _setup_model(self):
        """Initialize networks and optimizer."""
        # Get dimensions from environment
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        if hasattr(obs_space, "shape"):
            obs_dim = obs_space.shape[0]
        else:
            obs_dim = obs_space.n

        if hasattr(action_space, "n"):
            action_dim = action_space.n
        else:
            action_dim = action_space.shape[0]

        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Observation normalization
        if self.config.normalize_observations:
            self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_normalizer = None

        # Create network with improved architecture
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            features_dim=self.config.features_dim,
            dropout=self.config.dropout,
            use_orthogonal_init=self.config.use_orthogonal_init,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

        # Learning rate scheduler settings
        self.initial_lr = self.config.learning_rate
        self.final_lr = self.config.learning_rate_end
        self.target_total_timesteps = 1_000_000  # Will be updated in learn()

        # Buffer
        self.buffer = RolloutBuffer()

        # Training state
        self.total_timesteps = 0

    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.obs_normalizer is None:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs).astype(np.float32)

    def _update_lr(self):
        """Update learning rate with cosine annealing or linear decay."""
        if self.target_total_timesteps > 0:
            progress = min(1.0, self.total_timesteps / self.target_total_timesteps)

            lr_schedule = getattr(self.config, 'lr_schedule', 'linear')
            warmup_ratio = getattr(self.config, 'warmup_ratio', 0.05)

            if lr_schedule == "cosine":
                # Cosine annealing with warmup
                if progress < warmup_ratio:
                    # Linear warmup
                    new_lr = self.initial_lr * (progress / warmup_ratio)
                else:
                    # Cosine annealing
                    cos_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
                    new_lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (
                        1 + np.cos(np.pi * cos_progress)
                    )
            else:
                # Linear decay
                new_lr = self.initial_lr + progress * (self.final_lr - self.initial_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Select action."""
        # Normalize observation
        obs_normalized = self._normalize_obs(obs, update=not deterministic)

        self.network.eval()  # Disable dropout for inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)

            if action_mask is not None:
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            else:
                mask_tensor = None

            logits, value = self.network(obs_tensor, mask_tensor)

            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample().item()
        self.network.train()  # Re-enable dropout

        return action, {"value": value.item()}

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0
        last_return = last_value

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _update(self) -> Dict[str, float]:
        """Perform PPO update."""
        # Normalize observations if normalizer is available
        raw_obs = np.array(self.buffer.observations)
        if self.obs_normalizer is not None:
            normalized_obs = np.array([self.obs_normalizer.normalize(o) for o in raw_obs])
        else:
            normalized_obs = raw_obs

        # Convert buffer to tensors
        obs = torch.FloatTensor(normalized_obs).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        masks = torch.BoolTensor(np.array(self.buffer.action_masks)).to(self.device)

        # Compute last value for GAE
        with torch.no_grad():
            _, last_value = self.network(obs[-1:])
            last_value = last_value.item()

        # Compute advantages and returns
        advantages, returns = self._compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value,
        )

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (only if we have enough samples and std > 0)
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        else:
            advantages = advantages - advantages.mean()

        # PPO update epochs
        batch_size = self.config.batch_size
        n_samples = len(obs)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            # Random permutation
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = masks[batch_indices]

                # Forward pass
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_obs,
                    batch_masks,
                    batch_actions,
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_range = self.config.clip_range

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        # Update learning rate
        self._update_lr()

        # Clear buffer
        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy_loss": total_entropy_loss / n_updates,
        }

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ) -> "CustomMaskedPPO":
        """Train the model."""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0

        # Calculate target timesteps (current + requested)
        start_timesteps = self.total_timesteps
        target_timesteps = start_timesteps + total_timesteps

        # Set target for LR scheduling
        self.target_total_timesteps = target_timesteps

        if progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_timesteps, desc=f"Training {self.name}")
            except ImportError:
                pbar = None
                progress_bar = False
        else:
            pbar = None

        while self.total_timesteps < target_timesteps:
            # Collect rollout
            for _ in range(self.config.n_steps):
                action_mask = info.get("valid_action_mask")
                if action_mask is None:
                    action_mask = np.ones(self.env.action_space.n, dtype=bool)

                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                    action, log_prob, _, value = self.network.get_action_and_value(
                        obs_tensor, mask_tensor
                    )
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()

                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store in buffer
                self.buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    action_mask=action_mask,
                )

                episode_reward += reward
                episode_length += 1
                self.total_timesteps += 1

                if pbar:
                    pbar.update(1)

                if done:
                    placement = info.get("placement", 8)
                    self.metrics.add_episode(episode_reward, episode_length, placement)

                    # Log
                    if len(self.metrics.placements) % log_interval == 0:
                        stats = self.metrics.get_recent_stats(100)
                        if pbar:
                            pbar.set_postfix({
                                "placement": f"{stats.get('avg_placement', 0):.2f}",
                                "top4": f"{stats.get('top4_rate', 0):.1%}",
                            })

                    obs, info = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs

                if self.total_timesteps >= total_timesteps:
                    break

            # Update policy
            if len(self.buffer) > 0:
                losses = self._update()
                self.metrics.policy_losses.append(losses["policy_loss"])
                self.metrics.value_losses.append(losses["value_loss"])
                self.metrics.entropy_losses.append(losses["entropy_loss"])

        if pbar:
            pbar.close()

        return self

    def save(self, path: str):
        """Save model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_timesteps": self.total_timesteps,
        }

        # Save observation normalizer state
        if self.obs_normalizer is not None:
            save_dict["obs_normalizer"] = self.obs_normalizer.get_state()

        torch.save(save_dict, str(save_path) + ".pt")

    def load(self, path: str) -> "CustomMaskedPPO":
        """Load model."""
        checkpoint = torch.load(str(path) + ".pt", map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)

        # Load observation normalizer state
        if self.obs_normalizer is not None and "obs_normalizer" in checkpoint:
            self.obs_normalizer.set_state(checkpoint["obs_normalizer"])

        return self

    @property
    def name(self) -> str:
        return "CustomMaskedPPO"
