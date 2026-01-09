"""
Dueling DQN Model.

Implements Dueling DQN with action masking and prioritized experience replay.
Separates state value and action advantage estimation.
"""

import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .base import BaseRLModel, ModelConfig, TrainingMetrics, RunningMeanStd


@dataclass
class Transition:
    """Single transition in replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer (2017 - Fortunato et al.)

    Replaces epsilon-greedy exploration with parametric noise.
    Learns the optimal amount of noise for exploration.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NStepReplayBuffer:
    """
    N-step Prioritized Experience Replay buffer.

    Stores n-step returns for more efficient credit assignment.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.n_step_buffer: deque = deque(maxlen=n_step)
        self.max_priority = 1.0

    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        """Calculate n-step return and get final state."""
        reward = 0.0
        for idx, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * transition.reward
            if transition.done:
                return reward, transition.next_state, True
        return reward, self.n_step_buffer[-1].next_state, self.n_step_buffer[-1].done

    def add(self, transition: "Transition"):
        """Add transition with n-step processing."""
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate n-step return
        n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()

        # Create n-step transition
        first = self.n_step_buffer[0]
        n_step_transition = Transition(
            state=first.state,
            action=first.action,
            reward=n_step_reward,
            next_state=n_step_next_state,
            done=n_step_done,
            action_mask=first.action_mask,
            next_action_mask=self.n_step_buffer[-1].next_action_mask,
        )

        self.buffer.append(n_step_transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[List["Transition"], np.ndarray, List[int]]:
        """Sample batch with prioritized sampling."""
        n = len(self.buffer)
        if n == 0:
            return [], np.array([]), []

        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        self.frame += 1

        transitions = [self.buffer[i] for i in indices]
        return transitions, weights.astype(np.float32), list(indices)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority + 1e-6)

    def flush_n_step_buffer(self):
        """Flush remaining transitions in n-step buffer at episode end."""
        while len(self.n_step_buffer) > 0:
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            first = self.n_step_buffer[0]
            n_step_transition = Transition(
                state=first.state,
                action=first.action,
                reward=n_step_reward,
                next_state=n_step_next_state,
                done=n_step_done,
                action_mask=first.action_mask,
                next_action_mask=self.n_step_buffer[-1].next_action_mask if len(self.n_step_buffer) > 0 else first.next_action_mask,
            )
            self.buffer.append(n_step_transition)
            self.priorities.append(self.max_priority)
            self.n_step_buffer.popleft()

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to TD error.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, transition: Transition):
        """Add transition with max priority."""
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, List[int]]:
        """Sample batch with prioritized sampling."""
        n = len(self.buffer)
        if n == 0:
            return [], np.array([]), []

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)

        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        self.frame += 1

        transitions = [self.buffer[i] for i in indices]
        return transitions, weights.astype(np.float32), list(indices)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority + 1e-6)

    def __len__(self):
        return len(self.buffer)


def orthogonal_init(layer, gain: float = np.sqrt(2)):
    """Apply orthogonal initialization to a layer."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DuelingNetwork(nn.Module):
    """
    Dueling DQN network architecture with Noisy Networks support.

    Separates Q(s,a) into V(s) + A(s,a) - mean(A(s,a))
    Optionally uses NoisyLinear for exploration (2017 - Fortunato et al.)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        features_dim: int = 256,
        dropout: float = 0.1,
        use_orthogonal_init: bool = True,
        use_noisy: bool = False,
        noisy_std_init: float = 0.5,
    ):
        super().__init__()

        self.dropout_rate = dropout
        self.use_noisy = use_noisy

        # Feature extractor with dropout
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

        # Choose Linear type based on noisy setting
        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        # Value stream
        if use_noisy:
            self.value_hidden = NoisyLinear(features_dim, hidden_dims[-1], noisy_std_init)
            self.value_out = NoisyLinear(hidden_dims[-1], 1, noisy_std_init)
        else:
            self.value_hidden = nn.Linear(features_dim, hidden_dims[-1])
            self.value_out = nn.Linear(hidden_dims[-1], 1)

        # Advantage stream
        if use_noisy:
            self.advantage_hidden = NoisyLinear(features_dim, hidden_dims[-1], noisy_std_init)
            self.advantage_out = NoisyLinear(hidden_dims[-1], action_dim, noisy_std_init)
        else:
            self.advantage_hidden = nn.Linear(features_dim, hidden_dims[-1])
            self.advantage_out = nn.Linear(hidden_dims[-1], action_dim)

        self.value_dropout = nn.Dropout(dropout)
        self.advantage_dropout = nn.Dropout(dropout)

        # Apply orthogonal initialization (only for non-noisy layers)
        if use_orthogonal_init and not use_noisy:
            self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
            orthogonal_init(self.value_out, gain=1.0)
            orthogonal_init(self.advantage_out, gain=0.01)

    def reset_noise(self):
        """Reset noise for noisy layers."""
        if self.use_noisy:
            self.value_hidden.reset_noise()
            self.value_out.reset_noise()
            self.advantage_hidden.reset_noise()
            self.advantage_out.reset_noise()

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Q-values.

        Args:
            obs: Observation tensor [batch, obs_dim]
            action_mask: Boolean mask for valid actions

        Returns:
            Q-values [batch, action_dim]
        """
        features = self.features(obs)

        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_dropout(value)
        value = self.value_out(value)

        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_dropout(advantage)
        advantage = self.advantage_out(advantage)

        # Dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Apply action mask
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float("-inf"))

        return q_values


class DuelingDQNModel(BaseRLModel):
    """
    Advanced Dueling DQN with modern techniques (2020-2024).

    Key features:
    - Dueling architecture for better value estimation
    - Prioritized N-step replay for efficient sampling
    - Double DQN for reduced overestimation
    - Noisy Networks for learned exploration (replaces ε-greedy)
    - Munchausen RL for improved bootstrapping
    - Cosine annealing LR schedule
    - Action masking for invalid actions
    """

    def __init__(self, env, config: Optional[ModelConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        super().__init__(env, config)

    def _setup_model(self):
        """Initialize networks, optimizer, and replay buffer."""
        # Get dimensions
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        obs_dim = obs_space.shape[0]
        self.action_dim = action_space.n

        # Device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Observation normalization
        if self.config.normalize_observations:
            self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_normalizer = None

        # Check if using noisy networks
        self.use_noisy = self.config.use_noisy_networks
        self.use_munchausen = self.config.use_munchausen

        # Networks with Noisy Networks support
        self.q_network = DuelingNetwork(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            features_dim=self.config.features_dim,
            dropout=self.config.dropout,
            use_orthogonal_init=self.config.use_orthogonal_init,
            use_noisy=self.use_noisy,
            noisy_std_init=self.config.noisy_std_init,
        ).to(self.device)

        self.target_network = DuelingNetwork(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            features_dim=self.config.features_dim,
            dropout=self.config.dropout,
            use_orthogonal_init=self.config.use_orthogonal_init,
            use_noisy=self.use_noisy,
            noisy_std_init=self.config.noisy_std_init,
        ).to(self.device)

        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

        # Learning rate scheduler settings
        self.initial_lr = self.config.learning_rate
        self.final_lr = self.config.learning_rate_end
        self.lr_schedule = self.config.lr_schedule
        self.warmup_ratio = self.config.warmup_ratio

        # N-step Replay buffer
        self.replay_buffer = NStepReplayBuffer(
            capacity=self.config.buffer_size,
            n_step=self.config.n_step_returns,
            gamma=self.config.gamma,
        )

        # Munchausen RL parameters
        self.munchausen_alpha = self.config.munchausen_alpha
        self.munchausen_tau = self.config.munchausen_tau
        self.munchausen_clip = self.config.munchausen_clip

        # Exploration (only used if not using noisy networks)
        self.epsilon = self.config.exploration_initial_eps if not self.use_noisy else 0.0
        self.epsilon_decay = (
            self.config.exploration_initial_eps - self.config.exploration_final_eps
        ) / (self.config.exploration_fraction * 1_000_000) if not self.use_noisy else 0.0

        self.total_timesteps = 0
        self.target_total_timesteps = 1_000_000  # Will be updated in learn()
        self.updates = 0

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

            if self.lr_schedule == "cosine":
                # Cosine annealing with warmup
                if progress < self.warmup_ratio:
                    # Linear warmup
                    new_lr = self.initial_lr * (progress / self.warmup_ratio)
                else:
                    # Cosine annealing
                    cos_progress = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)
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
        """Select action using Noisy Networks or epsilon-greedy."""
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)

        valid_actions = np.where(action_mask)[0]

        if len(valid_actions) == 0:
            return 0, None  # Default to PASS

        # Normalize observation
        obs_normalized = self._normalize_obs(obs, update=not deterministic)

        # For Noisy Networks: reset noise before each action selection
        if self.use_noisy and not deterministic:
            self.q_network.reset_noise()

        # Epsilon-greedy (only if not using noisy networks)
        if not self.use_noisy and not deterministic and random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            if deterministic:
                self.q_network.eval()  # Disable noise and dropout
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                q_values = self.q_network(obs_tensor, mask_tensor)
                action = q_values.argmax(dim=-1).item()
            if deterministic:
                self.q_network.train()

        return action, {"q_value": None}

    def _update(self) -> Dict[str, float]:
        """Perform DQN update with Munchausen RL (2020)."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Reset noise for noisy networks
        if self.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Sample batch
        transitions, weights, indices = self.replay_buffer.sample(self.config.batch_size)

        if len(transitions) == 0:
            return {}

        # Prepare batch tensors with normalization
        raw_states = np.array([t.state for t in transitions])
        raw_next_states = np.array([t.next_state for t in transitions])

        # Normalize states if normalizer is available
        if self.obs_normalizer is not None:
            states = np.array([self.obs_normalizer.normalize(s) for s in raw_states])
            next_states = np.array([self.obs_normalizer.normalize(s) for s in raw_next_states])
        else:
            states = raw_states
            next_states = raw_next_states

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in transitions]).to(self.device)
        action_masks = torch.BoolTensor(np.array([t.action_mask for t in transitions])).to(self.device)
        next_masks = torch.BoolTensor(np.array([t.next_action_mask for t in transitions])).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_all = self.q_network(states, action_masks)
        current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN with Munchausen RL
        with torch.no_grad():
            # Next state Q values
            next_q_online = self.q_network(next_states, next_masks)
            next_actions = next_q_online.argmax(dim=-1)

            next_q_target = self.target_network(next_states, next_masks)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Handle -inf for masked actions
            next_q = torch.where(
                torch.isinf(next_q),
                torch.zeros_like(next_q),
                next_q,
            )

            if self.use_munchausen:
                # Munchausen RL: add scaled log-policy to reward
                # τ * log π(a|s) where π is softmax policy
                q_for_policy = current_q_all.clone()
                q_for_policy = torch.where(
                    action_masks,
                    q_for_policy,
                    torch.full_like(q_for_policy, -1e8)
                )

                # Compute log-policy with temperature
                log_policy = F.log_softmax(q_for_policy / self.munchausen_tau, dim=-1)
                log_policy_action = log_policy.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Clip log-policy to prevent extreme values
                munchausen_bonus = self.munchausen_alpha * torch.clamp(
                    self.munchausen_tau * log_policy_action,
                    min=self.munchausen_clip,
                    max=0.0
                )

                # Soft value for next state (expected Q under softmax policy)
                next_q_soft = self.target_network(next_states, next_masks)
                next_q_soft = torch.where(
                    next_masks,
                    next_q_soft,
                    torch.full_like(next_q_soft, -1e8)
                )
                next_policy = F.softmax(next_q_soft / self.munchausen_tau, dim=-1)
                next_log_policy = F.log_softmax(next_q_soft / self.munchausen_tau, dim=-1)

                # V(s') = Σ π(a'|s') * (Q(s',a') - τ * log π(a'|s'))
                next_v = (next_policy * (next_q_soft - self.munchausen_tau * next_log_policy)).sum(dim=-1)
                next_v = torch.where(torch.isnan(next_v), torch.zeros_like(next_v), next_v)

                # Munchausen target: r + α * τ * log π(a|s) + γ * V(s')
                target_q = rewards + munchausen_bonus + self.config.gamma * next_v * (1 - dones)
            else:
                # Standard Double DQN target
                target_q = rewards + self.config.gamma * next_q * (1 - dones)

        # TD error for prioritized replay
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Weighted loss
        loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.updates += 1

        # Update learning rate
        self._update_lr()

        # Soft update target network
        if self.updates % self.config.target_update_interval == 0:
            self._soft_update()

        return {"loss": loss.item(), "td_error": td_errors.mean()}

    def _soft_update(self):
        """Soft update target network."""
        tau = self.config.tau
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ) -> "DuelingDQNModel":
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

        # Update epsilon decay based on actual total timesteps
        self.epsilon_decay = (
            self.config.exploration_initial_eps - self.config.exploration_final_eps
        ) / (self.config.exploration_fraction * total_timesteps)

        while self.total_timesteps < target_timesteps:
            action_mask = info.get("valid_action_mask")
            if action_mask is None:
                action_mask = np.ones(self.action_dim, dtype=bool)

            # Select action
            action, _ = self.predict(obs, action_mask, deterministic=False)

            # Step environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated

            next_action_mask = next_info.get("valid_action_mask")
            if next_action_mask is None:
                next_action_mask = np.ones(self.action_dim, dtype=bool)

            # Store transition
            transition = Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )
            self.replay_buffer.add(transition)

            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1

            # Update epsilon
            self.epsilon = max(
                self.config.exploration_final_eps,
                self.epsilon - self.epsilon_decay,
            )

            if pbar:
                pbar.update(1)

            # Learn
            if len(self.replay_buffer) >= self.config.batch_size:
                losses = self._update()
                if losses:
                    self.metrics.losses.append(losses.get("loss", 0))

            if done:
                placement = next_info.get("placement", 8)
                self.metrics.add_episode(episode_reward, episode_length, placement)

                # Log
                if len(self.metrics.placements) % log_interval == 0:
                    stats = self.metrics.get_recent_stats(100)
                    if pbar:
                        pbar.set_postfix({
                            "placement": f"{stats.get('avg_placement', 0):.2f}",
                            "top4": f"{stats.get('top4_rate', 0):.1%}",
                            "eps": f"{self.epsilon:.3f}",
                        })

                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
                info = next_info

        if pbar:
            pbar.close()

        return self

    def save(self, path: str):
        """Save model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_timesteps": self.total_timesteps,
            "epsilon": self.epsilon,
        }

        # Save observation normalizer state
        if self.obs_normalizer is not None:
            save_dict["obs_normalizer"] = self.obs_normalizer.get_state()

        torch.save(save_dict, str(save_path) + ".pt")

    def load(self, path: str) -> "DuelingDQNModel":
        """Load model."""
        checkpoint = torch.load(str(path) + ".pt", map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.epsilon = checkpoint.get("epsilon", self.config.exploration_final_eps)

        # Load observation normalizer state
        if self.obs_normalizer is not None and "obs_normalizer" in checkpoint:
            self.obs_normalizer.set_state(checkpoint["obs_normalizer"])

        return self

    @property
    def name(self) -> str:
        return "DuelingDQN"
