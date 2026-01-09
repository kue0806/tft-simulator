"""
Transformer-based PPO Model (2024 Modern Architecture).

Uses self-attention mechanism to process TFT game state.
Particularly effective for capturing relationships between units and traits.

Modern Improvements (2021-2024):
- RMSNorm: More efficient than LayerNorm (LLaMA, 2023)
- Pre-LN: Pre-Layer Normalization for stable training
- RoPE: Rotary Position Embedding (RoFormer, 2021)
- SwiGLU: Gated activation function (PaLM, 2022)
- Observation Normalization: Running mean/std normalization
- Cosine LR with Warmup: Better convergence
"""

import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .base import BaseRLModel, ModelConfig, RunningMeanStd, TrainingMetrics
from .custom_masked_ppo import RolloutBuffer


# =============================================================================
# Modern Transformer Components (2021-2024)
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (2019, used in LLaMA 2023).

    More efficient than LayerNorm - no mean subtraction, no bias.
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer (2021).

    Encodes absolute position through rotation, while enabling
    relative position awareness via dot product properties.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Shape: [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Sequence length
        """
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (PaLM 2022, LLaMA 2023).

    Gated Linear Unit with Swish activation.
    SwiGLU(x) = Swish(xW) * (xV)

    Better performance than ReLU/GELU in transformers.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        # Gate and up projection
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_up = nn.Linear(in_features, hidden_features, bias=bias)
        # Down projection
        self.w_down = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModernMultiHeadAttention(nn.Module):
    """
    Modern Multi-Head Attention with RoPE and Pre-LN.

    Improvements:
    - RoPE instead of absolute position encoding
    - Pre-Layer Normalization for stable training
    - Optional dropout for regularization
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class ModernTransformerBlock(nn.Module):
    """
    Modern Transformer Block with Pre-LN, RMSNorm, RoPE, and SwiGLU.

    Pre-LN architecture (GPT-2 style):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Pre-LN with RMSNorm
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # Modern attention with RoPE
        self.attention = ModernMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        # SwiGLU FFN (hidden_dim is typically 8/3 * d_model for efficiency)
        ffn_hidden = int(d_ff * 2 / 3)  # Adjusted for SwiGLU's 3-way split
        self.ffn = SwiGLU(d_model, ffn_hidden, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN + Attention + Residual
        x = x + self.dropout(self.attention(self.attn_norm(x), mask))
        # Pre-LN + FFN + Residual
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class ModernTransformerEncoder(nn.Module):
    """
    Modern Transformer Encoder (2024) for processing game state.

    Key improvements over standard TransformerEncoder:
    - RMSNorm instead of LayerNorm
    - Pre-LN architecture for stable training
    - RoPE for position encoding (no separate position embedding)
    - SwiGLU activation in FFN
    - Learnable CLS token for aggregation

    Treats different parts of the state (board, bench, shop, etc.)
    as separate tokens for self-attention.
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_tokens: int = 16,  # Number of tokens to split observation into
        use_rope: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_tokens = n_tokens

        # Calculate token dimension
        self.token_dim = obs_dim // n_tokens
        if obs_dim % n_tokens != 0:
            self.token_dim += 1

        # Input projection with orthogonal initialization
        self.input_proj = nn.Linear(self.token_dim, d_model)
        nn.init.orthogonal_(self.input_proj.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.input_proj.bias)

        # Modern transformer blocks
        self.layers = nn.ModuleList([
            ModernTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rope=use_rope,
                max_seq_len=n_tokens + 1,  # +1 for CLS token
            )
            for _ in range(n_layers)
        ])

        # Final RMSNorm (Pre-LN requires final norm)
        self.final_norm = RMSNorm(d_model)

        # CLS token for aggregation (scaled initialization)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process observation through modern transformer.

        Args:
            x: Observation tensor [batch, obs_dim]

        Returns:
            Aggregated features [batch, d_model]
        """
        batch_size = x.size(0)

        # Pad observation to be divisible by n_tokens
        padded_dim = self.n_tokens * self.token_dim
        if x.size(1) < padded_dim:
            padding = torch.zeros(batch_size, padded_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Reshape to tokens: [batch, n_tokens, token_dim]
        x = x.view(batch_size, self.n_tokens, self.token_dim)

        # Project to d_model
        x = self.input_proj(x)

        # Add CLS token at position 0
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, n_tokens + 1, d_model]

        # Apply transformer layers (RoPE handles position encoding internally)
        for layer in self.layers:
            x = layer(x)

        # Final normalization (Pre-LN requires this)
        x = self.final_norm(x)

        # Return CLS token representation
        return x[:, 0, :]


# Legacy class for backward compatibility
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (legacy, kept for compatibility)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Legacy TransformerEncoder for backward compatibility
TransformerEncoder = ModernTransformerEncoder


class TransformerActorCritic(nn.Module):
    """
    Modern Actor-Critic network with Transformer encoder.

    Improvements (2021-2024):
    - Modern transformer encoder with RoPE, RMSNorm, SwiGLU
    - Orthogonal initialization for heads
    - RMSNorm in heads instead of LayerNorm
    - GELU activation (better than ReLU)
    - Dropout for regularization
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        hidden_dims: List[int] = [128],
        dropout: float = 0.1,
        use_orthogonal_init: bool = True,
    ):
        super().__init__()

        self.use_orthogonal_init = use_orthogonal_init

        # Modern Transformer encoder
        self.encoder = ModernTransformerEncoder(
            obs_dim=obs_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_rope=True,
        )

        # Actor head with modern design
        actor_layers = []
        prev_dim = d_model
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            if use_orthogonal_init:
                nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
                nn.init.zeros_(linear.bias)
            actor_layers.extend([
                linear,
                nn.GELU(),
                RMSNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final actor layer (smaller init for policy stability)
        actor_out = nn.Linear(prev_dim, action_dim)
        if use_orthogonal_init:
            nn.init.orthogonal_(actor_out.weight, gain=0.01)
            nn.init.zeros_(actor_out.bias)
        actor_layers.append(actor_out)
        self.actor = nn.Sequential(*actor_layers)

        # Critic head with modern design
        critic_layers = []
        prev_dim = d_model
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            if use_orthogonal_init:
                nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
                nn.init.zeros_(linear.bias)
            critic_layers.extend([
                linear,
                nn.GELU(),
                RMSNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final critic layer (identity-like init)
        critic_out = nn.Linear(prev_dim, 1)
        if use_orthogonal_init:
            nn.init.orthogonal_(critic_out.weight, gain=1.0)
            nn.init.zeros_(critic_out.bias)
        critic_layers.append(critic_out)
        self.critic = nn.Sequential(*critic_layers)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.encoder(obs)

        logits = self.actor(features)
        value = self.critic(features)

        if action_mask is not None:
            # Ensure at least one action is valid to prevent NaN
            if not action_mask.any(dim=-1).all():
                action_mask = action_mask.clone()
                action_mask[:, 0] = True
            # Use finite value instead of -inf to prevent NaN after softmax
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits, value.squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        logits, value = self.forward(obs, action_mask)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class TransformerPPO(BaseRLModel):
    """
    Modern PPO with Transformer-based policy network (2024).

    Key features:
    - Self-attention for capturing unit/trait relationships
    - Better at processing structured game state
    - Can learn positional importance of board/bench slots

    Modern Improvements (2021-2024):
    - RMSNorm instead of LayerNorm
    - Pre-LN architecture for stable training
    - RoPE for position encoding
    - SwiGLU activation in FFN
    - Observation normalization (RunningMeanStd)
    - Cosine LR schedule with warmup
    - Orthogonal weight initialization
    """

    def __init__(self, env, config: Optional[ModelConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        super().__init__(env, config)

    def _setup_model(self):
        """Initialize Modern Transformer network."""
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        # Device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Observation normalization
        self.normalize_observations = self.config.normalize_observations
        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))

        # Create modern network
        self.network = TransformerActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            use_orthogonal_init=self.config.use_orthogonal_init,
        ).to(self.device)

        # Optimizer with weight decay for transformer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            eps=1e-5,
            betas=(0.9, 0.95),  # LLaMA-style betas
        )

        # Learning rate schedule settings
        self.lr_schedule = self.config.lr_schedule
        self.initial_lr = self.config.learning_rate
        self.final_lr = self.config.learning_rate_end
        self.warmup_ratio = self.config.warmup_ratio
        self.total_training_steps = 1_000_000  # Will be updated in learn()

        # Buffer
        self.buffer = RolloutBuffer()

        self.total_timesteps = 0

    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize observation using running statistics."""
        if not self.normalize_observations:
            return obs
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _update_lr(self):
        """Update learning rate with cosine schedule and warmup."""
        if self.total_training_steps <= 0:
            return

        progress = self.total_timesteps / self.total_training_steps

        if self.lr_schedule == "cosine":
            # Cosine annealing with warmup
            if progress < self.warmup_ratio:
                # Linear warmup
                new_lr = self.initial_lr * (progress / self.warmup_ratio)
            else:
                # Cosine decay
                cos_progress = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)
                cos_progress = min(cos_progress, 1.0)
                new_lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (
                    1 + np.cos(np.pi * cos_progress)
                )
        elif self.lr_schedule == "linear":
            # Linear decay with warmup
            if progress < self.warmup_ratio:
                new_lr = self.initial_lr * (progress / self.warmup_ratio)
            else:
                decay_progress = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)
                new_lr = self.initial_lr - (self.initial_lr - self.final_lr) * decay_progress
        else:
            return  # No schedule

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def predict(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Select action with normalized observation."""
        # Normalize observation
        obs_normalized = self._normalize_obs(obs, update=False)

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

        return action, {"value": value.item()}

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0

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
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        masks = torch.BoolTensor(np.array(self.buffer.action_masks)).to(self.device)

        with torch.no_grad():
            _, last_value = self.network(obs[-1:])
            last_value = last_value.item()

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

        batch_size = self.config.batch_size
        n_samples = len(obs)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0

        for _ in range(self.config.n_epochs):
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

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_obs,
                    batch_masks,
                    batch_actions,
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_range = self.config.clip_range

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

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

        self.buffer.reset()

        # Get current LR
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy_loss": total_entropy_loss / n_updates,
            "learning_rate": current_lr,
        }

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100,
        progress_bar: bool = True,
    ) -> "TransformerPPO":
        """Train the model with modern techniques."""
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0

        # Calculate target timesteps (current + requested)
        start_timesteps = self.total_timesteps
        target_timesteps = start_timesteps + total_timesteps

        # Set total training steps for LR schedule
        self.total_training_steps = target_timesteps

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
            for _ in range(self.config.n_steps):
                action_mask = info.get("valid_action_mask")
                if action_mask is None:
                    action_mask = np.ones(self.env.action_space.n, dtype=bool)

                # Normalize observation and update running stats
                obs_normalized = self._normalize_obs(obs, update=True)

                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                    mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                    action, log_prob, _, value = self.network.get_action_and_value(
                        obs_tensor, mask_tensor
                    )
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store normalized observation in buffer
                self.buffer.add(
                    obs=obs_normalized,
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

                # Update learning rate
                self._update_lr()

                if pbar:
                    pbar.update(1)

                if done:
                    placement = info.get("placement", 8)
                    self.metrics.add_episode(episode_reward, episode_length, placement)

                    if len(self.metrics.placements) % log_interval == 0:
                        stats = self.metrics.get_recent_stats(100)
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        if pbar:
                            pbar.set_postfix({
                                "placement": f"{stats.get('avg_placement', 0):.2f}",
                                "top4": f"{stats.get('top4_rate', 0):.1%}",
                                "lr": f"{current_lr:.2e}",
                            })

                    obs, info = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs

                if self.total_timesteps >= target_timesteps:
                    break

            if len(self.buffer) > 0:
                losses = self._update()
                self.metrics.policy_losses.append(losses["policy_loss"])
                self.metrics.value_losses.append(losses["value_loss"])
                self.metrics.entropy_losses.append(losses["entropy_loss"])

        if pbar:
            pbar.close()

        return self

    def save(self, path: str):
        """Save model with all training state."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_timesteps": self.total_timesteps,
            "total_training_steps": self.total_training_steps,
        }

        # Save observation normalization state
        if self.normalize_observations and hasattr(self, "obs_rms"):
            save_dict["obs_rms_state"] = self.obs_rms.get_state()

        torch.save(save_dict, str(save_path) + ".pt")

    def load(self, path: str) -> "TransformerPPO":
        """Load model with all training state."""
        checkpoint = torch.load(str(path) + ".pt", map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.total_training_steps = checkpoint.get("total_training_steps", 1_000_000)

        # Load observation normalization state
        if "obs_rms_state" in checkpoint and self.normalize_observations:
            self.obs_rms.set_state(checkpoint["obs_rms_state"])

        return self

    @property
    def name(self) -> str:
        return "TransformerPPO"
