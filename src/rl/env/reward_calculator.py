"""
TFT Reward Calculator.

Reward composition:
1. Placement rewards: Final ranking-based reward
2. Round rewards: Win/loss, HP change
3. Shaping rewards: Synergies, upgrades, economy, unit acquisition
4. Action rewards: Buy, sell, level up decisions
"""

import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from .action_space import ActionType


@dataclass
class RewardConfig:
    """Reward configuration with dense reward support."""

    # Placement rewards (by rank) - INCREASED for stronger signal
    placement_rewards: Dict[int, float] = field(
        default_factory=lambda: {
            1: 15.0,   # 1st (increased from 10)
            2: 10.0,   # 2nd (increased from 6)
            3: 6.0,    # 3rd (increased from 3)
            4: 3.0,    # 4th (increased from 1)
            5: -1.0,   # 5th
            6: -4.0,   # 6th
            7: -8.0,   # 7th
            8: -15.0,  # 8th (increased penalty)
        }
    )

    # Round rewards - INCREASED for better signal
    win_reward: float = 2.0        # Increased from 0.5
    lose_reward: float = -0.5      # Increased penalty from -0.2
    hp_loss_penalty: float = -0.08  # Per HP lost (increased from -0.05)

    # Level up reward
    level_up_reward: float = 0.5   # Per level gained (increased from 0.3)

    # Unit acquisition rewards (by cost)
    unit_acquisition_rewards: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.05,   # 1-cost unit
            2: 0.10,   # 2-cost unit
            3: 0.20,   # 3-cost unit
            4: 0.40,   # 4-cost unit
            5: 0.80,   # 5-cost unit
        }
    )

    # Unit upgrade rewards (star level)
    unit_upgrade_rewards: Dict[int, float] = field(
        default_factory=lambda: {
            2: 0.5,    # 2-star upgrade
            3: 2.0,    # 3-star upgrade (much stronger signal)
        }
    )

    # Synergy rewards
    synergy_activation_reward: float = 0.3    # New synergy activated
    synergy_upgrade_reward: float = 0.2       # Synergy level increased

    # Economy rewards
    interest_threshold_reward: float = 0.1    # Reaching 10/20/30/40/50 gold
    economy_efficiency_reward: float = 0.05   # Maintaining gold for interest

    # Board management
    board_unit_reward: float = 0.1            # Per unit on board (up to level cap)
    full_board_bonus: float = 0.3             # Board is full (units == level)

    # Penalties
    invalid_action_penalty: float = -1.0      # Increased from -0.5
    empty_board_penalty: float = -1.0         # Increased from -0.5
    sell_penalty: float = -0.05               # Small penalty for selling (to discourage churn)
    excess_bench_penalty: float = -0.1        # Bench is full (can't buy more)

    # Streak bonuses
    win_streak_bonus: Dict[int, float] = field(
        default_factory=lambda: {
            2: 0.1,
            3: 0.2,
            4: 0.3,
            5: 0.5,  # 5+ win streak
        }
    )
    lose_streak_bonus: Dict[int, float] = field(
        default_factory=lambda: {
            2: 0.05,
            3: 0.1,
            4: 0.15,
            5: 0.2,  # 5+ lose streak (still some value from streak gold)
        }
    )


class RewardCalculator:
    """
    Dense reward calculator for TFT RL environment.

    Provides frequent, informative rewards to help agents learn faster.
    Tracks state changes to reward improvements and penalize mistakes.

    Usage:
        calc = RewardCalculator()
        reward = calc.calculate(player, action_type, action_valid, round_result, done, placement)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        # Previous state tracking (for shaping)
        self._prev_hp: int = 100
        self._prev_gold: int = 0
        self._prev_level: int = 1
        self._prev_board_count: int = 0
        self._prev_bench_count: int = 0
        self._prev_synergy_count: int = 0
        self._prev_active_synergies: set = set()
        self._prev_unit_stars: Dict[str, int] = {}  # champion_id -> star level
        self._prev_interest_tier: int = 0  # 0, 10, 20, 30, 40, 50

    def calculate(
        self,
        player: "PlayerState",
        action_type: "ActionType",
        action_valid: bool,
        round_result: Optional[Dict[str, Any]] = None,
        done: bool = False,
        placement: Optional[int] = None,
        action_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate reward with dense shaping.

        Args:
            player: Current player state.
            action_type: Action type performed.
            action_valid: Whether action was valid.
            round_result: Round result (win/loss, HP change, etc.)
            done: Whether game is over.
            placement: Final placement (if game over).
            action_info: Additional info about the action (unit bought, etc.)

        Returns:
            float: Total reward.
        """
        c = self.config
        reward = 0.0

        # 1. Invalid action penalty
        if not action_valid:
            reward += c.invalid_action_penalty
            return reward

        # 2. Placement reward (game over)
        if done and placement is not None:
            reward += c.placement_rewards.get(placement, 0)
            return reward

        # 3. Action-specific rewards
        reward += self._calculate_action_reward(action_type, action_info)

        # 4. Round reward (if round ended)
        if round_result:
            reward += self._calculate_round_reward(round_result, player)

        # 5. Shaping rewards (state improvement)
        reward += self._calculate_shaping_reward(player)

        # 6. Economy rewards
        reward += self._calculate_economy_reward(player)

        # 7. Synergy rewards
        reward += self._calculate_synergy_reward(player)

        # Update previous state
        self._update_prev_state(player)

        return reward

    def _calculate_action_reward(
        self,
        action_type: "ActionType",
        action_info: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reward for specific action types."""
        c = self.config
        reward = 0.0

        if action_info is None:
            return reward

        # Import here to avoid circular dependency
        from .action_space import ActionType as AT

        # Unit acquisition reward
        if action_type == AT.BUY and action_info.get("success", False):
            unit_cost = action_info.get("unit_cost", 1)
            reward += c.unit_acquisition_rewards.get(unit_cost, 0.05)

        # Unit upgrade reward (when buying completes a 2-star or 3-star)
        if action_info.get("upgrade_star"):
            star = action_info["upgrade_star"]
            reward += c.unit_upgrade_rewards.get(star, 0)

        # Selling penalty (small, to discourage unnecessary churn)
        if action_type in [AT.SELL_BENCH, AT.SELL_BOARD]:
            reward += c.sell_penalty

        return reward

    def _calculate_round_reward(
        self,
        result: Dict[str, Any],
        player: "PlayerState"
    ) -> float:
        """Calculate round win/loss reward."""
        c = self.config
        reward = 0.0

        if result.get("won", False):
            reward += c.win_reward

            # Win streak bonus
            win_streak = getattr(player, "win_streak", 0)
            if win_streak >= 5:
                reward += c.win_streak_bonus.get(5, 0)
            elif win_streak in c.win_streak_bonus:
                reward += c.win_streak_bonus[win_streak]
        else:
            reward += c.lose_reward
            hp_lost = result.get("hp_lost", 0)
            reward += hp_lost * c.hp_loss_penalty

            # Lose streak has some value (streak gold)
            lose_streak = getattr(player, "lose_streak", 0)
            if lose_streak >= 5:
                reward += c.lose_streak_bonus.get(5, 0)
            elif lose_streak in c.lose_streak_bonus:
                reward += c.lose_streak_bonus[lose_streak]

        return reward

    def _calculate_shaping_reward(self, player: "PlayerState") -> float:
        """Calculate reward shaping based on state changes."""
        c = self.config
        reward = 0.0

        # Get units safely
        units = getattr(player, "units", None)
        if units is None:
            return reward

        # Board and bench state
        board = getattr(units, "board", {})
        bench = getattr(units, "bench", {})
        board_count = len(board) if isinstance(board, dict) else 0
        bench_count = len(bench) if isinstance(bench, dict) else 0

        # Empty board penalty
        if board_count == 0:
            reward += c.empty_board_penalty

        # Level up reward
        current_level = getattr(player, "level", 1)
        if current_level > self._prev_level:
            reward += (current_level - self._prev_level) * c.level_up_reward

        # Board unit reward (incentivize placing units)
        if board_count > self._prev_board_count:
            units_added = board_count - self._prev_board_count
            reward += units_added * c.board_unit_reward

        # Full board bonus
        if board_count == current_level and board_count > 0:
            reward += c.full_board_bonus

        # Bench full penalty (can't buy more)
        if bench_count >= 9:
            reward += c.excess_bench_penalty

        return reward

    def _calculate_economy_reward(self, player: "PlayerState") -> float:
        """Calculate economy-related rewards."""
        c = self.config
        reward = 0.0

        current_gold = getattr(player, "gold", 0)

        # Interest threshold reward (reaching 10/20/30/40/50)
        current_tier = min(current_gold // 10, 5)
        if current_tier > self._prev_interest_tier:
            tiers_gained = current_tier - self._prev_interest_tier
            reward += tiers_gained * c.interest_threshold_reward

        # Economy efficiency (maintaining gold for interest)
        if current_gold >= 50:
            reward += c.economy_efficiency_reward

        return reward

    def _calculate_synergy_reward(self, player: "PlayerState") -> float:
        """Calculate synergy-related rewards."""
        c = self.config
        reward = 0.0

        units = getattr(player, "units", None)
        if units is None:
            return reward

        # Try to get active synergies
        try:
            if hasattr(units, "get_active_synergies"):
                active_synergies = units.get_active_synergies()
            elif hasattr(units, "active_synergies"):
                active_synergies = units.active_synergies
            else:
                return reward

            if not active_synergies:
                return reward

            # Get current active synergy names
            current_synergies = set()
            for trait_id, synergy_data in active_synergies.items():
                is_active = False
                if hasattr(synergy_data, "is_active"):
                    is_active = synergy_data.is_active
                elif isinstance(synergy_data, dict):
                    is_active = synergy_data.get("is_active", False)

                if is_active:
                    current_synergies.add(trait_id)

            # Reward for new synergy activations
            new_synergies = current_synergies - self._prev_active_synergies
            if new_synergies:
                reward += len(new_synergies) * c.synergy_activation_reward

            # Store for next calculation
            self._prev_active_synergies = current_synergies

        except Exception:
            # Silently handle any errors in synergy calculation
            pass

        return reward

    def _update_prev_state(self, player: "PlayerState"):
        """Save previous state for next calculation."""
        self._prev_hp = getattr(player, "hp", 100)
        self._prev_gold = getattr(player, "gold", 0)
        self._prev_level = getattr(player, "level", 1)
        self._prev_interest_tier = min(self._prev_gold // 10, 5)

        units = getattr(player, "units", None)
        if units:
            board = getattr(units, "board", {})
            bench = getattr(units, "bench", {})
            self._prev_board_count = len(board) if isinstance(board, dict) else 0
            self._prev_bench_count = len(bench) if isinstance(bench, dict) else 0

    def reset(self):
        """Reset for new episode."""
        self._prev_hp = 100
        self._prev_gold = 0
        self._prev_level = 1
        self._prev_board_count = 0
        self._prev_bench_count = 0
        self._prev_synergy_count = 0
        self._prev_active_synergies = set()
        self._prev_unit_stars = {}
        self._prev_interest_tier = 0

    def get_reward_breakdown(
        self,
        player: "PlayerState",
        action_type: "ActionType",
        action_valid: bool,
        round_result: Optional[Dict[str, Any]] = None,
        done: bool = False,
        placement: Optional[int] = None,
        action_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        Useful for debugging and analysis.
        """
        breakdown = {
            "action_reward": 0.0,
            "round_reward": 0.0,
            "shaping_reward": 0.0,
            "economy_reward": 0.0,
            "synergy_reward": 0.0,
            "placement_reward": 0.0,
            "penalty": 0.0,
            "total": 0.0,
        }

        if not action_valid:
            breakdown["penalty"] = self.config.invalid_action_penalty
            breakdown["total"] = breakdown["penalty"]
            return breakdown

        if done and placement is not None:
            breakdown["placement_reward"] = self.config.placement_rewards.get(placement, 0)
            breakdown["total"] = breakdown["placement_reward"]
            return breakdown

        breakdown["action_reward"] = self._calculate_action_reward(action_type, action_info)

        if round_result:
            breakdown["round_reward"] = self._calculate_round_reward(round_result, player)

        breakdown["shaping_reward"] = self._calculate_shaping_reward(player)
        breakdown["economy_reward"] = self._calculate_economy_reward(player)
        breakdown["synergy_reward"] = self._calculate_synergy_reward(player)

        breakdown["total"] = sum([
            breakdown["action_reward"],
            breakdown["round_reward"],
            breakdown["shaping_reward"],
            breakdown["economy_reward"],
            breakdown["synergy_reward"],
        ])

        return breakdown
