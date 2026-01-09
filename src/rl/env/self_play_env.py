"""
TFT Self-Play Environment.

Extends TFTEnv to support self-play training where all players
use trained RL models as opponents instead of random bots.

Supports:
- League training with mixed opponent pools
- Trained models as opponents (e.g., 2x PPO, 2x DQN, 2x Transformer)
- Curriculum learning with mix of random bots and self-play
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
import copy

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from .tft_env import TFTEnv
from .action_space import ActionType
from .state_encoder import EncoderConfig
from .action_space import ActionConfig
from .reward_calculator import RewardConfig
from ..models.base import BaseRLModel


class SelfPlayEnv(TFTEnv):
    """
    Self-Play TFT Environment.

    All players use learned policies (current or past versions).
    Supports curriculum learning with mix of random bots and self-play.

    Usage with trained models:
        models = {"PPO": ppo_model, "DQN": dqn_model}
        distribution = {"PPO": 3, "DQN": 4}  # 3 PPO opponents, 4 DQN opponents
        env = SelfPlayEnv(opponent_models=models, model_distribution=distribution)

    Usage with policy function:
        env = SelfPlayEnv(num_players=8)
        env.set_opponent_policy(policy_fn)  # Set learned policy
        obs, info = env.reset()

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        num_players: int = 8,
        agent_player_idx: int = 0,
        max_rounds: int = 50,
        render_mode: Optional[str] = None,
        encoder_config: Optional[EncoderConfig] = None,
        action_config: Optional[ActionConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        # Self-play specific params
        self_play_ratio: float = 0.5,  # Ratio of self-play vs random bots
        place_units_for_bots: bool = True,  # Auto-place units on board
        # Trained model opponents
        opponent_models: Optional[Dict[str, "BaseRLModel"]] = None,
        model_distribution: Optional[Dict[str, int]] = None,
        opponent_deterministic: bool = True,
    ):
        super().__init__(
            num_players=num_players,
            agent_player_idx=agent_player_idx,
            max_rounds=max_rounds,
            render_mode=render_mode,
            encoder_config=encoder_config,
            action_config=action_config,
            reward_config=reward_config,
        )

        self.self_play_ratio = self_play_ratio
        self.place_units_for_bots = place_units_for_bots
        self.opponent_deterministic = opponent_deterministic

        # Opponent policy function: obs -> action
        self._opponent_policy: Optional[Callable[[np.ndarray], int]] = None

        # Track which opponents use self-play vs random
        self._opponent_types: Dict[int, str] = {}  # 'self_play', 'random', or model name

        # Trained model opponents
        self._opponent_models = opponent_models or {}
        self._model_distribution = model_distribution

        # Map player_idx -> (model, model_name)
        self._player_models: Dict[int, Tuple["BaseRLModel", str]] = {}

        # Stats tracking
        self.opponent_stats: Dict[str, Dict] = {}
        for name in self._opponent_models.keys():
            self.opponent_stats[name] = {
                "wins": 0, "top4": 0, "games": 0, "total_placement": 0
            }

        # Assign models to players if provided
        if opponent_models and model_distribution:
            self._assign_models_to_players()

    def _assign_models_to_players(self):
        """Assign trained models to opponent player slots."""
        if not self._opponent_models or not self._model_distribution:
            return

        player_idx = 0
        for model_name, count in self._model_distribution.items():
            if model_name not in self._opponent_models:
                continue
            model = self._opponent_models[model_name]
            for _ in range(count):
                if player_idx == self.agent_player_idx:
                    player_idx += 1
                if player_idx >= self.num_players:
                    break
                self._player_models[player_idx] = (model, model_name)
                self._opponent_types[player_idx] = model_name
                player_idx += 1

        # Fill remaining slots with random bots
        for i in range(self.num_players):
            if i != self.agent_player_idx and i not in self._player_models:
                self._opponent_types[i] = 'random'

    def set_opponent_policy(self, policy_fn: Optional[Callable[[np.ndarray], int]]):
        """
        Set the policy function for self-play opponents.

        Args:
            policy_fn: Function that takes observation and returns action.
                       Can be None to use random policy.
        """
        self._opponent_policy = policy_fn

    def set_self_play_ratio(self, ratio: float):
        """Set the ratio of self-play opponents (0.0 to 1.0)."""
        self.self_play_ratio = max(0.0, min(1.0, ratio))

    def set_opponent_models(
        self,
        opponent_models: Dict[str, "BaseRLModel"],
        model_distribution: Dict[str, int],
    ):
        """
        Set trained models as opponents.

        Args:
            opponent_models: Dict of model_name -> trained model
            model_distribution: Dict of model_name -> number of opponents

        Example:
            >>> models = {"PPO": ppo_model, "DQN": dqn_model}
            >>> distribution = {"PPO": 3, "DQN": 4}
            >>> env.set_opponent_models(models, distribution)
        """
        self._opponent_models = opponent_models
        self._model_distribution = model_distribution

        # Reset stats
        self.opponent_stats = {}
        for name in opponent_models.keys():
            self.opponent_stats[name] = {
                "wins": 0, "top4": 0, "games": 0, "total_placement": 0
            }

        self._assign_models_to_players()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and assign opponent types."""
        obs, info = super().reset(seed=seed, options=options)

        # Re-assign models to players if we have trained models
        if self._opponent_models and self._model_distribution:
            self._assign_models_to_players()
        else:
            # Assign opponent types based on self_play_ratio (only if no trained models)
            self._opponent_types = {}
            for i in range(self.num_players):
                if i == self.agent_player_idx:
                    continue
                if np.random.random() < self.self_play_ratio and self._opponent_policy is not None:
                    self._opponent_types[i] = 'self_play'
                else:
                    self._opponent_types[i] = 'random'

        return obs, info

    def _update_bot_players(self):
        """
        Update bot players using trained models, self-play policy, or random heuristics.
        """
        for i, player in enumerate(self.game.players):
            if i == self.agent_player_idx:
                continue

            is_alive = getattr(player, "is_alive", True)
            if not is_alive:
                continue

            opponent_type = self._opponent_types.get(i, 'random')

            # Check if this player uses a trained model
            if i in self._player_models:
                self._update_trained_model_opponent(i, player)
            elif opponent_type == 'self_play' and self._opponent_policy is not None:
                self._update_self_play_opponent(i, player)
            else:
                self._update_random_opponent(i, player)

    def _update_trained_model_opponent(self, player_idx: int, player):
        """Update opponent using a trained RL model."""
        model, model_name = self._player_models[player_idx]
        shop = self.shops.get(player_idx)
        if not shop:
            return

        # Get observation from opponent's perspective
        obs = self.state_encoder.encode(player, self.game, player_idx)

        # Get valid action mask
        action_mask = self.action_space_handler.get_valid_actions(player, self.game)

        # Let opponent take multiple actions
        for _ in range(self._max_actions_per_round):
            try:
                # Get action from trained model
                action, _ = model.predict(
                    obs,
                    action_mask=action_mask,
                    deterministic=self.opponent_deterministic,
                )

                # Decode action
                action_type, params = self.action_space_handler.decode_action(action)

                # Execute action
                if action_type == ActionType.PASS:
                    break

                self._execute_opponent_action(player, player_idx, action_type, params)

                # Update observation and mask
                obs = self.state_encoder.encode(player, self.game, player_idx)
                action_mask = self.action_space_handler.get_valid_actions(player, self.game)

            except Exception as e:
                # On error, break and continue with random behavior
                break

        # Auto-place units on board if enabled
        if self.place_units_for_bots:
            self._auto_place_units(player)

        # Give income
        income = self._calculate_income(player)
        player.gold = getattr(player, "gold", 0) + income

        # Refresh shop
        is_locked = getattr(shop, "is_locked", False)
        if not is_locked:
            shop.refresh()

    def _update_self_play_opponent(self, player_idx: int, player):
        """Update opponent using learned policy."""
        shop = self.shops.get(player_idx)
        if not shop:
            return

        # Get observation from opponent's perspective
        obs = self.state_encoder.encode(player, self.game, player_idx)

        # Let opponent take multiple actions (similar to agent)
        for _ in range(self._max_actions_per_round):
            # Get action from policy
            action = self._opponent_policy(obs)

            # Decode action
            action_type, params = self.action_space_handler.decode_action(action)

            # Execute action
            if action_type == ActionType.PASS:
                break

            self._execute_opponent_action(player, player_idx, action_type, params)

            # Update observation
            obs = self.state_encoder.encode(player, self.game, player_idx)

        # Auto-place units on board if enabled
        if self.place_units_for_bots:
            self._auto_place_units(player)

        # Give income
        income = self._calculate_income(player)
        player.gold = getattr(player, "gold", 0) + income

        # Refresh shop
        is_locked = getattr(shop, "is_locked", False)
        if not is_locked:
            shop.refresh()

    def _update_random_opponent(self, player_idx: int, player):
        """Update opponent using random/heuristic policy."""
        shop = self.shops.get(player_idx)
        if not shop:
            return

        gold = getattr(player, "gold", 0)

        # Random buy
        slots = getattr(shop, "slots", [])
        for slot_idx, slot in enumerate(slots):
            if slot is None:
                continue

            is_purchased = getattr(slot, "is_purchased", False)
            if is_purchased:
                continue

            champion = getattr(slot, "champion", slot)
            if champion is None:
                continue

            cost = getattr(champion, "cost", 999)
            if gold >= cost and np.random.random() < 0.7:
                # Purchase champion
                if hasattr(shop, "purchase"):
                    purchased = shop.purchase(slot_idx)
                    if purchased:
                        player.gold = gold - cost
                        units = getattr(player, "units", None)
                        if units and hasattr(units, "add_to_bench"):
                            units.add_to_bench(purchased)
                gold = getattr(player, "gold", 0)

        # Auto-place units on board
        if self.place_units_for_bots:
            self._auto_place_units(player)

        # Random level up
        gold = getattr(player, "gold", 0)
        level = getattr(player, "level", 1)
        if gold >= 4 and level < 10 and np.random.random() < 0.3:
            player.gold = gold - 4
            if hasattr(player, "add_xp"):
                player.add_xp(4)
            shop.player_level = getattr(player, "level", 1)

        # Give income
        income = self._calculate_income(player)
        player.gold = getattr(player, "gold", 0) + income

        # Refresh shop
        is_locked = getattr(shop, "is_locked", False)
        if not is_locked:
            shop.refresh()

    def _execute_opponent_action(
        self, player, player_idx: int, action_type: ActionType, params: Any
    ) -> bool:
        """Execute action for opponent player."""
        c = self.action_space_handler.config
        shop = self.shops.get(player_idx)

        if action_type == ActionType.PASS:
            return True

        elif action_type == ActionType.BUY:
            slot_idx = params
            if shop and 0 <= slot_idx < c.shop_size:
                return self._execute_buy(shop, slot_idx, player)
            return False

        elif action_type == ActionType.SELL_BENCH:
            bench_idx = params
            if 0 <= bench_idx < c.bench_size:
                return self._execute_sell_bench(player, bench_idx)
            return False

        elif action_type == ActionType.SELL_BOARD:
            board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return self._execute_sell_board(player, pos)

        elif action_type == ActionType.PLACE:
            bench_idx, board_pos = params
            pos = self.action_space_handler.idx_to_pos(board_pos)
            return self._execute_place(player, bench_idx, pos)

        elif action_type == ActionType.REFRESH:
            gold = getattr(player, "gold", 0)
            if gold >= 2 and shop:
                player.gold = gold - 2
                shop.refresh()
                return True
            return False

        elif action_type == ActionType.BUY_XP:
            gold = getattr(player, "gold", 0)
            level = getattr(player, "level", 1)
            if gold >= 4 and level < 10:
                player.gold = gold - 4
                if hasattr(player, "add_xp"):
                    player.add_xp(4)
                else:
                    player.xp = getattr(player, "xp", 0) + 4
                if shop:
                    shop.player_level = getattr(player, "level", 1)
                return True
            return False

        return False

    def _auto_place_units(self, player):
        """
        Auto-place bench units onto empty board positions.
        Used for bots to ensure they have units on board.
        """
        units = getattr(player, "units", None)
        if units is None:
            return

        bench = getattr(units, "bench", [])
        board = getattr(units, "board", {})
        level = getattr(player, "level", 1)

        # Get board config
        c = self.action_space_handler.config
        max_row = c.board_rows
        max_col = c.board_cols

        # Find empty positions
        all_positions = [(r, c) for r in range(max_row) for c in range(max_col)]
        empty_positions = [pos for pos in all_positions if pos not in board]

        # Place bench units on board
        for bench_idx, unit in enumerate(bench):
            if unit is None:
                continue

            if len(board) >= level:
                break

            if not empty_positions:
                break

            # Place unit
            pos = empty_positions.pop(0)
            if hasattr(units, "place_from_bench"):
                units.place_from_bench(bench_idx, pos)
            else:
                board[pos] = unit
                bench[bench_idx] = None

    def get_opponent_observations(self) -> Dict[int, np.ndarray]:
        """
        Get observations for all opponent players.

        Useful for external evaluation or analysis.
        """
        observations = {}
        for i, player in enumerate(self.game.players):
            if i == self.agent_player_idx:
                continue
            if getattr(player, "is_alive", True):
                observations[i] = self.state_encoder.encode(
                    player, self.game, i
                )
        return observations

    def get_opponent_types(self) -> Dict[int, str]:
        """Get the type ('self_play' or 'random') of each opponent."""
        return self._opponent_types.copy()

    def _record_final_placements(self):
        """Record final placements for all model opponents."""
        if self.game is None:
            return

        # Get final placements
        players_with_hp = [
            (i, p) for i, p in enumerate(self.game.players)
            if hasattr(p, 'hp')
        ]
        players_sorted = sorted(players_with_hp, key=lambda x: x[1].hp, reverse=True)

        for rank, (player_idx, player) in enumerate(players_sorted, 1):
            if player_idx in self._player_models:
                _, model_name = self._player_models[player_idx]
                if model_name in self.opponent_stats:
                    self.opponent_stats[model_name]["games"] += 1
                    self.opponent_stats[model_name]["total_placement"] += rank
                    if rank == 1:
                        self.opponent_stats[model_name]["wins"] += 1
                    if rank <= 4:
                        self.opponent_stats[model_name]["top4"] += 1

    def get_opponent_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all opponent models.

        Returns:
            Dict mapping model_name -> stats dict with:
                - games: total games played
                - win_rate: fraction of wins
                - top4_rate: fraction of top 4 finishes
                - avg_placement: average placement
        """
        result = {}
        for name, stats in self.opponent_stats.items():
            games = stats["games"]
            if games > 0:
                result[name] = {
                    "games": games,
                    "win_rate": stats["wins"] / games,
                    "top4_rate": stats["top4"] / games,
                    "avg_placement": stats["total_placement"] / games,
                }
            else:
                result[name] = {
                    "games": 0, "win_rate": 0.0, "top4_rate": 0.0, "avg_placement": 0.0
                }
        return result


def create_league_env(
    trained_models: Dict[str, "BaseRLModel"],
    model_distribution: Optional[Dict[str, int]] = None,
    training_player_idx: int = 0,
    **env_kwargs
) -> SelfPlayEnv:
    """
    Create a self-play environment with specified model distribution.

    Args:
        trained_models: Dict of model_name -> trained model
        model_distribution: Dict of model_name -> number of opponents using that model
                           If None, distributes evenly across 7 opponent slots
        training_player_idx: Index of training player (default: 0)
        **env_kwargs: Additional environment kwargs (num_players, max_rounds, etc.)

    Returns:
        SelfPlayEnv configured with opponent models

    Example:
        >>> models = {"PPO": ppo_model, "DQN": dqn_model, "Transformer": transformer_model}
        >>> distribution = {"PPO": 2, "DQN": 2, "Transformer": 3}
        >>> env = create_league_env(models, distribution)

        # This creates:
        # Player 0: Training agent
        # Players 1-2: PPO model
        # Players 3-4: DQN model
        # Players 5-7: Transformer model
    """
    num_players = env_kwargs.get("num_players", 8)
    num_opponents = num_players - 1

    if model_distribution is None:
        # Default: distribute evenly across opponent slots
        num_models = len(trained_models)
        if num_models == 0:
            model_distribution = {}
        else:
            slots_per_model = num_opponents // num_models
            remainder = num_opponents % num_models

            model_distribution = {}
            for i, name in enumerate(trained_models.keys()):
                count = slots_per_model + (1 if i < remainder else 0)
                model_distribution[name] = count

    return SelfPlayEnv(
        agent_player_idx=training_player_idx,
        opponent_models=trained_models,
        model_distribution=model_distribution,
        **env_kwargs
    )


# Register environment
try:
    gym.register(
        id="TFT-SelfPlay-v0",
        entry_point="src.rl.env.self_play_env:SelfPlayEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass  # Already registered or gym not available
