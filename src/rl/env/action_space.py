"""
TFT Action Space Definition.

Action Types:
1. PASS - Do nothing
2. BUY - Buy from shop slot 0-4
3. SELL_BENCH - Sell bench unit
4. SELL_BOARD - Sell board unit
5. PLACE - Place bench unit on board
6. REFRESH - Refresh shop
7. BUY_XP - Buy experience
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import IntEnum

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

if TYPE_CHECKING:
    from src.core.game_state import PlayerState, GameState


class ActionType(IntEnum):
    """Action types."""

    PASS = 0
    BUY = 1  # buy_slot_idx
    SELL_BENCH = 2  # sell_bench_idx
    SELL_BOARD = 3  # sell_board_pos
    PLACE = 4  # place_bench_idx_to_board_pos
    MOVE = 5  # move_from_pos_to_pos
    REFRESH = 6  # shop reroll
    BUY_XP = 7  # buy experience
    SELECT_AUGMENT = 8  # select augment option (0, 1, or 2)


@dataclass
class ActionConfig:
    """Action space configuration."""

    shop_size: int = 5
    bench_size: int = 9
    board_rows: int = 4
    board_cols: int = 7
    num_augment_choices: int = 3  # Number of augment options to choose from

    @property
    def board_size(self) -> int:
        return self.board_rows * self.board_cols


class ActionSpace:
    """
    TFT Action Space.

    Maps action indices to (ActionType, params) tuples.
    """

    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()

        # Build action mapping
        self._build_action_mapping()

        # Gym action space
        self.gym_space = spaces.Discrete(self.num_actions)

    def _build_action_mapping(self):
        """Build action index -> (type, params) mapping."""
        c = self.config
        self._idx_to_action: List[Tuple[ActionType, Any]] = []
        self._action_to_idx: Dict[Tuple, int] = {}

        idx = 0

        # 0: PASS
        self._idx_to_action.append((ActionType.PASS, None))
        self._action_to_idx[(ActionType.PASS, None)] = idx
        idx += 1

        # 1-5: BUY (shop slots 0-4)
        for slot in range(c.shop_size):
            self._idx_to_action.append((ActionType.BUY, slot))
            self._action_to_idx[(ActionType.BUY, slot)] = idx
            idx += 1

        # 6-14: SELL_BENCH (bench slots 0-8)
        for bench_idx in range(c.bench_size):
            self._idx_to_action.append((ActionType.SELL_BENCH, bench_idx))
            self._action_to_idx[(ActionType.SELL_BENCH, bench_idx)] = idx
            idx += 1

        # 15-42: SELL_BOARD (board positions 0-27)
        for board_pos in range(c.board_size):
            self._idx_to_action.append((ActionType.SELL_BOARD, board_pos))
            self._action_to_idx[(ActionType.SELL_BOARD, board_pos)] = idx
            idx += 1

        # 43-294: PLACE (bench 9 × board 28 = 252)
        for bench_idx in range(c.bench_size):
            for board_pos in range(c.board_size):
                self._idx_to_action.append((ActionType.PLACE, (bench_idx, board_pos)))
                self._action_to_idx[(ActionType.PLACE, (bench_idx, board_pos))] = idx
                idx += 1

        # 295: REFRESH
        self._idx_to_action.append((ActionType.REFRESH, None))
        self._action_to_idx[(ActionType.REFRESH, None)] = idx
        idx += 1

        # 296: BUY_XP
        self._idx_to_action.append((ActionType.BUY_XP, None))
        self._action_to_idx[(ActionType.BUY_XP, None)] = idx
        idx += 1

        # 297-299: SELECT_AUGMENT (options 0, 1, 2)
        for augment_idx in range(c.num_augment_choices):
            self._idx_to_action.append((ActionType.SELECT_AUGMENT, augment_idx))
            self._action_to_idx[(ActionType.SELECT_AUGMENT, augment_idx)] = idx
            idx += 1

        self.num_actions = idx

    def decode_action(self, action_idx: int) -> Tuple[ActionType, Any]:
        """Decode action index to (type, params)."""
        if 0 <= action_idx < len(self._idx_to_action):
            return self._idx_to_action[action_idx]
        return (ActionType.PASS, None)

    def encode_action(self, action_type: ActionType, params: Any = None) -> int:
        """Encode (type, params) to action index."""
        key = (action_type, params)
        return self._action_to_idx.get(key, 0)

    def get_valid_actions(
        self, player: "PlayerState", game: "GameState"
    ) -> np.ndarray:
        """
        Get valid action mask for current state.

        Args:
            player: Current player state.
            game: Full game state.

        Returns:
            np.ndarray: shape (num_actions,), 1 if valid, 0 if invalid.
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)
        c = self.config

        # PASS is always valid
        mask[0] = 1.0

        # Get player attributes safely
        gold = getattr(player, "gold", 0)
        level = getattr(player, "level", 1)
        units = getattr(player, "units", None)
        player_id = getattr(player, "player_id", 0)

        if units is None:
            return mask

        board = getattr(units, "board", {})
        bench = getattr(units, "bench", [])

        # BUY: gold sufficient + bench space + shop has unit
        shop = None
        if hasattr(game, "get_shop_for_player"):
            shop = game.get_shop_for_player(player_id)
        elif hasattr(game, "shops"):
            shop = game.shops.get(player_id)

        bench_space = sum(1 for b in bench if b is None) if bench else 0

        if shop and bench_space > 0:
            slots = getattr(shop, "slots", [])
            for slot_idx, slot in enumerate(slots[: c.shop_size]):
                if slot is None:
                    continue

                is_purchased = getattr(slot, "is_purchased", False)
                if is_purchased:
                    continue

                champion = getattr(slot, "champion", slot)
                if champion is None:
                    continue

                cost = getattr(champion, "cost", 999)
                if gold >= cost:
                    action_idx = self.encode_action(ActionType.BUY, slot_idx)
                    mask[action_idx] = 1.0

        # SELL_BENCH: bench has unit
        for bench_idx, unit in enumerate(bench[: c.bench_size]):
            if unit is not None:
                action_idx = self.encode_action(ActionType.SELL_BENCH, bench_idx)
                mask[action_idx] = 1.0

        # SELL_BOARD: board has unit
        if isinstance(board, dict):
            for pos, unit in board.items():
                if isinstance(pos, tuple) and len(pos) == 2:
                    board_pos = pos[0] * c.board_cols + pos[1]
                    if board_pos < c.board_size:
                        action_idx = self.encode_action(ActionType.SELL_BOARD, board_pos)
                        mask[action_idx] = 1.0

        # PLACE: bench unit -> empty board position
        board_positions = set()
        if isinstance(board, dict):
            for pos in board.keys():
                if isinstance(pos, tuple) and len(pos) == 2:
                    board_positions.add(pos[0] * c.board_cols + pos[1])

        max_units = level  # Level = max placed units

        if len(board) < max_units:
            for bench_idx, unit in enumerate(bench[: c.bench_size]):
                if unit is not None:
                    for board_pos in range(c.board_size):
                        if board_pos not in board_positions:
                            action_idx = self.encode_action(
                                ActionType.PLACE, (bench_idx, board_pos)
                            )
                            mask[action_idx] = 1.0

        # REFRESH: gold >= 2
        if gold >= 2:
            action_idx = self.encode_action(ActionType.REFRESH, None)
            mask[action_idx] = 1.0

        # BUY_XP: gold >= 4 + level < 10
        if gold >= 4 and level < 10:
            action_idx = self.encode_action(ActionType.BUY_XP, None)
            mask[action_idx] = 1.0

        # SELECT_AUGMENT: only valid during augment selection phase
        # This is handled separately via augment_choices parameter
        # Augment actions are NOT valid during normal gameplay

        return mask

    def get_augment_action_mask(
        self, num_available_choices: int = 3
    ) -> np.ndarray:
        """
        Get action mask for augment selection phase.

        During augment selection, ONLY augment actions are valid.
        All other actions are masked out.

        Args:
            num_available_choices: Number of augment options available (usually 3).

        Returns:
            np.ndarray: shape (num_actions,), 1 if valid, 0 if invalid.
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)

        # Only augment selection actions are valid
        for i in range(min(num_available_choices, self.config.num_augment_choices)):
            action_idx = self.encode_action(ActionType.SELECT_AUGMENT, i)
            mask[action_idx] = 1.0

        return mask

    def sample_valid_action(self, mask: np.ndarray) -> int:
        """Sample random valid action from mask."""
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            return 0  # PASS
        return np.random.choice(valid_indices)

    def pos_to_idx(self, row: int, col: int) -> int:
        """(row, col) -> board position index."""
        return row * self.config.board_cols + col

    def idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """Board position index -> (row, col)."""
        return (idx // self.config.board_cols, idx % self.config.board_cols)
