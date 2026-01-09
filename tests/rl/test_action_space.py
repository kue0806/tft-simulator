"""Tests for TFT Action Space."""

import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock

from src.rl.env.action_space import ActionSpace, ActionType, ActionConfig


class TestActionConfig:
    """Test ActionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ActionConfig()

        assert config.shop_size == 5
        assert config.bench_size == 9
        assert config.board_rows == 4
        assert config.board_cols == 7

    def test_board_size_property(self):
        """Test board_size property calculation."""
        config = ActionConfig()
        assert config.board_size == 28  # 4 * 7

        config = ActionConfig(board_rows=3, board_cols=5)
        assert config.board_size == 15  # 3 * 5


class TestActionType:
    """Test ActionType enum."""

    def test_action_type_values(self):
        """Test action type enum values."""
        assert ActionType.PASS == 0
        assert ActionType.BUY == 1
        assert ActionType.SELL_BENCH == 2
        assert ActionType.SELL_BOARD == 3
        assert ActionType.PLACE == 4
        assert ActionType.MOVE == 5
        assert ActionType.REFRESH == 6
        assert ActionType.BUY_XP == 7


class TestActionSpace:
    """Test ActionSpace."""

    def test_initialization(self):
        """Test action space initialization."""
        action_space = ActionSpace()

        assert action_space.config is not None
        assert action_space.num_actions > 0
        assert action_space.gym_space is not None

    def test_num_actions(self):
        """Test total number of actions."""
        action_space = ActionSpace()
        c = action_space.config

        # Calculate expected actions:
        # PASS: 1
        # BUY: shop_size (5)
        # SELL_BENCH: bench_size (9)
        # SELL_BOARD: board_size (28)
        # PLACE: bench_size * board_size (9 * 28 = 252)
        # REFRESH: 1
        # BUY_XP: 1
        expected = 1 + c.shop_size + c.bench_size + c.board_size + (c.bench_size * c.board_size) + 1 + 1

        assert action_space.num_actions == expected

    def test_decode_pass(self):
        """Test decode PASS action."""
        action_space = ActionSpace()
        action_type, params = action_space.decode_action(0)

        assert action_type == ActionType.PASS
        assert params is None

    def test_decode_buy(self):
        """Test decode BUY actions."""
        action_space = ActionSpace()

        for slot in range(5):
            action_idx = 1 + slot  # BUY starts at index 1
            action_type, params = action_space.decode_action(action_idx)

            assert action_type == ActionType.BUY
            assert params == slot

    def test_decode_sell_bench(self):
        """Test decode SELL_BENCH actions."""
        action_space = ActionSpace()

        for bench_idx in range(9):
            action_idx = 6 + bench_idx  # SELL_BENCH starts at index 6
            action_type, params = action_space.decode_action(action_idx)

            assert action_type == ActionType.SELL_BENCH
            assert params == bench_idx

    def test_decode_sell_board(self):
        """Test decode SELL_BOARD actions."""
        action_space = ActionSpace()

        for board_pos in range(28):
            action_idx = 15 + board_pos  # SELL_BOARD starts at index 15
            action_type, params = action_space.decode_action(action_idx)

            assert action_type == ActionType.SELL_BOARD
            assert params == board_pos

    def test_decode_place(self):
        """Test decode PLACE actions."""
        action_space = ActionSpace()

        # Test first PLACE action (bench 0 -> board 0)
        action_idx = 43  # PLACE starts at index 43
        action_type, params = action_space.decode_action(action_idx)

        assert action_type == ActionType.PLACE
        assert params == (0, 0)

        # Test another PLACE action (bench 1 -> board 5)
        action_idx = 43 + (1 * 28) + 5
        action_type, params = action_space.decode_action(action_idx)

        assert action_type == ActionType.PLACE
        assert params == (1, 5)

    def test_decode_refresh(self):
        """Test decode REFRESH action."""
        action_space = ActionSpace()

        # REFRESH is at index: 1 + 5 + 9 + 28 + 252 = 295
        action_idx = action_space.encode_action(ActionType.REFRESH, None)
        action_type, params = action_space.decode_action(action_idx)

        assert action_type == ActionType.REFRESH
        assert params is None

    def test_decode_buy_xp(self):
        """Test decode BUY_XP action."""
        action_space = ActionSpace()

        action_idx = action_space.encode_action(ActionType.BUY_XP, None)
        action_type, params = action_space.decode_action(action_idx)

        assert action_type == ActionType.BUY_XP
        assert params is None

    def test_decode_invalid_index(self):
        """Test decode with invalid index."""
        action_space = ActionSpace()

        # Too high index should return PASS
        action_type, params = action_space.decode_action(9999)
        assert action_type == ActionType.PASS
        assert params is None

        # Negative index should return PASS
        action_type, params = action_space.decode_action(-1)
        assert action_type == ActionType.PASS
        assert params is None

    def test_encode_decode_roundtrip(self):
        """Test encode then decode returns same action."""
        action_space = ActionSpace()

        # Test PASS
        idx = action_space.encode_action(ActionType.PASS, None)
        action_type, params = action_space.decode_action(idx)
        assert action_type == ActionType.PASS

        # Test BUY
        idx = action_space.encode_action(ActionType.BUY, 3)
        action_type, params = action_space.decode_action(idx)
        assert action_type == ActionType.BUY
        assert params == 3

        # Test PLACE
        idx = action_space.encode_action(ActionType.PLACE, (2, 10))
        action_type, params = action_space.decode_action(idx)
        assert action_type == ActionType.PLACE
        assert params == (2, 10)

    def test_encode_invalid_action(self):
        """Test encode with invalid action returns 0."""
        action_space = ActionSpace()

        # Invalid params should return 0 (PASS)
        idx = action_space.encode_action(ActionType.BUY, 100)
        assert idx == 0

    def test_pos_to_idx(self):
        """Test position to index conversion."""
        action_space = ActionSpace()

        assert action_space.pos_to_idx(0, 0) == 0
        assert action_space.pos_to_idx(0, 6) == 6
        assert action_space.pos_to_idx(1, 0) == 7
        assert action_space.pos_to_idx(3, 6) == 27

    def test_idx_to_pos(self):
        """Test index to position conversion."""
        action_space = ActionSpace()

        assert action_space.idx_to_pos(0) == (0, 0)
        assert action_space.idx_to_pos(6) == (0, 6)
        assert action_space.idx_to_pos(7) == (1, 0)
        assert action_space.idx_to_pos(27) == (3, 6)

    def test_sample_valid_action(self):
        """Test random valid action sampling."""
        action_space = ActionSpace()

        # Create mask with only some valid actions
        mask = np.zeros(action_space.num_actions, dtype=np.float32)
        mask[0] = 1.0  # PASS
        mask[1] = 1.0  # BUY slot 0

        # Sample should be one of valid actions
        for _ in range(10):
            action = action_space.sample_valid_action(mask)
            assert action in [0, 1]

    def test_sample_valid_action_empty_mask(self):
        """Test sampling with empty mask returns PASS."""
        action_space = ActionSpace()

        mask = np.zeros(action_space.num_actions, dtype=np.float32)
        action = action_space.sample_valid_action(mask)

        assert action == 0  # PASS


class TestGetValidActions:
    """Test get_valid_actions method."""

    @pytest.fixture
    def action_space(self):
        return ActionSpace()

    @pytest.fixture
    def mock_player(self):
        """Create mock player."""
        player = MagicMock()
        player.gold = 10
        player.level = 3
        player.player_id = 0

        # Mock units
        player.units = MagicMock()
        player.units.board = {}
        player.units.bench = [None] * 9

        return player

    @pytest.fixture
    def mock_game(self):
        """Create mock game."""
        game = MagicMock()
        game.get_shop_for_player = MagicMock(return_value=None)
        game.shops = {}
        return game

    def test_pass_always_valid(self, action_space, mock_player, mock_game):
        """Test PASS action is always valid."""
        mask = action_space.get_valid_actions(mock_player, mock_game)

        assert mask[0] == 1.0  # PASS

    def test_refresh_valid_with_gold(self, action_space, mock_player, mock_game):
        """Test REFRESH valid when gold >= 2."""
        mock_player.gold = 2
        mask = action_space.get_valid_actions(mock_player, mock_game)

        refresh_idx = action_space.encode_action(ActionType.REFRESH, None)
        assert mask[refresh_idx] == 1.0

    def test_refresh_invalid_without_gold(self, action_space, mock_player, mock_game):
        """Test REFRESH invalid when gold < 2."""
        mock_player.gold = 1
        mask = action_space.get_valid_actions(mock_player, mock_game)

        refresh_idx = action_space.encode_action(ActionType.REFRESH, None)
        assert mask[refresh_idx] == 0.0

    def test_buy_xp_valid_with_gold_and_level(self, action_space, mock_player, mock_game):
        """Test BUY_XP valid when gold >= 4 and level < 10."""
        mock_player.gold = 4
        mock_player.level = 5
        mask = action_space.get_valid_actions(mock_player, mock_game)

        buy_xp_idx = action_space.encode_action(ActionType.BUY_XP, None)
        assert mask[buy_xp_idx] == 1.0

    def test_buy_xp_invalid_at_max_level(self, action_space, mock_player, mock_game):
        """Test BUY_XP invalid when level == 10."""
        mock_player.gold = 10
        mock_player.level = 10
        mask = action_space.get_valid_actions(mock_player, mock_game)

        buy_xp_idx = action_space.encode_action(ActionType.BUY_XP, None)
        assert mask[buy_xp_idx] == 0.0

    def test_sell_bench_valid(self, action_space, mock_player, mock_game):
        """Test SELL_BENCH valid when bench has unit."""
        # Put unit at bench slot 2
        mock_unit = MagicMock()
        mock_player.units.bench[2] = mock_unit

        mask = action_space.get_valid_actions(mock_player, mock_game)

        sell_idx = action_space.encode_action(ActionType.SELL_BENCH, 2)
        assert mask[sell_idx] == 1.0

        # Empty slot should be invalid
        sell_idx_empty = action_space.encode_action(ActionType.SELL_BENCH, 0)
        assert mask[sell_idx_empty] == 0.0

    def test_sell_board_valid(self, action_space, mock_player, mock_game):
        """Test SELL_BOARD valid when board has unit."""
        mock_unit = MagicMock()
        mock_player.units.board = {(1, 2): mock_unit}

        mask = action_space.get_valid_actions(mock_player, mock_game)

        board_pos = action_space.pos_to_idx(1, 2)
        sell_idx = action_space.encode_action(ActionType.SELL_BOARD, board_pos)
        assert mask[sell_idx] == 1.0

    def test_place_valid(self, action_space, mock_player, mock_game):
        """Test PLACE valid when bench has unit and board has space."""
        mock_player.level = 3  # Can place 3 units
        mock_player.units.board = {}  # Empty board

        # Put unit at bench slot 0
        mock_unit = MagicMock()
        mock_player.units.bench[0] = mock_unit

        mask = action_space.get_valid_actions(mock_player, mock_game)

        # Should be able to place to any board position
        place_idx = action_space.encode_action(ActionType.PLACE, (0, 0))
        assert mask[place_idx] == 1.0

    def test_place_invalid_board_full(self, action_space, mock_player, mock_game):
        """Test PLACE invalid when board is at max capacity."""
        mock_player.level = 2  # Can only place 2 units

        # Fill board to capacity
        mock_player.units.board = {(0, 0): MagicMock(), (0, 1): MagicMock()}

        # Put unit at bench
        mock_player.units.bench[0] = MagicMock()

        mask = action_space.get_valid_actions(mock_player, mock_game)

        # No PLACE actions should be valid
        place_idx = action_space.encode_action(ActionType.PLACE, (0, 2))
        assert mask[place_idx] == 0.0

    def test_buy_valid_with_shop(self, action_space, mock_player, mock_game):
        """Test BUY valid when shop has affordable champion."""
        # Create shop with affordable champion
        mock_shop = MagicMock()
        mock_slot = MagicMock()
        mock_slot.is_purchased = False
        mock_slot.champion = MagicMock()
        mock_slot.champion.cost = 2
        mock_shop.slots = [mock_slot, None, None, None, None]

        mock_game.get_shop_for_player = MagicMock(return_value=mock_shop)

        mock_player.gold = 5
        mock_player.units.bench = [None] * 9  # Empty bench

        mask = action_space.get_valid_actions(mock_player, mock_game)

        buy_idx = action_space.encode_action(ActionType.BUY, 0)
        assert mask[buy_idx] == 1.0

    def test_buy_invalid_not_enough_gold(self, action_space, mock_player, mock_game):
        """Test BUY invalid when not enough gold."""
        mock_shop = MagicMock()
        mock_slot = MagicMock()
        mock_slot.is_purchased = False
        mock_slot.champion = MagicMock()
        mock_slot.champion.cost = 5
        mock_shop.slots = [mock_slot]

        mock_game.get_shop_for_player = MagicMock(return_value=mock_shop)

        mock_player.gold = 2  # Not enough
        mock_player.units.bench = [None] * 9

        mask = action_space.get_valid_actions(mock_player, mock_game)

        buy_idx = action_space.encode_action(ActionType.BUY, 0)
        assert mask[buy_idx] == 0.0

    def test_buy_invalid_bench_full(self, action_space, mock_player, mock_game):
        """Test BUY invalid when bench is full."""
        mock_shop = MagicMock()
        mock_slot = MagicMock()
        mock_slot.is_purchased = False
        mock_slot.champion = MagicMock()
        mock_slot.champion.cost = 1
        mock_shop.slots = [mock_slot]

        mock_game.get_shop_for_player = MagicMock(return_value=mock_shop)

        mock_player.gold = 10
        mock_player.units.bench = [MagicMock() for _ in range(9)]  # Full bench

        mask = action_space.get_valid_actions(mock_player, mock_game)

        buy_idx = action_space.encode_action(ActionType.BUY, 0)
        assert mask[buy_idx] == 0.0

    def test_mask_shape(self, action_space, mock_player, mock_game):
        """Test mask has correct shape."""
        mask = action_space.get_valid_actions(mock_player, mock_game)

        assert mask.shape == (action_space.num_actions,)
        assert mask.dtype == np.float32
