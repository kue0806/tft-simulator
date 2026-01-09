"""Tests for TFT State Encoder."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.rl.env.state_encoder import StateEncoder, EncoderConfig


class TestEncoderConfig:
    """Test EncoderConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EncoderConfig()

        assert config.champion_embed_dim == 32
        assert config.item_embed_dim == 16
        assert config.trait_embed_dim == 8
        assert config.board_size == 28
        assert config.bench_size == 9
        assert config.shop_size == 5
        assert config.num_champions == 100
        assert config.num_items == 46
        assert config.num_traits == 44
        assert config.max_players == 8
        assert config.max_gold == 100
        assert config.max_hp == 100
        assert config.max_level == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = EncoderConfig(
            champion_embed_dim=64,
            board_size=35,
            max_gold=200,
        )

        assert config.champion_embed_dim == 64
        assert config.board_size == 35
        assert config.max_gold == 200


class TestStateEncoder:
    """Test StateEncoder."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_player(self):
        """Create mock player."""
        player = MagicMock()
        player.player_id = 0
        player.gold = 50
        player.hp = 80
        player.level = 5
        player.xp = 20
        player.streak = 2

        # Mock units
        player.units = MagicMock()
        player.units.board = {}
        player.units.bench = [None] * 9
        player.units.get_active_synergies = MagicMock(return_value={})

        return player

    @pytest.fixture
    def mock_game(self):
        """Create mock game."""
        game = MagicMock()
        game.stage_manager = MagicMock()
        game.stage_manager.stage = 3
        game.stage_manager.round = 2
        game.stage_manager.is_pvp_round = MagicMock(return_value=True)
        game.stage_manager.is_carousel_round = MagicMock(return_value=False)
        game.stage_manager.is_augment_round = MagicMock(return_value=False)
        game.stage_manager.get_total_rounds = MagicMock(return_value=15)

        game.players = [MagicMock() for _ in range(8)]
        for i, p in enumerate(game.players):
            p.player_id = i
            p.is_alive = True
            p.hp = 100 - i * 10
            p.level = 3 + i % 3
            p.gold = 20 + i * 5
            p.units = MagicMock()
            p.units.board = {}
            p.units.get_active_synergies = MagicMock(return_value={})

        game.get_shop_for_player = MagicMock(return_value=None)
        game.shops = {}

        return game

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.config is not None
        assert encoder.state_dim > 0
        assert isinstance(encoder._champion_to_idx, dict)

    def test_state_dim_calculation(self, encoder):
        """Test state dimension calculation."""
        c = encoder.config

        # Calculate expected dimension
        unit_dim = c.champion_embed_dim + 3 + c.item_embed_dim

        # Opponent encoding dimension
        if c.encode_opponent_units:
            # Legacy: full unit encoding
            opponent_units_per_player = (c.board_size + c.bench_size) * c.opponent_unit_dim
            opponent_dim = (c.max_players - 1) * opponent_units_per_player
        else:
            # New: compact summary encoding (7 opponents * 15 dims)
            opponent_dim = (c.max_players - 1) * c.opponent_summary_dim

        expected = (
            c.board_size * unit_dim  # board
            + c.bench_size * unit_dim  # bench
            + c.shop_size * (c.champion_embed_dim + 5)  # shop
            + 10  # economy
            + c.num_traits * 2  # synergy
            + 10  # stage
            + (c.max_players - 1) * 5  # other players
            + opponent_dim  # opponent encoding
        )

        assert encoder.state_dim == expected

    def test_encode_returns_array(self, encoder, mock_player, mock_game):
        """Test encode returns numpy array."""
        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert state.shape == (encoder.state_dim,)

    def test_encode_normalized_values(self, encoder, mock_player, mock_game):
        """Test encoded values are reasonable."""
        state = encoder.encode(mock_player, mock_game)

        # Most values should be normalized (between -1 and 2)
        # Allow some slack for edge cases
        assert np.all(state >= -10)
        assert np.all(state <= 10)

    def test_encode_economy(self, encoder, mock_player, mock_game):
        """Test economy encoding."""
        mock_player.gold = 50
        mock_player.hp = 80
        mock_player.level = 5

        state = encoder.encode(mock_player, mock_game)

        # Economy values should be normalized
        assert state is not None

    def test_encode_with_board_units(self, encoder, mock_player, mock_game):
        """Test encoding with units on board."""
        # Add a unit to board
        mock_unit = MagicMock()
        mock_unit.champion = MagicMock()
        mock_unit.champion.id = "test_champion"
        mock_unit.star_level = 2
        mock_unit.items = []

        mock_player.units.board = {(0, 0): mock_unit}

        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)
        assert state.shape == (encoder.state_dim,)

    def test_encode_with_bench_units(self, encoder, mock_player, mock_game):
        """Test encoding with units on bench."""
        mock_unit = MagicMock()
        mock_unit.champion = MagicMock()
        mock_unit.champion.id = "test_champion"
        mock_unit.star_level = 1
        mock_unit.items = []

        mock_player.units.bench = [mock_unit] + [None] * 8

        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)
        assert state.shape == (encoder.state_dim,)

    def test_encode_with_shop(self, encoder, mock_player, mock_game):
        """Test encoding with shop data."""
        mock_shop = MagicMock()
        mock_slot = MagicMock()
        mock_slot.is_purchased = False
        mock_slot.champion = MagicMock()
        mock_slot.champion.id = "shop_champion"
        mock_slot.champion.cost = 3
        mock_shop.slots = [mock_slot, None, None, None, None]

        mock_game.get_shop_for_player = MagicMock(return_value=mock_shop)

        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)

    def test_encode_with_synergies(self, encoder, mock_player, mock_game):
        """Test encoding with active synergies."""
        mock_player.units.get_active_synergies = MagicMock(
            return_value={
                "trait1": {"count": 3, "is_active": True},
                "trait2": {"count": 2, "is_active": False},
            }
        )

        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)

    def test_encode_other_players(self, encoder, mock_player, mock_game):
        """Test encoding other players' info."""
        # Mark some players as dead
        mock_game.players[5].is_alive = False
        mock_game.players[6].is_alive = False

        state = encoder.encode(mock_player, mock_game, player_idx=0)

        assert isinstance(state, np.ndarray)

    def test_encode_stage_info(self, encoder, mock_player, mock_game):
        """Test encoding stage information."""
        mock_game.stage_manager.stage = 4
        mock_game.stage_manager.round = 5
        mock_game.stage_manager.is_pvp_round = MagicMock(return_value=True)

        state = encoder.encode(mock_player, mock_game)

        assert isinstance(state, np.ndarray)

    def test_get_champion_embedding(self, encoder):
        """Test champion embedding generation."""
        embed = encoder._get_champion_embedding(0)

        assert isinstance(embed, np.ndarray)
        assert embed.shape == (encoder.config.champion_embed_dim,)
        assert np.sum(embed) == 1.0  # Should be one-hot like

    def test_encode_items(self, encoder):
        """Test item encoding."""
        mock_items = [
            MagicMock(id="item1"),
            MagicMock(id="item2"),
        ]

        embed = encoder._encode_items(mock_items)

        assert isinstance(embed, np.ndarray)
        assert embed.shape == (encoder.config.item_embed_dim,)

    def test_encode_no_items(self, encoder):
        """Test encoding with no items."""
        embed = encoder._encode_items([])

        assert isinstance(embed, np.ndarray)
        assert embed.shape == (encoder.config.item_embed_dim,)
        assert np.sum(embed) == 0.0

    def test_custom_config_encoder(self):
        """Test encoder with custom config."""
        config = EncoderConfig(
            champion_embed_dim=16,
            board_size=20,
        )
        encoder = StateEncoder(config)

        assert encoder.config.champion_embed_dim == 16
        assert encoder.config.board_size == 20

    def test_encode_consistency(self, encoder, mock_player, mock_game):
        """Test encode produces consistent results."""
        state1 = encoder.encode(mock_player, mock_game)
        state2 = encoder.encode(mock_player, mock_game)

        np.testing.assert_array_equal(state1, state2)

    def test_encode_different_players(self, encoder, mock_game):
        """Test encoding different players produces different results."""
        player1 = MagicMock()
        player1.gold = 50
        player1.hp = 100
        player1.level = 5
        player1.xp = 0
        player1.streak = 0
        player1.player_id = 0
        player1.units = MagicMock()
        player1.units.board = {}
        player1.units.bench = [None] * 9
        player1.units.get_active_synergies = MagicMock(return_value={})

        player2 = MagicMock()
        player2.gold = 10
        player2.hp = 50
        player2.level = 3
        player2.xp = 0
        player2.streak = 0
        player2.player_id = 1
        player2.units = MagicMock()
        player2.units.board = {}
        player2.units.bench = [None] * 9
        player2.units.get_active_synergies = MagicMock(return_value={})

        state1 = encoder.encode(player1, mock_game, 0)
        state2 = encoder.encode(player2, mock_game, 1)

        # States should be different
        assert not np.array_equal(state1, state2)
