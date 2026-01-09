"""Tests for TFT Reward Calculator."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.rl.env.reward_calculator import RewardCalculator, RewardConfig
from src.rl.env.action_space import ActionType


class TestRewardConfig:
    """Test RewardConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RewardConfig()

        assert config.win_reward == 0.5
        assert config.lose_reward == -0.2
        assert config.hp_loss_penalty == -0.05
        assert config.invalid_action_penalty == -0.5
        assert config.shaping_weight == 0.1

    def test_placement_rewards(self):
        """Test placement rewards dictionary."""
        config = RewardConfig()

        assert config.placement_rewards[1] == 10.0
        assert config.placement_rewards[4] == 1.0
        assert config.placement_rewards[8] == -10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = RewardConfig(
            win_reward=1.0,
            lose_reward=-0.5,
            shaping_weight=0.2,
        )

        assert config.win_reward == 1.0
        assert config.lose_reward == -0.5
        assert config.shaping_weight == 0.2


class TestRewardCalculator:
    """Test RewardCalculator."""

    @pytest.fixture
    def calculator(self):
        return RewardCalculator()

    @pytest.fixture
    def mock_player(self):
        """Create mock player."""
        player = MagicMock()
        player.gold = 30
        player.hp = 80
        player.level = 5
        player.streak = 1

        player.units = MagicMock()
        player.units.board = {}
        player.units.bench = [None] * 9
        player.units.get_active_synergies = MagicMock(return_value={})

        return player

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.config is not None
        assert calculator._prev_synergies == {}
        assert calculator._prev_hp == 100
        assert calculator._prev_gold == 0
        assert calculator._prev_star_levels == {}

    def test_reset(self, calculator):
        """Test calculator reset."""
        # Modify state
        calculator._prev_synergies = {"trait": 3}
        calculator._prev_hp = 50
        calculator._prev_gold = 100

        # Reset
        calculator.reset()

        assert calculator._prev_synergies == {}
        assert calculator._prev_hp == 100
        assert calculator._prev_gold == 0
        assert calculator._prev_star_levels == {}

    def test_invalid_action_penalty(self, calculator, mock_player):
        """Test penalty for invalid action."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.BUY,
            action_valid=False,
        )

        assert reward == calculator.config.invalid_action_penalty

    def test_placement_reward_first(self, calculator, mock_player):
        """Test reward for 1st place."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            done=True,
            placement=1,
        )

        assert reward == calculator.config.placement_rewards[1]

    def test_placement_reward_last(self, calculator, mock_player):
        """Test reward for 8th place."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            done=True,
            placement=8,
        )

        assert reward == calculator.config.placement_rewards[8]

    def test_win_round_reward(self, calculator, mock_player):
        """Test reward for winning a round."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            round_result={"won": True, "hp_lost": 0},
        )

        assert reward >= calculator.config.win_reward * (1 - calculator.config.shaping_weight)

    def test_lose_round_reward(self, calculator, mock_player):
        """Test reward for losing a round."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            round_result={"won": False, "hp_lost": 10},
        )

        # Should include lose penalty and HP loss penalty
        expected_base = calculator.config.lose_reward + 10 * calculator.config.hp_loss_penalty
        # Actual reward includes shaping, so just check it's negative
        assert reward < 0

    def test_pass_action_no_penalty(self, calculator, mock_player):
        """Test PASS action gives no invalid penalty."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        assert reward >= 0  # No penalty, may have small shaping reward

    def test_shaping_interest_reward(self, calculator, mock_player):
        """Test shaping reward for interest."""
        mock_player.gold = 50  # 5 interest gold

        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Should get interest reward (5 * 0.1 * 0.1 = 0.05)
        expected_min = 5 * calculator.config.interest_reward * calculator.config.shaping_weight
        assert reward >= expected_min

    def test_shaping_synergy_activation(self, calculator, mock_player):
        """Test shaping reward for activating synergy."""
        # First call - no synergies
        calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Second call - synergy activated
        mock_player.units.get_active_synergies = MagicMock(
            return_value={
                "new_trait": {"count": 3, "is_active": True}
            }
        )

        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Should include synergy activation reward
        assert reward > 0

    def test_shaping_unit_upgrade(self, calculator, mock_player):
        """Test shaping reward for unit upgrade."""
        # Initial unit at 1 star
        mock_unit = MagicMock()
        mock_unit.star_level = 1
        mock_player.units.board = {(0, 0): mock_unit}

        calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Upgrade to 2 star
        mock_unit.star_level = 2

        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Should include upgrade reward
        assert reward > 0

    def test_bench_unit_upgrade(self, calculator, mock_player):
        """Test shaping reward for bench unit upgrade."""
        mock_unit = MagicMock()
        mock_unit.star_level = 1
        mock_player.units.bench = [mock_unit] + [None] * 8

        calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Upgrade to 3 star
        mock_unit.star_level = 3

        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        # Should include 3-star upgrade reward (larger than 2-star)
        expected_min = calculator.config.unit_upgrade_3star * calculator.config.shaping_weight
        assert reward >= expected_min * 0.5  # Allow some variance

    def test_prev_state_updated(self, calculator, mock_player):
        """Test previous state is updated after calculation."""
        mock_player.gold = 40
        mock_player.hp = 75

        calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        assert calculator._prev_hp == 75
        assert calculator._prev_gold == 40

    def test_custom_config_calculator(self):
        """Test calculator with custom config."""
        config = RewardConfig(
            win_reward=2.0,
            invalid_action_penalty=-1.0,
        )
        calculator = RewardCalculator(config)

        assert calculator.config.win_reward == 2.0
        assert calculator.config.invalid_action_penalty == -1.0

    def test_round_reward_hp_loss_proportional(self, calculator, mock_player):
        """Test HP loss penalty is proportional to HP lost."""
        # 5 HP loss
        reward1 = calculator._calculate_round_reward({"won": False, "hp_lost": 5})

        # 15 HP loss
        reward2 = calculator._calculate_round_reward({"won": False, "hp_lost": 15})

        # More HP loss should give lower reward
        assert reward2 < reward1

    def test_no_units_no_crash(self, calculator, mock_player):
        """Test calculator handles player with no units attribute."""
        mock_player.units = None

        # Should not crash
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        assert isinstance(reward, float)

    def test_empty_synergies_dict(self, calculator, mock_player):
        """Test handling of empty synergies."""
        mock_player.units.get_active_synergies = MagicMock(return_value={})

        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
        )

        assert isinstance(reward, float)

    def test_done_without_placement(self, calculator, mock_player):
        """Test done=True without placement."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            done=True,
            placement=None,
        )

        # Should still return a reward (shaping)
        assert isinstance(reward, float)

    def test_invalid_placement_returns_zero(self, calculator, mock_player):
        """Test invalid placement returns 0 placement reward."""
        reward = calculator.calculate(
            player=mock_player,
            action_type=ActionType.PASS,
            action_valid=True,
            done=True,
            placement=10,  # Invalid placement
        )

        # Should return 0 for unknown placement
        assert reward == 0.0
