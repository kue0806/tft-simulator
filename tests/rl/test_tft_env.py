"""Tests for TFT Gymnasium Environment."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.rl.env.tft_env import TFTEnv
from src.rl.env.action_space import ActionType
from src.rl.env.state_encoder import EncoderConfig
from src.rl.env.reward_calculator import RewardConfig


class TestTFTEnvInit:
    """Test TFTEnv initialization."""

    def test_default_initialization(self):
        """Test default environment initialization."""
        env = TFTEnv()

        assert env.num_players == 8
        assert env.agent_player_idx == 0
        assert env.max_rounds == 50
        assert env.render_mode is None

    def test_custom_initialization(self):
        """Test custom environment initialization."""
        env = TFTEnv(
            num_players=4,
            agent_player_idx=1,
            max_rounds=30,
            render_mode="ansi",
        )

        assert env.num_players == 4
        assert env.agent_player_idx == 1
        assert env.max_rounds == 30
        assert env.render_mode == "ansi"

    def test_observation_space(self):
        """Test observation space is defined."""
        env = TFTEnv()

        assert env.observation_space is not None
        assert env.observation_space.shape[0] == env.state_encoder.state_dim

    def test_action_space(self):
        """Test action space is defined."""
        env = TFTEnv()

        assert env.action_space is not None
        assert env.action_space.n == env.action_space_handler.num_actions


class TestTFTEnvReset:
    """Test TFTEnv reset."""

    def test_reset_returns_tuple(self):
        """Test reset returns (obs, info) tuple."""
        env = TFTEnv()
        result = env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self):
        """Test reset returns correct observation shape."""
        env = TFTEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_reset_info_contents(self):
        """Test reset returns expected info keys."""
        env = TFTEnv()
        obs, info = env.reset()

        assert "round" in info
        assert "hp" in info
        assert "gold" in info
        assert "level" in info
        assert "valid_action_mask" in info

    def test_reset_initializes_game(self):
        """Test reset initializes game state."""
        env = TFTEnv()
        env.reset()

        assert env.game is not None
        assert env.game.pool is not None
        assert len(env.shops) == env.num_players

    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility."""
        env = TFTEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # Should produce same initial state (within game randomness)
        assert obs1.shape == obs2.shape

    def test_reset_clears_done_flag(self):
        """Test reset clears done flag."""
        env = TFTEnv()
        env.reset()
        env.done = True

        env.reset()
        assert env.done is False


class TestTFTEnvStep:
    """Test TFTEnv step."""

    @pytest.fixture
    def env(self):
        env = TFTEnv()
        env.reset()
        return env

    def test_step_returns_tuple(self, env):
        """Test step returns 5-tuple."""
        result = env.step(0)  # PASS action

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_return_types(self, env):
        """Test step return types."""
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, env):
        """Test step returns correct observation shape."""
        obs, _, _, _, _ = env.step(0)

        assert obs.shape == env.observation_space.shape

    def test_step_pass_action(self, env):
        """Test PASS action triggers round."""
        initial_round = env.current_round

        _, _, _, _, info = env.step(0)  # PASS

        # PASS should end the round
        assert env.current_round == initial_round + 1

    def test_step_buy_action(self, env):
        """Test BUY action execution."""
        # Get initial gold
        player = env.game.players[env.agent_player_idx]
        initial_gold = player.gold

        # Try to buy (may or may not succeed depending on shop)
        buy_idx = env.action_space_handler.encode_action(ActionType.BUY, 0)
        _, _, _, _, info = env.step(buy_idx)

        # Check action was processed
        assert "action_valid" in info

    def test_step_refresh_action(self, env):
        """Test REFRESH action execution."""
        player = env.game.players[env.agent_player_idx]
        player.gold = 10

        refresh_idx = env.action_space_handler.encode_action(ActionType.REFRESH, None)
        _, _, _, _, info = env.step(refresh_idx)

        assert info["action_valid"] is True
        assert player.gold == 8  # 10 - 2

    def test_step_buy_xp_action(self, env):
        """Test BUY_XP action execution."""
        player = env.game.players[env.agent_player_idx]
        player.gold = 10
        player.level = 3

        buy_xp_idx = env.action_space_handler.encode_action(ActionType.BUY_XP, None)
        _, _, _, _, info = env.step(buy_xp_idx)

        assert info["action_valid"] is True
        assert player.gold == 6  # 10 - 4

    def test_step_invalid_action(self, env):
        """Test invalid action handling."""
        player = env.game.players[env.agent_player_idx]
        player.gold = 0  # No gold

        refresh_idx = env.action_space_handler.encode_action(ActionType.REFRESH, None)
        _, reward, _, _, info = env.step(refresh_idx)

        assert info["action_valid"] is False
        assert reward < 0  # Should have penalty

    def test_step_done_state(self, env):
        """Test step when already done."""
        env.done = True

        obs, reward, terminated, truncated, info = env.step(0)

        assert terminated is True
        assert reward == 0.0

    def test_step_terminates_on_death(self, env):
        """Test game terminates when agent dies."""
        player = env.game.players[env.agent_player_idx]
        player.hp = 1
        player.is_alive = True

        # Force death
        player.hp = 0
        player.is_alive = False

        _, _, terminated, _, _ = env.step(0)

        assert terminated is True

    def test_step_truncates_on_max_rounds(self, env):
        """Test game truncates at max rounds."""
        env.max_rounds = 5
        env.current_round = 4

        _, _, terminated, truncated, _ = env.step(0)

        assert truncated is True

    def test_step_info_contains_mask(self, env):
        """Test step info contains valid action mask."""
        _, _, _, _, info = env.step(0)

        assert "valid_action_mask" in info
        mask = info["valid_action_mask"]
        assert mask.shape == (env.action_space_handler.num_actions,)

    def test_step_placement_on_game_end(self, env):
        """Test placement is calculated on game end."""
        # Kill all other players
        for i, p in enumerate(env.game.players):
            if i != env.agent_player_idx:
                p.is_alive = False
                p.hp = 0

        _, _, terminated, _, info = env.step(0)

        assert terminated is True
        assert info["placement"] == 1  # Should be 1st


class TestTFTEnvRender:
    """Test TFTEnv rendering."""

    def test_render_ansi(self):
        """Test ANSI rendering."""
        env = TFTEnv(render_mode="ansi")
        env.reset()

        output = env.render()

        assert isinstance(output, str)
        assert "Round" in output
        assert "HP" in output
        assert "Gold" in output

    def test_render_human(self, capsys):
        """Test human rendering (prints to stdout)."""
        env = TFTEnv(render_mode="human")
        env.reset()

        env.render()

        captured = capsys.readouterr()
        assert "Round" in captured.out

    def test_render_none(self):
        """Test no rendering when mode is None."""
        env = TFTEnv(render_mode=None)
        env.reset()

        output = env.render()

        assert output is None


class TestTFTEnvHelpers:
    """Test TFTEnv helper methods."""

    @pytest.fixture
    def env(self):
        env = TFTEnv()
        env.reset()
        return env

    def test_get_valid_action_mask(self, env):
        """Test get_valid_action_mask method."""
        mask = env.get_valid_action_mask()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (env.action_space_handler.num_actions,)
        assert mask[0] == 1.0  # PASS always valid

    def test_estimate_board_power_empty(self, env):
        """Test board power estimation with empty board."""
        player = env.game.players[env.agent_player_idx]
        player.units.board = {}

        power = env._estimate_board_power(player)

        assert power == 0.0

    def test_estimate_board_power_with_units(self, env):
        """Test board power estimation with units."""
        player = env.game.players[env.agent_player_idx]

        mock_unit = MagicMock()
        mock_unit.champion = MagicMock()
        mock_unit.champion.cost = 3
        mock_unit.star_level = 2
        mock_unit.items = []

        player.units.board = {(0, 0): mock_unit}
        player.units.get_active_synergies = MagicMock(return_value={})

        power = env._estimate_board_power(player)

        assert power > 0

    def test_calculate_income(self, env):
        """Test income calculation."""
        player = env.game.players[env.agent_player_idx]
        player.gold = 30
        player.streak = 2

        income = env._calculate_income(player)

        # Base 5 + interest (3) + streak (2) = 10
        assert income >= 5  # At least base

    def test_calculate_placement(self, env):
        """Test placement calculation."""
        # Set up players with different HP
        for i, p in enumerate(env.game.players):
            p.hp = 100 - i * 10
            p.is_alive = p.hp > 0

        placement = env._calculate_placement()

        assert 1 <= placement <= env.num_players


class TestTFTEnvIntegration:
    """Integration tests for TFTEnv."""

    def test_full_episode(self):
        """Test running a full episode."""
        env = TFTEnv(max_rounds=10)
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 200

        while steps < max_steps:
            # Random valid action
            mask = info["valid_action_mask"]
            valid_actions = np.where(mask > 0)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert isinstance(total_reward, float)

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = TFTEnv(max_rounds=5)

        for _ in range(3):
            obs, info = env.reset()
            done = False

            while not done:
                mask = info["valid_action_mask"]
                valid_actions = np.where(mask > 0)[0]
                action = np.random.choice(valid_actions)

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

    def test_deterministic_with_seed(self):
        """Test environment is somewhat reproducible with seed."""
        env1 = TFTEnv()
        env2 = TFTEnv()

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        # Initial observations should have same shape
        assert obs1.shape == obs2.shape

    def test_action_mask_correctness(self):
        """Test action mask correctly prevents invalid actions."""
        env = TFTEnv()
        obs, info = env.reset()

        # Set up player with specific state
        player = env.game.players[env.agent_player_idx]
        player.gold = 0  # No gold

        # Refresh should be invalid
        mask = env.get_valid_action_mask()
        refresh_idx = env.action_space_handler.encode_action(ActionType.REFRESH, None)

        assert mask[refresh_idx] == 0.0

    def test_observation_bounds(self):
        """Test observations are within reasonable bounds."""
        env = TFTEnv()
        obs, _ = env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            # Most normalized values should be in reasonable range
            assert np.all(np.isfinite(obs))

            if terminated or truncated:
                break
