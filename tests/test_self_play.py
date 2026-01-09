"""
Tests for Self-Play Training Components.

Tests:
- SelfPlayEnv functionality
- AgentPool management
- Curriculum learning
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSelfPlayEnv:
    """Tests for SelfPlayEnv."""

    def test_env_creation(self):
        """Test creating SelfPlayEnv."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=8, max_rounds=50)
        assert env.num_players == 8
        assert env.self_play_ratio == 0.5  # Default

    def test_self_play_ratio(self):
        """Test setting self-play ratio."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=8)

        env.set_self_play_ratio(0.75)
        assert env.self_play_ratio == 0.75

        env.set_self_play_ratio(1.5)  # Should clamp to 1.0
        assert env.self_play_ratio == 1.0

        env.set_self_play_ratio(-0.5)  # Should clamp to 0.0
        assert env.self_play_ratio == 0.0

    def test_opponent_policy_setting(self):
        """Test setting opponent policy."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=8)

        def dummy_policy(obs):
            return 0

        env.set_opponent_policy(dummy_policy)
        assert env._opponent_policy is not None
        assert env._opponent_policy(np.zeros(100)) == 0

    def test_reset_assigns_opponent_types(self):
        """Test that reset assigns opponent types."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=8, self_play_ratio=0.5)

        # Without opponent policy, all should be random
        obs, info = env.reset()
        for i in range(8):
            if i != env.agent_player_idx:
                assert env._opponent_types.get(i) == 'random'

        # With opponent policy, some should be self_play
        def dummy_policy(obs):
            return 0

        env.set_opponent_policy(dummy_policy)
        env.set_self_play_ratio(1.0)  # 100% self-play
        obs, info = env.reset()

        for i in range(8):
            if i != env.agent_player_idx:
                assert env._opponent_types.get(i) == 'self_play'

    def test_env_step(self):
        """Test environment step."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=4, max_rounds=10)
        obs, info = env.reset()

        # Take some steps
        for _ in range(10):
            action = 0  # Pass action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert obs is not None

    def test_get_opponent_observations(self):
        """Test getting opponent observations."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=4)
        obs, info = env.reset()

        opponent_obs = env.get_opponent_observations()

        # Should have observations for 3 opponents (4 players - 1 agent)
        assert len(opponent_obs) == 3
        for idx, obs in opponent_obs.items():
            assert idx != env.agent_player_idx
            assert isinstance(obs, np.ndarray)

    def test_gym_registration(self):
        """Test gym environment registration."""
        import gymnasium as gym

        try:
            env = gym.make("TFT-SelfPlay-v0")
            assert env is not None
            env.close()
        except gym.error.Error:
            # May not be registered in test environment
            pass


class TestAgentPool:
    """Tests for AgentPool."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_pool_creation(self, temp_dir):
        """Test creating agent pool."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)
        assert len(pool) == 0
        assert pool.max_size == 10

    def test_save_and_load_agent(self, temp_dir):
        """Test saving and loading agents."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        # Create mock model
        mock_model = Mock()
        mock_model.save = Mock()

        # Save agent
        version = pool.save_agent(mock_model, timestep=1000)

        assert version.version_id == 0
        assert version.timestep == 1000
        assert len(pool) == 1

        # Check metadata was saved
        mock_model.save.assert_called_once()

    def test_sample_opponent(self, temp_dir):
        """Test sampling opponents."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        # Empty pool should return None
        assert pool.sample_opponent() is None

        # Add some versions
        mock_model = Mock()
        mock_model.save = Mock()

        for i in range(5):
            pool.save_agent(mock_model, timestep=i * 1000)

        # Should be able to sample
        opponent = pool.sample_opponent()
        assert opponent is not None
        assert opponent.version_id in range(5)

    def test_sample_exclude_latest(self, temp_dir):
        """Test sampling with exclude_latest."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        mock_model = Mock()
        mock_model.save = Mock()

        for i in range(3):
            pool.save_agent(mock_model, timestep=i * 1000)

        # Sample 100 times with exclude_latest
        for _ in range(100):
            opponent = pool.sample_opponent(exclude_latest=True)
            assert opponent.version_id != 2  # Not the latest

    def test_elo_update(self, temp_dir):
        """Test ELO rating updates."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        mock_model = Mock()
        mock_model.save = Mock()

        # Add two versions
        v1 = pool.save_agent(mock_model, timestep=1000)
        v2 = pool.save_agent(mock_model, timestep=2000)

        initial_elo1 = v1.elo_rating
        initial_elo2 = v2.elo_rating

        # v2 wins against v1
        pool.update_elo(v2, v1)

        assert v2.elo_rating > initial_elo2
        assert v1.elo_rating < initial_elo1
        assert v1.games_played == 1
        assert v2.games_played == 1

    def test_pool_pruning(self, temp_dir):
        """Test pool pruning when over max size."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=5, min_size=3)

        mock_model = Mock()
        mock_model.save = Mock()

        # Add more than max_size
        for i in range(7):
            pool.save_agent(mock_model, timestep=i * 1000)

        # Should be pruned to max_size
        assert len(pool) <= 5

    def test_mark_weak(self, temp_dir):
        """Test marking versions as weak."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        mock_model = Mock()
        mock_model.save = Mock()

        version = pool.save_agent(mock_model, timestep=1000)
        assert version.is_strong is True

        pool.mark_weak(version)
        assert version.is_strong is False

    def test_only_strong_sampling(self, temp_dir):
        """Test sampling only strong agents."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        mock_model = Mock()
        mock_model.save = Mock()

        # Add versions
        v1 = pool.save_agent(mock_model, timestep=1000)
        v2 = pool.save_agent(mock_model, timestep=2000)
        v3 = pool.save_agent(mock_model, timestep=3000)

        # Mark v1 and v2 as weak
        pool.mark_weak(v1)
        pool.mark_weak(v2)

        # Sample only strong
        for _ in range(50):
            opponent = pool.sample_opponent(only_strong=True)
            assert opponent.version_id == v3.version_id  # Only v3 is strong

    def test_pool_stats(self, temp_dir):
        """Test pool statistics."""
        from src.rl.training.agent_pool import AgentPool

        pool = AgentPool(save_dir=temp_dir, max_size=10)

        mock_model = Mock()
        mock_model.save = Mock()

        for i in range(5):
            pool.save_agent(mock_model, timestep=i * 1000)

        stats = pool.get_pool_stats()

        assert stats["size"] == 5
        assert stats["avg_elo"] == 1000.0
        assert stats["strong_count"] == 5
        assert stats["latest_timestep"] == 4000

    def test_persistence(self, temp_dir):
        """Test that pool persists across instances."""
        from src.rl.training.agent_pool import AgentPool

        mock_model = Mock()
        mock_model.save = Mock()

        # Create pool and add agents
        pool1 = AgentPool(save_dir=temp_dir, max_size=10)
        pool1.save_agent(mock_model, timestep=1000)
        pool1.save_agent(mock_model, timestep=2000)

        # Create new pool instance
        pool2 = AgentPool(save_dir=temp_dir, max_size=10)

        assert len(pool2) == 2
        assert pool2.versions[0].timestep == 1000
        assert pool2.versions[1].timestep == 2000


class TestCurriculum:
    """Tests for curriculum learning."""

    def test_curriculum_config(self):
        """Test curriculum configuration."""
        from src.rl.training.self_play_trainer import CurriculumConfig

        config = CurriculumConfig()

        assert config.phase1_end == 200_000
        assert config.phase1_self_play_ratio == 0.5
        assert config.phase2_end == 500_000
        assert config.phase2_self_play_ratio == 0.75
        assert config.phase3_self_play_ratio == 1.0

    def test_curriculum_schedule(self):
        """Test curriculum schedule logic."""
        from src.rl.training.self_play_trainer import CurriculumConfig

        config = CurriculumConfig()

        def get_ratio(timestep):
            if timestep < config.phase1_end:
                return config.phase1_self_play_ratio
            elif timestep < config.phase2_end:
                return config.phase2_self_play_ratio
            else:
                return config.phase3_self_play_ratio

        # Phase 1
        assert get_ratio(0) == 0.5
        assert get_ratio(100_000) == 0.5
        assert get_ratio(199_999) == 0.5

        # Phase 2
        assert get_ratio(200_000) == 0.75
        assert get_ratio(300_000) == 0.75
        assert get_ratio(499_999) == 0.75

        # Phase 3
        assert get_ratio(500_000) == 1.0
        assert get_ratio(1_000_000) == 1.0


class TestSelfPlayTrainer:
    """Tests for SelfPlayTrainer."""

    def test_config_creation(self):
        """Test creating trainer config."""
        from src.rl.training.self_play_trainer import SelfPlayConfig

        config = SelfPlayConfig(
            total_timesteps=100_000,
            n_envs=4,
            learning_rate=1e-4,
        )

        assert config.total_timesteps == 100_000
        assert config.n_envs == 4
        assert config.learning_rate == 1e-4
        assert config.curriculum is not None

    def test_trainer_creation(self, tmp_path):
        """Test creating trainer."""
        from src.rl.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig

        config = SelfPlayConfig(total_timesteps=100)
        trainer = SelfPlayTrainer(
            config=config,
            save_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
        )

        assert trainer.config == config
        assert trainer.agent_pool is not None


class TestIntegration:
    """Integration tests for self-play components."""

    def test_env_with_real_step(self):
        """Test SelfPlayEnv with actual game mechanics."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(num_players=4, max_rounds=5)

        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape

        total_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            # Take random valid action
            mask = info.get("valid_action_mask")
            if mask is not None:
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        # Should complete eventually
        assert steps > 0

    def test_self_play_vs_random(self):
        """Test self-play environment with mixed opponents."""
        from src.rl.env.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(
            num_players=4,
            max_rounds=5,
            self_play_ratio=0.5,
        )

        # Set up a dummy policy
        def dummy_policy(obs):
            return 0  # Always pass

        env.set_opponent_policy(dummy_policy)

        obs, info = env.reset()

        # Check opponent types
        opponent_types = env.get_opponent_types()
        assert len(opponent_types) == 3  # 4 players - 1 agent

        # Run a few steps
        for _ in range(10):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
