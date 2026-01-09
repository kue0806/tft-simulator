"""Tests for Unlock Manager system."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.unlock_manager import (
    UnlockManager,
    PlayerUnlockState,
    RerollEvaluator,
    SoulsEvaluator,
    TraitUnitsEvaluator,
    UnitStarEvaluator,
    WinStreakWithUnitEvaluator,
    LossStreakWithUnitEvaluator,
    reset_unlock_manager,
)


class TestPlayerUnlockState:
    """Tests for PlayerUnlockState dataclass."""

    def test_initial_state(self):
        """Test initial state is empty."""
        state = PlayerUnlockState()

        assert len(state.unlocked_champions) == 0
        assert state.souls_collected == 0
        assert state.sunshards_collected == 0
        assert state.reroll_count == 0
        assert state.current_stage == "1-1"

    def test_track_souls(self):
        """Test soul tracking."""
        state = PlayerUnlockState()
        state.souls_collected = 50
        assert state.souls_collected == 50

    def test_track_streak(self):
        """Test streak tracking."""
        state = PlayerUnlockState()
        state.streak_with_unit["azir"] = (3, 0)  # 3 win streak

        win, loss = state.streak_with_unit["azir"]
        assert win == 3
        assert loss == 0


class TestRerollEvaluator:
    """Tests for RerollEvaluator."""

    def test_reroll_before_stage(self):
        """Test reroll condition before stage."""
        evaluator = RerollEvaluator()
        player = MagicMock()
        state = PlayerUnlockState()
        state.reroll_count = 4
        state.current_stage = "2-3"

        params = {"reroll_count": 4, "before_stage": "2-4"}
        result = evaluator.evaluate(player, state, params)

        assert result is True

    def test_reroll_not_enough(self):
        """Test reroll condition not met."""
        evaluator = RerollEvaluator()
        player = MagicMock()
        state = PlayerUnlockState()
        state.reroll_count = 2
        state.current_stage = "2-3"

        params = {"reroll_count": 4, "before_stage": "2-4"}
        result = evaluator.evaluate(player, state, params)

        assert result is False

    def test_reroll_after_stage(self):
        """Test reroll fails after target stage."""
        evaluator = RerollEvaluator()
        player = MagicMock()
        state = PlayerUnlockState()
        state.reroll_count = 10
        state.current_stage = "3-1"

        params = {"reroll_count": 4, "before_stage": "2-4"}
        result = evaluator.evaluate(player, state, params)

        assert result is False


class TestSoulsEvaluator:
    """Tests for SoulsEvaluator."""

    def test_souls_met(self):
        """Test souls condition met."""
        evaluator = SoulsEvaluator()
        player = MagicMock()
        state = PlayerUnlockState()
        state.souls_collected = 25

        params = {"soul_count": 20}
        result = evaluator.evaluate(player, state, params)

        assert result is True

    def test_souls_not_met(self):
        """Test souls condition not met."""
        evaluator = SoulsEvaluator()
        player = MagicMock()
        state = PlayerUnlockState()
        state.souls_collected = 15

        params = {"soul_count": 20}
        result = evaluator.evaluate(player, state, params)

        assert result is False


class TestTraitUnitsEvaluator:
    """Tests for TraitUnitsEvaluator."""

    def test_trait_units_met(self):
        """Test trait units condition met."""
        evaluator = TraitUnitsEvaluator()

        # Create mock player with units on board
        player = MagicMock()
        unit1 = MagicMock()
        unit1.champion.id = "garen"
        unit1.champion.traits = ["demacia", "juggernaut"]

        unit2 = MagicMock()
        unit2.champion.id = "jarvan_iv"
        unit2.champion.traits = ["demacia", "warden"]

        unit3 = MagicMock()
        unit3.champion.id = "lux"
        unit3.champion.traits = ["demacia", "invoker"]

        player.units.board = {
            (0, 0): unit1,
            (1, 0): unit2,
            (2, 0): unit3,
        }
        player.level = 5

        state = PlayerUnlockState()
        params = {"trait_id": "demacia", "unit_count": 3}
        result = evaluator.evaluate(player, state, params)

        assert result is True

    def test_trait_units_level_requirement(self):
        """Test trait units with level requirement not met."""
        evaluator = TraitUnitsEvaluator()

        player = MagicMock()
        player.units.board = {}
        player.level = 5

        state = PlayerUnlockState()
        params = {"trait_id": "demacia", "unit_count": 3, "min_level": 7}
        result = evaluator.evaluate(player, state, params)

        assert result is False


class TestUnitStarEvaluator:
    """Tests for UnitStarEvaluator."""

    def test_unit_star_met(self):
        """Test unit star condition met."""
        evaluator = UnitStarEvaluator()

        player = MagicMock()
        unit = MagicMock()
        unit.champion.id = "yasuo"
        unit.star_level = 3

        player.units.board = {(0, 0): unit}

        state = PlayerUnlockState()
        params = {"champion_id": "yasuo", "star_level": 3}
        result = evaluator.evaluate(player, state, params)

        assert result is True

    def test_unit_star_not_met(self):
        """Test unit star condition not met."""
        evaluator = UnitStarEvaluator()

        player = MagicMock()
        unit = MagicMock()
        unit.champion.id = "yasuo"
        unit.star_level = 2

        player.units.board = {(0, 0): unit}

        state = PlayerUnlockState()
        params = {"champion_id": "yasuo", "star_level": 3}
        result = evaluator.evaluate(player, state, params)

        assert result is False


class TestWinStreakWithUnitEvaluator:
    """Tests for WinStreakWithUnitEvaluator."""

    def test_win_streak_met(self):
        """Test win streak condition met."""
        evaluator = WinStreakWithUnitEvaluator()

        player = MagicMock()
        state = PlayerUnlockState()
        state.streak_with_unit["azir"] = (4, 0)  # 4 wins

        params = {"champion_id": "azir", "streak_counts": [2, 4, 5]}
        result = evaluator.evaluate(player, state, params)

        assert result is True

    def test_win_streak_not_met(self):
        """Test win streak not in required counts."""
        evaluator = WinStreakWithUnitEvaluator()

        player = MagicMock()
        state = PlayerUnlockState()
        state.streak_with_unit["azir"] = (3, 0)  # 3 wins, not in [2, 4, 5]

        params = {"champion_id": "azir", "streak_counts": [2, 4, 5]}
        result = evaluator.evaluate(player, state, params)

        assert result is False


class TestLossStreakWithUnitEvaluator:
    """Tests for LossStreakWithUnitEvaluator."""

    def test_loss_streak_met(self):
        """Test loss streak condition met."""
        evaluator = LossStreakWithUnitEvaluator()

        player = MagicMock()
        state = PlayerUnlockState()
        state.streak_with_unit["azir"] = (0, 5)  # 5 losses

        params = {"champion_id": "azir", "streak_counts": [2, 4, 5]}
        result = evaluator.evaluate(player, state, params)

        assert result is True


class TestUnlockManager:
    """Tests for UnlockManager."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock ChampionPool."""
        pool = MagicMock()
        pool.add_unlockable_to_pool = MagicMock()
        pool.get_champion = MagicMock(return_value=None)
        return pool

    @pytest.fixture
    def manager(self, mock_pool):
        """Create UnlockManager with mock pool."""
        reset_unlock_manager()
        # Patch get_unlockable_champions to return empty list
        with patch("src.core.unlock_manager.get_unlockable_champions") as mock:
            mock.return_value = []
            manager = UnlockManager(mock_pool)
        return manager

    def test_get_player_state(self, manager):
        """Test getting player state creates new state."""
        state = manager.get_player_state(0)

        assert isinstance(state, PlayerUnlockState)
        assert len(state.unlocked_champions) == 0

    def test_add_souls(self, manager):
        """Test adding souls."""
        manager.add_souls(0, 10)
        manager.add_souls(0, 15)

        state = manager.get_player_state(0)
        assert state.souls_collected == 25

    def test_add_sunshards(self, manager):
        """Test adding sunshards."""
        manager.add_sunshards(0, 100)

        state = manager.get_player_state(0)
        assert state.sunshards_collected == 100

    def test_spend_serpents(self, manager):
        """Test spending serpents."""
        manager.spend_serpents(0, 50)
        manager.spend_serpents(0, 100)

        state = manager.get_player_state(0)
        assert state.serpents_spent == 150

    def test_record_reroll(self, manager):
        """Test recording rerolls."""
        manager.record_reroll(0)
        manager.record_reroll(0)

        state = manager.get_player_state(0)
        assert state.reroll_count == 2

    def test_record_combat_result_win(self, manager):
        """Test recording win result."""
        manager.record_combat_result(0, won=True, units_on_board=["azir", "garen"])

        state = manager.get_player_state(0)
        assert state.streak_with_unit["azir"] == (1, 0)
        assert state.streak_with_unit["garen"] == (1, 0)

    def test_record_combat_result_loss(self, manager):
        """Test recording loss result."""
        manager.record_combat_result(0, won=False, units_on_board=["azir"])

        state = manager.get_player_state(0)
        assert state.streak_with_unit["azir"] == (0, 1)

    def test_record_alternating_results(self, manager):
        """Test alternating win/loss tracking."""
        manager.record_combat_result(0, won=True, units_on_board=["azir"])
        manager.record_combat_result(0, won=False, units_on_board=["azir"])
        manager.record_combat_result(0, won=True, units_on_board=["azir"])

        state = manager.get_player_state(0)
        assert state.alternate_count["azir"] == 2

    def test_record_unit_sold(self, manager):
        """Test recording unit sold."""
        manager.record_unit_sold(0, "jarvan_iv", 2)
        manager.record_unit_sold(0, "garen", 2)
        manager.record_unit_sold(0, "lux", 2)

        state = manager.get_player_state(0)
        assert state.sold_units["jarvan_iv:2"] == 1
        assert state.sold_units["garen:2"] == 1
        assert state.sold_units["lux:2"] == 1

    def test_update_stage(self, manager):
        """Test updating stage."""
        manager.update_stage(0, "3-5")

        state = manager.get_player_state(0)
        assert state.current_stage == "3-5"

    def test_reset_player(self, manager):
        """Test resetting player state."""
        manager.add_souls(0, 100)
        manager.record_reroll(0)

        manager.reset_player(0)

        state = manager.get_player_state(0)
        assert state.souls_collected == 0
        assert state.reroll_count == 0

    def test_pending_shop_unlock(self, manager):
        """Test pending shop unlock tracking."""
        # Manually add a pending unlock
        manager.pending_shop_unlocks[0] = ["bard", "graves"]

        first = manager.get_pending_shop_unlock(0)
        assert first == "bard"

        second = manager.get_pending_shop_unlock(0)
        assert second == "graves"

        third = manager.get_pending_shop_unlock(0)
        assert third is None


class TestShopUnlockIntegration:
    """Tests for shop integration with unlock system."""

    def test_shop_pending_unlock(self):
        """Test shop pending unlock sets correctly."""
        from src.core.shop import Shop

        # Create mock pool
        pool = MagicMock()
        pool.get_champion = MagicMock(return_value=None)
        pool.take = MagicMock(return_value=0)
        pool.return_champion = MagicMock()
        pool.get_available_champions_of_cost = MagicMock(return_value=[])

        shop = Shop(pool, player_level=5)
        shop.set_pending_unlock("bard")

        assert shop._pending_unlock_champion_id == "bard"

    def test_shop_clears_pending_on_refresh(self):
        """Test shop clears pending unlock after refresh."""
        from src.core.shop import Shop

        pool = MagicMock()
        pool.get_champion = MagicMock(return_value=None)
        pool.take = MagicMock(return_value=0)
        pool.return_champion = MagicMock()
        pool.get_available_champions_of_cost = MagicMock(return_value=[])

        shop = Shop(pool, player_level=5)
        shop.set_pending_unlock("bard")
        shop.refresh()

        assert shop._pending_unlock_champion_id is None
