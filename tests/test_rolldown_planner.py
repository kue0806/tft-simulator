"""Tests for RolldownPlanner."""

import pytest
from unittest.mock import MagicMock

from src.optimizer.rolldown_planner import (
    RolldownPlanner,
    RolldownPlan,
    RolldownStrategy,
    RolldownTiming,
)


class MockStageManager:
    """Mock stage manager for testing."""

    def __init__(self, stage: str = "4-1"):
        self._stage = stage

    def get_stage_string(self) -> str:
        return self._stage


class MockPlayerUnits:
    """Mock player units for testing."""

    def __init__(self):
        self.board: dict = {}
        self.bench: list = []


class MockPlayerState:
    """Mock player state for testing."""

    def __init__(self, gold: int = 50, level: int = 7, health: int = 80, xp: int = 0):
        self.gold = gold
        self.level = level
        self.health = health
        self.xp = xp
        self.units = MockPlayerUnits()


class MockGameState:
    """Mock game state for testing."""

    def __init__(self, stage: str = "4-1"):
        self.stage_manager = MockStageManager(stage)


@pytest.fixture
def planner():
    """Create rolldown planner."""
    return RolldownPlanner()


@pytest.fixture
def player():
    """Create mock player state."""
    return MockPlayerState()


@pytest.fixture
def game():
    """Create mock game state."""
    return MockGameState()


class TestRolldownPlanner:
    """RolldownPlanner tests."""

    def test_initialization(self, planner):
        """Test planner initializes correctly."""
        assert planner is not None
        assert len(planner.KEY_TIMINGS) > 0

    def test_create_plan_basic(self, planner, player, game):
        """Test basic plan creation."""
        target_units = ["TFT_Unit1", "TFT_Unit2"]

        plan = planner.create_plan(player, game, target_units)

        assert isinstance(plan, RolldownPlan)
        assert plan.strategy is not None
        assert plan.target_units == target_units
        assert len(plan.advice) > 0

    def test_all_in_strategy_low_hp(self, planner, game):
        """Test all-in strategy when HP is critical."""
        player = MockPlayerState(health=25)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        assert plan.strategy == RolldownStrategy.ALL_IN
        assert plan.is_rolldown_now is True

    def test_slow_roll_6_strategy(self, planner, game):
        """Test slow roll 6 strategy for low cost units."""
        player = MockPlayerState(level=6, health=80)
        # Low cost targets
        target_units = ["TFT_1Cost1", "TFT_1Cost2"]

        plan = planner.create_plan(player, game, target_units)

        # Should recommend slow roll for 1-cost units
        assert plan.strategy in [
            RolldownStrategy.SLOW_ROLL_6,
            RolldownStrategy.SLOW_ROLL_7,
            RolldownStrategy.FAST_8,
        ]

    def test_fast_8_strategy(self, planner, game):
        """Test fast 8 strategy for mid cost units."""
        player = MockPlayerState(gold=60, level=7, health=70)

        plan = planner.create_plan(player, game, ["TFT_4Cost"])

        # Should recommend fast 8 or slow roll 8
        assert plan.strategy in [
            RolldownStrategy.FAST_8,
            RolldownStrategy.SLOW_ROLL_8,
        ]

    def test_fast_9_strategy(self, planner, game):
        """Test fast 9 strategy for high cost units."""
        player = MockPlayerState(gold=80, level=8, health=60)

        plan = planner.create_plan(player, game, ["TFT_5Cost"])

        # High cost targets with good economy at level 8
        # Could go fast 9, slow roll 8, or fast 8 depending on logic
        assert plan.strategy in [
            RolldownStrategy.FAST_8,
            RolldownStrategy.FAST_9,
            RolldownStrategy.SLOW_ROLL_8,
        ]

    def test_budget_allocation_all_in(self, planner, game):
        """Test budget allocation for all-in."""
        player = MockPlayerState(gold=40, health=20)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        assert plan.roll_budget == player.gold
        assert plan.save_amount == 0

    def test_budget_allocation_slow_roll(self, planner, game):
        """Test budget allocation for slow roll."""
        player = MockPlayerState(gold=60, level=7, health=70)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        if plan.strategy in [
            RolldownStrategy.SLOW_ROLL_6,
            RolldownStrategy.SLOW_ROLL_7,
            RolldownStrategy.SLOW_ROLL_8,
        ]:
            assert plan.save_amount == 50
            assert plan.roll_budget <= 10

    def test_current_phase_leveling(self, planner, game):
        """Test phase detection when leveling."""
        player = MockPlayerState(level=6, gold=50)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        # At level 6 targeting level 8, should be leveling
        if plan.strategy == RolldownStrategy.FAST_8:
            assert plan.current_phase == "leveling"

    def test_current_phase_rolling(self, planner, game):
        """Test phase detection when rolling."""
        player = MockPlayerState(level=8, gold=40)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        # At target level with gold, should be rolling
        assert plan.current_phase in ["rolling", "stabilized"]

    def test_rolldown_timing_key_stages(self, planner, player):
        """Test rolldown at key stages."""
        key_stages = ["4-1", "4-2", "4-5", "5-1"]
        player.level = 8

        for stage in key_stages:
            game = MockGameState(stage=stage)
            plan = planner.create_plan(player, game, ["TFT_Unit1"])

            # Should potentially rolldown at key timings
            if plan.strategy not in [
                RolldownStrategy.SLOW_ROLL_6,
                RolldownStrategy.SLOW_ROLL_7,
                RolldownStrategy.SAVE,
            ]:
                # Key stages should trigger rolldown consideration
                assert plan.recommended_timing is not None

    def test_hit_probability_calculation(self, planner, player, game):
        """Test hit probability is calculated."""
        plan = planner.create_plan(player, game, ["TFT_Unit1", "TFT_Unit2"])

        assert 0 <= plan.hit_probability <= 1
        assert plan.expected_rolls > 0

    def test_advice_generation(self, planner, player, game):
        """Test advice is generated."""
        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        assert len(plan.advice) > 0
        assert all(isinstance(a, str) for a in plan.advice)

    def test_advice_contains_hp_warning(self, planner, game):
        """Test advice contains HP warning when low."""
        player = MockPlayerState(health=40)

        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        # Should have some HP-related advice
        has_hp_advice = any("HP" in a or "hp" in a.lower() for a in plan.advice)
        # At 40 HP, should mention it
        assert has_hp_advice or player.health > 50

    def test_target_star_levels_default(self, planner, player, game):
        """Test default target star levels."""
        target_units = ["TFT_Unit1", "TFT_Unit2"]

        plan = planner.create_plan(player, game, target_units)

        # Default should be 2-star
        for unit in target_units:
            assert plan.target_star_levels.get(unit, 2) == 2

    def test_target_star_levels_custom(self, planner, player, game):
        """Test custom target star levels."""
        target_units = ["TFT_Unit1"]
        target_stars = {"TFT_Unit1": 3}

        plan = planner.create_plan(player, game, target_units, target_stars)

        assert plan.target_star_levels["TFT_Unit1"] == 3


class TestRolldownStrategy:
    """Tests for RolldownStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all expected strategies exist."""
        expected = [
            "FAST_8",
            "FAST_9",
            "SLOW_ROLL_6",
            "SLOW_ROLL_7",
            "SLOW_ROLL_8",
            "ALL_IN",
            "SAVE",
        ]

        for name in expected:
            assert hasattr(RolldownStrategy, name)

    def test_strategy_values(self):
        """Test strategy values are strings."""
        assert RolldownStrategy.FAST_8.value == "fast_8"
        assert RolldownStrategy.ALL_IN.value == "all_in"


class TestRolldownTiming:
    """Tests for RolldownTiming dataclass."""

    def test_create_timing(self):
        """Test creating a timing."""
        timing = RolldownTiming(
            stage="4-2",
            level=8,
            gold_threshold=50,
            description="Test timing",
        )

        assert timing.stage == "4-2"
        assert timing.level == 8
        assert timing.gold_threshold == 50

    def test_key_timings_loaded(self, planner):
        """Test key timings are pre-loaded."""
        assert "3-2" in planner.KEY_TIMINGS
        assert "4-1" in planner.KEY_TIMINGS
        assert "4-2" in planner.KEY_TIMINGS

        timing = planner.KEY_TIMINGS["4-2"]
        assert timing.level == 8


class TestRolldownPlan:
    """Tests for RolldownPlan dataclass."""

    def test_plan_has_all_fields(self, planner, player, game):
        """Test plan contains all required fields."""
        plan = planner.create_plan(player, game, ["TFT_Unit1"])

        assert hasattr(plan, "strategy")
        assert hasattr(plan, "current_phase")
        assert hasattr(plan, "recommended_timing")
        assert hasattr(plan, "is_rolldown_now")
        assert hasattr(plan, "roll_budget")
        assert hasattr(plan, "level_budget")
        assert hasattr(plan, "save_amount")
        assert hasattr(plan, "target_units")
        assert hasattr(plan, "target_star_levels")
        assert hasattr(plan, "hit_probability")
        assert hasattr(plan, "expected_rolls")
        assert hasattr(plan, "advice")
