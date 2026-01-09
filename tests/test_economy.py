"""Tests for the Economy System."""

import pytest

from src.core.economy import (
    EconomyCalculator,
    EconomyState,
    IncomeBreakdown,
    RolldownCalculator,
)
from src.core.stage_manager import StageManager, RoundType, RoundInfo
from src.core.economy_advisor import EconomyAdvisor, EconomyStrategy, EconomyAdvice


class TestEconomyCalculator:
    """Tests for EconomyCalculator functionality."""

    @pytest.fixture
    def calc(self):
        """Create a fresh EconomyCalculator."""
        return EconomyCalculator()

    def test_interest_calculation_basic(self, calc):
        """10g = 1 interest, 25g = 2, etc."""
        assert calc.calculate_interest(0) == 0
        assert calc.calculate_interest(5) == 0
        assert calc.calculate_interest(10) == 1
        assert calc.calculate_interest(15) == 1
        assert calc.calculate_interest(20) == 2
        assert calc.calculate_interest(25) == 2
        assert calc.calculate_interest(30) == 3

    def test_interest_calculation_max(self, calc):
        """50g = 5 interest (max), 100g still = 5."""
        assert calc.calculate_interest(50) == 5
        assert calc.calculate_interest(60) == 5
        assert calc.calculate_interest(100) == 5

    def test_streak_bonus_none(self, calc):
        """No bonus for 0-1 streak."""
        assert calc.calculate_streak_bonus(0) == 0
        assert calc.calculate_streak_bonus(1) == 0

    def test_streak_bonus_small(self, calc):
        """2 streak = 0, 3 streak = +1 gold (TFT Set 16)."""
        assert calc.calculate_streak_bonus(2) == 0  # No bonus for 2 streak
        assert calc.calculate_streak_bonus(3) == 1

    def test_streak_bonus_medium(self, calc):
        """4 streak = +1 gold (TFT Set 16)."""
        assert calc.calculate_streak_bonus(4) == 1

    def test_streak_bonus_large(self, calc):
        """5 streak = +2 gold, 6+ streak = +3 gold (TFT Set 16)."""
        assert calc.calculate_streak_bonus(5) == 2
        assert calc.calculate_streak_bonus(6) == 3
        assert calc.calculate_streak_bonus(10) == 3
        assert calc.calculate_streak_bonus(20) == 3

    def test_round_income_base_only(self, calc):
        """Base income is 5 gold + 1 for PvP win."""
        state = EconomyState(gold=0, win_streak=0, loss_streak=0)
        income = calc.calculate_round_income(state, won_combat=True)

        assert income.base_income == 5
        assert income.interest == 0
        assert income.streak_bonus == 0
        assert income.pvp_win_bonus == 1  # +1 for winning PvP
        assert income.total == 6  # 5 base + 1 PvP win

    def test_round_income_with_interest(self, calc):
        """Income with interest + PvP win bonus."""
        state = EconomyState(gold=50, win_streak=0, loss_streak=0)
        income = calc.calculate_round_income(state, won_combat=True)

        assert income.base_income == 5
        assert income.interest == 5
        assert income.streak_bonus == 0
        assert income.pvp_win_bonus == 1
        assert income.total == 11  # 5 base + 5 interest + 1 PvP win

    def test_round_income_with_streak(self, calc):
        """Income with streak bonus + PvP win."""
        state = EconomyState(gold=0, win_streak=6, loss_streak=0)  # 6+ for +3
        income = calc.calculate_round_income(state, won_combat=True)

        assert income.base_income == 5
        assert income.interest == 0
        assert income.streak_bonus == 3
        assert income.pvp_win_bonus == 1
        assert income.total == 9  # 5 base + 3 streak + 1 PvP win

    def test_round_income_full(self, calc):
        """Total income = base + interest + streak + PvP win."""
        state = EconomyState(gold=50, win_streak=6, loss_streak=0)  # 6+ for +3
        income = calc.calculate_round_income(state, won_combat=True)

        assert income.base_income == 5
        assert income.interest == 5
        assert income.streak_bonus == 3
        assert income.pvp_win_bonus == 1
        assert income.total == 14  # 5 base + 5 interest + 3 streak + 1 PvP win

    def test_round_income_loss_streak(self, calc):
        """Loss streak gives bonus too."""
        state = EconomyState(gold=30, win_streak=0, loss_streak=4)
        income = calc.calculate_round_income(state, won_combat=False)

        assert income.base_income == 5
        assert income.interest == 3
        assert income.streak_bonus == 1  # 4 streak = +1 in Set 16
        assert income.total == 9

    def test_round_income_pve_bonus(self, calc):
        """PvE rounds can give bonus gold."""
        state = EconomyState(gold=0)
        income = calc.calculate_round_income(state, won_combat=True, is_pve=True, pve_gold=5)

        assert income.pve_bonus == 5
        assert income.total == 10  # 5 base + 5 pve

    def test_xp_needed(self, calc):
        """XP needed for each level (TFT Set 16)."""
        assert calc.calculate_xp_needed(1) == 0  # Starts at level 2 with 0 XP needed
        assert calc.calculate_xp_needed(2) == 2  # To level 3
        assert calc.calculate_xp_needed(3) == 6  # To level 4
        assert calc.calculate_xp_needed(4) == 10  # To level 5
        assert calc.calculate_xp_needed(5) == 20  # To level 6
        assert calc.calculate_xp_needed(6) == 36  # To level 7
        assert calc.calculate_xp_needed(7) == 60  # To level 8 (Set 16)
        assert calc.calculate_xp_needed(8) == 68  # To level 9 (Set 16)
        assert calc.calculate_xp_needed(9) == 68  # To level 10 (Set 16)
        assert calc.calculate_xp_needed(10) == 0  # Max level

    def test_gold_to_level_same(self, calc):
        """No gold needed if already at target."""
        assert calc.calculate_gold_to_level(5, 0, 5) == 0
        assert calc.calculate_gold_to_level(5, 0, 3) == 0

    def test_gold_to_level_one_level(self, calc):
        """Gold to level up once."""
        # Level 2 -> 3 needs 2 XP = 4 gold (1 purchase) in Set 16
        assert calc.calculate_gold_to_level(2, 0, 3) == 4

    def test_gold_to_level_multiple(self, calc):
        """Gold to level up multiple times."""
        # Level 2 -> 5: 2 + 6 + 10 = 18 XP = 20 gold (5 purchases)
        gold = calc.calculate_gold_to_level(2, 0, 5)
        assert gold == 20

    def test_gold_to_level_with_progress(self, calc):
        """Gold to level with existing XP progress."""
        # Level 3 -> 4 needs 6 XP, if we have 4 XP, need 2 more = 4 gold (1 purchase rounds up)
        assert calc.calculate_gold_to_level(3, 4, 4) == 4

    def test_rounds_to_level(self, calc):
        """Rounds needed with passive XP."""
        # Level 2 -> 3 needs 2 XP, 2 XP per round = 1 round
        assert calc.calculate_rounds_to_level(2, 0, 3) == 1

        # Level 2 -> 4 needs 2 + 6 = 8 XP, 2 XP per round = 4 rounds
        assert calc.calculate_rounds_to_level(2, 0, 4) == 4

    def test_simulate_economy_length(self, calc):
        """Simulation returns correct number of states."""
        initial = EconomyState(gold=10, level=1)
        states = calc.simulate_economy(initial, rounds=10)

        assert len(states) == 11  # Initial + 10 rounds

    def test_simulate_economy_gold_increases(self, calc):
        """Gold should generally increase over time."""
        initial = EconomyState(gold=10, level=1)
        states = calc.simulate_economy(initial, rounds=10, strategy="econ")

        # With econ strategy, gold should increase
        assert states[-1].gold > states[0].gold


class TestRolldownCalculator:
    """Tests for RolldownCalculator functionality."""

    @pytest.fixture
    def calc(self):
        """Create a fresh RolldownCalculator."""
        return RolldownCalculator()

    def test_rolldown_budget_healthy(self, calc):
        """Healthy player (>50 HP) keeps 20 reserve."""
        budget = calc.calculate_rolldown_budget(50, health=80, stage="4-2")
        assert budget == 30  # 50 - 20 reserve

    def test_rolldown_budget_low_hp(self, calc):
        """Low HP (30-50) keeps 10 reserve."""
        budget = calc.calculate_rolldown_budget(50, health=40, stage="4-2")
        assert budget == 40  # 50 - 10 reserve

    def test_rolldown_budget_critical(self, calc):
        """Critical HP (<30) rolls almost everything."""
        budget = calc.calculate_rolldown_budget(50, health=20, stage="4-2")
        assert budget == 48  # 50 - 2 (reroll cost)

    def test_expected_rolls(self, calc):
        """Calculate number of rolls from budget."""
        assert calc.expected_rolls(10) == 5
        assert calc.expected_rolls(20) == 10
        assert calc.expected_rolls(50) == 25

    def test_optimal_rolldown_stage(self, calc):
        """Calculate best rolldown timing."""
        state = EconomyState(gold=30, level=6)
        result = calc.calculate_optimal_rolldown_stage(state, target_cost=4)

        assert "recommended_stage" in result
        assert "recommended_level" in result
        assert "expected_gold" in result
        assert "all_options" in result

    def test_optimal_rolldown_for_5_cost(self, calc):
        """Higher level needed for 5-cost."""
        state = EconomyState(gold=50, level=8)
        result = calc.calculate_optimal_rolldown_stage(state, target_cost=5)

        # Should recommend level 9 for 5-costs
        assert result["recommended_level"] >= 8


class TestStageManager:
    """Tests for StageManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh StageManager."""
        return StageManager()

    def test_initial_stage(self, manager):
        """Starts at stage 1-1."""
        assert manager.current_stage == 1
        assert manager.current_round == 1
        assert manager.get_stage_string() == "1-1"

    def test_advance_round_same_stage(self, manager):
        """Advancing within same stage."""
        manager.advance_round()
        assert manager.get_stage_string() == "1-2"

        manager.advance_round()
        assert manager.get_stage_string() == "1-3"

    def test_advance_round_stage_transition(self, manager):
        """Advancing from stage 1 to stage 2."""
        # Stage 1 has 4 rounds
        for _ in range(4):
            manager.advance_round()

        assert manager.current_stage == 2
        assert manager.current_round == 1
        assert manager.get_stage_string() == "2-1"

    def test_advance_round_stage_2_transition(self, manager):
        """Advancing from stage 2 to stage 3."""
        # Go to stage 2
        for _ in range(4):
            manager.advance_round()

        # Stage 2 has 7 rounds
        for _ in range(7):
            manager.advance_round()

        assert manager.current_stage == 3
        assert manager.current_round == 1

    def test_round_type_pve(self, manager):
        """Stage 1-2 through 1-4 are PvE (1-1 is carousel)."""
        # 1-1 is carousel, advance to 1-2
        manager.advance_round()
        info = manager.get_current_round_info()
        assert info.round_type == RoundType.PVE

    def test_round_type_pvp(self, manager):
        """Stage 2+ is mostly PvP."""
        # Go to stage 2-1
        for _ in range(4):
            manager.advance_round()

        info = manager.get_current_round_info()
        assert info.round_type == RoundType.PVP

    def test_round_type_carousel(self, manager):
        """Carousel rounds identified correctly."""
        # Go to 2-4 (carousel)
        for _ in range(4 + 3):  # Through stage 1, then 3 more
            manager.advance_round()

        assert manager.get_stage_string() == "2-4"
        info = manager.get_current_round_info()
        assert info.is_carousel is True

    def test_round_type_augment(self, manager):
        """Augment rounds identified correctly."""
        # Go to 2-1 (augment)
        for _ in range(4):
            manager.advance_round()

        assert manager.get_stage_string() == "2-1"
        info = manager.get_current_round_info()
        assert info.is_augment is True

    def test_passive_xp(self, manager):
        """No passive XP in 1-1, 2 XP from 1-2 onwards."""
        info = manager.get_current_round_info()
        assert info.passive_xp == 0  # 1-1 (carousel)

        # Go to 1-2 (first PvE with XP)
        manager.advance_round()
        info = manager.get_current_round_info()
        assert info.passive_xp == 2  # 1-2+

    def test_is_pve_round(self, manager):
        """Check is_pve flag for various rounds."""
        # 1-1 is carousel, not PvE
        assert manager.is_pve_round() is False

        # 1-2 is PvE
        manager.advance_round()
        assert manager.is_pve_round() is True

    def test_rounds_until_same_stage(self, manager):
        """Rounds until target in same stage."""
        rounds = manager.get_rounds_until("1-4")
        assert rounds == 3  # 1-1 to 1-4

    def test_rounds_until_next_stage(self, manager):
        """Rounds until target in next stage."""
        rounds = manager.get_rounds_until("2-1")
        assert rounds == 4  # 4 rounds in stage 1

    def test_rounds_until_far_stage(self, manager):
        """Rounds until far target."""
        # 1-1 to 3-2: 3 remaining in stage 1 (1-1 to 1-4) + 7 (stage 2) + 2 (stage 3) = 12
        rounds = manager.get_rounds_until("3-2")
        assert rounds == 3 + 7 + 2

    def test_is_rolldown_timing(self, manager):
        """Rolldown timing detection."""
        assert manager.is_rolldown_timing() is False  # 1-1

        # Go to 3-2 (rolldown timing)
        manager.set_stage("3-2")
        assert manager.is_rolldown_timing() is True

        # Go to 4-2 (rolldown timing)
        manager.set_stage("4-2")
        assert manager.is_rolldown_timing() is True

    def test_is_level_timing(self, manager):
        """Level timing detection."""
        assert manager.is_level_timing() is False  # 1-1

        # Go to 2-1 (level 4 timing)
        manager.set_stage("2-1")
        assert manager.is_level_timing() is True

        # Go to 4-2 (level 8 timing)
        manager.set_stage("4-2")
        assert manager.is_level_timing() is True

    def test_set_stage(self, manager):
        """Set stage directly."""
        manager.set_stage("4-5")
        assert manager.current_stage == 4
        assert manager.current_round == 5
        assert manager.get_stage_string() == "4-5"

    def test_recommended_level(self, manager):
        """Get recommended level for stage."""
        assert manager.get_recommended_level() == 3  # Early game default

        manager.set_stage("2-5")
        assert manager.get_recommended_level() == 5

        manager.set_stage("4-2")
        assert manager.get_recommended_level() == 8

    def test_is_pve_round(self, manager):
        """PvE round detection."""
        assert manager.is_pve_round() is False  # 1-1 is carousel, not PvE

        manager.set_stage("1-2")
        assert manager.is_pve_round() is True  # 1-2 is PvE

        manager.set_stage("2-1")
        assert manager.is_pve_round() is False

        manager.set_stage("2-7")
        assert manager.is_pve_round() is True  # Krugs


class TestEconomyAdvisor:
    """Tests for EconomyAdvisor functionality."""

    @pytest.fixture
    def advisor(self):
        """Create a fresh EconomyAdvisor."""
        return EconomyAdvisor()

    def test_critical_hp_advice(self, advisor):
        """Critical HP (<20) recommends all-in."""
        advice = advisor.get_advice(
            gold=50, level=7, xp=0, health=15, stage="4-2"
        )
        assert advice.strategy == EconomyStrategy.ALL_IN
        assert advice.action == "all_in"
        assert advice.priority == "high"

    def test_low_hp_advice(self, advisor):
        """Low HP (20-40) recommends rolling."""
        advice = advisor.get_advice(
            gold=50, level=7, xp=0, health=35, stage="4-2"
        )
        assert advice.action == "roll"
        assert advice.priority == "high"

    def test_loss_streak_advice(self, advisor):
        """Long loss streak with good health recommends saving."""
        advice = advisor.get_advice(
            gold=30, level=5, xp=0, health=80, stage="2-5",
            loss_streak=5
        )
        assert advice.strategy == EconomyStrategy.LOSS_STREAK
        assert advice.action == "save"

    def test_win_streak_advice(self, advisor):
        """Win streak recommends maintaining strength."""
        advice = advisor.get_advice(
            gold=40, level=6, xp=0, health=90, stage="3-2",
            win_streak=5, board_strength=0.8
        )
        assert advice.strategy == EconomyStrategy.WIN_STREAK

    def test_standard_level_timing(self, advisor):
        """Standard timing can recommend leveling."""
        advice = advisor.get_advice(
            gold=30, level=5, xp=0, health=70, stage="3-2"
        )
        # At 3-2 with level 5, should consider leveling to 6
        assert advice.action in ["level", "save"]

    def test_slow_roll_advice(self, advisor):
        """Weak board early recommends slow roll."""
        advice = advisor.get_advice(
            gold=55, level=5, xp=0, health=80, stage="2-5",
            board_strength=0.3
        )
        assert advice.strategy == EconomyStrategy.SLOW_ROLL
        # Should roll above 50
        assert advice.action == "roll"
        assert advice.gold_to_keep == 50

    def test_rolldown_recommendation(self, advisor):
        """Rolldown recommendation."""
        result = advisor.get_rolldown_recommendation(
            gold=50, level=8, health=60, target_champions=None, stage="4-2"
        )

        assert "should_rolldown" in result
        assert "budget" in result
        assert "num_rolls" in result
        assert result["should_rolldown"] is True  # 4-2 at level 8

    def test_level_recommendation(self, advisor):
        """Level recommendation."""
        result = advisor.get_level_recommendation(
            gold=30, level=6, xp=0, stage="4-1", health=70
        )

        assert "should_level" in result
        assert "target_level" in result
        assert "gold_needed" in result


class TestIncomeBreakdown:
    """Tests for IncomeBreakdown dataclass."""

    def test_calculate_total(self):
        """Total calculation."""
        breakdown = IncomeBreakdown(
            base_income=5,
            interest=3,
            streak_bonus=2,
            pve_bonus=1
        )
        total = breakdown.calculate_total()

        assert total == 11
        assert breakdown.total == 11

    def test_default_values(self):
        """Default values."""
        breakdown = IncomeBreakdown()

        assert breakdown.base_income == 5
        assert breakdown.interest == 0
        assert breakdown.streak_bonus == 0
        assert breakdown.pve_bonus == 0
        assert breakdown.total == 0


class TestEconomyState:
    """Tests for EconomyState dataclass."""

    def test_default_values(self):
        """Default values."""
        state = EconomyState()

        assert state.gold == 0
        assert state.level == 1
        assert state.xp == 0
        assert state.win_streak == 0
        assert state.loss_streak == 0
        assert state.stage == "1-1"

    def test_custom_values(self):
        """Custom initialization."""
        state = EconomyState(
            gold=50,
            level=7,
            xp=10,
            win_streak=3,
            stage="4-2"
        )

        assert state.gold == 50
        assert state.level == 7
        assert state.xp == 10
        assert state.win_streak == 3
        assert state.stage == "4-2"


class TestRoundInfo:
    """Tests for RoundInfo dataclass."""

    def test_round_info_creation(self):
        """Create RoundInfo."""
        info = RoundInfo(
            stage="2-3",
            stage_num=2,
            round_num=3,
            round_type=RoundType.PVP,
            passive_gold=5,
            passive_xp=2
        )

        assert info.stage == "2-3"
        assert info.stage_num == 2
        assert info.round_num == 3
        assert info.round_type == RoundType.PVP
        assert info.passive_gold == 5
        assert info.passive_xp == 2
        assert info.is_carousel is False
        assert info.is_augment is False
