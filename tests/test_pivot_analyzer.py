"""Tests for PivotAnalyzer."""

import pytest
from unittest.mock import MagicMock, patch

from src.optimizer.pivot_analyzer import (
    PivotAnalyzer,
    PivotAdvice,
    PivotOption,
    PivotReason,
)
from src.optimizer.comp_builder import CompTemplate, CompStyle


class MockChampion:
    """Mock champion for testing."""

    def __init__(self, champion_id: str, name: str, cost: int, traits: list):
        self.id = champion_id
        self.name = name
        self.cost = cost
        self.traits = traits


class MockChampionInstance:
    """Mock champion instance for testing."""

    def __init__(self, champion: MockChampion, star_level: int = 1):
        self.champion = champion
        self.star_level = star_level

    def get_sell_value(self) -> int:
        return self.champion.cost * (3 ** (self.star_level - 1))


class MockPlayerUnits:
    """Mock player units for testing."""

    def __init__(self):
        self.board: dict = {}
        self.bench: list = []


class MockPlayerState:
    """Mock player state for testing."""

    def __init__(self, gold: int = 50, health: int = 80):
        self.gold = gold
        self.health = health
        self.units = MockPlayerUnits()


class MockStageManager:
    """Mock stage manager for testing."""

    def get_stage_string(self) -> str:
        return "4-1"


class MockGameState:
    """Mock game state for testing."""

    def __init__(self):
        self.stage_manager = MockStageManager()


def create_test_template(name: str, core_units: list) -> CompTemplate:
    """Create a test composition template."""
    return CompTemplate(
        name=name,
        style=CompStyle.STANDARD,
        core_units=core_units,
        flex_units=[],
        carry=core_units[0] if core_units else "",
        items_priority={},
        target_synergies={},
        tier="A",
        difficulty="medium",
        description="Test",
        power_spikes=["4-1"],
    )


@pytest.fixture
def analyzer():
    """Create pivot analyzer with mocked comp builder."""
    with patch("src.optimizer.pivot_analyzer.CompBuilder") as mock_builder:
        mock_instance = MagicMock()
        mock_instance.templates = [
            create_test_template("Comp A", ["TFT_A1", "TFT_A2"]),
            create_test_template("Comp B", ["TFT_B1", "TFT_B2"]),
        ]
        mock_instance.recommend.return_value = []
        mock_builder.return_value = mock_instance

        analyzer = PivotAnalyzer()
        analyzer.comp_builder = mock_instance
        return analyzer


@pytest.fixture
def player():
    """Create mock player state."""
    return MockPlayerState()


@pytest.fixture
def game():
    """Create mock game state."""
    return MockGameState()


class TestPivotAnalyzer:
    """PivotAnalyzer tests."""

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.comp_builder is not None

    def test_analyze_healthy_comp(self, analyzer, player, game):
        """Test analysis of healthy composition."""
        current_comp = create_test_template("Current", ["TFT_Unit1", "TFT_Unit2"])

        # Player has all core units
        champ1 = MockChampion("TFT_Unit1", "Unit1", 3, ["Trait1"])
        champ2 = MockChampion("TFT_Unit2", "Unit2", 3, ["Trait2"])
        player.units.board = {
            "u1": MockChampionInstance(champ1, star_level=2),
            "u2": MockChampionInstance(champ2, star_level=2),
        }

        advice = analyzer.analyze(player, game, current_comp)

        assert isinstance(advice, PivotAdvice)
        assert advice.current_comp_health > 50

    def test_analyze_critical_hp(self, analyzer, game):
        """Test analysis with critical HP."""
        player = MockPlayerState(health=25)

        advice = analyzer.analyze(player, game)

        assert advice.should_pivot is True
        assert advice.urgency == "immediate"
        assert PivotReason.HP_CRITICAL in [
            r for opt in advice.options for r in opt.reasons
        ] or not advice.options

    def test_analyze_low_comp_health(self, analyzer, player, game):
        """Test analysis with poor composition health."""
        current_comp = create_test_template("Current", ["TFT_A", "TFT_B", "TFT_C"])

        # Player has none of the core units
        player.units.board = {}

        advice = analyzer.analyze(player, game, current_comp)

        # With no core units, comp health should be low
        assert advice.current_comp_health < 50

    def test_analyze_contested_units(self, analyzer, player, game):
        """Test analysis with contested units."""
        current_comp = create_test_template("Current", ["TFT_A", "TFT_B", "TFT_C"])
        contested = ["TFT_A", "TFT_B", "TFT_C"]  # All core units contested

        advice = analyzer.analyze(player, game, current_comp, contested)

        # Heavy contestation should reduce health
        assert advice.current_comp_health < 70 or advice.should_pivot

    def test_calculate_pivot_cost(self, analyzer, player):
        """Test pivot cost calculation."""
        from_comp = create_test_template("From", ["TFT_A", "TFT_B"])
        to_comp = create_test_template("To", ["TFT_B", "TFT_C"])

        # Player has TFT_A and TFT_B
        champ_a = MockChampion("TFT_A", "A", 3, ["Trait1"])
        champ_b = MockChampion("TFT_B", "B", 3, ["Trait2"])
        player.units.board = {
            "u1": MockChampionInstance(champ_a),
            "u2": MockChampionInstance(champ_b),
        }

        option = analyzer.calculate_pivot_cost(player, from_comp, to_comp)

        assert isinstance(option, PivotOption)
        assert "TFT_B" in option.shared_units  # Shared between both
        assert "TFT_A" in option.units_to_sell  # Only in from_comp
        assert "TFT_C" in option.units_to_buy  # Only in to_comp

    def test_pivot_option_risk_levels(self, analyzer, player):
        """Test pivot option risk assessment."""
        from_comp = create_test_template("From", ["TFT_A"])
        to_comp = create_test_template("To", ["TFT_B"])

        # High gold = higher success probability = lower risk
        player.gold = 80
        option = analyzer.calculate_pivot_cost(player, from_comp, to_comp)
        assert option.risk_level in ["low", "medium", "high"]

        # Low gold = lower success probability = higher risk
        player.gold = 10
        option = analyzer.calculate_pivot_cost(player, from_comp, to_comp)
        # Risk is based on success_probability, which depends on gold
        assert option.risk_level in ["medium", "high"]

    def test_pivot_option_success_probability(self, analyzer, player):
        """Test success probability calculation."""
        from_comp = create_test_template("From", [])
        to_comp = create_test_template("To", ["TFT_A", "TFT_B"])

        player.gold = 100  # High gold
        option = analyzer.calculate_pivot_cost(player, from_comp, to_comp)
        high_gold_prob = option.success_probability

        player.gold = 10  # Low gold
        option = analyzer.calculate_pivot_cost(player, from_comp, to_comp)
        low_gold_prob = option.success_probability

        assert high_gold_prob > low_gold_prob

    def test_find_natural_pivots(self, analyzer, player):
        """Test finding natural pivot options."""
        current = create_test_template("Current", ["TFT_A", "TFT_B", "TFT_C"])

        # Set up templates with overlapping units
        analyzer.comp_builder.templates = [
            create_test_template("Overlap", ["TFT_A", "TFT_B", "TFT_D"]),  # 2 shared
            create_test_template("Different", ["TFT_X", "TFT_Y", "TFT_Z"]),  # 0 shared
            current,
        ]

        natural_pivots = analyzer.find_natural_pivots(player, current)

        # Should find Overlap but not Different
        assert any(t.name == "Overlap" for t in natural_pivots)
        assert not any(t.name == "Different" for t in natural_pivots)

    def test_get_transition_path(self, analyzer, player):
        """Test transition path generation."""
        from_comp = create_test_template("From", ["TFT_A", "TFT_B"])
        to_comp = create_test_template("To", ["TFT_B", "TFT_C"])

        champ_a = MockChampion("TFT_A", "A", 3, ["Trait1"])
        champ_b = MockChampion("TFT_B", "B", 3, ["Trait2"])
        player.units.board = {
            "u1": MockChampionInstance(champ_a),
            "u2": MockChampionInstance(champ_b),
        }

        steps = analyzer.get_transition_path(player, from_comp, to_comp)

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert all(isinstance(s, str) for s in steps)

    def test_advice_explanation(self, analyzer, player, game):
        """Test advice explanation generation."""
        advice = analyzer.analyze(player, game)

        assert isinstance(advice.explanation, str)
        assert len(advice.explanation) > 0

    def test_pivot_options_sorted(self, analyzer, player, game):
        """Test pivot options are sorted by viability."""
        # Set up comp builder to return recommendations
        mock_recs = [
            MagicMock(template=create_test_template("A", ["TFT_A"])),
            MagicMock(template=create_test_template("B", ["TFT_B"])),
        ]
        analyzer.comp_builder.recommend.return_value = mock_recs

        player.health = 30  # Trigger pivot

        advice = analyzer.analyze(player, game)

        # Options should be sorted (best first)
        if len(advice.options) >= 2:
            for i in range(len(advice.options) - 1):
                ratio_i = advice.options[i].success_probability / (
                    advice.options[i].total_cost + 1
                )
                ratio_j = advice.options[i + 1].success_probability / (
                    advice.options[i + 1].total_cost + 1
                )
                assert ratio_i >= ratio_j


class TestPivotReason:
    """Tests for PivotReason enum."""

    def test_all_reasons_exist(self):
        """Test all expected reasons exist."""
        expected = [
            "CONTESTED",
            "LOW_ROLLS",
            "HP_CRITICAL",
            "BETTER_ITEMS",
            "HIGHROLL",
            "LOBBY_READ",
        ]

        for name in expected:
            assert hasattr(PivotReason, name)


class TestPivotOption:
    """Tests for PivotOption dataclass."""

    def test_create_option(self):
        """Test creating a pivot option."""
        template = create_test_template("Target", ["TFT_A"])

        option = PivotOption(
            target_comp=template,
            from_comp=None,
            shared_units=["TFT_Shared"],
            units_to_sell=["TFT_Sell"],
            units_to_buy=["TFT_Buy"],
            gold_loss=5,
            roll_cost=30,
            total_cost=35,
            success_probability=0.7,
            risk_level="medium",
            reasons=[PivotReason.LOW_ROLLS],
        )

        assert option.target_comp == template
        assert option.total_cost == 35
        assert option.success_probability == 0.7


class TestPivotAdvice:
    """Tests for PivotAdvice dataclass."""

    def test_create_advice(self):
        """Test creating pivot advice."""
        advice = PivotAdvice(
            should_pivot=True,
            urgency="soon",
            current_comp_health=45.0,
            options=[],
            recommendation=None,
            explanation="Test explanation",
        )

        assert advice.should_pivot is True
        assert advice.urgency == "soon"
        assert advice.current_comp_health == 45.0
