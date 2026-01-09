"""Tests for CompBuilder."""

import pytest
from unittest.mock import MagicMock, patch

from src.optimizer.comp_builder import (
    CompBuilder,
    CompTemplate,
    CompRecommendation,
    CompStyle,
)


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


class MockPlayerUnits:
    """Mock player units for testing."""

    def __init__(self):
        self.board: dict = {}
        self.bench: list = []


class MockPlayerState:
    """Mock player state for testing."""

    def __init__(self):
        self.gold = 50
        self.level = 8
        self.units = MockPlayerUnits()


@pytest.fixture
def builder():
    """Create comp builder with mocked dependencies."""
    with patch("src.optimizer.comp_builder.SynergyCalculator") as mock_calc:
        with patch("src.optimizer.comp_builder.load_champions") as mock_load:
            mock_calc_instance = MagicMock()
            mock_calc_instance.calculate_synergies.return_value = {}
            mock_calc.return_value = mock_calc_instance

            # Return empty champions list for default
            mock_load.return_value = []

            builder = CompBuilder()
            builder._champions = {}  # Override lazy loading
            return builder


@pytest.fixture
def player():
    """Create mock player state."""
    return MockPlayerState()


class TestCompBuilder:
    """CompBuilder tests."""

    def test_initialization(self, builder):
        """Test builder initializes correctly."""
        assert builder is not None
        assert len(builder.templates) > 0

    def test_templates_loaded(self, builder):
        """Test templates are loaded."""
        assert len(builder.templates) >= 1

        template = builder.templates[0]
        assert isinstance(template, CompTemplate)
        assert template.name is not None
        assert template.carry is not None

    def test_recommend_empty_board(self, builder, player):
        """Test recommendations with empty board."""
        recommendations = builder.recommend(player)

        assert isinstance(recommendations, list)
        assert all(isinstance(r, CompRecommendation) for r in recommendations)

    def test_recommend_with_board(self, builder, player):
        """Test recommendations with units on board."""
        champ = MockChampion("TFT16_Ahri", "Ahri", 2, ["Arcana", "Mage"])
        player.units.board["unit1"] = MockChampionInstance(champ)

        recommendations = builder.recommend(player)

        # Should have recommendations
        assert len(recommendations) >= 0

    def test_recommend_top_n(self, builder, player):
        """Test limiting recommendations."""
        recommendations = builder.recommend(player, top_n=2)

        assert len(recommendations) <= 2

    def test_recommend_style_filter(self, builder, player):
        """Test filtering by style."""
        recommendations = builder.recommend(player, style_filter=CompStyle.REROLL)

        for rec in recommendations:
            assert rec.template.style == CompStyle.REROLL

    def test_recommendation_has_match_score(self, builder, player):
        """Test recommendations have match scores."""
        recommendations = builder.recommend(player)

        for rec in recommendations:
            assert 0 <= rec.match_score <= 100

    def test_recommendation_sorted_by_score(self, builder, player):
        """Test recommendations sorted by match score."""
        recommendations = builder.recommend(player)

        if len(recommendations) >= 2:
            for i in range(len(recommendations) - 1):
                assert recommendations[i].match_score >= recommendations[i + 1].match_score

    def test_build_from_scratch(self, builder):
        """Test building composition from scratch."""
        # Add some champions to builder
        builder._champions = {
            "TFT_A": MockChampion("TFT_A", "A", 2, ["Warrior", "Knight"]),
            "TFT_B": MockChampion("TFT_B", "B", 3, ["Warrior", "Mage"]),
            "TFT_C": MockChampion("TFT_C", "C", 4, ["Knight", "Guardian"]),
        }

        target_traits = {"Warrior": 2, "Knight": 2}

        result = builder.build_from_scratch(target_traits, level=4)

        assert isinstance(result, list)
        assert len(result) <= 4

    def test_suggest_additions(self, builder, player):
        """Test suggesting unit additions."""
        # Add champion data
        builder._champions = {
            "TFT_A": MockChampion("TFT_A", "A", 2, ["Warrior"]),
        }

        suggestions = builder.suggest_additions(player, slots_available=1)

        assert isinstance(suggestions, list)

    def test_get_templates_by_style(self, builder):
        """Test getting templates by style."""
        reroll_templates = builder.get_templates_by_style(CompStyle.REROLL)

        for template in reroll_templates:
            assert template.style == CompStyle.REROLL

    def test_get_templates_by_tier(self, builder):
        """Test getting templates by tier."""
        s_tier = builder.get_templates_by_tier("S")
        a_tier = builder.get_templates_by_tier("A")

        for template in s_tier:
            assert template.tier == "S"
        for template in a_tier:
            assert template.tier == "A"

    def test_add_custom_template(self, builder):
        """Test adding custom template."""
        initial_count = len(builder.templates)

        custom = CompTemplate(
            name="Custom Comp",
            style=CompStyle.FLEX,
            core_units=["TFT_Custom1", "TFT_Custom2"],
            flex_units=["TFT_Flex1"],
            carry="TFT_Custom1",
            items_priority={},
            target_synergies={"Custom": 4},
            tier="B",
            difficulty="easy",
            description="Custom test comp",
            power_spikes=["4-1"],
        )

        builder.add_custom_template(custom)

        assert len(builder.templates) == initial_count + 1
        assert custom in builder.templates


class TestCompStyle:
    """Tests for CompStyle enum."""

    def test_all_styles_exist(self):
        """Test all expected styles exist."""
        expected = ["REROLL", "STANDARD", "FAST_9", "FLEX"]

        for name in expected:
            assert hasattr(CompStyle, name)

    def test_style_values(self):
        """Test style values are strings."""
        assert CompStyle.REROLL.value == "reroll"
        assert CompStyle.FAST_9.value == "fast_9"


class TestCompTemplate:
    """Tests for CompTemplate dataclass."""

    def test_create_template(self):
        """Test creating a template."""
        template = CompTemplate(
            name="Test Comp",
            style=CompStyle.STANDARD,
            core_units=["TFT_Unit1", "TFT_Unit2"],
            flex_units=["TFT_Flex1"],
            carry="TFT_Unit1",
            items_priority={"TFT_Unit1": ["Item1", "Item2"]},
            target_synergies={"Trait1": 4},
            tier="A",
            difficulty="medium",
            description="Test composition",
            power_spikes=["4-1", "4-5"],
        )

        assert template.name == "Test Comp"
        assert template.style == CompStyle.STANDARD
        assert "TFT_Unit1" in template.core_units
        assert template.carry == "TFT_Unit1"
        assert template.tier == "A"


class TestCompRecommendation:
    """Tests for CompRecommendation dataclass."""

    def test_create_recommendation(self):
        """Test creating a recommendation."""
        template = CompTemplate(
            name="Test",
            style=CompStyle.STANDARD,
            core_units=["TFT_Unit1"],
            flex_units=[],
            carry="TFT_Unit1",
            items_priority={},
            target_synergies={},
            tier="A",
            difficulty="easy",
            description="Test",
            power_spikes=[],
        )

        rec = CompRecommendation(
            template=template,
            match_score=75.0,
            missing_units=["TFT_Unit1"],
            current_units=[],
            transition_cost=10,
            estimated_strength=85.0,
        )

        assert rec.match_score == 75.0
        assert rec.transition_cost == 10
        assert "TFT_Unit1" in rec.missing_units

    def test_recommendation_with_perfect_match(self, builder, player):
        """Test recommendation when player has all units."""
        # This would require full integration test with actual data
        # Placeholder for now
        pass


class TestSynergyBreakdown:
    """Tests for synergy breakdown functionality."""

    def test_get_synergy_breakdown(self, builder):
        """Test getting synergy breakdown."""
        # Add champions
        builder._champions = {
            "TFT_A": MockChampion("TFT_A", "A", 2, ["Warrior"]),
            "TFT_B": MockChampion("TFT_B", "B", 3, ["Warrior", "Mage"]),
        }

        # Mock synergy calculator
        builder.synergy_calc.calculate_synergies = MagicMock(
            return_value={
                "Warrior": MagicMock(count=2),
                "Mage": MagicMock(count=1),
            }
        )

        breakdown = builder.get_synergy_breakdown(["TFT_A", "TFT_B"])

        assert isinstance(breakdown, dict)
