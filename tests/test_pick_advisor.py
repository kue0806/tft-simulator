"""Tests for PickAdvisor."""

import pytest
from unittest.mock import MagicMock, patch

from src.optimizer.pick_advisor import (
    PickAdvisor,
    PickAdvice,
    PickRecommendation,
    PickReason,
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
        self.level = 7
        self.health = 80
        self.units = MockPlayerUnits()
        self.shop = MockShop()


class MockShop:
    """Mock shop for testing."""

    def __init__(self):
        self.slots = [None] * 5


@pytest.fixture
def advisor():
    """Create pick advisor with mocked synergy calculator."""
    with patch("src.optimizer.pick_advisor.SynergyCalculator") as mock_calc:
        mock_instance = MagicMock()
        mock_instance.calculate_synergies.return_value = {}
        mock_instance.preview_add_champion.return_value = {}
        mock_calc.return_value = mock_instance
        return PickAdvisor(synergy_calculator=mock_instance)


@pytest.fixture
def player():
    """Create mock player state."""
    return MockPlayerState()


class TestPickAdvisor:
    """PickAdvisor tests."""

    def test_initialization(self, advisor):
        """Test advisor initializes correctly."""
        assert advisor is not None
        assert advisor.weights[PickReason.UPGRADE_3STAR] == 100
        assert advisor.weights[PickReason.UPGRADE_2STAR] == 50

    def test_analyze_empty_shop(self, advisor, player):
        """Test analysis with empty shop."""
        advice = advisor.analyze(player)

        assert isinstance(advice, PickAdvice)
        assert len(advice.recommendations) == 0
        assert advice.should_refresh is True
        assert advice.refresh_reason is not None

    def test_analyze_with_champion(self, advisor, player):
        """Test analysis with champion in shop."""
        champion = MockChampion("TFT_Test", "Test", 3, ["Warrior"])
        player.shop.slots[0] = champion

        # Add a copy on bench to trigger pair bonus
        instance = MockChampionInstance(champion, star_level=1)
        player.units.bench.append(instance)

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert rec.champion_id == "TFT_Test"
        assert rec.copies_owned == 1
        assert PickReason.ECONOMY_PAIR in rec.reasons

    def test_recommend_upgrade_2star(self, advisor, player):
        """Test recommending 2-star upgrade."""
        champion = MockChampion("TFT_Test", "Test", 2, ["Warrior"])
        player.shop.slots[0] = champion

        # Add 2 copies to trigger 2-star recommendation
        for _ in range(2):
            instance = MockChampionInstance(champion, star_level=1)
            player.units.bench.append(instance)

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert PickReason.UPGRADE_2STAR in rec.reasons
        assert rec.copies_owned == 2

    def test_recommend_upgrade_3star(self, advisor, player):
        """Test recommending 3-star upgrade."""
        champion = MockChampion("TFT_Test", "Test", 1, ["Warrior"])
        player.shop.slots[0] = champion

        # Add 6 copies (two 2-stars worth) to trigger 3-star recommendation
        instance1 = MockChampionInstance(champion, star_level=2)
        instance2 = MockChampionInstance(champion, star_level=2)
        player.units.bench.extend([instance1, instance2])

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert PickReason.UPGRADE_3STAR in rec.reasons
        assert rec.copies_owned == 6

    def test_recommend_strong_unit(self, advisor, player):
        """Test recommending strong high-cost unit."""
        champion = MockChampion("TFT_5Cost", "Legendary", 5, ["Dragon"])
        player.shop.slots[0] = champion

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert PickReason.STRONG_UNIT in rec.reasons

    def test_recommend_core_carry(self, advisor, player):
        """Test recommending core carry from target comp."""
        champion = MockChampion("TFT_Carry", "Carry", 4, ["Assassin"])
        player.shop.slots[0] = champion

        advice = advisor.analyze(player, target_comp=["TFT_Carry", "TFT_Tank"])

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert PickReason.CORE_CARRY in rec.reasons

    def test_gold_to_save_calculation(self, advisor, player):
        """Test gold to save calculation for interest."""
        player.gold = 48

        advice = advisor.analyze(player)

        # Should recommend saving to 50 for next interest threshold
        assert advice.gold_to_save == 50

    def test_gold_to_save_at_threshold(self, advisor, player):
        """Test gold to save when at threshold."""
        player.gold = 50

        advice = advisor.analyze(player)

        assert advice.gold_to_save == 50

    def test_should_refresh_low_score(self, advisor, player):
        """Test refresh recommendation with low scores."""
        # Empty shop should trigger refresh
        advice = advisor.analyze(player)

        assert advice.should_refresh is True

    def test_multiple_recommendations_sorted(self, advisor, player):
        """Test multiple recommendations are sorted by score."""
        champ1 = MockChampion("TFT_Cheap", "Cheap", 1, ["Warrior"])
        champ2 = MockChampion("TFT_Expensive", "Expensive", 4, ["Dragon"])

        player.shop.slots[0] = champ1
        player.shop.slots[1] = champ2

        # Add copies for upgrade potential
        player.units.bench.append(MockChampionInstance(champ1))
        player.units.bench.append(MockChampionInstance(champ1))

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 2
        # Should be sorted by score descending
        for i in range(len(advice.recommendations) - 1):
            assert advice.recommendations[i].score >= advice.recommendations[i + 1].score

    def test_get_affordable_recommendations(self, advisor, player):
        """Test filtering by affordability."""
        champ1 = MockChampion("TFT_Cheap", "Cheap", 1, ["Warrior"])
        champ2 = MockChampion("TFT_Mid", "Mid", 3, ["Mage"])
        champ3 = MockChampion("TFT_Expensive", "Expensive", 5, ["Dragon"])

        player.shop.slots = [champ1, champ2, champ3, None, None]
        player.gold = 4

        # Add copies for recommendations
        player.units.bench.append(MockChampionInstance(champ1))
        player.units.bench.append(MockChampionInstance(champ2))
        player.units.bench.append(MockChampionInstance(champ3))

        advice = advisor.analyze(player)
        affordable = advisor.get_affordable_recommendations(advice, player.gold)

        # Only cheap and mid should be affordable
        for rec in affordable:
            assert rec.cost <= player.gold

    def test_get_priority_buys(self, advisor, player):
        """Test getting priority buys within budget."""
        champ1 = MockChampion("TFT_Unit1", "Unit1", 2, ["Warrior"])
        champ2 = MockChampion("TFT_Unit2", "Unit2", 3, ["Mage"])

        player.shop.slots = [champ1, champ2, None, None, None]
        player.gold = 10

        player.units.bench.append(MockChampionInstance(champ1))
        player.units.bench.append(MockChampionInstance(champ2))

        advice = advisor.analyze(player)
        priority = advisor.get_priority_buys(advice, player.gold, preserve_interest=False)

        # Should return buys that fit within budget
        total_cost = sum(rec.cost for rec in priority)
        assert total_cost <= player.gold

    def test_get_priority_buys_preserve_interest(self, advisor, player):
        """Test priority buys while preserving interest."""
        champ1 = MockChampion("TFT_Unit1", "Unit1", 2, ["Warrior"])

        player.shop.slots = [champ1, None, None, None, None]
        player.gold = 52

        player.units.bench.append(MockChampionInstance(champ1))

        advice = advisor.analyze(player)
        priority = advisor.get_priority_buys(advice, player.gold, preserve_interest=True)

        # Should preserve 50 gold for interest
        total_cost = sum(rec.cost for rec in priority)
        assert total_cost <= 2  # Can only spend 2 gold to stay at 50

    def test_copies_needed_calculation(self, advisor, player):
        """Test copies needed for upgrade calculation."""
        champion = MockChampion("TFT_Test", "Test", 3, ["Warrior"])
        player.shop.slots[0] = champion

        # 1 copy owned - need 2 more
        player.units.bench.append(MockChampionInstance(champion))

        advice = advisor.analyze(player)

        assert len(advice.recommendations) >= 1
        rec = advice.recommendations[0]
        assert rec.copies_owned == 1
        assert rec.copies_needed == 2

    def test_score_adjusted_by_cost(self, advisor, player):
        """Test that scores are adjusted for cost efficiency."""
        cheap = MockChampion("TFT_Cheap", "Cheap", 1, ["Warrior"])
        expensive = MockChampion("TFT_Expensive", "Expensive", 5, ["Warrior"])

        player.shop.slots[0] = cheap
        player.shop.slots[1] = expensive

        # Same upgrade potential
        player.units.bench.append(MockChampionInstance(cheap))
        player.units.bench.append(MockChampionInstance(expensive))

        advice = advisor.analyze(player)

        # Cheap unit should have higher score due to cost efficiency
        if len(advice.recommendations) >= 2:
            cheap_rec = next(r for r in advice.recommendations if r.cost == 1)
            expensive_rec = next(r for r in advice.recommendations if r.cost == 5)
            # With same raw score, cheap should be 5x more efficient
            assert cheap_rec.score > expensive_rec.score


class TestPickReason:
    """Tests for PickReason enum."""

    def test_all_reasons_exist(self):
        """Test all expected reasons exist."""
        expected = [
            "UPGRADE_2STAR",
            "UPGRADE_3STAR",
            "SYNERGY_ACTIVATE",
            "SYNERGY_UPGRADE",
            "CORE_CARRY",
            "STRONG_UNIT",
            "ECONOMY_PAIR",
            "PIVOT_OPTION",
        ]

        for name in expected:
            assert hasattr(PickReason, name)


class TestPickRecommendation:
    """Tests for PickRecommendation dataclass."""

    def test_create_recommendation(self):
        """Test creating a recommendation."""
        rec = PickRecommendation(
            champion_id="TFT_Test",
            champion_name="Test",
            shop_index=0,
            score=50.0,
            reasons=[PickReason.UPGRADE_2STAR],
            synergy_delta={"Warrior": 1},
            cost=3,
            copies_owned=2,
            copies_needed=1,
        )

        assert rec.champion_id == "TFT_Test"
        assert rec.score == 50.0
        assert PickReason.UPGRADE_2STAR in rec.reasons
