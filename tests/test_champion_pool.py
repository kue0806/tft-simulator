"""Tests for Champion Pool functionality."""

import pytest

from src.data.loaders import load_champions
from src.core.champion_pool import ChampionPool
from src.core.constants import POOL_SIZE


@pytest.fixture
def pool():
    """Create a fresh champion pool for testing."""
    champions = load_champions()
    return ChampionPool(champions)


class TestChampionPoolInitialization:
    """Tests for pool initialization."""

    def test_pool_initialization(self, pool):
        """Verify correct initial counts for each cost tier."""
        # Check a 1-cost champion
        assert pool.get_available("anivia") == POOL_SIZE[1]  # 30

        # Check a 4-cost champion
        assert pool.get_available("garen") == POOL_SIZE[4]  # 10

        # Check a 5-cost champion
        assert pool.get_available("azir") == POOL_SIZE[5]  # 9

    def test_pool_excludes_unlockables(self, pool):
        """Verify unlockable champions start outside the pool."""
        # Baron Nashor is unlockable
        assert pool.get_available("baron_nashor") == 0

    def test_pool_has_champions_data(self, pool):
        """Verify champion data is accessible."""
        garen = pool.get_champion("garen")
        assert garen is not None
        assert garen.name == "Garen"
        assert garen.cost == 4

    def test_champions_by_cost(self, pool):
        """Verify champions are grouped by cost correctly."""
        one_costs = pool.get_champions_by_cost(1)
        assert len(one_costs) > 0
        assert all(c.cost == 1 for c in one_costs)


class TestChampionPoolTakeReturn:
    """Tests for taking and returning champions."""

    def test_take_champion(self, pool):
        """Taking reduces pool count."""
        initial = pool.get_available("jinx")
        taken = pool.take("jinx", 1)

        assert taken == 1
        assert pool.get_available("jinx") == initial - 1

    def test_take_multiple(self, pool):
        """Taking multiple reduces pool count correctly."""
        initial = pool.get_available("jinx")
        taken = pool.take("jinx", 3)

        assert taken == 3
        assert pool.get_available("jinx") == initial - 3

    def test_return_champion(self, pool):
        """Returning increases pool count."""
        # First take some
        pool.take("jinx", 3)
        after_take = pool.get_available("jinx")

        # Then return
        pool.return_champion("jinx", 2)
        assert pool.get_available("jinx") == after_take + 2

    def test_pool_depletion(self, pool):
        """Cannot take more than available."""
        # Take all jinx (should be 10 for 3-cost)
        initial = pool.get_available("jinx")
        taken = pool.take("jinx", initial + 5)

        assert taken == initial
        assert pool.get_available("jinx") == 0

    def test_take_depleted_returns_zero(self, pool):
        """Taking from depleted pool returns 0."""
        # Deplete the pool
        pool.take("jinx", pool.get_available("jinx"))

        # Try to take more
        taken = pool.take("jinx", 1)
        assert taken == 0

    def test_return_does_not_exceed_max(self, pool):
        """Returning cannot exceed max pool size."""
        initial = pool.get_available("jinx")

        # Try to return more than max
        pool.return_champion("jinx", 100)
        assert pool.get_available("jinx") == initial

    def test_take_nonexistent_champion(self, pool):
        """Taking nonexistent champion returns 0."""
        taken = pool.take("nonexistent_champion", 1)
        assert taken == 0


class TestChampionPoolProbability:
    """Tests for probability calculations."""

    def test_probability_calculation(self, pool):
        """Verify probability math is correct."""
        # At level 7, 3-cost odds are 40%
        # If there's equal distribution, probability should be reasonable
        prob = pool.get_probability("jinx", level=7)

        assert 0 < prob < 1

    def test_probability_zero_for_depleted(self, pool):
        """Depleted champions have zero probability."""
        pool.take("jinx", pool.get_available("jinx"))
        prob = pool.get_probability("jinx", level=7)

        assert prob == 0.0

    def test_probability_zero_for_wrong_level(self, pool):
        """5-costs have 0% at low levels."""
        # At level 3, 5-cost odds are 0%
        prob = pool.get_probability("azir", level=3)
        assert prob == 0.0

    def test_probability_increases_with_level_for_high_costs(self, pool):
        """Higher level = higher probability for high cost units."""
        prob_7 = pool.get_probability("garen", level=7)
        prob_9 = pool.get_probability("garen", level=9)

        assert prob_9 > prob_7

    def test_total_available_by_cost(self, pool):
        """Total available calculation is correct."""
        total_1cost = pool.get_total_available_by_cost(1)
        num_1cost_champs = len(pool.get_champions_by_cost(1))

        assert total_1cost == num_1cost_champs * POOL_SIZE[1]


class TestChampionPoolUnlockables:
    """Tests for unlockable champion functionality."""

    def test_add_unlockable_to_pool(self, pool):
        """Can add unlockable champions to pool."""
        from src.data.loaders import get_champion_by_id

        # Get an unlockable champion
        baron = get_champion_by_id("baron_nashor")
        assert baron is not None
        assert baron.is_unlockable

        # Initially not in pool
        assert pool.get_available("baron_nashor") == 0

        # Add to pool
        pool.add_unlockable_to_pool(baron)

        # Now should be available
        assert pool.get_available("baron_nashor") == POOL_SIZE.get(baron.cost, 0)


class TestChampionPoolDetailedOdds:
    """Tests for detailed odds calculation."""

    def test_calculate_champion_odds(self, pool):
        """Detailed odds calculation returns expected format."""
        odds = pool.calculate_champion_odds("jinx", player_level=7)

        assert "single_slot" in odds
        assert "any_slot" in odds
        assert "copies_remaining" in odds
        assert "tier_total" in odds

        assert odds["single_slot"] > 0
        assert odds["any_slot"] > odds["single_slot"]  # any_slot >= single_slot
        assert odds["copies_remaining"] == POOL_SIZE[3]  # jinx is 3-cost

    def test_odds_decrease_when_pool_depleted(self, pool):
        """Odds decrease as copies are taken from pool."""
        initial_odds = pool.calculate_champion_odds("jinx", player_level=7)

        # Take half the copies
        pool.take("jinx", 9)

        after_odds = pool.calculate_champion_odds("jinx", player_level=7)

        assert after_odds["single_slot"] < initial_odds["single_slot"]
        assert after_odds["any_slot"] < initial_odds["any_slot"]
        assert after_odds["copies_remaining"] == POOL_SIZE[3] - 9

    def test_odds_zero_when_depleted(self, pool):
        """Odds are zero when champion is completely depleted."""
        pool.take("jinx", pool.get_available("jinx"))

        odds = pool.calculate_champion_odds("jinx", player_level=7)

        assert odds["single_slot"] == 0
        assert odds["any_slot"] == 0
        assert odds["copies_remaining"] == 0

    def test_odds_zero_for_wrong_level(self, pool):
        """Odds are zero when level can't roll that tier."""
        # 5-cost at level 3 should be 0
        odds = pool.calculate_champion_odds("azir", player_level=3)

        assert odds["single_slot"] == 0
        assert odds["any_slot"] == 0


class TestChampionPoolContested:
    """Tests for contested champion detection."""

    def test_is_contested_false_for_full_pool(self, pool):
        """Champions with full pool are not contested."""
        assert pool.is_contested("jinx") is False

    def test_is_contested_true_when_depleted(self, pool):
        """Champions are contested when many copies taken."""
        # Take more than 50% of copies
        pool.take("jinx", 10)  # 10 out of 18

        assert pool.is_contested("jinx") is True

    def test_is_contested_threshold(self, pool):
        """Custom threshold works correctly."""
        # Take 6 out of 18 (33%)
        pool.take("jinx", 6)

        # At 50% threshold, not contested
        assert pool.is_contested("jinx", threshold=0.5) is False

        # At 30% threshold, is contested
        assert pool.is_contested("jinx", threshold=0.3) is True


class TestChampionPoolCopiesNeeded:
    """Tests for copies needed calculation."""

    def test_copies_needed_3star(self, pool):
        """3-star requires 9 copies."""
        assert pool.calculate_copies_needed("jinx", target_star=3) == 9

    def test_copies_needed_2star(self, pool):
        """2-star requires 3 copies."""
        assert pool.calculate_copies_needed("jinx", target_star=2) == 3


class TestChampionPoolUtilities:
    """Tests for utility methods."""

    def test_get_max_pool_size(self, pool):
        """Max pool size returns correct values."""
        assert pool.get_max_pool_size(1) == 30
        assert pool.get_max_pool_size(3) == 18
        assert pool.get_max_pool_size(5) == 9

    def test_get_champion_count_by_cost(self, pool):
        """Champion count by cost returns positive value."""
        count = pool.get_champion_count_by_cost(1)
        assert count > 0

    def test_get_expected_pool_total(self, pool):
        """Expected pool total calculation is correct."""
        # For 1-cost: num_champions * 30
        expected = pool.get_champion_count_by_cost(1) * 30
        assert pool.get_expected_pool_total(1) == expected

    def test_pool_state_snapshot(self, pool):
        """Pool state returns copy of current counts."""
        state = pool.get_pool_state()

        assert isinstance(state, dict)
        assert "jinx" in state
        assert state["jinx"] == POOL_SIZE[3]

        # Modifying returned state doesn't affect pool
        state["jinx"] = 0
        assert pool.get_available("jinx") == POOL_SIZE[3]
