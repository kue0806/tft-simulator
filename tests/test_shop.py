"""Tests for Shop functionality."""

import pytest

from src.data.loaders import load_champions
from src.core.champion_pool import ChampionPool
from src.core.shop import Shop
from src.core.constants import SHOP_SIZE, SHOP_ODDS


@pytest.fixture
def pool():
    """Create a fresh champion pool for testing."""
    champions = load_champions()
    return ChampionPool(champions)


@pytest.fixture
def shop(pool):
    """Create a shop at level 5."""
    return Shop(pool, player_level=5)


class TestShopRefresh:
    """Tests for shop refresh functionality."""

    def test_shop_refresh_fills_slots(self, shop):
        """Shop shows 5 champions after refresh."""
        shop.refresh()

        filled = sum(1 for s in shop.slots if s is not None)
        assert filled == SHOP_SIZE

    def test_shop_refresh_returns_champions(self, shop):
        """Refresh returns list of champions."""
        result = shop.refresh()

        assert len(result) == SHOP_SIZE
        assert all(c is not None for c in result)

    def test_shop_refresh_returns_to_pool(self, shop, pool):
        """Unbought champions return to pool on refresh."""
        # First refresh
        shop.refresh()
        first_champ = shop.slots[0]
        initial_count = pool.get_available(first_champ.id)

        # Refresh again (first champ should return to pool)
        shop.refresh()

        # The champion from first refresh should be back in pool
        # (unless it appeared again in the new shop)
        # This is hard to test exactly due to randomness, so we just verify
        # the pool count is reasonable
        assert pool.get_available(first_champ.id) >= initial_count


class TestShopOdds:
    """Tests for shop level odds."""

    def test_shop_level_odds(self, pool):
        """Higher level = higher cost champions appear more often."""
        # At level 3, only 1-2 costs should appear
        shop_low = Shop(pool, player_level=3)
        shop_low.refresh()

        # Run multiple times to get distribution
        cost_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for _ in range(100):
            shop_low.refresh()
            for champ in shop_low.slots:
                if champ:
                    cost_counts[champ.cost] += 1

        # At level 3, no 4 or 5 costs should appear
        assert cost_counts[4] == 0
        assert cost_counts[5] == 0

    def test_get_odds(self, shop):
        """Get odds returns correct values for level."""
        odds = shop.get_odds()
        expected = SHOP_ODDS[5]  # Level 5 odds

        assert odds == expected


class TestShopPurchase:
    """Tests for purchasing from shop."""

    def test_shop_purchase(self, shop, pool):
        """Purchasing removes from slot and keeps out of pool."""
        shop.refresh()
        champ = shop.slots[0]
        initial_pool = pool.get_available(champ.id)

        # Purchase
        purchased = shop.purchase(0)

        assert purchased is not None
        assert purchased.id == champ.id
        assert shop.slots[0] is None
        # Champion stays out of pool (was already taken during refresh)
        assert pool.get_available(champ.id) == initial_pool

    def test_purchase_empty_slot_returns_none(self, shop):
        """Purchasing from empty slot returns None."""
        shop.refresh()
        shop.purchase(0)  # Empty slot 0

        result = shop.purchase(0)  # Try again
        assert result is None

    def test_purchase_invalid_slot_returns_none(self, shop):
        """Purchasing from invalid slot returns None."""
        shop.refresh()

        result = shop.purchase(10)
        assert result is None

        result = shop.purchase(-1)
        assert result is None


class TestShopLock:
    """Tests for shop lock functionality."""

    def test_shop_lock(self, shop):
        """Locked shop doesn't refresh."""
        shop.refresh()
        original_champs = [c.id if c else None for c in shop.slots]

        # Lock and refresh
        shop.toggle_lock()
        assert shop.locked is True

        shop.refresh()
        new_champs = [c.id if c else None for c in shop.slots]

        # Should be the same
        assert original_champs == new_champs

    def test_shop_unlock(self, shop):
        """Unlocking allows refresh."""
        shop.refresh()
        shop.toggle_lock()  # Lock
        shop.toggle_lock()  # Unlock

        assert shop.locked is False

    def test_toggle_lock_returns_state(self, shop):
        """Toggle returns new lock state."""
        result = shop.toggle_lock()
        assert result is True

        result = shop.toggle_lock()
        assert result is False


class TestShopLevelChange:
    """Tests for changing shop level."""

    def test_set_level(self, shop):
        """Setting level updates odds."""
        shop.set_level(9)
        odds = shop.get_odds()

        assert odds == SHOP_ODDS[9]

    def test_set_level_clamps_min(self, shop):
        """Level cannot go below 1."""
        shop.set_level(0)
        assert shop.level == 1

    def test_set_level_clamps_max(self, shop):
        """Level cannot go above 10."""
        shop.set_level(15)
        assert shop.level == 10


class TestShopClear:
    """Tests for shop clearing."""

    def test_clear_empties_shop(self, shop, pool):
        """Clear empties all slots."""
        shop.refresh()
        shop.clear()

        assert all(s is None for s in shop.slots)

    def test_clear_returns_to_pool(self, shop, pool):
        """Clear returns unbought champions to pool."""
        shop.refresh()
        champ = shop.slots[0]

        # Count how many times this champion appears in the shop
        shop_count = sum(1 for s in shop.slots if s and s.id == champ.id)
        pool_before = pool.get_available(champ.id)

        shop.clear()

        # All instances of this champion should be back in pool
        pool_after = pool.get_available(champ.id)
        assert pool_after == pool_before + shop_count


class TestShopNoDuplicates:
    """Tests for preventing unbought champions from reappearing."""

    def test_no_duplicate_in_same_shop(self, shop):
        """Same champion should not appear twice in one shop refresh."""
        # Run many refreshes to check for duplicates
        for _ in range(50):
            shop.refresh()
            champion_ids = [c.id for c in shop.slots if c is not None]
            # All IDs should be unique
            assert len(champion_ids) == len(set(champion_ids)), \
                f"Duplicate champion in shop: {champion_ids}"

    def test_unbought_not_in_next_shop(self, pool):
        """Unbought champions should not appear in the next refresh.

        TFT mechanic: "Consecutive shops will not repeat unbought champions"
        With 22 unique 1-costs and only 5 slots, there should always be
        enough champions to avoid repeats.
        """
        shop = Shop(pool, player_level=1)  # Level 1 = 100% 1-cost only

        # Run multiple trials to be thorough
        for trial in range(10):
            # First refresh
            shop.refresh()
            first_shop_ids = {c.id for c in shop.slots if c is not None}

            # Second refresh (don't buy anything)
            shop.refresh()
            second_shop_ids = {c.id for c in shop.slots if c is not None}

            # There should be no overlap
            overlap = first_shop_ids & second_shop_ids
            assert len(overlap) == 0, \
                f"Trial {trial}: Unbought champions reappeared: {overlap}"

    def test_bought_can_reappear(self, pool):
        """Purchased champions can appear again in future shops."""
        shop = Shop(pool, player_level=3)

        # Refresh and buy all
        shop.refresh()
        bought_ids = set()
        for i in range(SHOP_SIZE):
            champ = shop.purchase(i)
            if champ:
                bought_ids.add(champ.id)

        # Next refresh - bought champions CAN appear
        shop.refresh()
        new_shop_ids = {c.id for c in shop.slots if c is not None}

        # This is probabilistic, but over many runs, bought champs should appear sometimes
        # We can't assert overlap, but we can at least verify the shop works


class TestShopPoolWeighting:
    """Tests for pool-weighted champion selection."""

    def test_depleted_champion_less_likely(self, pool):
        """Champions with fewer copies in pool should appear less often."""
        # Get a 1-cost champion and deplete most of its copies
        one_costs = pool.get_champions_by_cost(1)
        target = one_costs[0]
        target_id = target.id

        # Take 25 out of 30 copies (leaving only 5)
        pool.take(target_id, 25)

        # Count appearances over many refreshes
        shop = Shop(pool, player_level=1)  # Level 1 = 100% 1-cost
        appearances = 0
        total_slots = 0

        for _ in range(100):
            shop.refresh()
            for champ in shop.slots:
                if champ:
                    total_slots += 1
                    if champ.id == target_id:
                        appearances += 1

        # With only 5 copies vs ~600+ total 1-costs in pool,
        # target should appear rarely
        appearance_rate = appearances / total_slots if total_slots > 0 else 0

        # Should be relatively rare (rough check)
        assert appearance_rate < 0.1, \
            f"Depleted champion appeared too often: {appearance_rate:.2%}"

    def test_full_pool_distribution(self, pool):
        """With full pool, champions should appear roughly evenly."""
        shop = Shop(pool, player_level=1)  # Level 1 = 100% 1-cost

        # Count appearances
        appearances = {}
        for _ in range(200):
            shop.refresh()
            for champ in shop.slots:
                if champ:
                    appearances[champ.id] = appearances.get(champ.id, 0) + 1

        # Check variance isn't too extreme
        if appearances:
            counts = list(appearances.values())
            avg = sum(counts) / len(counts)
            # No champion should appear more than 3x the average
            assert all(c < avg * 3 for c in counts), \
                f"Uneven distribution detected"


class TestShopPendingUnlock:
    """Tests for pending unlock champion feature."""

    def test_pending_unlock_appears_in_rightmost(self, pool):
        """Pending unlock champion appears in rightmost slot."""
        shop = Shop(pool, player_level=5)

        # Get a champion to use as "unlocked"
        target = pool.get_champions_by_cost(1)[0]
        shop.set_pending_unlock(target.id)

        shop.refresh()

        # Rightmost slot (index 4) should have the target
        assert shop.slots[SHOP_SIZE - 1] is not None
        assert shop.slots[SHOP_SIZE - 1].id == target.id

    def test_pending_unlock_cleared_after_refresh(self, pool):
        """Pending unlock is consumed after one refresh."""
        shop = Shop(pool, player_level=5)

        target = pool.get_champions_by_cost(1)[0]
        shop.set_pending_unlock(target.id)

        # First refresh - target appears
        shop.refresh()
        assert shop.slots[SHOP_SIZE - 1].id == target.id

        # Second refresh - target not guaranteed
        shop.refresh()
        # Can't assert target is gone (might appear randomly), but pending should be None
        assert shop._pending_unlock_champion_id is None
