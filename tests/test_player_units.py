"""Tests for Player Units functionality."""

import pytest

from src.data.loaders import get_champion_by_id
from src.core.player_units import ChampionInstance, PlayerUnits
from src.core.constants import BENCH_SIZE


@pytest.fixture
def player_units():
    """Create a fresh PlayerUnits for testing."""
    return PlayerUnits()


@pytest.fixture
def garen():
    """Get Garen champion data."""
    return get_champion_by_id("garen")


@pytest.fixture
def jinx():
    """Get Jinx champion data."""
    return get_champion_by_id("jinx")


class TestChampionInstance:
    """Tests for ChampionInstance class."""

    def test_instance_creation(self, garen):
        """Can create a champion instance."""
        instance = ChampionInstance(champion=garen)

        assert instance.champion == garen
        assert instance.star_level == 1
        assert len(instance.items) == 0
        assert instance.position is None

    def test_get_stats(self, garen):
        """Get stats returns correct values."""
        instance = ChampionInstance(champion=garen)
        stats = instance.get_stats()

        assert stats["health"] == garen.stats.health[0]  # 1-star health
        assert stats["attack_damage"] == garen.stats.attack_damage[0]

    def test_get_stats_2_star(self, garen):
        """2-star stats are correct."""
        instance = ChampionInstance(champion=garen, star_level=2)
        stats = instance.get_stats()

        assert stats["health"] == garen.stats.health[1]  # 2-star health
        assert stats["attack_damage"] == garen.stats.attack_damage[1]

    def test_sell_value_1_star(self, garen):
        """1-star sell value equals cost."""
        instance = ChampionInstance(champion=garen)
        assert instance.get_sell_value() == garen.cost

    def test_sell_value_2_star(self, garen):
        """2-star sell value is cost * 3."""
        instance = ChampionInstance(champion=garen, star_level=2)
        assert instance.get_sell_value() == garen.cost * 3

    def test_sell_value_3_star(self, garen):
        """3-star sell value is cost * 9."""
        instance = ChampionInstance(champion=garen, star_level=3)
        assert instance.get_sell_value() == garen.cost * 9

    def test_can_add_item(self, garen):
        """Can add up to 3 items."""
        instance = ChampionInstance(champion=garen)

        assert instance.can_add_item() is True
        assert instance.max_items == 3

    def test_repr(self, garen):
        """String representation is readable."""
        instance = ChampionInstance(champion=garen, star_level=2)
        repr_str = repr(instance)

        assert "Garen" in repr_str
        assert "★★" in repr_str


class TestPlayerUnitsBasic:
    """Tests for basic PlayerUnits operations."""

    def test_initial_state(self, player_units):
        """PlayerUnits starts empty."""
        assert player_units.get_total_units() == 0
        assert player_units.get_bench_count() == 0
        assert player_units.get_board_count() == 0

    def test_add_to_bench(self, player_units, garen):
        """Can add champion to bench."""
        instance = player_units.add_to_bench(garen)

        assert instance is not None
        assert instance.champion == garen
        assert player_units.get_bench_count() == 1

    def test_bench_full(self, player_units, garen, jinx):
        """Cannot add to full bench."""
        # Fill bench with different champions to avoid auto-upgrade
        from src.data.loaders import load_champions
        champions = [c for c in load_champions() if not c.is_unlockable][:BENCH_SIZE]

        for champ in champions:
            player_units.add_to_bench(champ)

        # Try to add one more
        result = player_units.add_to_bench(garen)
        assert result is None
        assert player_units.get_bench_count() == BENCH_SIZE

    def test_has_bench_space(self, player_units, garen):
        """has_bench_space returns correct value."""
        assert player_units.has_bench_space() is True

        # Fill bench with different champions to avoid auto-upgrade
        from src.data.loaders import load_champions
        champions = [c for c in load_champions() if not c.is_unlockable][:BENCH_SIZE]

        for champ in champions:
            player_units.add_to_bench(champ)

        assert player_units.has_bench_space() is False


class TestPlayerUnitsUpgrade:
    """Tests for champion upgrade functionality."""

    def test_upgrade_to_2_star(self, player_units, garen):
        """3 copies combine into 2-star."""
        # Add 3 copies
        for _ in range(3):
            player_units.add_to_bench(garen)

        # Should auto-upgrade to 2-star
        instances = player_units.get_all_instances()
        assert len(instances) == 1
        assert instances[0].star_level == 2

    def test_can_upgrade_check(self, player_units, garen):
        """can_upgrade returns correct value."""
        # Add 2 copies
        player_units.add_to_bench(garen)
        player_units.add_to_bench(garen)

        assert player_units.can_upgrade(garen.id) is False

        # Add 3rd copy
        player_units.add_to_bench(garen)

        # After auto-upgrade, we have a 2-star and can't upgrade further with just that
        assert player_units.can_upgrade(garen.id) is False

    def test_upgrade_to_3_star(self, player_units, garen):
        """9 copies combine into 3-star."""
        # Add 9 copies (will create 3x 2-star, then 1x 3-star)
        for _ in range(9):
            player_units.add_to_bench(garen)

        instances = player_units.get_all_instances()
        assert len(instances) == 1
        assert instances[0].star_level == 3

    def test_upgrade_preserves_items(self, player_units, garen):
        """Items are preserved during upgrade."""
        from src.data.loaders import get_item_by_id

        # Add champion with item
        instance1 = player_units.add_to_bench(garen)
        item = get_item_by_id("bf_sword")
        if item:
            instance1.add_item(item)

        # Add 2 more copies
        player_units.add_to_bench(garen)
        player_units.add_to_bench(garen)

        # Check upgraded unit has the item
        instances = player_units.get_all_instances()
        assert len(instances) == 1
        assert len(instances[0].items) >= 1


class TestPlayerUnitsSell:
    """Tests for selling champions."""

    def test_sell_returns_gold(self, player_units, garen):
        """Selling returns correct gold value."""
        instance = player_units.add_to_bench(garen)
        gold = player_units.sell(instance)

        assert gold == garen.cost
        assert player_units.get_total_units() == 0

    def test_sell_from_bench(self, player_units, garen):
        """Can sell from specific bench slot."""
        player_units.add_to_bench(garen)

        instance, gold = player_units.sell_from_bench(0)
        assert instance is not None
        assert gold == garen.cost

    def test_sell_empty_bench_slot(self, player_units):
        """Selling empty slot returns None, 0."""
        instance, gold = player_units.sell_from_bench(0)
        assert instance is None
        assert gold == 0


class TestPlayerUnitsBoard:
    """Tests for board placement."""

    def test_place_on_board(self, player_units, garen):
        """Can place unit on board."""
        instance = player_units.add_to_bench(garen)
        result = player_units.place_on_board(instance, 0, 0)

        assert result is True
        assert instance.position == (0, 0)
        assert player_units.get_board_count() == 1
        assert player_units.get_bench_count() == 0

    def test_place_on_occupied_position(self, player_units, garen, jinx):
        """Cannot place on occupied position."""
        instance1 = player_units.add_to_bench(garen)
        player_units.place_on_board(instance1, 0, 0)

        instance2 = player_units.add_to_bench(jinx)
        result = player_units.place_on_board(instance2, 0, 0)

        assert result is False
        assert player_units.get_bench_count() == 1

    def test_place_out_of_bounds(self, player_units, garen):
        """Cannot place outside board bounds."""
        instance = player_units.add_to_bench(garen)

        result = player_units.place_on_board(instance, 10, 10)
        assert result is False

    def test_remove_from_board(self, player_units, garen):
        """Can remove unit from board."""
        instance = player_units.add_to_bench(garen)
        player_units.place_on_board(instance, 0, 0)

        removed = player_units.remove_from_board(0, 0)

        assert removed is not None
        assert removed == instance
        assert player_units.get_board_count() == 0
        assert player_units.get_bench_count() == 1


class TestPlayerUnitsClear:
    """Tests for clearing all units."""

    def test_clear_returns_all_instances(self, player_units, garen, jinx):
        """Clear returns all owned instances."""
        player_units.add_to_bench(garen)
        player_units.add_to_bench(jinx)

        instances = player_units.clear()

        assert len(instances) == 2
        assert player_units.get_total_units() == 0
