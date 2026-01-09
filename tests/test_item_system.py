"""Tests for the Item System."""

import pytest

from src.data.loaders import get_champion_by_id, get_item_by_id, load_components
from src.core.item_manager import ItemManager, ItemInstance
from src.core.stat_calculator import StatCalculator, CalculatedStats
from src.core.bis_calculator import BiSCalculator, ItemRecommendation
from src.core.player_units import ChampionInstance


@pytest.fixture
def item_manager():
    """Create a fresh ItemManager."""
    return ItemManager()


@pytest.fixture
def stat_calculator():
    """Create a fresh StatCalculator."""
    return StatCalculator()


@pytest.fixture
def bis_calculator():
    """Create a fresh BiSCalculator."""
    return BiSCalculator()


@pytest.fixture
def jinx():
    """Get Jinx champion for testing (AD carry)."""
    return get_champion_by_id("jinx")


@pytest.fixture
def garen():
    """Get Garen champion for testing (tank)."""
    return get_champion_by_id("garen")


@pytest.fixture
def bf_sword():
    """Get BF Sword component."""
    return get_item_by_id("bf_sword")


@pytest.fixture
def recurve_bow():
    """Get Recurve Bow component."""
    return get_item_by_id("recurve_bow")


@pytest.fixture
def chain_vest():
    """Get Chain Vest component."""
    return get_item_by_id("chain_vest")


class TestItemManager:
    """Tests for ItemManager functionality."""

    def test_add_to_inventory(self, item_manager, bf_sword):
        """Add item to inventory."""
        instance = item_manager.add_to_inventory(bf_sword)

        assert instance is not None
        assert instance.item == bf_sword
        assert instance in item_manager.inventory
        assert item_manager.get_inventory_count() == 1

    def test_remove_from_inventory(self, item_manager, bf_sword):
        """Remove item from inventory."""
        instance = item_manager.add_to_inventory(bf_sword)
        result = item_manager.remove_from_inventory(instance)

        assert result is True
        assert instance not in item_manager.inventory
        assert item_manager.get_inventory_count() == 0

    def test_equip_item(self, item_manager, jinx, bf_sword):
        """Equip item to champion."""
        champion = ChampionInstance(champion=jinx)
        instance = item_manager.add_to_inventory(bf_sword)

        result = item_manager.equip_item(instance, champion)

        assert result is True
        assert instance in champion.items
        assert instance not in item_manager.inventory

    def test_max_items_limit(self, item_manager, jinx):
        """Cannot equip more than 3 items."""
        champion = ChampionInstance(champion=jinx)

        # Equip 3 combined items (so they don't auto-combine)
        combined_ids = ["giant_slayer", "infinity_edge", "bloodthirster"]
        for item_id in combined_ids:
            item = get_item_by_id(item_id)
            instance = item_manager.add_to_inventory(item)
            item_manager.equip_item(instance, champion)

        assert len(champion.items) == 3

        # Try to equip 4th
        fourth_item = get_item_by_id("rabadons_deathcap")
        instance4 = item_manager.add_to_inventory(fourth_item)
        result = item_manager.equip_item(instance4, champion)

        assert result is False
        assert len(champion.items) == 3

    def test_auto_combine(self, item_manager, jinx, bf_sword, recurve_bow):
        """Components auto-combine when equipped."""
        champion = ChampionInstance(champion=jinx)

        # Add first component
        bf_instance = item_manager.add_to_inventory(bf_sword)
        item_manager.equip_item(bf_instance, champion)

        # Add second component - should auto-combine into Giant Slayer
        bow_instance = item_manager.add_to_inventory(recurve_bow)
        item_manager.equip_item(bow_instance, champion)

        # Should have 1 combined item, not 2 components
        assert len(champion.items) == 1
        assert champion.items[0].item.id == "giant_slayer"

    def test_try_combine_manual(self, item_manager, bf_sword, recurve_bow):
        """Manually combine two components."""
        bf_instance = item_manager.add_to_inventory(bf_sword)
        bow_instance = item_manager.add_to_inventory(recurve_bow)

        combined = item_manager.try_combine(bf_instance, bow_instance)

        assert combined is not None
        assert combined.item.id == "giant_slayer"
        assert item_manager.get_inventory_count() == 1

    def test_try_combine_invalid(self, item_manager, bf_sword):
        """Cannot combine same component twice (unless valid recipe)."""
        bf1 = item_manager.add_to_inventory(bf_sword)
        bf2 = item_manager.add_to_inventory(bf_sword)

        combined = item_manager.try_combine(bf1, bf2)

        # BF + BF = Deathblade (valid recipe)
        assert combined is not None
        assert combined.item.id == "deathblade"

    def test_get_available_recipes(self, item_manager, bf_sword, recurve_bow):
        """Get all buildable recipes from inventory."""
        item_manager.add_to_inventory(bf_sword)
        item_manager.add_to_inventory(recurve_bow)

        recipes = item_manager.get_available_recipes()

        assert len(recipes) == 1
        assert recipes[0][2].id == "giant_slayer"

    def test_can_build_item(self, item_manager, bf_sword, recurve_bow):
        """Check if item can be built."""
        item_manager.add_to_inventory(bf_sword)
        item_manager.add_to_inventory(recurve_bow)

        assert item_manager.can_build_item("giant_slayer") is True
        assert item_manager.can_build_item("infinity_edge") is False

    def test_unequip_item(self, item_manager, jinx, bf_sword):
        """Unequip item from champion."""
        champion = ChampionInstance(champion=jinx)
        instance = item_manager.add_to_inventory(bf_sword)
        item_manager.equip_item(instance, champion)

        result = item_manager.unequip_item(instance, champion)

        assert result is True
        assert instance not in champion.items
        assert instance in item_manager.inventory


class TestStatCalculator:
    """Tests for StatCalculator functionality."""

    def test_base_stats_1_star(self, stat_calculator, jinx):
        """Calculate 1-star base stats."""
        champion = ChampionInstance(champion=jinx, star_level=1)

        stats = stat_calculator.calculate_stats(champion)

        assert stats.max_health == jinx.stats.health[0]
        assert stats.attack_damage == jinx.stats.attack_damage[0]
        assert stats.attack_speed == jinx.stats.attack_speed

    def test_base_stats_2_star(self, stat_calculator, jinx):
        """2-star has scaled stats."""
        champion = ChampionInstance(champion=jinx, star_level=2)

        stats = stat_calculator.calculate_stats(champion)

        assert stats.max_health == jinx.stats.health[1]
        assert stats.attack_damage == jinx.stats.attack_damage[1]

    def test_base_stats_3_star(self, stat_calculator, jinx):
        """3-star has scaled stats."""
        champion = ChampionInstance(champion=jinx, star_level=3)

        stats = stat_calculator.calculate_stats(champion)

        assert stats.max_health == jinx.stats.health[2]
        assert stats.attack_damage == jinx.stats.attack_damage[2]

    def test_item_stats_applied(self, stat_calculator, jinx, bf_sword):
        """Item stats add to champion stats."""
        champion = ChampionInstance(champion=jinx)
        champion.items.append(bf_sword)

        stats = stat_calculator.calculate_stats(champion)

        expected_ad = jinx.stats.attack_damage[0] + bf_sword.stats.ad
        assert stats.attack_damage == expected_ad

    def test_multiple_items_stack(self, stat_calculator, jinx, bf_sword):
        """Multiple item stats stack correctly."""
        champion = ChampionInstance(champion=jinx)
        champion.items.append(bf_sword)
        champion.items.append(bf_sword)  # Two BF Swords

        stats = stat_calculator.calculate_stats(champion)

        expected_ad = jinx.stats.attack_damage[0] + (bf_sword.stats.ad * 2)
        assert stats.attack_damage == expected_ad

    def test_trait_bonuses_applied(self, stat_calculator, jinx):
        """Trait bonuses add to stats."""
        champion = ChampionInstance(champion=jinx)
        trait_bonuses = {"armor": 20, "magic_resist": 20}

        stats = stat_calculator.calculate_stats(champion, trait_bonuses)

        assert stats.armor == jinx.stats.armor + 20
        assert stats.magic_resist == jinx.stats.magic_resist + 20

    def test_effective_health(self, stat_calculator, garen):
        """EHP calculation with armor/MR."""
        champion = ChampionInstance(champion=garen)

        stats = stat_calculator.calculate_stats(champion)
        ehp = stat_calculator.get_effective_health(stats)

        # EHP should be higher than raw HP due to armor/MR
        assert ehp > stats.max_health

    def test_dps_calculation(self, stat_calculator, jinx):
        """DPS calculation with crit."""
        champion = ChampionInstance(champion=jinx)

        stats = stat_calculator.calculate_stats(champion)
        dps = stat_calculator.get_dps(stats)

        # DPS should be positive
        assert dps > 0

        # Basic DPS formula check
        expected_base = stats.attack_damage * stats.attack_speed
        assert dps >= expected_base  # Crit should increase DPS


class TestBiSCalculator:
    """Tests for BiSCalculator functionality."""

    def test_determine_role_tank(self, bis_calculator, garen):
        """Tank traits -> tank role."""
        champion = ChampionInstance(champion=garen)

        role = bis_calculator._determine_role(champion)

        assert role == "tank"

    def test_determine_role_carry(self, bis_calculator, jinx):
        """Carry traits -> carry role."""
        champion = ChampionInstance(champion=jinx)

        role = bis_calculator._determine_role(champion)

        # Jinx should be ad_carry based on range/traits
        assert role in ["ad_carry", "ap_carry"]

    def test_bis_returns_three_items(self, bis_calculator, jinx):
        """BiS returns 3 item recommendations."""
        champion = ChampionInstance(champion=jinx)

        recommendations = bis_calculator.get_bis(champion)

        assert len(recommendations) == 3
        assert all(isinstance(r, ItemRecommendation) for r in recommendations)

    def test_bis_priorities(self, bis_calculator, jinx):
        """BiS items have correct priorities."""
        champion = ChampionInstance(champion=jinx)

        recommendations = bis_calculator.get_bis(champion)

        assert recommendations[0].priority == 1
        assert recommendations[1].priority == 2
        assert recommendations[2].priority == 3

    def test_bis_has_reasons(self, bis_calculator, jinx):
        """BiS recommendations have reasons."""
        champion = ChampionInstance(champion=jinx)

        recommendations = bis_calculator.get_bis(champion)

        for rec in recommendations:
            assert len(rec.reasons) > 0

    def test_suggest_components(self, bis_calculator):
        """Suggest components for target item."""
        components = load_components()
        giant_slayer = get_item_by_id("giant_slayer")

        suggestion = bis_calculator.suggest_components(giant_slayer, components)

        assert suggestion is not None
        comp1, comp2 = suggestion
        assert comp1.id in ["bf_sword", "recurve_bow"]
        assert comp2.id in ["bf_sword", "recurve_bow"]

    def test_get_missing_components(self, bis_calculator):
        """Get missing components for target item."""
        bf_sword = get_item_by_id("bf_sword")
        giant_slayer = get_item_by_id("giant_slayer")

        # Only have BF Sword, missing Recurve Bow
        missing = bis_calculator.get_missing_components(giant_slayer, [bf_sword])

        assert "recurve_bow" in missing


class TestChampionInstanceItems:
    """Tests for ChampionInstance item methods."""

    def test_get_item_traits(self, jinx):
        """Get traits from equipped emblems."""
        champion = ChampionInstance(champion=jinx)

        # Jinx has Zaun trait by default, no emblem items
        traits = champion.get_item_traits()
        assert len(traits) == 0

    def test_get_all_traits(self, jinx):
        """Get all traits including base."""
        champion = ChampionInstance(champion=jinx)

        traits = champion.get_all_traits()

        # Should include Jinx's base traits
        assert all(t in traits for t in jinx.traits)

    def test_has_item(self, jinx, bf_sword):
        """Check if champion has specific item."""
        champion = ChampionInstance(champion=jinx)
        champion.items.append(bf_sword)

        assert champion.has_item("bf_sword") is True
        assert champion.has_item("infinity_edge") is False

    def test_get_calculated_stats(self, jinx, bf_sword):
        """Get stats using StatCalculator."""
        champion = ChampionInstance(champion=jinx)
        champion.items.append(bf_sword)

        stats = champion.get_calculated_stats()

        assert isinstance(stats, CalculatedStats)
        expected_ad = jinx.stats.attack_damage[0] + bf_sword.stats.ad
        assert stats.attack_damage == expected_ad


class TestItemInstance:
    """Tests for ItemInstance class."""

    def test_is_component(self, bf_sword):
        """Component items are identified correctly."""
        instance = ItemInstance(item=bf_sword)

        assert instance.is_component is True
        assert instance.is_combined is False

    def test_is_combined(self):
        """Combined items are identified correctly."""
        giant_slayer = get_item_by_id("giant_slayer")
        instance = ItemInstance(item=giant_slayer)

        assert instance.is_component is False
        assert instance.is_combined is True

    def test_equipped_tracking(self, bf_sword, jinx):
        """Track which champion item is equipped to."""
        instance = ItemInstance(item=bf_sword)

        assert instance.equipped_to is None

        instance.equipped_to = jinx.id
        assert instance.equipped_to == jinx.id
