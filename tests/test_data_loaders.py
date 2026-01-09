"""Tests for data loaders."""

import pytest

from src.data.loaders import (
    load_champions,
    get_champion_by_id,
    get_champions_by_cost,
    get_champions_by_trait,
    get_base_champions,
    get_unlockable_champions,
    load_traits,
    load_origins,
    load_classes,
    get_trait_by_id,
    load_items,
    load_components,
    load_combined_items,
    get_item_by_id,
    get_recipe,
)


class TestChampionLoader:
    """Tests for champion loading functionality."""

    def test_load_champions_returns_list(self):
        champions = load_champions()
        assert isinstance(champions, list)
        assert len(champions) > 0

    def test_load_champions_count(self):
        champions = load_champions()
        # Should have approximately 99 champions (70 base + 29 unlockable)
        assert len(champions) >= 90

    def test_get_champion_by_id_found(self):
        champion = get_champion_by_id("garen")
        assert champion is not None
        assert champion.name == "Garen"
        assert champion.cost == 4

    def test_get_champion_by_id_not_found(self):
        champion = get_champion_by_id("nonexistent_champion")
        assert champion is None

    def test_get_champions_by_cost(self):
        one_costs = get_champions_by_cost(1)
        assert len(one_costs) > 0
        assert all(c.cost == 1 for c in one_costs)

    def test_get_champions_by_trait(self):
        demacia_champs = get_champions_by_trait("demacia")
        assert len(demacia_champs) > 0
        assert all("demacia" in c.traits for c in demacia_champs)

    def test_get_base_champions(self):
        base = get_base_champions()
        assert len(base) >= 60
        assert all(not c.is_unlockable for c in base)

    def test_get_unlockable_champions(self):
        unlockables = get_unlockable_champions()
        assert len(unlockables) > 0
        assert all(c.is_unlockable for c in unlockables)

    def test_champion_has_required_fields(self):
        champion = get_champion_by_id("lux")
        assert champion is not None
        assert champion.id == "lux"
        assert champion.name == "Lux"
        assert champion.cost > 0
        assert len(champion.traits) > 0
        assert champion.stats is not None
        assert champion.ability is not None


class TestTraitLoader:
    """Tests for trait loading functionality."""

    def test_load_traits_returns_list(self):
        traits = load_traits()
        assert isinstance(traits, list)
        assert len(traits) > 0

    def test_load_origins(self):
        origins = load_origins()
        assert len(origins) > 0
        assert all(t.type == "origin" for t in origins)

    def test_load_classes(self):
        classes = load_classes()
        assert len(classes) > 0
        assert all(t.type == "class" for t in classes)

    def test_get_trait_by_id_found(self):
        trait = get_trait_by_id("demacia")
        assert trait is not None
        assert trait.name == "Demacia"

    def test_get_trait_by_id_not_found(self):
        trait = get_trait_by_id("nonexistent_trait")
        assert trait is None

    def test_trait_has_breakpoints(self):
        trait = get_trait_by_id("arcanist")
        assert trait is not None
        assert len(trait.breakpoints) > 0


class TestItemLoader:
    """Tests for item loading functionality."""

    def test_load_items_returns_list(self):
        items = load_items()
        assert isinstance(items, list)
        assert len(items) > 0

    def test_load_components(self):
        components = load_components()
        assert len(components) == 9  # 9 base components

    def test_load_combined_items(self):
        combined = load_combined_items()
        assert len(combined) > 0

    def test_get_item_by_id_found(self):
        item = get_item_by_id("bf_sword")
        assert item is not None
        assert item.name == "B.F. Sword"

    def test_get_item_by_id_not_found(self):
        item = get_item_by_id("nonexistent_item")
        assert item is None

    def test_get_recipe(self):
        # BF + BF = Deathblade
        recipe = get_recipe("bf_sword", "bf_sword")
        assert recipe is not None
        assert recipe.name == "Deathblade"

    def test_get_recipe_order_independent(self):
        # Test that order doesn't matter
        recipe1 = get_recipe("bf_sword", "recurve_bow")
        recipe2 = get_recipe("recurve_bow", "bf_sword")
        assert recipe1 == recipe2

    def test_combined_item_has_components(self):
        item = get_item_by_id("infinity_edge")
        assert item is not None
        assert item.components is not None
        assert len(item.components) == 2
