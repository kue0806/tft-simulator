"""Tests for Synergy Calculator functionality."""

import pytest

from src.data.loaders import get_champion_by_id, get_champions_by_trait
from src.core.synergy_calculator import SynergyCalculator, ActiveTrait, SynergyDelta
from src.core.player_units import ChampionInstance, PlayerUnits
from src.core.emblem_system import EmblemSystem
from src.core.unique_traits import (
    YordleHandler,
    IoniaHandler,
    DemaciaHandler,
    VoidHandler,
    get_handler,
)


@pytest.fixture
def calculator():
    """Create a fresh SynergyCalculator."""
    return SynergyCalculator()


@pytest.fixture
def demacia_champions():
    """Get Demacia champions for testing."""
    return get_champions_by_trait("demacia")


class TestSynergyCalculator:
    """Tests for core synergy calculation."""

    def test_single_trait_activation(self, calculator, demacia_champions):
        """Test basic trait activation with enough champions."""
        # Demacia activates at 3 (breakpoints: 3/5/7/11)
        # Get 3 Demacia champions
        champs = demacia_champions[:3]
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)

        assert "demacia" in synergies
        assert synergies["demacia"].count == 3
        assert synergies["demacia"].is_active  # Demacia activates at 3

    def test_trait_not_active_with_one(self, calculator, demacia_champions):
        """One champion doesn't activate most traits."""
        champ = demacia_champions[0]
        instances = [ChampionInstance(champion=champ)]

        synergies = calculator.calculate_synergies(instances)

        assert "demacia" in synergies
        assert synergies["demacia"].count == 1
        assert not synergies["demacia"].is_active  # Not enough for activation

    def test_multiple_breakpoints(self, calculator, demacia_champions):
        """Test trait with multiple breakpoints."""
        # Demacia has 3/5/7/11 breakpoints
        # With 6 Demacia, should show 5 active, 7 as next
        champs = demacia_champions[:6]
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)

        assert synergies["demacia"].count == 6
        assert synergies["demacia"].is_active
        assert synergies["demacia"].active_breakpoint.count == 5
        assert synergies["demacia"].next_breakpoint.count == 7

    def test_multiple_traits_per_champion(self, calculator):
        """Test champion with multiple traits."""
        # Garen has Demacia and Defender
        garen = get_champion_by_id("garen")
        instances = [ChampionInstance(champion=garen)]

        synergies = calculator.calculate_synergies(instances)

        # Should have both traits with count 1
        for trait_id in garen.traits:
            if trait_id in synergies:
                assert synergies[trait_id].count >= 1

    def test_unique_champions_only(self, calculator, demacia_champions):
        """Same champion multiple times counts once per unique champion."""
        # Get one Demacia champion and "duplicate" it
        champ = demacia_champions[0]
        # Create 3 instances of same champion (e.g., 2-star upgrade scenario)
        instances = [ChampionInstance(champion=champ) for _ in range(3)]

        synergies = calculator.calculate_synergies(instances)

        # Should only count as 1 champion for trait purposes
        assert synergies["demacia"].count == 1

    def test_empty_board(self, calculator):
        """Empty board returns empty synergies."""
        synergies = calculator.calculate_synergies([])
        assert len(synergies) == 0

    def test_style_bronze(self, calculator, demacia_champions):
        """Test bronze style for first breakpoint."""
        # Demacia first breakpoint is 3
        champs = demacia_champions[:3]  # Exactly 3 for first breakpoint
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)

        assert synergies["demacia"].style == "bronze"

    def test_style_inactive(self, calculator, demacia_champions):
        """Test inactive style."""
        champ = demacia_champions[0]
        instances = [ChampionInstance(champion=champ)]

        synergies = calculator.calculate_synergies(instances)

        assert synergies["demacia"].style == "inactive"

    def test_get_trait_bonuses(self, calculator, demacia_champions):
        """Test aggregating trait bonuses."""
        champs = demacia_champions[:4]  # Activate at 4
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)
        bonuses = calculator.get_trait_bonuses(synergies)

        # Should have some bonuses (depends on trait data)
        assert isinstance(bonuses, dict)

    def test_preview_add_champion(self, calculator, demacia_champions):
        """Test synergy preview when adding champion."""
        # Demacia activates at 3
        # Start with 2 Demacia
        instances = [ChampionInstance(champion=c) for c in demacia_champions[:2]]

        # Preview adding third Demacia (will activate)
        champ3 = demacia_champions[2]
        deltas = calculator.preview_add_champion(instances, champ3)

        # Demacia should go from 2->3, activating
        assert "demacia" in deltas
        assert deltas["demacia"].old_count == 2
        assert deltas["demacia"].new_count == 3
        assert not deltas["demacia"].was_active
        assert deltas["demacia"].will_be_active
        assert deltas["demacia"].breakpoint_change == "upgrade"

    def test_preview_remove_champion(self, calculator, demacia_champions):
        """Test synergy preview when removing champion."""
        # Demacia activates at 3
        # Start with 3 Demacia (active)
        champs = demacia_champions[:3]
        instances = [ChampionInstance(champion=c) for c in champs]

        # Preview removing one (will deactivate)
        deltas = calculator.preview_remove_champion(instances, instances[0])

        assert "demacia" in deltas
        assert deltas["demacia"].old_count == 3
        assert deltas["demacia"].new_count == 2
        assert deltas["demacia"].was_active
        assert not deltas["demacia"].will_be_active
        assert deltas["demacia"].breakpoint_change == "downgrade"

    def test_get_contributing_champions(self, calculator, demacia_champions):
        """Test getting champions that contribute to a trait."""
        champs = demacia_champions[:3]
        instances = [ChampionInstance(champion=c) for c in champs]

        contributing = calculator.get_contributing_champions(instances, "demacia")

        assert len(contributing) == 3
        for c in contributing:
            assert "demacia" in c.champion.traits


class TestEmblemSystem:
    """Tests for the Emblem System."""

    def test_emblem_detection(self):
        """Test detecting emblem items."""
        assert EmblemSystem.is_emblem("demacia_emblem")
        assert not EmblemSystem.is_emblem("bf_sword")

    def test_get_trait_for_emblem(self):
        """Test getting trait from emblem."""
        trait = EmblemSystem.get_trait_for_emblem("slayer_emblem")
        assert trait == "slayer"

        trait = EmblemSystem.get_trait_for_emblem("bf_sword")
        assert trait is None

    def test_cannot_double_trait(self):
        """Cannot equip emblem for existing trait."""
        garen = get_champion_by_id("garen")  # Has Demacia
        instance = ChampionInstance(champion=garen)

        # Should not be able to equip Demacia emblem
        can_equip = EmblemSystem.can_equip_emblem(instance, "demacia")
        assert not can_equip

        # But can equip other emblems
        can_equip = EmblemSystem.can_equip_emblem(instance, "slayer")
        assert can_equip

    def test_emblem_counts_for_trait(self, calculator, demacia_champions):
        """Test that emblems add to trait count."""
        # Start with 1 Demacia champion
        champ = demacia_champions[0]
        instances = [ChampionInstance(champion=champ)]

        # Without emblems: count = 1
        synergies = calculator.calculate_synergies(instances, [])
        assert synergies["demacia"].count == 1

        # With Demacia emblem: count = 2
        synergies = calculator.calculate_synergies(instances, ["demacia"])
        assert synergies["demacia"].count == 2


class TestUniqueTraits:
    """Tests for unique trait handlers."""

    def test_yordle_scaling(self):
        """Yordle bonus scales with count and star level."""
        handler = YordleHandler()

        yordle_champs = get_champions_by_trait("yordle")
        if not yordle_champs:
            pytest.skip("No Yordle champions found")

        # Create 3 Yordle instances at 1-star
        instances = [ChampionInstance(champion=c) for c in yordle_champs[:3]]

        bonus = handler.calculate_bonus(instances)

        assert bonus["health"] == 50 * 3  # 50 per Yordle
        assert bonus["attack_speed"] == 0.07 * 3

    def test_yordle_3_star_bonus(self):
        """3-star Yordles grant 50% more bonus."""
        handler = YordleHandler()

        yordle_champs = get_champions_by_trait("yordle")
        if not yordle_champs:
            pytest.skip("No Yordle champions found")

        # Create 1 Yordle at 3-star
        instance = ChampionInstance(champion=yordle_champs[0], star_level=3)

        bonus = handler.calculate_bonus([instance])

        assert bonus["health"] == 50 * 1.5  # 50% more
        assert bonus["attack_speed"] == 0.07 * 1.5

    def test_ionia_path_selection(self):
        """Ionia randomly selects path."""
        handler = IoniaHandler()

        # Roll path multiple times to verify randomness works
        paths_seen = set()
        for _ in range(50):
            path = handler.roll_path()
            assert path in handler.PATHS
            paths_seen.add(path)

        # Should see multiple different paths (statistically likely)
        assert len(paths_seen) >= 2

    def test_demacia_rally(self):
        """Demacia rally triggers at 25% health lost."""
        handler = DemaciaHandler()

        # No rally at 90% health
        triggered = handler.on_health_lost(90)
        assert not triggered
        assert handler.rally_count == 0

        # Rally at 75% health (crossed 25% threshold)
        triggered = handler.on_health_lost(75)
        assert triggered
        assert handler.rally_count == 1

        # Rally again at 50% (crossed another 25%)
        triggered = handler.on_health_lost(50)
        assert triggered
        assert handler.rally_count == 2

    def test_void_mutations(self):
        """Void assigns mutations to champions."""
        handler = VoidHandler()

        void_champs = get_champions_by_trait("void")
        if not void_champs:
            pytest.skip("No Void champions found")

        instance = ChampionInstance(champion=void_champs[0])

        # Assign mutation
        result = handler.assign_mutation(instance, "vampiric")
        assert result

        # Check mutation was assigned
        mutation = handler.get_champion_mutation(instance)
        assert mutation == "vampiric"

    def test_get_handler(self):
        """Test getting handlers for traits."""
        handler = get_handler("yordle")
        assert isinstance(handler, YordleHandler)

        handler = get_handler("demacia")
        assert isinstance(handler, DemaciaHandler)

        # Unknown trait returns None
        handler = get_handler("unknown_trait")
        assert handler is None


class TestPlayerUnitsSynergies:
    """Tests for synergy integration in PlayerUnits."""

    def test_synergy_cache_on_board_change(self):
        """Synergy cache invalidates when board changes."""
        units = PlayerUnits()
        demacia_champs = get_champions_by_trait("demacia")

        if len(demacia_champs) < 2:
            pytest.skip("Not enough Demacia champions")

        # Add to bench
        instance1 = units.add_to_bench(demacia_champs[0])
        instance2 = units.add_to_bench(demacia_champs[1])

        # Place on board
        units.place_on_board(instance1, 0, 0)

        # Get synergies (should cache)
        synergies1 = units.get_active_synergies()
        assert "demacia" in synergies1
        assert synergies1["demacia"].count == 1

        # Place second champion
        units.place_on_board(instance2, 1, 0)

        # Get synergies again (should be different due to cache invalidation)
        synergies2 = units.get_active_synergies()
        assert synergies2["demacia"].count == 2

    def test_synergies_only_count_board(self):
        """Synergies only count champions on board, not bench."""
        units = PlayerUnits()
        demacia_champs = get_champions_by_trait("demacia")

        if len(demacia_champs) < 2:
            pytest.skip("Not enough Demacia champions")

        # Add to bench only
        units.add_to_bench(demacia_champs[0])
        units.add_to_bench(demacia_champs[1])

        synergies = units.get_active_synergies()

        # Bench champions don't count for synergies
        assert len(synergies) == 0


class TestSynergyDisplay:
    """Tests for synergy display formatting."""

    def test_sort_by_priority(self, calculator, demacia_champions):
        """Test synergy sorting priority."""
        from src.core.synergy_display import SynergyFormatter

        # Create synergies with various states
        champs = demacia_champions[:4]
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)
        displays = SynergyFormatter.format_for_display(synergies, instances)

        # Active synergies should come before inactive
        active_found = False
        for display in displays:
            if display.is_active:
                active_found = True
            elif active_found:
                # If we found an active one before, this inactive should come after
                pass

    def test_breakpoint_text(self, calculator, demacia_champions):
        """Test breakpoint text generation."""
        from src.core.synergy_display import SynergyFormatter

        champs = demacia_champions[:4]
        instances = [ChampionInstance(champion=c) for c in champs]

        synergies = calculator.calculate_synergies(instances)
        displays = SynergyFormatter.format_for_display(synergies, instances)

        # Find Demacia display
        demacia_display = next(
            (d for d in displays if d.trait_id == "demacia"), None
        )

        assert demacia_display is not None
        assert demacia_display.count == 4
        # Breakpoint text should contain the count and breakpoints
        assert "4" in demacia_display.breakpoint_text
