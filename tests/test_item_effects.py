"""Tests for Item Effects System."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.combat.hex_grid import Team, HexGrid, HexPosition
from src.combat.combat_unit import CombatUnit, CombatStats
from src.combat.item_effects import ItemEffectSystem, ItemEffectContext
from src.combat.status_effects import StatusEffectSystem, StatusEffectType, StatusEffect


@dataclass
class MockItem:
    """Simple mock item with id attribute."""
    id: str


def create_test_unit(
    unit_id: str,
    team: Team = Team.BLUE,
    hp: float = 1000,
    attack_damage: float = 100,
    items: list = None,
) -> CombatUnit:
    """Create a test combat unit with specified items."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=attack_damage,
        ability_power=100,
        armor=50,
        magic_resist=50,
        attack_speed=1.0,
        crit_chance=0.25,
        crit_damage=1.4,
        max_mana=100,
        current_mana=50,
        starting_mana=50,
        attack_range=1,
        dodge_chance=0.0,
        omnivamp=0.0,
        damage_amp=1.0,
        damage_reduction=0.0,
    )
    unit = CombatUnit(
        id=unit_id,
        name=f"Unit_{unit_id}",
        champion_id=f"TFT_Test_{unit_id}",
        star_level=1,
        team=team,
        stats=stats,
    )

    # Mock source_instance with items using dataclass instead of MagicMock
    if items:
        mock_items = [MockItem(id=item_id) for item_id in items]
        mock_instance = MagicMock()
        mock_instance.items = mock_items
        unit.source_instance = mock_instance

    return unit


@pytest.fixture
def system():
    """Create item effect system."""
    return ItemEffectSystem()


@pytest.fixture
def grid():
    """Create hex grid."""
    return HexGrid()


@pytest.fixture
def status_effects():
    """Create status effect system."""
    return StatusEffectSystem()


@pytest.fixture
def context(status_effects, grid):
    """Create item effect context."""
    return ItemEffectContext(
        attack_system=MagicMock(),
        status_effects=status_effects,
        target_selector=MagicMock(),
        all_units={},
        grid=grid,
        current_time=0.0,
    )


class TestCombatStartEffects:
    """Tests for combat start item effects."""

    def test_zekes_herald_buffs_adjacent_allies(self, system, status_effects, grid, context):
        """Zeke's Herald should grant AS buff to adjacent allies."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["zekes_herald"])
        ally = create_test_unit("ally", Team.BLUE)

        grid.place_unit("carrier", HexPosition(0, 0))
        grid.place_unit("ally", HexPosition(0, 1))  # Adjacent

        context.all_units = {"carrier": carrier, "ally": ally}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(ally)

        events = system.apply_combat_start_effects(carrier, context)

        # Ally should have AS buff
        assert status_effects.has_effect(ally, StatusEffectType.ATTACK_SPEED_BUFF)
        assert len([e for e in events if e["type"] == "zekes_herald"]) > 0

    def test_chalice_buffs_same_row(self, system, grid, context):
        """Chalice should grant AP to allies in same row."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["chalice_of_power"])
        ally = create_test_unit("ally", Team.BLUE)

        grid.place_unit("carrier", HexPosition(1, 0))
        grid.place_unit("ally", HexPosition(1, 3))  # Same row

        initial_ap = ally.stats.ability_power
        context.all_units = {"carrier": carrier, "ally": ally}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(ally)

        events = system.apply_combat_start_effects(carrier, context)

        # Ally should have more AP
        assert ally.stats.ability_power > initial_ap
        assert len([e for e in events if e["type"] == "chalice_of_power"]) > 0

    def test_locket_shields_same_row(self, system, status_effects, grid, context):
        """Locket should shield allies in same row."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["locket_of_the_iron_solari"])
        ally = create_test_unit("ally", Team.BLUE)

        grid.place_unit("carrier", HexPosition(2, 1))
        grid.place_unit("ally", HexPosition(2, 4))  # Same row

        context.all_units = {"carrier": carrier, "ally": ally}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(ally)

        events = system.apply_combat_start_effects(carrier, context)

        # Both should have shields
        assert status_effects.has_effect(carrier, StatusEffectType.SHIELD)
        assert status_effects.has_effect(ally, StatusEffectType.SHIELD)

    def test_frozen_heart_debuffs_enemies(self, system, status_effects, grid, context):
        """Frozen Heart should reduce enemy AS."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["frozen_heart"])
        enemy = create_test_unit("enemy", Team.RED)

        grid.place_unit("carrier", HexPosition(3, 3))
        grid.place_unit("enemy", HexPosition(4, 3))  # Within 2 hexes

        context.all_units = {"carrier": carrier, "enemy": enemy}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(enemy)

        events = system.apply_combat_start_effects(carrier, context)

        # Enemy should have AS debuff
        assert status_effects.has_effect(enemy, StatusEffectType.ATTACK_SPEED_DEBUFF)

    def test_shroud_increases_enemy_mana(self, system, grid, context):
        """Shroud should increase enemy max mana."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["shroud_of_stillness"])
        enemy = create_test_unit("enemy", Team.RED)

        grid.place_unit("carrier", HexPosition(3, 3))
        grid.place_unit("enemy", HexPosition(4, 3))

        initial_mana = enemy.stats.max_mana
        context.all_units = {"carrier": carrier, "enemy": enemy}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(enemy)

        events = system.apply_combat_start_effects(carrier, context)

        # Enemy should have increased max mana
        assert enemy.stats.max_mana > initial_mana

    def test_gargoyle_gains_stats_per_enemy(self, system, context):
        """Gargoyle should gain armor/MR per enemy."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["gargoyle_stoneplate"])
        enemy1 = create_test_unit("enemy1", Team.RED)
        enemy2 = create_test_unit("enemy2", Team.RED)

        initial_armor = carrier.stats.armor
        initial_mr = carrier.stats.magic_resist

        context.all_units = {"carrier": carrier, "enemy1": enemy1, "enemy2": enemy2}

        system.initialize_unit(carrier)

        events = system.apply_combat_start_effects(carrier, context)

        # Should have +20 armor/MR (10 per enemy)
        assert carrier.stats.armor == initial_armor + 20
        assert carrier.stats.magic_resist == initial_mr + 20


class TestOnHitEffects:
    """Tests for on-hit item effects."""

    def test_hextech_gunblade_heals(self, system, context):
        """Hextech Gunblade should heal for damage dealt."""
        attacker = create_test_unit("attacker", Team.BLUE, items=["hextech_gunblade"])
        attacker.stats.current_hp = 500  # Damaged
        target = create_test_unit("target", Team.RED)

        context.all_units = {"attacker": attacker, "target": target}
        system.initialize_unit(attacker)

        # Mock damage event
        damage_event = MagicMock()
        damage_event.final_damage = 100
        damage_event.is_critical = False

        events = system.apply_on_hit_effects(attacker, target, damage_event, context)

        # Should have healed (25 HP from 100 damage)
        assert attacker.stats.current_hp > 500
        gunblade_events = [e for e in events if "hextech_gunblade" in e["type"]]
        assert len(gunblade_events) > 0

    def test_guardbreaker_marks_shielded_targets(self, system, status_effects, context):
        """Guardbreaker should mark shielded targets."""
        attacker = create_test_unit("attacker", Team.BLUE, items=["guardbreaker"])
        target = create_test_unit("target", Team.RED)

        # Apply shield to target
        from src.combat.status_effects import create_shield
        shield = create_shield("test", 5.0, 200)
        status_effects.apply_effect(target, shield)

        context.all_units = {"attacker": attacker, "target": target}
        context.current_time = 1.0

        system.initialize_unit(attacker)

        damage_event = MagicMock()
        damage_event.final_damage = 50
        damage_event.is_critical = False

        events = system.apply_on_hit_effects(attacker, target, damage_event, context)

        # Should have marked event
        mark_events = [e for e in events if e["type"] == "guardbreaker_marked"]
        assert len(mark_events) > 0


class TestOnCastEffects:
    """Tests for on-cast item effects."""

    def test_morellonomicon_applies_burn(self, system, status_effects, context):
        """Morellonomicon should apply burn and grievous wounds."""
        caster = create_test_unit("caster", Team.BLUE, items=["morellonomicon"])
        target = create_test_unit("target", Team.RED)

        context.all_units = {"caster": caster, "target": target}
        system.initialize_unit(caster)

        events = system.apply_on_cast_effects(caster, context, ability_targets=[target])

        # Target should have burn and grievous wounds
        assert status_effects.has_effect(target, StatusEffectType.BURN)
        assert status_effects.has_effect(target, StatusEffectType.GRIEVOUS_WOUNDS)

    def test_ionic_spark_damages_enemy_casters(self, system, grid, context):
        """Ionic Spark should damage nearby enemies when they cast."""
        carrier = create_test_unit("carrier", Team.BLUE, items=["ionic_spark"])
        enemy_caster = create_test_unit("enemy", Team.RED)

        grid.place_unit("carrier", HexPosition(3, 3))
        grid.place_unit("enemy", HexPosition(4, 3))  # Within 2 hexes

        initial_hp = enemy_caster.stats.current_hp
        context.all_units = {"carrier": carrier, "enemy": enemy_caster}
        context.grid = grid

        system.initialize_unit(carrier)
        system.initialize_unit(enemy_caster)

        events = system.apply_ionic_spark_on_enemy_cast(enemy_caster, context)

        # Enemy should have taken damage
        assert enemy_caster.stats.current_hp < initial_hp
        assert len([e for e in events if e["type"] == "ionic_spark"]) > 0


class TestOnDamageTakenEffects:
    """Tests for on-damage-taken item effects."""

    def test_steraks_gage_triggers_below_threshold(self, system, status_effects, context):
        """Sterak's Gage should trigger when below 60% HP."""
        unit = create_test_unit("unit", Team.BLUE, hp=1000, items=["steraks_gage"])
        unit.stats.current_hp = 500  # 50% HP, below threshold

        context.all_units = {"unit": unit}
        system.initialize_unit(unit)

        events = system.apply_on_damage_taken_effects(unit, 100, context)

        # Should have shield and AD buff
        assert status_effects.has_effect(unit, StatusEffectType.SHIELD)
        sterak_events = [e for e in events if e["type"] == "steraks_gage"]
        assert len(sterak_events) > 0

    def test_bramble_vest_reflects_damage(self, system, context):
        """Bramble Vest should reflect damage to attackers."""
        unit = create_test_unit("unit", Team.BLUE, items=["bramble_vest"])
        attacker = create_test_unit("attacker", Team.RED)

        initial_hp = attacker.stats.current_hp
        context.all_units = {"unit": unit, "attacker": attacker}
        system.initialize_unit(unit)

        events = system.apply_on_damage_taken_effects(
            unit, 100, context, attacker=attacker, is_ability_damage=False
        )

        # Attacker should have taken reflected damage
        assert attacker.stats.current_hp < initial_hp
        bramble_events = [e for e in events if e["type"] == "bramble_vest"]
        assert len(bramble_events) > 0


class TestDamageModification:
    """Tests for damage modification effects."""

    def test_guardbreaker_bonus_vs_shielded(self, system, status_effects, context):
        """Guardbreaker should deal bonus damage to shielded enemies."""
        attacker = create_test_unit("attacker", Team.BLUE, items=["guardbreaker"])
        target = create_test_unit("target", Team.RED)

        # Apply shield to target
        from src.combat.status_effects import create_shield
        shield = create_shield("test", 5.0, 200)
        status_effects.apply_effect(target, shield)

        system.initialize_unit(attacker)

        base_damage = 100
        modified = system.modify_outgoing_damage(attacker, target, base_damage, context=context)

        # Should be 25% more damage
        assert modified == 125

    def test_bramble_reduces_attack_damage(self, system):
        """Bramble Vest should reduce attack damage taken."""
        unit = create_test_unit("unit", Team.BLUE, items=["bramble_vest"])
        system.initialize_unit(unit)

        base_damage = 100
        modified = system.modify_incoming_damage(unit, base_damage, is_ability_damage=False)

        # Should be 8% less
        assert modified == 92

    def test_dragons_claw_reduces_ability_damage(self, system):
        """Dragon's Claw should reduce ability damage taken."""
        unit = create_test_unit("unit", Team.BLUE, items=["dragons_claw"])
        system.initialize_unit(unit)

        base_damage = 100
        modified = system.modify_incoming_damage(unit, base_damage, is_ability_damage=True)

        # Should be 10% less
        assert modified == 90


class TestPeriodicEffects:
    """Tests for periodic item effects."""

    def test_warmogs_heals_over_time(self, system, context):
        """Warmog's should heal 3% max HP per second."""
        unit = create_test_unit("unit", Team.BLUE, hp=1000, items=["warmogs_armor"])
        unit.stats.current_hp = 500  # 50% HP

        context.all_units = {"unit": unit}
        system.initialize_unit(unit)

        # Simulate 1 second
        system.update_periodic_effects(unit, 1.0, context)

        # Should have healed 30 HP (3% of 1000)
        assert unit.stats.current_hp == 530

    def test_dragons_claw_periodic_heal(self, system, context):
        """Dragon's Claw should heal based on enemies every 2 seconds."""
        unit = create_test_unit("unit", Team.BLUE, hp=1000, items=["dragons_claw"])
        unit.stats.current_hp = 500
        enemy = create_test_unit("enemy", Team.RED)

        context.all_units = {"unit": unit, "enemy": enemy}
        system.initialize_unit(unit)

        # Simulate 2 seconds (trigger interval)
        system.update_periodic_effects(unit, 2.0, context)

        # Should have healed 1.2% max HP per enemy (12 HP)
        assert unit.stats.current_hp > 500
