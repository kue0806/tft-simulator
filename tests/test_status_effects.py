"""Tests for Status Effects System."""

import pytest
from src.combat.hex_grid import Team
from src.combat.combat_unit import CombatUnit, CombatStats, UnitState
from src.combat.status_effects import (
    StatusEffectSystem,
    StatusEffect,
    StatusEffectType,
    create_stun,
    create_burn,
    create_shield,
    create_attack_speed_buff,
    create_grievous_wounds,
)


def create_test_unit(
    unit_id: str,
    team: Team = Team.BLUE,
    hp: float = 1000,
) -> CombatUnit:
    """Create a test combat unit."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=100,
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
    return CombatUnit(
        id=unit_id,
        name=f"Unit_{unit_id}",
        champion_id="TFT_Test",
        star_level=1,
        team=team,
        stats=stats,
    )


class TestStatusEffectSystem:
    """StatusEffectSystem tests."""

    @pytest.fixture
    def system(self):
        """Create status effect system."""
        return StatusEffectSystem()

    def test_apply_stun(self, system):
        """Test applying stun effect."""
        unit = create_test_unit("unit1")
        stun = create_stun("source1", duration=2.0)

        result = system.apply_effect(unit, stun)

        assert result is True
        assert system.has_effect(unit, StatusEffectType.STUN)

    def test_stun_prevents_actions(self, system):
        """Test stun prevents all actions."""
        unit = create_test_unit("unit1")
        stun = create_stun("source1", duration=2.0)

        system.apply_effect(unit, stun)

        assert not system.can_act(unit)
        assert not system.can_attack(unit)
        assert not system.can_cast(unit)
        assert not system.can_move(unit)

    def test_burn_damage(self, system):
        """Test burn deals damage over time."""
        unit = create_test_unit("unit1")
        burn = create_burn("source1", duration=3.0, damage_per_tick=50)

        system.apply_effect(unit, burn)

        # Update for one tick
        events = system.update(unit, delta_time=1.0)

        # Should have taken burn damage
        dot_events = [e for e in events if e.get("type") == "dot_damage"]
        assert len(dot_events) == 1
        assert unit.stats.current_hp < 1000

    def test_shield_absorbs_damage(self, system):
        """Test shield absorbs damage."""
        unit = create_test_unit("unit1")
        shield = create_shield("source1", duration=5.0, shield_amount=200)

        system.apply_effect(unit, shield)

        # Try to absorb 100 damage
        remaining = system.absorb_damage(unit, 100)

        assert remaining == 0
        # Shield should have 100 left
        effects = system.get_effects(unit)
        shield_effect = next(e for e in effects if e.effect_type == StatusEffectType.SHIELD)
        assert shield_effect.remaining_shield == 100

    def test_shield_partially_blocks(self, system):
        """Test shield partially blocks damage."""
        unit = create_test_unit("unit1")
        shield = create_shield("source1", duration=5.0, shield_amount=100)

        system.apply_effect(unit, shield)

        # Try to deal 150 damage
        remaining = system.absorb_damage(unit, 150)

        assert remaining == 50  # 50 damage gets through

    def test_attack_speed_buff(self, system):
        """Test attack speed buff modifier."""
        unit = create_test_unit("unit1")
        buff = create_attack_speed_buff("source1", duration=5.0, bonus=0.5)

        system.apply_effect(unit, buff)

        modifiers = system.get_stat_modifiers(unit.id)

        assert modifiers["attack_speed"] == 0.5

    def test_effect_expiration(self, system):
        """Test effect expires after duration."""
        unit = create_test_unit("unit1")
        stun = create_stun("source1", duration=1.0)

        system.apply_effect(unit, stun)
        assert system.has_effect(unit, StatusEffectType.STUN)

        # Update past duration
        system.update(unit, delta_time=1.5)

        assert not system.has_effect(unit, StatusEffectType.STUN)

    def test_effect_stacking(self, system):
        """Test effect stacking."""
        unit = create_test_unit("unit1")

        # Create stackable effect
        effect1 = StatusEffect(
            effect_type=StatusEffectType.BURN,
            source_id="source1",
            duration=5.0,
            value=50,
            max_stacks=3,
        )
        effect2 = StatusEffect(
            effect_type=StatusEffectType.BURN,
            source_id="source1",
            duration=5.0,
            value=50,
            max_stacks=3,
        )

        system.apply_effect(unit, effect1)
        system.apply_effect(unit, effect2)

        effects = system.get_effects(unit)
        burn = next(e for e in effects if e.effect_type == StatusEffectType.BURN)

        assert burn.stacks == 2

    def test_remove_effect(self, system):
        """Test removing effect."""
        unit = create_test_unit("unit1")
        stun = create_stun("source1", duration=5.0)

        system.apply_effect(unit, stun)
        assert system.has_effect(unit, StatusEffectType.STUN)

        system.remove_effect(unit, StatusEffectType.STUN)
        assert not system.has_effect(unit, StatusEffectType.STUN)

    def test_clear_unit(self, system):
        """Test clearing all effects from unit."""
        unit = create_test_unit("unit1")

        system.apply_effect(unit, create_stun("s1", 5.0))
        system.apply_effect(unit, create_burn("s2", 5.0, 50))

        system.clear_unit(unit.id)

        assert len(system.get_effects(unit)) == 0

    def test_grievous_wounds_reduces_healing(self, system):
        """Test grievous wounds reduces heal over time."""
        unit = create_test_unit("unit1")
        unit.stats.current_hp = 500

        # Apply grievous wounds
        gw = create_grievous_wounds("source1", duration=10.0)
        system.apply_effect(unit, gw)

        # Apply heal over time
        hot = StatusEffect(
            effect_type=StatusEffectType.HEAL_OVER_TIME,
            source_id="source2",
            duration=5.0,
            value=100,
            tick_interval=1.0,
        )
        system.apply_effect(unit, hot)

        # Update to trigger healing
        system.update(unit, delta_time=1.0)

        # Should heal for 50 instead of 100 (50% reduction)
        assert unit.stats.current_hp == 550

    def test_silence_prevents_casting(self, system):
        """Test silence prevents casting but allows attacking."""
        unit = create_test_unit("unit1")
        silence = StatusEffect(
            effect_type=StatusEffectType.SILENCE,
            source_id="source1",
            duration=2.0,
        )

        system.apply_effect(unit, silence)

        assert system.can_attack(unit)
        assert not system.can_cast(unit)
        assert system.can_move(unit)

    def test_disarm_prevents_attacking(self, system):
        """Test disarm prevents attacking but allows casting."""
        unit = create_test_unit("unit1")
        disarm = StatusEffect(
            effect_type=StatusEffectType.DISARM,
            source_id="source1",
            duration=2.0,
        )

        system.apply_effect(unit, disarm)

        assert not system.can_attack(unit)
        assert system.can_cast(unit)
        assert system.can_move(unit)

    def test_root_prevents_movement(self, system):
        """Test root prevents movement but allows actions."""
        unit = create_test_unit("unit1")
        root = StatusEffect(
            effect_type=StatusEffectType.ROOT,
            source_id="source1",
            duration=2.0,
        )

        system.apply_effect(unit, root)

        assert system.can_attack(unit)
        assert system.can_cast(unit)
        assert not system.can_move(unit)

    def test_multiple_stat_modifiers(self, system):
        """Test multiple stat modifiers stack."""
        unit = create_test_unit("unit1")

        buff1 = create_attack_speed_buff("s1", 5.0, 0.3)
        buff2 = create_attack_speed_buff("s2", 5.0, 0.2)

        system.apply_effect(unit, buff1)
        system.apply_effect(unit, buff2)

        modifiers = system.get_stat_modifiers(unit.id)

        # Both buffs from different sources should add
        assert modifiers["attack_speed"] == pytest.approx(0.5)

    def test_armor_debuff(self, system):
        """Test armor reduction debuff."""
        unit = create_test_unit("unit1")

        debuff = StatusEffect(
            effect_type=StatusEffectType.ARMOR_DEBUFF,
            source_id="source1",
            duration=5.0,
            value=30,  # -30 armor
        )

        system.apply_effect(unit, debuff)

        modifiers = system.get_stat_modifiers(unit.id)

        assert modifiers["armor"] == -30


class TestStatusEffectHelpers:
    """Tests for status effect helper functions."""

    def test_create_stun(self):
        """Test stun creation helper."""
        stun = create_stun("source1", 2.5)

        assert stun.effect_type == StatusEffectType.STUN
        assert stun.source_id == "source1"
        assert stun.duration == 2.5

    def test_create_burn(self):
        """Test burn creation helper."""
        burn = create_burn("source1", 3.0, 75)

        assert burn.effect_type == StatusEffectType.BURN
        assert burn.duration == 3.0
        assert burn.value == 75
        assert burn.tick_interval == 1.0

    def test_create_shield(self):
        """Test shield creation helper."""
        shield = create_shield("source1", 4.0, 300)

        assert shield.effect_type == StatusEffectType.SHIELD
        assert shield.duration == 4.0
        assert shield.value == 300
        assert shield.remaining_shield == 300
