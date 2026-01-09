"""Tests for Attack System."""

import pytest
import random
from src.combat.hex_grid import Team
from src.combat.combat_unit import CombatUnit, CombatStats, UnitState
from src.combat.attack import (
    AttackSystem,
    AttackResult,
    DamageEvent,
    DamageType,
    MANA_ON_ATTACK,
    MANA_ON_DAMAGE_TAKEN,
    calculate_effective_hp,
    calculate_dps,
)


def create_test_unit(
    unit_id: str,
    team: Team = Team.BLUE,
    hp: float = 1000,
    ad: float = 100,
    armor: float = 50,
    mr: float = 50,
    attack_speed: float = 1.0,
    crit_chance: float = 0.0,
    crit_damage: float = 1.4,
) -> CombatUnit:
    """Create a test combat unit."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=ad,
        ability_power=100,
        armor=armor,
        magic_resist=mr,
        attack_speed=attack_speed,
        crit_chance=crit_chance,
        crit_damage=crit_damage,
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


class TestAttackSystem:
    """AttackSystem tests."""

    @pytest.fixture
    def attack_system(self):
        """Create attack system with fixed seed."""
        return AttackSystem(rng=random.Random(42))

    def test_can_attack_ready(self, attack_system):
        """Test can_attack when ready."""
        unit = create_test_unit("attacker")
        unit.attack_cooldown = 0

        assert attack_system.can_attack(unit)

    def test_can_attack_on_cooldown(self, attack_system):
        """Test can_attack when on cooldown."""
        unit = create_test_unit("attacker")
        unit.attack_cooldown = 0.5

        assert not attack_system.can_attack(unit)

    def test_can_attack_dead(self, attack_system):
        """Test can_attack when dead."""
        unit = create_test_unit("attacker")
        unit.state = UnitState.DEAD

        assert not attack_system.can_attack(unit)

    def test_can_attack_stunned(self, attack_system):
        """Test can_attack when stunned."""
        unit = create_test_unit("attacker")
        unit.state = UnitState.STUNNED

        assert not attack_system.can_attack(unit)

    def test_execute_attack_basic(self, attack_system):
        """Test basic attack execution."""
        attacker = create_test_unit("attacker", Team.BLUE, ad=100)
        target = create_test_unit("target", Team.RED, armor=50)

        result = attack_system.execute_attack(attacker, target)

        assert result.success is True
        assert result.damage_event is not None
        assert result.damage_event.final_damage > 0
        assert target.stats.current_hp < 1000

    def test_execute_attack_sets_cooldown(self, attack_system):
        """Test attack sets cooldown."""
        attacker = create_test_unit("attacker", attack_speed=2.0)  # 0.5s interval
        target = create_test_unit("target", Team.RED)

        attack_system.execute_attack(attacker, target)

        assert attacker.attack_cooldown == 0.5  # 1 / attack_speed

    def test_attack_grants_mana(self, attack_system):
        """Test attack grants mana to attacker."""
        attacker = create_test_unit("attacker")
        attacker.stats.current_mana = 50
        target = create_test_unit("target", Team.RED)

        attack_system.execute_attack(attacker, target)

        assert attacker.stats.current_mana == 50 + MANA_ON_ATTACK

    def test_damage_grants_mana_to_target(self, attack_system):
        """Test taking damage grants mana to target."""
        attacker = create_test_unit("attacker")
        target = create_test_unit("target", Team.RED)
        target.stats.current_mana = 50

        attack_system.execute_attack(attacker, target)

        assert target.stats.current_mana == 50 + MANA_ON_DAMAGE_TAKEN

    def test_armor_reduces_physical_damage(self, attack_system):
        """Test armor reduces physical damage."""
        attacker = create_test_unit("attacker", ad=100)
        target_no_armor = create_test_unit("target1", Team.RED, armor=0)
        target_armor = create_test_unit("target2", Team.RED, armor=100)

        # Attack target without armor
        result1 = attack_system.execute_attack(attacker, target_no_armor)
        attacker.attack_cooldown = 0

        # Attack target with armor
        result2 = attack_system.execute_attack(attacker, target_armor)

        # Armored target takes less damage
        assert result2.damage_event.final_damage < result1.damage_event.final_damage

    def test_critical_strike(self):
        """Test critical strike mechanics."""
        # Use deterministic RNG that always crits
        attack_system = AttackSystem(rng=random.Random(0))

        attacker = create_test_unit("attacker", ad=100, crit_chance=1.0, crit_damage=1.5)
        target = create_test_unit("target", Team.RED, armor=0)

        result = attack_system.execute_attack(attacker, target)

        # Should crit for 150 damage (100 * 1.5)
        assert result.damage_event.is_critical is True
        assert result.damage_event.raw_damage == 150

    def test_attack_kills_target(self, attack_system):
        """Test attack that kills target."""
        attacker = create_test_unit("attacker", ad=500)
        target = create_test_unit("target", Team.RED, hp=100, armor=0)

        result = attack_system.execute_attack(attacker, target)

        assert result.target_killed is True
        assert not target.is_alive
        assert attacker.kills == 1

    def test_update_cooldown(self, attack_system):
        """Test cooldown reduction."""
        unit = create_test_unit("unit")
        unit.attack_cooldown = 1.0

        attack_system.update_cooldown(unit, delta_time=0.3)

        assert unit.attack_cooldown == pytest.approx(0.7)

    def test_update_cooldown_to_zero(self, attack_system):
        """Test cooldown doesn't go negative."""
        unit = create_test_unit("unit")
        unit.attack_cooldown = 0.5

        attack_system.update_cooldown(unit, delta_time=1.0)

        assert unit.attack_cooldown == 0

    def test_damage_amp(self, attack_system):
        """Test damage amplification."""
        attacker = create_test_unit("attacker", ad=100)
        attacker.stats.damage_amp = 1.5  # 50% more damage
        target = create_test_unit("target", Team.RED, armor=0)

        result = attack_system.execute_attack(attacker, target)

        assert result.damage_event.raw_damage == 150

    def test_damage_reduction(self, attack_system):
        """Test target's damage reduction."""
        attacker = create_test_unit("attacker", ad=100)
        target = create_test_unit("target", Team.RED, armor=0)
        target.stats.damage_reduction = 0.2  # 20% reduction

        result = attack_system.execute_attack(attacker, target)

        # Damage is also reduced by armor first (even 0 armor applies formula)
        # Then damage_reduction is applied
        # With 0 armor: 100 * 1.0 * 0.8 = 80, but if target has default armor we need to account
        # The test creates target with armor=0, so final should be 80
        # Actually, looking at the creation: armor=0 by default param not being passed
        # Let's verify the actual calculation is correct
        assert result.damage_event.final_damage == pytest.approx(80, rel=0.01)

    def test_omnivamp(self, attack_system):
        """Test omnivamp healing."""
        attacker = create_test_unit("attacker", ad=100, hp=500)
        attacker.stats.current_hp = 400  # Damaged
        attacker.stats.omnivamp = 0.2  # 20% omnivamp
        target = create_test_unit("target", Team.RED, armor=0)

        attack_system.execute_attack(attacker, target)

        # Should heal for 20% of damage dealt
        assert attacker.stats.current_hp > 400

    def test_dodge(self):
        """Test dodge mechanics."""
        # Use RNG that will trigger dodge
        attack_system = AttackSystem(rng=random.Random(1))

        attacker = create_test_unit("attacker", ad=100)
        target = create_test_unit("target", Team.RED)
        target.stats.dodge_chance = 1.0  # Always dodge

        result = attack_system.execute_attack(attacker, target)

        assert result.damage_event.final_damage == 0

    def test_damage_tracking(self, attack_system):
        """Test damage dealt tracking."""
        attacker = create_test_unit("attacker", ad=100)
        target = create_test_unit("target", Team.RED, armor=0)

        attack_system.execute_attack(attacker, target)

        assert attacker.total_damage_dealt > 0
        assert target.total_damage_taken > 0


class TestCalculateDamage:
    """Tests for calculate_damage method."""

    @pytest.fixture
    def attack_system(self):
        return AttackSystem(rng=random.Random(42))

    def test_physical_damage(self, attack_system):
        """Test physical damage calculation."""
        source = create_test_unit("source")
        target = create_test_unit("target", Team.RED, armor=50)

        event = attack_system.calculate_damage(
            source, target, 100, DamageType.PHYSICAL
        )

        # 50 armor = 33.3% reduction
        expected = 100 * (100 / 150)
        assert event.final_damage == pytest.approx(expected, rel=0.01)

    def test_magical_damage(self, attack_system):
        """Test magical damage calculation."""
        source = create_test_unit("source")
        target = create_test_unit("target", Team.RED, mr=50)

        event = attack_system.calculate_damage(
            source, target, 100, DamageType.MAGICAL
        )

        # 50 MR = 33.3% reduction
        expected = 100 * (100 / 150)
        assert event.final_damage == pytest.approx(expected, rel=0.01)

    def test_true_damage(self, attack_system):
        """Test true damage ignores resistances."""
        source = create_test_unit("source")
        target = create_test_unit("target", Team.RED, armor=100, mr=100)

        event = attack_system.calculate_damage(
            source, target, 100, DamageType.TRUE
        )

        assert event.final_damage == 100


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_effective_hp(self):
        """Test effective HP calculation."""
        ehp = calculate_effective_hp(1000, 100, 50)

        # Should be higher than base HP
        assert ehp > 1000

    def test_calculate_dps(self):
        """Test DPS calculation."""
        dps = calculate_dps(
            attack_damage=100,
            attack_speed=1.0,
            crit_chance=0.25,
            crit_damage=1.4,
        )

        # 100 * (1 + 0.25 * 0.4) * 1.0 = 110
        assert dps == pytest.approx(110)
