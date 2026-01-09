"""Tests for CombatUnit."""

import pytest

from src.combat.combat_unit import CombatUnit, CombatStats, CombatResult, UnitState
from src.combat.hex_grid import Team


class TestCombatStats:
    """Tests for CombatStats dataclass."""

    def test_create_stats(self):
        """Create stats."""
        stats = CombatStats(
            max_hp=1000,
            current_hp=1000,
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
        assert stats.max_hp == 1000
        assert stats.attack_damage == 100


class TestCombatUnit:
    """Tests for CombatUnit class."""

    @pytest.fixture
    def basic_unit(self):
        """Create basic test unit."""
        stats = CombatStats(
            max_hp=1000,
            current_hp=1000,
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
            id="test-unit-1",
            name="Test Champion",
            champion_id="TFT_Test",
            star_level=1,
            team=Team.BLUE,
            stats=stats,
        )

    def test_is_alive(self, basic_unit):
        """Check if unit is alive."""
        assert basic_unit.is_alive
        basic_unit.stats.current_hp = 0
        assert not basic_unit.is_alive

    def test_is_alive_dead_state(self, basic_unit):
        """Dead state means not alive."""
        basic_unit.state = UnitState.DEAD
        assert not basic_unit.is_alive

    def test_can_act(self, basic_unit):
        """Check if unit can act."""
        assert basic_unit.can_act
        basic_unit.state = UnitState.STUNNED
        assert not basic_unit.can_act

    def test_can_act_dead(self, basic_unit):
        """Dead units cannot act."""
        basic_unit.state = UnitState.DEAD
        assert not basic_unit.can_act

    def test_attack_interval(self, basic_unit):
        """Attack interval calculation."""
        # Attack speed 1.0 = 1 attack per second
        assert basic_unit.attack_interval == 1.0

        # Attack speed 2.0 = 0.5 seconds between attacks
        basic_unit.stats.attack_speed = 2.0
        assert basic_unit.attack_interval == 0.5

    def test_attack_interval_zero_speed(self, basic_unit):
        """Zero attack speed gives infinite interval."""
        basic_unit.stats.attack_speed = 0
        assert basic_unit.attack_interval == float("inf")

    def test_take_physical_damage(self, basic_unit):
        """Physical damage (pre-mitigated by AttackSystem).

        Note: Armor/MR reduction is now handled in AttackSystem._calculate_attack_damage(),
        not in take_damage(). The amount passed to take_damage() is already mitigated.
        This test verifies that take_damage() applies only damage_reduction stat.
        """
        # take_damage receives already-mitigated damage (armor applied in AttackSystem)
        # So 100 damage in = 100 damage taken (with 0 damage_reduction)
        damage = basic_unit.take_damage(100, "physical")
        assert damage == 100
        assert basic_unit.stats.current_hp == 900  # 1000 - 100

    def test_take_magical_damage(self, basic_unit):
        """Magical damage (pre-mitigated by AttackSystem).

        Note: Armor/MR reduction is now handled in AttackSystem.calculate_damage(),
        not in take_damage(). The amount passed to take_damage() is already mitigated.
        """
        # take_damage receives already-mitigated damage
        damage = basic_unit.take_damage(100, "magical")
        assert damage == 100

    def test_take_true_damage(self, basic_unit):
        """True damage ignores defenses."""
        damage = basic_unit.take_damage(100, "true")
        assert damage == 100

    def test_damage_kills_unit(self, basic_unit):
        """Lethal damage kills unit."""
        basic_unit.take_damage(10000, "true")
        assert not basic_unit.is_alive
        assert basic_unit.state == UnitState.DEAD
        assert basic_unit.stats.current_hp == 0

    def test_damage_reduction(self, basic_unit):
        """Damage reduction effect."""
        basic_unit.stats.damage_reduction = 0.2  # 20% reduction
        damage = basic_unit.take_damage(100, "true")
        assert damage == 80

    def test_damage_to_dead_unit(self, basic_unit):
        """Dead units take no damage."""
        basic_unit.state = UnitState.DEAD
        basic_unit.stats.current_hp = 0
        damage = basic_unit.take_damage(100, "true")
        assert damage == 0

    def test_heal(self, basic_unit):
        """Heal unit."""
        basic_unit.stats.current_hp = 500
        healed = basic_unit.heal(200)
        assert healed == 200
        assert basic_unit.stats.current_hp == 700

    def test_heal_no_overheal(self, basic_unit):
        """Cannot heal above max HP."""
        basic_unit.stats.current_hp = 900
        healed = basic_unit.heal(200)
        assert healed == 100
        assert basic_unit.stats.current_hp == 1000

    def test_heal_dead_unit(self, basic_unit):
        """Dead units cannot be healed."""
        basic_unit.state = UnitState.DEAD
        basic_unit.stats.current_hp = 0
        healed = basic_unit.heal(500)
        assert healed == 0

    def test_gain_mana(self, basic_unit):
        """Gain mana."""
        basic_unit.stats.current_mana = 50
        basic_unit.gain_mana(30)
        assert basic_unit.stats.current_mana == 80

    def test_gain_mana_cap(self, basic_unit):
        """Mana capped at max."""
        basic_unit.stats.current_mana = 90
        basic_unit.gain_mana(30)
        assert basic_unit.stats.current_mana == 100

    def test_gain_mana_dead(self, basic_unit):
        """Dead units don't gain mana."""
        basic_unit.state = UnitState.DEAD
        basic_unit.stats.current_hp = 0
        basic_unit.stats.current_mana = 50
        basic_unit.gain_mana(30)
        assert basic_unit.stats.current_mana == 50

    def test_spend_mana(self, basic_unit):
        """Spend mana resets to starting mana."""
        basic_unit.stats.current_mana = 100
        basic_unit.spend_mana()
        assert basic_unit.stats.current_mana == basic_unit.stats.starting_mana

    def test_can_cast(self, basic_unit):
        """Check if unit can cast ability."""
        basic_unit.stats.current_mana = 50
        assert not basic_unit.can_cast

        basic_unit.stats.current_mana = 100
        assert basic_unit.can_cast

        basic_unit.is_casting = True
        assert not basic_unit.can_cast

    def test_can_attack(self, basic_unit):
        """Check if unit can attack."""
        assert basic_unit.can_attack

        basic_unit.attack_cooldown = 0.5
        assert not basic_unit.can_attack

        basic_unit.attack_cooldown = 0
        basic_unit.is_casting = True
        assert not basic_unit.can_attack

    def test_reset_for_combat(self, basic_unit):
        """Reset unit for new combat."""
        basic_unit.stats.current_hp = 500
        basic_unit.stats.current_mana = 100
        basic_unit.total_damage_dealt = 1000
        basic_unit.state = UnitState.ATTACKING
        basic_unit.kills = 5

        basic_unit.reset_for_combat()

        assert basic_unit.stats.current_hp == basic_unit.stats.max_hp
        assert basic_unit.stats.current_mana == basic_unit.stats.starting_mana
        assert basic_unit.total_damage_dealt == 0
        assert basic_unit.state == UnitState.IDLE
        assert basic_unit.kills == 0

    def test_damage_tracking(self, basic_unit):
        """Damage statistics tracking."""
        basic_unit.take_damage(100, "true")
        basic_unit.take_damage(50, "true")
        assert basic_unit.total_damage_taken == 150

    def test_healing_tracking(self, basic_unit):
        """Healing statistics tracking."""
        basic_unit.stats.current_hp = 500
        basic_unit.heal(100)
        basic_unit.heal(50)
        assert basic_unit.total_healing_done == 150

    def test_is_targetable(self, basic_unit):
        """Check if unit is targetable."""
        assert basic_unit.is_targetable

        basic_unit.state = UnitState.DEAD
        assert not basic_unit.is_targetable

    def test_repr(self, basic_unit):
        """String representation."""
        repr_str = repr(basic_unit)
        assert "Test Champion" in repr_str
        assert "*" in repr_str  # Star level


class TestCombatResult:
    """Tests for CombatResult dataclass."""

    def test_create_result(self):
        """Create combat result."""
        result = CombatResult(
            winner=Team.BLUE,
            winning_units_remaining=3,
            losing_units_remaining=0,
            rounds_taken=50,
            total_damage_to_loser=15.0,
        )
        assert result.winner == Team.BLUE
        assert result.winning_units_remaining == 3
        assert result.total_damage_to_loser == 15.0

    def test_result_with_unit_stats(self):
        """Combat result with unit statistics."""
        result = CombatResult(
            winner=Team.RED,
            winning_units_remaining=2,
            losing_units_remaining=0,
            rounds_taken=100,
            total_damage_to_loser=20.0,
            unit_stats={
                "unit1": {"damage_dealt": 5000, "damage_taken": 1000, "kills": 3},
                "unit2": {"damage_dealt": 3000, "damage_taken": 500, "kills": 2},
            },
        )
        assert len(result.unit_stats) == 2
        assert result.unit_stats["unit1"]["kills"] == 3


class TestUnitState:
    """Tests for UnitState enum."""

    def test_all_states(self):
        """All unit states exist."""
        assert UnitState.IDLE
        assert UnitState.MOVING
        assert UnitState.ATTACKING
        assert UnitState.CASTING
        assert UnitState.STUNNED
        assert UnitState.DEAD
