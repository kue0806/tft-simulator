"""Tests for Targeting System."""

import pytest
import random
from src.combat.hex_grid import HexGrid, HexPosition, Team
from src.combat.combat_unit import CombatUnit, CombatStats, UnitState
from src.combat.targeting import (
    TargetSelector,
    TargetingContext,
    TargetingPriority,
)


def create_test_unit(
    unit_id: str,
    team: Team,
    hp: float = 1000,
    mana: float = 50,
    attack_range: int = 1,
    attack_damage: float = 100,
    armor: float = 50,
) -> CombatUnit:
    """Create a test combat unit."""
    stats = CombatStats(
        max_hp=hp,
        current_hp=hp,
        attack_damage=attack_damage,
        ability_power=100,
        armor=armor,
        magic_resist=50,
        attack_speed=1.0,
        crit_chance=0.25,
        crit_damage=1.4,
        max_mana=100,
        current_mana=mana,
        starting_mana=50,
        attack_range=attack_range,
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


class TestTargetSelector:
    """TargetSelector tests."""

    @pytest.fixture
    def setup_battle(self):
        """Basic battle setup with multiple units."""
        grid = HexGrid()
        units = {}

        # BLUE team unit (row 0-3)
        blue1 = create_test_unit("blue1", Team.BLUE)
        grid.place_unit("blue1", HexPosition(3, 3))
        units["blue1"] = blue1

        # RED team units (row 4-7) at different distances
        red1 = create_test_unit("red1", Team.RED)  # Distance 1
        red2 = create_test_unit("red2", Team.RED, hp=500)  # Distance 2, low HP
        red3 = create_test_unit("red3", Team.RED, hp=1500)  # Distance 3, high HP

        grid.place_unit("red1", HexPosition(4, 3))  # Distance 1
        grid.place_unit("red2", HexPosition(5, 3))  # Distance 2
        grid.place_unit("red3", HexPosition(6, 3))  # Distance 3

        units["red1"] = red1
        units["red2"] = red2
        units["red3"] = red3

        context = TargetingContext(grid=grid, all_units=units)
        selector = TargetSelector(context, rng=random.Random(42))

        return grid, units, selector

    def test_find_nearest_target(self, setup_battle):
        """Test nearest enemy targeting."""
        grid, units, selector = setup_battle

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.NEAREST
        )

        assert target == "red1"  # Distance 1

    def test_find_farthest_target(self, setup_battle):
        """Test farthest enemy targeting (assassin style)."""
        grid, units, selector = setup_battle

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.FARTHEST
        )

        assert target == "red3"  # Distance 3

    def test_find_lowest_hp_target(self, setup_battle):
        """Test lowest HP targeting."""
        grid, units, selector = setup_battle

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.LOWEST_HP
        )

        assert target == "red2"  # HP 500

    def test_find_highest_hp_target(self, setup_battle):
        """Test highest HP targeting."""
        grid, units, selector = setup_battle

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.HIGHEST_HP
        )

        assert target == "red3"  # HP 1500

    def test_no_target_same_team(self, setup_battle):
        """Same team units are not valid targets."""
        grid, units, selector = setup_battle

        # Add another blue unit
        blue2 = create_test_unit("blue2", Team.BLUE)
        grid.place_unit("blue2", HexPosition(2, 3))
        units["blue2"] = blue2

        # blue2's target should be a RED unit
        target = selector.find_target(
            units["blue2"], priority=TargetingPriority.NEAREST
        )

        assert target in ["red1", "red2", "red3"]

    def test_no_target_dead_unit(self, setup_battle):
        """Dead units are not valid targets."""
        grid, units, selector = setup_battle

        # Kill red1
        units["red1"].state = UnitState.DEAD
        units["red1"].stats.current_hp = 0

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.NEAREST
        )

        assert target == "red2"  # Next nearest

    def test_is_in_range_true(self, setup_battle):
        """Test range check - in range."""
        grid, units, selector = setup_battle

        # blue1(3,3) -> red1(4,3) = distance 1
        assert selector.is_in_range(units["blue1"], "red1", attack_range=1)

    def test_is_in_range_false(self, setup_battle):
        """Test range check - out of range."""
        grid, units, selector = setup_battle

        # blue1(3,3) -> red2(5,3) = distance 2
        assert not selector.is_in_range(units["blue1"], "red2", attack_range=1)

    def test_get_units_in_range(self, setup_battle):
        """Test getting all units in range."""
        grid, units, selector = setup_battle

        in_range = selector.get_units_in_range(units["blue1"], range_=2)

        assert "red1" in in_range  # Distance 1
        assert "red2" in in_range  # Distance 2
        assert "red3" not in in_range  # Distance 3

    def test_filter_function(self, setup_battle):
        """Test custom filter function."""
        grid, units, selector = setup_battle

        # Only target units with HP > 1000
        target = selector.find_target(
            units["blue1"],
            priority=TargetingPriority.NEAREST,
            filter_fn=lambda u: u.stats.current_hp > 1000,
        )

        assert target == "red3"  # Only one with HP > 1000

    def test_tiebreak_by_position(self):
        """Test deterministic tiebreaker when distances are equal."""
        grid = HexGrid()
        units = {}

        blue = create_test_unit("blue", Team.BLUE)
        grid.place_unit("blue", HexPosition(3, 3))
        units["blue"] = blue

        # Two enemies at equal distance
        red1 = create_test_unit("red1", Team.RED)
        red2 = create_test_unit("red2", Team.RED)

        grid.place_unit("red1", HexPosition(4, 4))  # Distance 1
        grid.place_unit("red2", HexPosition(4, 3))  # Distance 1

        units["red1"] = red1
        units["red2"] = red2

        context = TargetingContext(grid=grid, all_units=units)
        selector = TargetSelector(context)

        target = selector.find_target(units["blue"], priority=TargetingPriority.NEAREST)

        # Same row, lower col wins
        assert target == "red2"

    def test_acquire_target_keeps_current(self, setup_battle):
        """Test that acquire_target keeps current valid target."""
        grid, units, selector = setup_battle

        # Set current target
        units["blue1"].current_target_id = "red2"

        # Acquire should keep red2 since it's still valid
        target = selector.acquire_target(units["blue1"], keep_current=True)

        assert target == "red2"

    def test_acquire_target_finds_new_when_current_dead(self, setup_battle):
        """Test acquiring new target when current is dead."""
        grid, units, selector = setup_battle

        # Set current target and kill it
        units["blue1"].current_target_id = "red1"
        units["red1"].state = UnitState.DEAD
        units["red1"].stats.current_hp = 0

        # Should find new target
        target = selector.acquire_target(units["blue1"], keep_current=True)

        assert target in ["red2", "red3"]

    def test_lowest_hp_percent_target(self, setup_battle):
        """Test lowest HP percentage targeting."""
        grid, units, selector = setup_battle

        # Modify HP percentages
        units["red1"].stats.current_hp = 800  # 80%
        units["red2"].stats.max_hp = 1000
        units["red2"].stats.current_hp = 300  # 60% (was 500/500=100%)
        units["red3"].stats.current_hp = 1400  # ~93%

        target = selector.find_target(
            units["blue1"], priority=TargetingPriority.LOWEST_HP_PERCENT
        )

        # red2 has lowest HP% now
        assert target == "red2"

    def test_random_target(self, setup_battle):
        """Test random targeting produces different results."""
        grid, units, selector = setup_battle

        targets = set()
        for _ in range(50):
            target = selector.find_target(
                units["blue1"], priority=TargetingPriority.RANDOM
            )
            targets.add(target)

        # Should hit multiple targets with random selection
        assert len(targets) >= 2
