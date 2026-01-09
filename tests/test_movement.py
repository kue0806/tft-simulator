"""Tests for Movement System."""

import pytest
from src.combat.hex_grid import HexGrid, HexPosition, Team
from src.combat.combat_unit import CombatUnit, CombatStats
from src.combat.movement import MovementSystem, MOVE_TIME_PER_HEX


def create_test_unit(unit_id: str, team: Team = Team.BLUE) -> CombatUnit:
    """Create a test combat unit."""
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
        id=unit_id,
        name=f"Unit_{unit_id}",
        champion_id="TFT_Test",
        star_level=1,
        team=team,
        stats=stats,
    )


class TestMovementSystem:
    """MovementSystem tests."""

    @pytest.fixture
    def setup(self):
        """Create grid and movement system."""
        grid = HexGrid()
        movement = MovementSystem(grid, move_time_per_hex=0.5)
        return grid, movement

    def test_start_move_to_position(self, setup):
        """Test starting movement to position."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))

        result = movement.start_move_to_position(unit, HexPosition(2, 0))

        assert result is True
        assert movement.is_moving("unit1")

    def test_start_move_same_position(self, setup):
        """Test movement to same position fails."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)

        result = movement.start_move_to_position(unit, pos)

        assert result is False
        assert not movement.is_moving("unit1")

    def test_update_moves_unit(self, setup):
        """Test update actually moves unit."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(1, 0))

        # Update with enough time to complete
        movement.update(unit, delta_time=1.0)

        assert grid.get_unit_position("unit1") == HexPosition(1, 0)

    def test_update_partial_progress(self, setup):
        """Test partial movement progress."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(2, 0))

        # Half of one hex movement
        movement.update(unit, delta_time=0.25)

        progress = movement.get_movement_progress("unit1")
        assert 0 < progress < 1.0
        assert movement.is_moving("unit1")

    def test_move_to_target_range(self, setup):
        """Test movement to target's attack range."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        target_pos = HexPosition(4, 4)

        result = movement.start_move_to_target(unit, target_pos, attack_range=1)

        assert result is True
        assert movement.is_moving("unit1")

    def test_already_in_range(self, setup):
        """Test no movement when already in range."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(3, 3))
        target_pos = HexPosition(4, 3)  # Distance 1

        result = movement.start_move_to_target(unit, target_pos, attack_range=1)

        assert result is False
        assert not movement.is_moving("unit1")

    def test_stop_movement(self, setup):
        """Test stopping movement."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(5, 5))

        assert movement.is_moving("unit1")

        movement.stop_movement("unit1")

        assert not movement.is_moving("unit1")

    def test_recalculate_path(self, setup):
        """Test path recalculation."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_target(unit, HexPosition(4, 4), attack_range=1)

        # Recalculate to new target
        result = movement.recalculate_path(unit, HexPosition(4, 0), attack_range=1)

        assert result is True

    def test_blocked_destination(self, setup):
        """Test movement to blocked destination."""
        grid, movement = setup

        unit1 = create_test_unit("unit1")
        unit2 = create_test_unit("unit2")

        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 0))  # Block destination

        result = movement.start_move_to_position(unit1, HexPosition(1, 0))

        assert result is False

    def test_clear_all_movement(self, setup):
        """Test clearing all movement states."""
        grid, movement = setup

        unit1 = create_test_unit("unit1")
        unit2 = create_test_unit("unit2")

        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(0, 1))

        movement.start_move_to_position(unit1, HexPosition(5, 5))
        movement.start_move_to_position(unit2, HexPosition(5, 4))

        movement.clear()

        assert not movement.is_moving("unit1")
        assert not movement.is_moving("unit2")

    def test_get_remaining_path(self, setup):
        """Test getting remaining path."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(3, 0))

        path = movement.get_remaining_path("unit1")

        assert len(path) == 3

    def test_movement_completes(self, setup):
        """Test movement completion."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(1, 0))

        # Update until complete
        completed = movement.update(unit, delta_time=1.0)

        assert completed is True
        assert not movement.is_moving("unit1")
        assert grid.get_unit_position("unit1") == HexPosition(1, 0)

    def test_multi_hex_movement(self, setup):
        """Test movement across multiple hexes."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        start = HexPosition(0, 0)
        goal = HexPosition(3, 0)
        grid.place_unit("unit1", start)
        movement.start_move_to_position(unit, goal)

        # Move one hex at a time
        for _ in range(10):  # More updates than needed
            completed = movement.update(unit, delta_time=0.5)
            if completed:
                break

        assert grid.get_unit_position("unit1") == goal

    def test_get_stats(self, setup):
        """Test movement statistics."""
        grid, movement = setup
        unit = create_test_unit("unit1")

        grid.place_unit("unit1", HexPosition(0, 0))
        movement.start_move_to_position(unit, HexPosition(2, 0))

        # Complete movement
        for _ in range(10):
            movement.update(unit, delta_time=0.5)

        stats = movement.get_stats("unit1")

        assert stats["total_distance_moved"] == 2
        assert stats["is_moving"] is False
