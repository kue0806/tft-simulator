"""Tests for HexGrid and HexPosition."""

import pytest

from src.combat.hex_grid import HexPosition, HexGrid, Team


class TestHexPosition:
    """Tests for HexPosition class."""

    def test_create_position(self):
        """Create position."""
        pos = HexPosition(0, 0)
        assert pos.row == 0
        assert pos.col == 0

    def test_position_equality(self):
        """Position equality."""
        pos1 = HexPosition(1, 2)
        pos2 = HexPosition(1, 2)
        pos3 = HexPosition(1, 3)
        assert pos1 == pos2
        assert pos1 != pos3

    def test_position_hashable(self):
        """Position can be used as dict key."""
        pos1 = HexPosition(1, 2)
        pos2 = HexPosition(1, 2)
        d = {pos1: "unit1"}
        assert d[pos2] == "unit1"

    def test_distance_same_position(self):
        """Distance to self is 0."""
        pos = HexPosition(3, 3)
        assert pos.distance_to(pos) == 0

    def test_distance_adjacent(self):
        """Distance to adjacent hex is 1."""
        pos = HexPosition(2, 2)
        for neighbor in pos.get_neighbors():
            assert pos.distance_to(neighbor) == 1

    def test_distance_two_hexes(self):
        """Distance of 2 hexes."""
        pos1 = HexPosition(0, 0)
        pos2 = HexPosition(2, 0)
        assert pos1.distance_to(pos2) == 2

    def test_get_neighbors_count(self):
        """Each hex has 6 neighbors."""
        pos = HexPosition(3, 3)  # Center position
        neighbors = pos.get_neighbors()
        assert len(neighbors) == 6

    def test_get_neighbors_even_row(self):
        """Neighbors for even row."""
        pos = HexPosition(2, 3)
        neighbors = pos.get_neighbors()
        expected = [
            HexPosition(1, 3),
            HexPosition(1, 2),  # Upper
            HexPosition(2, 2),
            HexPosition(2, 4),  # Left/Right
            HexPosition(3, 3),
            HexPosition(3, 2),  # Lower
        ]
        assert set(neighbors) == set(expected)

    def test_get_neighbors_odd_row(self):
        """Neighbors for odd row."""
        pos = HexPosition(3, 3)
        neighbors = pos.get_neighbors()
        expected = [
            HexPosition(2, 4),
            HexPosition(2, 3),  # Upper
            HexPosition(3, 2),
            HexPosition(3, 4),  # Left/Right
            HexPosition(4, 4),
            HexPosition(4, 3),  # Lower
        ]
        assert set(neighbors) == set(expected)

    def test_is_valid(self):
        """Validity check for board bounds."""
        assert HexPosition(0, 0).is_valid()
        assert HexPosition(7, 6).is_valid()
        assert not HexPosition(-1, 0).is_valid()
        assert not HexPosition(8, 0).is_valid()
        assert not HexPosition(0, 7).is_valid()

    def test_cube_conversion_roundtrip(self):
        """Cube coordinate conversion roundtrip."""
        for row in range(8):
            for col in range(7):
                pos = HexPosition(row, col)
                cube = pos.to_cube()
                back = HexPosition.from_cube(*cube)
                assert pos == back


class TestHexGrid:
    """Tests for HexGrid class."""

    def test_create_grid(self):
        """Create grid."""
        grid = HexGrid()
        assert grid.ROWS == 8
        assert grid.COLS == 7

    def test_place_unit(self):
        """Place unit on grid."""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        assert grid.place_unit("unit1", pos)
        assert grid.get_unit_at(pos) == "unit1"
        assert grid.get_unit_position("unit1") == pos

    def test_place_unit_occupied(self):
        """Cannot place on occupied position."""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)
        assert not grid.place_unit("unit2", pos)

    def test_place_unit_invalid_position(self):
        """Placing on invalid position raises error."""
        grid = HexGrid()
        with pytest.raises(ValueError):
            grid.place_unit("unit1", HexPosition(10, 10))

    def test_remove_unit(self):
        """Remove unit from grid."""
        grid = HexGrid()
        pos = HexPosition(0, 0)
        grid.place_unit("unit1", pos)

        removed_pos = grid.remove_unit("unit1")
        assert removed_pos == pos
        assert grid.get_unit_at(pos) is None
        assert grid.get_unit_position("unit1") is None

    def test_remove_nonexistent_unit(self):
        """Remove non-existent unit returns None."""
        grid = HexGrid()
        assert grid.remove_unit("nonexistent") is None

    def test_move_unit(self):
        """Move unit to new position."""
        grid = HexGrid()
        pos1 = HexPosition(0, 0)
        pos2 = HexPosition(1, 1)
        grid.place_unit("unit1", pos1)

        assert grid.move_unit("unit1", pos2)
        assert grid.get_unit_at(pos1) is None
        assert grid.get_unit_at(pos2) == "unit1"
        assert grid.get_unit_position("unit1") == pos2

    def test_move_unit_to_occupied(self):
        """Cannot move to occupied position."""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))

        assert not grid.move_unit("unit1", HexPosition(1, 1))

    def test_get_valid_neighbors(self):
        """Get valid neighbors within board."""
        grid = HexGrid()

        # Corner has fewer valid neighbors
        corner = HexPosition(0, 0)
        valid = grid.get_valid_neighbors(corner)
        assert len(valid) < 6

        # Center has all 6 valid
        center = HexPosition(3, 3)
        valid = grid.get_valid_neighbors(center)
        assert len(valid) == 6

    def test_get_empty_neighbors(self):
        """Get empty neighboring hexes."""
        grid = HexGrid()
        center = HexPosition(3, 3)

        # Initially all empty
        empty = grid.get_empty_neighbors(center)
        assert len(empty) == 6

        # Fill one, now 5 empty
        grid.place_unit("unit1", HexPosition(2, 3))
        empty = grid.get_empty_neighbors(center)
        assert len(empty) == 5

    def test_get_units_in_range(self):
        """Get units within range."""
        grid = HexGrid()
        center = HexPosition(3, 3)
        grid.place_unit("center", center)
        grid.place_unit("adjacent", HexPosition(2, 3))  # Distance 1
        grid.place_unit("far", HexPosition(0, 0))  # Far away

        # Range 1: center + adjacent
        in_range = grid.get_units_in_range(center, 1)
        assert "center" in in_range
        assert "adjacent" in in_range
        assert "far" not in in_range

    def test_get_all_units(self):
        """Get all units on grid."""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))

        all_units = grid.get_all_units()
        assert len(all_units) == 2
        assert "unit1" in all_units
        assert "unit2" in all_units

    def test_clear(self):
        """Clear grid."""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.clear()

        assert len(grid.get_all_units()) == 0

    def test_team_for_position(self):
        """Team region detection."""
        grid = HexGrid()

        # Rows 0-3 are BLUE
        assert grid.get_team_for_position(HexPosition(0, 0)) == Team.BLUE
        assert grid.get_team_for_position(HexPosition(3, 6)) == Team.BLUE

        # Rows 4-7 are RED
        assert grid.get_team_for_position(HexPosition(4, 0)) == Team.RED
        assert grid.get_team_for_position(HexPosition(7, 6)) == Team.RED

    def test_mirror_position(self):
        """Mirror position calculation."""
        # (0, 0) -> (7, 6)
        assert HexGrid.mirror_position(HexPosition(0, 0)) == HexPosition(7, 6)

        # (0, 6) -> (7, 0)
        assert HexGrid.mirror_position(HexPosition(0, 6)) == HexPosition(7, 0)

        # (3, 3) -> (4, 3)
        assert HexGrid.mirror_position(HexPosition(3, 3)) == HexPosition(4, 3)

        # Double mirror returns original
        pos = HexPosition(2, 4)
        assert HexGrid.mirror_position(HexGrid.mirror_position(pos)) == pos

    def test_is_position_empty(self):
        """Check if position is empty."""
        grid = HexGrid()
        pos = HexPosition(0, 0)

        assert grid.is_position_empty(pos)

        grid.place_unit("unit1", pos)
        assert not grid.is_position_empty(pos)

    def test_get_units_by_team(self):
        """Get units filtered by team."""
        grid = HexGrid()
        grid.place_unit("blue1", HexPosition(0, 0))  # BLUE
        grid.place_unit("blue2", HexPosition(1, 1))  # BLUE
        grid.place_unit("red1", HexPosition(4, 0))  # RED

        blue_units = grid.get_units_by_team(Team.BLUE)
        assert len(blue_units) == 2
        assert "blue1" in blue_units
        assert "blue2" in blue_units
        assert "red1" not in blue_units

        red_units = grid.get_units_by_team(Team.RED)
        assert len(red_units) == 1
        assert "red1" in red_units


class TestTeam:
    """Tests for Team enum."""

    def test_team_values(self):
        """Team values."""
        assert Team.BLUE.value == "blue"
        assert Team.RED.value == "red"
