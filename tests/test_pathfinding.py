"""Tests for Pathfinding System."""

import pytest
from src.combat.hex_grid import HexGrid, HexPosition
from src.combat.pathfinding import PathFinder, get_blocked_positions, get_walkable_neighbors


class TestPathFinder:
    """PathFinder tests."""

    @pytest.fixture
    def grid_and_finder(self):
        """Create grid and finder."""
        grid = HexGrid()
        finder = PathFinder(grid)
        return grid, finder

    def test_find_path_simple(self, grid_and_finder):
        """Test simple path finding."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        goal = HexPosition(0, 2)

        path = finder.find_path(start, goal)

        assert path is not None
        assert len(path) == 2  # 2 hex moves
        assert path[-1] == goal

    def test_find_path_same_position(self, grid_and_finder):
        """Test path to same position returns empty."""
        grid, finder = grid_and_finder

        pos = HexPosition(3, 3)
        path = finder.find_path(pos, pos)

        assert path == []

    def test_find_path_around_obstacle(self, grid_and_finder):
        """Test pathfinding around obstacles."""
        grid, finder = grid_and_finder

        start = HexPosition(2, 0)
        goal = HexPosition(2, 2)

        # Block direct path
        blocked = {HexPosition(2, 1)}

        path = finder.find_path(start, goal, blocked)

        assert path is not None
        assert HexPosition(2, 1) not in path
        assert path[-1] == goal

    def test_find_path_completely_blocked(self, grid_and_finder):
        """Test when path is completely blocked."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)

        # Surround start position
        blocked = set()
        for neighbor in start.get_neighbors():
            if neighbor.is_valid(grid.ROWS, grid.COLS):
                blocked.add(neighbor)

        # Try to reach a far position
        path = finder.find_path(start, HexPosition(4, 4), blocked)

        assert path is None

    def test_find_path_to_range(self, grid_and_finder):
        """Test finding path to within attack range."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        target = HexPosition(4, 4)

        path = finder.find_path_to_range(start, target, attack_range=1)

        assert path is not None
        if path:
            final_pos = path[-1]
            assert final_pos.distance_to(target) <= 1

    def test_find_path_already_in_range(self, grid_and_finder):
        """Test when already in attack range."""
        grid, finder = grid_and_finder

        start = HexPosition(3, 3)
        target = HexPosition(4, 3)  # Distance 1

        path = finder.find_path_to_range(start, target, attack_range=1)

        assert path == []  # No movement needed

    def test_get_next_step(self, grid_and_finder):
        """Test getting only the next step."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        goal = HexPosition(2, 2)

        next_step = finder.get_next_step(start, goal)

        assert next_step is not None
        assert start.distance_to(next_step) == 1

    def test_path_length_optimal(self, grid_and_finder):
        """Test that path is optimal length."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        goal = HexPosition(3, 0)

        path = finder.find_path(start, goal)

        assert path is not None
        assert len(path) == start.distance_to(goal)

    def test_find_path_diagonal(self, grid_and_finder):
        """Test diagonal movement."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        goal = HexPosition(3, 3)

        path = finder.find_path(start, goal)

        assert path is not None
        # Each step should be adjacent
        current = start
        for pos in path:
            assert current.distance_to(pos) == 1
            current = pos

    def test_get_next_step_to_range(self, grid_and_finder):
        """Test getting next step to get in range."""
        grid, finder = grid_and_finder

        start = HexPosition(0, 0)
        target = HexPosition(5, 5)

        next_step = finder.get_next_step_to_range(start, target, attack_range=2)

        assert next_step is not None
        assert start.distance_to(next_step) == 1


class TestGetBlockedPositions:
    """Tests for get_blocked_positions helper."""

    def test_empty_grid(self):
        """Test with empty grid."""
        grid = HexGrid()
        blocked = get_blocked_positions(grid)
        assert len(blocked) == 0

    def test_with_units(self):
        """Test with units on grid."""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))

        blocked = get_blocked_positions(grid)

        assert HexPosition(0, 0) in blocked
        assert HexPosition(1, 1) in blocked
        assert len(blocked) == 2

    def test_exclude_unit(self):
        """Test excluding specific unit."""
        grid = HexGrid()
        grid.place_unit("unit1", HexPosition(0, 0))
        grid.place_unit("unit2", HexPosition(1, 1))

        blocked = get_blocked_positions(grid, exclude_unit_id="unit1")

        assert HexPosition(0, 0) not in blocked
        assert HexPosition(1, 1) in blocked
        assert len(blocked) == 1


class TestGetWalkableNeighbors:
    """Tests for get_walkable_neighbors helper."""

    def test_center_position(self):
        """Test walkable neighbors in center."""
        grid = HexGrid()
        pos = HexPosition(3, 3)

        neighbors = get_walkable_neighbors(grid, pos)

        assert len(neighbors) == 6  # All 6 valid

    def test_corner_position(self):
        """Test walkable neighbors in corner."""
        grid = HexGrid()
        pos = HexPosition(0, 0)

        neighbors = get_walkable_neighbors(grid, pos)

        assert len(neighbors) < 6

    def test_with_blocked(self):
        """Test with blocked positions."""
        grid = HexGrid()
        pos = HexPosition(3, 3)
        blocked = {HexPosition(3, 4), HexPosition(2, 3)}

        neighbors = get_walkable_neighbors(grid, pos, blocked)

        assert HexPosition(3, 4) not in neighbors
        assert HexPosition(2, 3) not in neighbors
        assert len(neighbors) == 4
