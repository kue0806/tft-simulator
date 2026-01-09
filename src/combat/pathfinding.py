"""A* Pathfinding for TFT Combat.

Implements A* pathfinding on the hex grid for unit movement.
Handles obstacle avoidance and path-to-range calculations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from heapq import heappush, heappop

from .hex_grid import HexGrid, HexPosition


@dataclass(order=True)
class PathNode:
    """A* search node."""

    f_score: float  # g + h (total estimated cost)
    position: HexPosition = field(compare=False)
    g_score: float = field(compare=False)  # Actual cost from start
    parent: Optional["PathNode"] = field(default=None, compare=False)


class PathFinder:
    """
    A* pathfinding on hex grid.

    Usage:
        finder = PathFinder(grid)
        path = finder.find_path(start, goal, blocked_positions)
    """

    def __init__(self, grid: HexGrid):
        """
        Initialize pathfinder.

        Args:
            grid: The hex grid to pathfind on.
        """
        self.grid = grid

    def find_path(
        self,
        start: HexPosition,
        goal: HexPosition,
        blocked: Optional[Set[HexPosition]] = None,
        max_iterations: int = 1000,
    ) -> Optional[List[HexPosition]]:
        """
        Find shortest path from start to goal.

        Args:
            start: Starting position.
            goal: Target position.
            blocked: Set of blocked positions (other units, etc.).
            max_iterations: Maximum search iterations (prevents infinite loops).

        Returns:
            Path as list of positions (excluding start, including goal),
            or None if no path exists.
        """
        if start == goal:
            return []

        if blocked is None:
            blocked = set()

        # A* initialization
        open_set: List[PathNode] = []
        closed_set: Set[HexPosition] = set()
        g_scores: Dict[HexPosition, float] = {start: 0}

        start_node = PathNode(
            f_score=self._heuristic(start, goal), position=start, g_score=0
        )
        heappush(open_set, start_node)

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heappop(open_set)

            # Goal reached
            if current.position == goal:
                return self._reconstruct_path(current)

            # Skip if already processed
            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            # Explore neighbors
            for neighbor_pos in current.position.get_neighbors():
                # Check validity
                if not neighbor_pos.is_valid(self.grid.ROWS, self.grid.COLS):
                    continue

                # Check blocked (goal is allowed even if blocked)
                if neighbor_pos != goal and neighbor_pos in blocked:
                    continue

                if neighbor_pos in closed_set:
                    continue

                # Calculate g score (all moves cost 1)
                tentative_g = current.g_score + 1

                # Skip if we've found a better path already
                if neighbor_pos in g_scores and tentative_g >= g_scores[neighbor_pos]:
                    continue

                g_scores[neighbor_pos] = tentative_g
                f_score = tentative_g + self._heuristic(neighbor_pos, goal)

                neighbor_node = PathNode(
                    f_score=f_score,
                    position=neighbor_pos,
                    g_score=tentative_g,
                    parent=current,
                )
                heappush(open_set, neighbor_node)

        # No path found
        return None

    def find_path_to_range(
        self,
        start: HexPosition,
        target: HexPosition,
        attack_range: int,
        blocked: Optional[Set[HexPosition]] = None,
    ) -> Optional[List[HexPosition]]:
        """
        Find path to get within attack range of target.

        Instead of pathfinding to the target itself (which is occupied),
        this finds a path to the closest empty position within attack range.

        Args:
            start: Starting position.
            target: Target's position.
            attack_range: Attack range to get within.
            blocked: Set of blocked positions.

        Returns:
            Path to a position within attack range, or None if impossible.
        """
        if blocked is None:
            blocked = set()

        # Already in range
        if start.distance_to(target) <= attack_range:
            return []

        # Find all valid positions within attack range of target
        valid_goals = []
        for row in range(self.grid.ROWS):
            for col in range(self.grid.COLS):
                pos = HexPosition(row, col)
                if pos == start:
                    continue
                if pos in blocked:
                    continue
                if pos.distance_to(target) <= attack_range:
                    valid_goals.append(pos)

        if not valid_goals:
            return None

        # Find shortest path to any valid goal
        # Sort goals by distance from start for efficiency
        valid_goals.sort(key=lambda p: start.distance_to(p))

        best_path = None
        best_length = float("inf")

        for goal in valid_goals:
            # Skip if this goal can't possibly be shorter
            if start.distance_to(goal) >= best_length:
                continue

            path = self.find_path(start, goal, blocked)
            if path is not None and len(path) < best_length:
                best_path = path
                best_length = len(path)

                # Early termination if we found optimal path
                if best_length == start.distance_to(goal):
                    break

        return best_path

    def get_next_step(
        self,
        start: HexPosition,
        goal: HexPosition,
        blocked: Optional[Set[HexPosition]] = None,
    ) -> Optional[HexPosition]:
        """
        Get just the first step of a path.

        More efficient when only immediate movement is needed.

        Args:
            start: Starting position.
            goal: Target position.
            blocked: Set of blocked positions.

        Returns:
            Next position to move to, or None if no path.
        """
        path = self.find_path(start, goal, blocked)
        if path and len(path) > 0:
            return path[0]
        return None

    def get_next_step_to_range(
        self,
        start: HexPosition,
        target: HexPosition,
        attack_range: int,
        blocked: Optional[Set[HexPosition]] = None,
    ) -> Optional[HexPosition]:
        """
        Get next step to get within attack range.

        Args:
            start: Starting position.
            target: Target's position.
            attack_range: Attack range to get within.
            blocked: Set of blocked positions.

        Returns:
            Next position to move to, or None if already in range or no path.
        """
        path = self.find_path_to_range(start, target, attack_range, blocked)
        if path and len(path) > 0:
            return path[0]
        return None

    def _heuristic(self, a: HexPosition, b: HexPosition) -> float:
        """Heuristic function: hex distance."""
        return float(a.distance_to(b))

    def _reconstruct_path(self, node: PathNode) -> List[HexPosition]:
        """Reconstruct path from goal node (excluding start)."""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path


def get_blocked_positions(
    grid: HexGrid, exclude_unit_id: Optional[str] = None
) -> Set[HexPosition]:
    """
    Get all blocked positions on the grid.

    Args:
        grid: The hex grid.
        exclude_unit_id: Unit to exclude from blocked set (self).

    Returns:
        Set of blocked positions.
    """
    blocked = set()
    for unit_id, pos in grid.get_all_units().items():
        if unit_id != exclude_unit_id:
            blocked.add(pos)
    return blocked


def get_walkable_neighbors(
    grid: HexGrid,
    position: HexPosition,
    blocked: Optional[Set[HexPosition]] = None,
) -> List[HexPosition]:
    """
    Get all walkable neighbor positions.

    Args:
        grid: The hex grid.
        position: Center position.
        blocked: Set of blocked positions.

    Returns:
        List of walkable neighbor positions.
    """
    if blocked is None:
        blocked = set()

    neighbors = []
    for neighbor in position.get_neighbors():
        if neighbor.is_valid(grid.ROWS, grid.COLS) and neighbor not in blocked:
            neighbors.append(neighbor)
    return neighbors
