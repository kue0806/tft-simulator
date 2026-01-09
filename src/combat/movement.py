"""Movement System for TFT Combat.

Handles tick-based unit movement across the hex grid.
Integrates with pathfinding for autonomous unit movement.
"""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

from .hex_grid import HexGrid, HexPosition
from .pathfinding import PathFinder, get_blocked_positions

if TYPE_CHECKING:
    from .combat_unit import CombatUnit


# TFT movement constants
# All units move at the same speed in TFT
BASE_MOVE_SPEED = 550  # Units per second (approximate)
HEX_SIZE = 180  # Hex size in game units (approximate)
MOVE_TIME_PER_HEX = HEX_SIZE / BASE_MOVE_SPEED  # ~0.327 seconds per hex


@dataclass
class MovementState:
    """Tracks unit's movement state."""

    unit_id: str
    current_path: List[HexPosition] = field(default_factory=list)
    progress: float = 0.0  # Progress to next hex (0.0 to 1.0)
    is_moving: bool = False
    total_distance_moved: int = 0  # Stats tracking

    @property
    def next_position(self) -> Optional[HexPosition]:
        """Get the next hex in path."""
        if self.current_path:
            return self.current_path[0]
        return None

    @property
    def remaining_path_length(self) -> int:
        """Get remaining hexes to traverse."""
        return len(self.current_path)


class MovementSystem:
    """
    Unit movement manager.

    Handles movement initiation, progress updates, and path recalculation.
    Works in tick-based simulation with delta time updates.

    Usage:
        movement = MovementSystem(grid)
        movement.start_move_to_target(unit, target_pos, attack_range=1)
        # Each tick:
        completed = movement.update(unit, delta_time=0.033)
    """

    def __init__(
        self, grid: HexGrid, move_time_per_hex: float = MOVE_TIME_PER_HEX
    ):
        """
        Initialize movement system.

        Args:
            grid: The hex grid.
            move_time_per_hex: Time to move one hex (seconds).
        """
        self.grid = grid
        self.pathfinder = PathFinder(grid)
        self.move_time_per_hex = move_time_per_hex

        # Movement state per unit
        self._movement_states: dict[str, MovementState] = {}

    def start_move_to_target(
        self,
        unit: "CombatUnit",
        target_pos: HexPosition,
        attack_range: int = 1,
    ) -> bool:
        """
        Start moving unit toward target's attack range.

        Args:
            unit: The unit to move.
            target_pos: Target's position.
            attack_range: Attack range to get within.

        Returns:
            True if movement started, False if already in range or no path.
        """
        unit_pos = self.grid.get_unit_position(unit.id)
        if unit_pos is None:
            return False

        # Already in range
        if unit_pos.distance_to(target_pos) <= attack_range:
            self._stop_movement(unit.id)
            return False

        # Find path
        blocked = get_blocked_positions(self.grid, exclude_unit_id=unit.id)
        path = self.pathfinder.find_path_to_range(
            unit_pos, target_pos, attack_range, blocked
        )

        if not path:
            return False

        # Initialize movement state
        self._movement_states[unit.id] = MovementState(
            unit_id=unit.id,
            current_path=path,
            progress=0.0,
            is_moving=True,
        )

        return True

    def start_move_to_position(
        self, unit: "CombatUnit", goal: HexPosition
    ) -> bool:
        """
        Start moving unit to a specific position.

        Args:
            unit: The unit to move.
            goal: Target position.

        Returns:
            True if movement started, False if at goal, blocked, or no path.
        """
        unit_pos = self.grid.get_unit_position(unit.id)
        if unit_pos is None:
            return False

        if unit_pos == goal:
            return False

        blocked = get_blocked_positions(self.grid, exclude_unit_id=unit.id)

        # Check if goal is blocked (can't move to occupied position)
        if goal in blocked:
            return False

        path = self.pathfinder.find_path(unit_pos, goal, blocked)

        if not path:
            return False

        self._movement_states[unit.id] = MovementState(
            unit_id=unit.id,
            current_path=path,
            progress=0.0,
            is_moving=True,
        )

        return True

    def update(self, unit: "CombatUnit", delta_time: float) -> bool:
        """
        Update unit movement by delta time.

        Args:
            unit: The unit to update.
            delta_time: Time elapsed (seconds).

        Returns:
            True if movement completed or stopped, False if still moving.
        """
        state = self._movement_states.get(unit.id)
        if state is None or not state.is_moving:
            return True

        if not state.current_path:
            self._stop_movement(unit.id)
            return True

        # Update progress
        progress_delta = delta_time / self.move_time_per_hex
        state.progress += progress_delta

        # Process completed hex movements
        while state.progress >= 1.0 and state.current_path:
            state.progress -= 1.0
            next_pos = state.current_path.pop(0)

            # Execute move on grid
            if self.grid.move_unit(unit.id, next_pos):
                state.total_distance_moved += 1
            else:
                # Move failed (position occupied) - need path recalculation
                state.is_moving = False
                state.current_path = []
                return False

        # Check if path completed
        if not state.current_path:
            self._stop_movement(unit.id)
            return True

        return False

    def update_all(self, units: List["CombatUnit"], delta_time: float) -> dict[str, bool]:
        """
        Update movement for multiple units.

        Args:
            units: List of units to update.
            delta_time: Time elapsed (seconds).

        Returns:
            Dict mapping unit_id to completion status.
        """
        results = {}
        for unit in units:
            results[unit.id] = self.update(unit, delta_time)
        return results

    def is_moving(self, unit_id: str) -> bool:
        """Check if unit is currently moving."""
        state = self._movement_states.get(unit_id)
        return state is not None and state.is_moving

    def get_movement_progress(self, unit_id: str) -> float:
        """Get current hex movement progress (0.0 to 1.0)."""
        state = self._movement_states.get(unit_id)
        if state is None:
            return 0.0
        return state.progress

    def get_remaining_path(self, unit_id: str) -> List[HexPosition]:
        """Get remaining path for unit."""
        state = self._movement_states.get(unit_id)
        if state is None:
            return []
        return list(state.current_path)

    def get_movement_state(self, unit_id: str) -> Optional[MovementState]:
        """Get full movement state for unit."""
        return self._movement_states.get(unit_id)

    def stop_movement(self, unit_id: str) -> None:
        """Stop unit movement (external call)."""
        self._stop_movement(unit_id)

    def _stop_movement(self, unit_id: str) -> None:
        """Internal: stop unit movement."""
        if unit_id in self._movement_states:
            state = self._movement_states[unit_id]
            state.is_moving = False
            state.current_path = []
            state.progress = 0.0

    def recalculate_path(
        self,
        unit: "CombatUnit",
        target_pos: HexPosition,
        attack_range: int = 1,
    ) -> bool:
        """
        Recalculate path after obstacle changes.

        Args:
            unit: The unit to recalculate for.
            target_pos: Target's position.
            attack_range: Attack range to get within.

        Returns:
            True if new path found, False otherwise.
        """
        self._stop_movement(unit.id)
        return self.start_move_to_target(unit, target_pos, attack_range)

    def needs_path_recalculation(
        self,
        unit: "CombatUnit",
        target_pos: HexPosition,
        attack_range: int = 1,
    ) -> bool:
        """
        Check if unit needs path recalculation.

        This happens when:
        - Unit is not moving but not in range
        - Path is blocked by new obstacles

        Args:
            unit: The unit to check.
            target_pos: Target's position.
            attack_range: Attack range.

        Returns:
            True if recalculation needed.
        """
        unit_pos = self.grid.get_unit_position(unit.id)
        if unit_pos is None:
            return False

        # Already in range - no movement needed
        if unit_pos.distance_to(target_pos) <= attack_range:
            return False

        state = self._movement_states.get(unit.id)

        # Not moving and not in range - need path
        if state is None or not state.is_moving:
            return True

        # Check if current path is still valid
        if state.current_path:
            blocked = get_blocked_positions(self.grid, exclude_unit_id=unit.id)
            for pos in state.current_path:
                if pos in blocked:
                    return True

        return False

    def clear(self) -> None:
        """Clear all movement states."""
        self._movement_states.clear()

    def get_stats(self, unit_id: str) -> dict:
        """Get movement statistics for unit."""
        state = self._movement_states.get(unit_id)
        if state is None:
            return {"total_distance_moved": 0, "is_moving": False}
        return {
            "total_distance_moved": state.total_distance_moved,
            "is_moving": state.is_moving,
            "remaining_path_length": state.remaining_path_length,
        }
