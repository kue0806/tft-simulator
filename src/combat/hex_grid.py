"""Hex Grid System for TFT Combat.

Implements offset coordinates (odd-r layout) for the TFT hexagonal board.
Supports distance calculation, neighbor finding, and pathfinding.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class Team(Enum):
    """Team identification."""

    BLUE = "blue"  # Player 1 (bottom, rows 0-3)
    RED = "red"  # Player 2 (top, rows 4-7)


@dataclass(frozen=True)
class HexPosition:
    """
    Hex coordinate using offset coordinates (odd-r layout).

    In odd-r layout, odd rows are shifted right by 0.5 hexes.

    Attributes:
        row: Row index (0-7, 0-3 is BLUE team, 4-7 is RED team)
        col: Column index (0-6)
    """

    row: int
    col: int

    def to_cube(self) -> Tuple[int, int, int]:
        """
        Convert to cube coordinates for distance calculations.

        Cube coordinates satisfy x + y + z = 0.

        Returns:
            Tuple of (x, y, z) cube coordinates.
        """
        # odd-r offset to cube
        x = self.col - (self.row - (self.row & 1)) // 2
        z = self.row
        y = -x - z
        return (x, y, z)

    @classmethod
    def from_cube(cls, x: int, y: int, z: int) -> "HexPosition":
        """
        Create position from cube coordinates.

        Args:
            x: Cube x coordinate.
            y: Cube y coordinate.
            z: Cube z coordinate.

        Returns:
            HexPosition in offset coordinates.
        """
        col = x + (z - (z & 1)) // 2
        row = z
        return cls(row, col)

    def distance_to(self, other: "HexPosition") -> int:
        """
        Calculate hex distance to another position.

        Args:
            other: Target position.

        Returns:
            Distance in hex units.
        """
        ax, ay, az = self.to_cube()
        bx, by, bz = other.to_cube()
        return (abs(ax - bx) + abs(ay - by) + abs(az - bz)) // 2

    def get_neighbors(self) -> List["HexPosition"]:
        """
        Get all 6 adjacent hex positions (ignoring board bounds).

        Returns:
            List of 6 neighboring HexPositions.
        """
        # odd-r layout directions differ by row parity
        if self.row % 2 == 0:  # Even row
            directions = [
                (-1, 0),
                (-1, -1),  # Upper right, upper left
                (0, -1),
                (0, 1),  # Left, right
                (1, 0),
                (1, -1),  # Lower right, lower left
            ]
        else:  # Odd row
            directions = [
                (-1, 1),
                (-1, 0),  # Upper right, upper left
                (0, -1),
                (0, 1),  # Left, right
                (1, 1),
                (1, 0),  # Lower right, lower left
            ]

        neighbors = []
        for dr, dc in directions:
            neighbors.append(HexPosition(self.row + dr, self.col + dc))
        return neighbors

    def is_valid(self, rows: int = 8, cols: int = 7) -> bool:
        """
        Check if position is within board bounds.

        Args:
            rows: Number of rows (default 8 for combat board).
            cols: Number of columns (default 7).

        Returns:
            True if position is valid.
        """
        return 0 <= self.row < rows and 0 <= self.col < cols

    def __repr__(self) -> str:
        return f"Hex({self.row}, {self.col})"


class HexGrid:
    """
    TFT combat board (8x7 hex grid).

    The board consists of two 4x7 player areas:
    - Rows 0-3: BLUE team area
    - Rows 4-7: RED team area (opponent's board mirrored)
    """

    ROWS = 8
    COLS = 7
    BLUE_ROWS = range(0, 4)  # Player 1 area
    RED_ROWS = range(4, 8)  # Player 2 area

    def __init__(self):
        """Initialize empty grid."""
        # Map of position to unit ID
        self._grid: dict[HexPosition, Optional[str]] = {}
        # Reverse map of unit ID to position
        self._unit_positions: dict[str, HexPosition] = {}

    def place_unit(self, unit_id: str, position: HexPosition) -> bool:
        """
        Place a unit at a hex position.

        If the unit is already placed elsewhere, it will be moved.

        Args:
            unit_id: Unique identifier for the unit.
            position: Target position.

        Returns:
            True if successful, False if position is occupied.

        Raises:
            ValueError: If position is outside board bounds.
        """
        if not position.is_valid(self.ROWS, self.COLS):
            raise ValueError(f"Invalid position: {position}")

        if self._grid.get(position) is not None:
            return False

        # Remove from old position if exists
        if unit_id in self._unit_positions:
            old_pos = self._unit_positions[unit_id]
            self._grid[old_pos] = None

        self._grid[position] = unit_id
        self._unit_positions[unit_id] = position
        return True

    def remove_unit(self, unit_id: str) -> Optional[HexPosition]:
        """
        Remove a unit from the grid.

        Args:
            unit_id: Unit to remove.

        Returns:
            Previous position if unit existed, None otherwise.
        """
        if unit_id not in self._unit_positions:
            return None

        position = self._unit_positions.pop(unit_id)
        self._grid[position] = None
        return position

    def move_unit(self, unit_id: str, new_position: HexPosition) -> bool:
        """
        Move a unit to a new position.

        Args:
            unit_id: Unit to move.
            new_position: Target position.

        Returns:
            True if successful, False if invalid or occupied.
        """
        if not new_position.is_valid(self.ROWS, self.COLS):
            return False

        if self._grid.get(new_position) is not None:
            return False

        if unit_id not in self._unit_positions:
            return False

        old_position = self._unit_positions[unit_id]
        self._grid[old_position] = None
        self._grid[new_position] = unit_id
        self._unit_positions[unit_id] = new_position
        return True

    def get_unit_at(self, position: HexPosition) -> Optional[str]:
        """
        Get unit ID at a position.

        Args:
            position: Position to check.

        Returns:
            Unit ID or None if empty.
        """
        return self._grid.get(position)

    def get_unit_position(self, unit_id: str) -> Optional[HexPosition]:
        """
        Get a unit's current position.

        Args:
            unit_id: Unit to find.

        Returns:
            Position or None if not on grid.
        """
        return self._unit_positions.get(unit_id)

    def get_valid_neighbors(self, position: HexPosition) -> List[HexPosition]:
        """
        Get neighboring positions that are within board bounds.

        Args:
            position: Center position.

        Returns:
            List of valid neighboring positions.
        """
        return [n for n in position.get_neighbors() if n.is_valid(self.ROWS, self.COLS)]

    def get_empty_neighbors(self, position: HexPosition) -> List[HexPosition]:
        """
        Get neighboring positions that are empty.

        Args:
            position: Center position.

        Returns:
            List of empty neighboring positions.
        """
        return [
            n for n in self.get_valid_neighbors(position) if self._grid.get(n) is None
        ]

    def get_units_in_range(self, position: HexPosition, range_: int) -> List[str]:
        """
        Get all unit IDs within a certain range.

        Args:
            position: Center position.
            range_: Maximum distance (inclusive).

        Returns:
            List of unit IDs within range.
        """
        units = []
        for uid, pos in self._unit_positions.items():
            if position.distance_to(pos) <= range_:
                units.append(uid)
        return units

    def get_all_units(self) -> dict[str, HexPosition]:
        """
        Get all units and their positions.

        Returns:
            Copy of unit position mapping.
        """
        return self._unit_positions.copy()

    def clear(self) -> None:
        """Clear all units from the grid."""
        self._grid.clear()
        self._unit_positions.clear()

    def get_team_for_position(self, position: HexPosition) -> Team:
        """
        Get the team that owns a board region.

        Args:
            position: Position to check.

        Returns:
            Team that owns the region.
        """
        if position.row in self.BLUE_ROWS:
            return Team.BLUE
        return Team.RED

    @staticmethod
    def mirror_position(position: HexPosition) -> HexPosition:
        """
        Calculate the mirrored position for placing opponent's board.

        When an opponent's board is placed on the combat grid, their
        positions are mirrored both horizontally and vertically.

        Example:
            Opponent's (0, 3) -> Our grid's (7, 3)
            Opponent's (3, 0) -> Our grid's (4, 6)

        Args:
            position: Original position on opponent's board.

        Returns:
            Mirrored position on the combat grid.
        """
        # Mirror both row (0->7, 1->6, etc.) and column (0->6, 1->5, etc.)
        new_row = 7 - position.row
        new_col = 6 - position.col
        return HexPosition(new_row, new_col)

    def is_position_empty(self, position: HexPosition) -> bool:
        """
        Check if a position is empty.

        Args:
            position: Position to check.

        Returns:
            True if empty or invalid position.
        """
        return self._grid.get(position) is None

    def get_units_by_team(self, team: Team) -> List[str]:
        """
        Get all unit IDs belonging to a team.

        Args:
            team: Team to filter by.

        Returns:
            List of unit IDs on that team's side.
        """
        units = []
        for uid, pos in self._unit_positions.items():
            if self.get_team_for_position(pos) == team:
                units.append(uid)
        return units
