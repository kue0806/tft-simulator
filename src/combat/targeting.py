"""Targeting System for TFT Combat.

Handles target selection logic including:
- Nearest enemy targeting (default)
- Farthest enemy targeting (assassins)
- Lowest/Highest HP targeting
- Custom filters for ability targeting
"""

from dataclasses import dataclass
from typing import Optional, List, Callable, TYPE_CHECKING
from enum import Enum, auto
import random
import heapq

from src.core.constants import TARGET_PRIORITY_BY_ROLE

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .hex_grid import HexGrid, HexPosition, Team


class TargetingPriority(Enum):
    """Targeting priority types."""

    NEAREST = auto()  # Closest enemy (default)
    FARTHEST = auto()  # Farthest enemy (assassin)
    LOWEST_HP = auto()  # Lowest current HP
    HIGHEST_HP = auto()  # Highest current HP
    LOWEST_HP_PERCENT = auto()  # Lowest HP percentage
    HIGHEST_MANA = auto()  # Highest current mana
    LOWEST_ARMOR = auto()  # Lowest armor (armor shred)
    HIGHEST_ATTACK_DAMAGE = auto()  # Highest AD (priority target)
    RANDOM = auto()  # Random selection


@dataclass
class TargetingContext:
    """Context needed for targeting decisions."""

    grid: "HexGrid"
    all_units: dict[str, "CombatUnit"]  # unit_id -> CombatUnit

    def get_unit_position(self, unit_id: str) -> Optional["HexPosition"]:
        """Get unit position from grid."""
        return self.grid.get_unit_position(unit_id)

    def get_unit(self, unit_id: str) -> Optional["CombatUnit"]:
        """Get unit by ID."""
        return self.all_units.get(unit_id)


class TargetSelector:
    """
    Target selection system.

    Handles finding appropriate targets for attacks and abilities
    based on various targeting priorities.

    Usage:
        context = TargetingContext(grid=grid, all_units=units)
        selector = TargetSelector(context)
        target_id = selector.find_target(attacker, priority=TargetingPriority.NEAREST)
    """

    def __init__(
        self, context: TargetingContext, rng: Optional[random.Random] = None
    ):
        """
        Initialize target selector.

        Args:
            context: Targeting context with grid and units.
            rng: Random number generator for deterministic simulation.
        """
        self.context = context
        self.rng = rng or random.Random()

    def find_target(
        self,
        attacker: "CombatUnit",
        priority: TargetingPriority = TargetingPriority.NEAREST,
        filter_fn: Optional[Callable[["CombatUnit"], bool]] = None,
    ) -> Optional[str]:
        """
        Find a target for the attacker.

        Args:
            attacker: The attacking unit.
            priority: Targeting priority type.
            filter_fn: Optional filter function for additional constraints.

        Returns:
            Target unit ID or None if no valid target.
        """
        # Get attacker position
        attacker_pos = self.context.get_unit_position(attacker.id)
        if attacker_pos is None:
            return None

        # Get valid enemy candidates
        enemies = self._get_valid_enemies(attacker, filter_fn)
        if not enemies:
            return None

        # Select based on priority
        if priority == TargetingPriority.NEAREST:
            return self._select_nearest(attacker_pos, enemies)
        elif priority == TargetingPriority.FARTHEST:
            return self._select_farthest(attacker_pos, enemies)
        elif priority == TargetingPriority.LOWEST_HP:
            return self._select_lowest_hp(enemies)
        elif priority == TargetingPriority.HIGHEST_HP:
            return self._select_highest_hp(enemies)
        elif priority == TargetingPriority.LOWEST_HP_PERCENT:
            return self._select_lowest_hp_percent(enemies)
        elif priority == TargetingPriority.HIGHEST_MANA:
            return self._select_highest_mana(enemies)
        elif priority == TargetingPriority.LOWEST_ARMOR:
            return self._select_lowest_armor(enemies)
        elif priority == TargetingPriority.HIGHEST_ATTACK_DAMAGE:
            return self._select_highest_ad(enemies)
        elif priority == TargetingPriority.RANDOM:
            return self._select_random(enemies)
        else:
            return self._select_nearest(attacker_pos, enemies)

    def find_target_in_range(
        self,
        attacker: "CombatUnit",
        attack_range: Optional[int] = None,
        priority: TargetingPriority = TargetingPriority.NEAREST,
        filter_fn: Optional[Callable[["CombatUnit"], bool]] = None,
    ) -> Optional[str]:
        """
        Find a target within attack range.

        Args:
            attacker: The attacking unit.
            attack_range: Maximum range (uses unit's attack_range if None).
            priority: Targeting priority type.
            filter_fn: Optional filter function.

        Returns:
            Target unit ID or None if no valid target in range.
        """
        range_ = (
            attack_range if attack_range is not None else attacker.stats.attack_range
        )

        # Filter to only targets in range
        def in_range_filter(unit: "CombatUnit") -> bool:
            if not self.is_in_range(attacker, unit.id, range_):
                return False
            if filter_fn and not filter_fn(unit):
                return False
            return True

        attacker_pos = self.context.get_unit_position(attacker.id)
        if attacker_pos is None:
            return None

        enemies = self._get_valid_enemies(attacker, in_range_filter)
        if not enemies:
            return None

        # Select based on priority
        if priority == TargetingPriority.NEAREST:
            return self._select_nearest(attacker_pos, enemies)
        elif priority == TargetingPriority.FARTHEST:
            return self._select_farthest(attacker_pos, enemies)
        elif priority == TargetingPriority.LOWEST_HP:
            return self._select_lowest_hp(enemies)
        elif priority == TargetingPriority.HIGHEST_HP:
            return self._select_highest_hp(enemies)
        elif priority == TargetingPriority.LOWEST_HP_PERCENT:
            return self._select_lowest_hp_percent(enemies)
        elif priority == TargetingPriority.RANDOM:
            return self._select_random(enemies)
        else:
            return self._select_nearest(attacker_pos, enemies)

    def _get_valid_enemies(
        self,
        attacker: "CombatUnit",
        filter_fn: Optional[Callable[["CombatUnit"], bool]] = None,
    ) -> List["CombatUnit"]:
        """Get list of valid enemy targets."""
        enemies = []
        for unit_id, unit in self.context.all_units.items():
            # Skip same team
            if unit.team == attacker.team:
                continue
            # Skip non-targetable units
            if not unit.is_targetable:
                continue
            # Apply custom filter
            if filter_fn and not filter_fn(unit):
                continue
            enemies.append(unit)
        return enemies

    def _select_nearest(
        self, from_pos: "HexPosition", candidates: List["CombatUnit"]
    ) -> Optional[str]:
        """
        Select nearest enemy with role-based tiebreaker.

        Tiebreaker priority (when same distance):
        1. Tank (highest priority - targeted first)
        2. Fighter/Marksman/Caster/Specialist
        3. Assassin (lowest priority - targeted last)
        """
        if not candidates:
            return None

        def sort_key(unit: "CombatUnit"):
            pos = self.context.get_unit_position(unit.id)
            if pos is None:
                return (float("inf"), float("inf"), float("inf"), float("inf"))
            distance = from_pos.distance_to(pos)
            # Role priority (lower = higher priority to be targeted)
            role_priority = TARGET_PRIORITY_BY_ROLE.get(unit.role, 2)
            # Tiebreaker: distance -> role priority -> row -> col
            return (distance, role_priority, pos.row, pos.col)

        # Use heapq for O(n) instead of sort's O(n log n)
        best = min(candidates, key=sort_key)
        return best.id

    def _select_farthest(
        self, from_pos: "HexPosition", candidates: List["CombatUnit"]
    ) -> Optional[str]:
        """Select farthest enemy (for assassins)."""
        if not candidates:
            return None

        def sort_key(unit: "CombatUnit"):
            pos = self.context.get_unit_position(unit.id)
            if pos is None:
                return (float("-inf"), 0, 0)
            distance = from_pos.distance_to(pos)
            # Sort by distance descending
            return (-distance, pos.row, pos.col)

        # Use min with negated distance for O(n)
        best = min(candidates, key=sort_key)
        return best.id

    def _select_lowest_hp(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select enemy with lowest current HP."""
        if not candidates:
            return None
        best = min(candidates, key=lambda u: (u.stats.current_hp, u.id))
        return best.id

    def _select_highest_hp(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select enemy with highest current HP."""
        if not candidates:
            return None
        best = max(candidates, key=lambda u: (u.stats.current_hp, -hash(u.id)))
        return best.id

    def _select_lowest_hp_percent(
        self, candidates: List["CombatUnit"]
    ) -> Optional[str]:
        """Select enemy with lowest HP percentage."""
        if not candidates:
            return None

        def hp_percent(unit: "CombatUnit") -> float:
            if unit.stats.max_hp <= 0:
                return 1.0
            return unit.stats.current_hp / unit.stats.max_hp

        best = min(candidates, key=lambda u: (hp_percent(u), u.id))
        return best.id

    def _select_highest_mana(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select enemy with highest current mana."""
        if not candidates:
            return None
        best = max(candidates, key=lambda u: (u.stats.current_mana, -hash(u.id)))
        return best.id

    def _select_lowest_armor(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select enemy with lowest armor."""
        if not candidates:
            return None
        best = min(candidates, key=lambda u: (u.stats.armor, u.id))
        return best.id

    def _select_highest_ad(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select enemy with highest attack damage."""
        if not candidates:
            return None
        best = max(candidates, key=lambda u: (u.stats.attack_damage, -hash(u.id)))
        return best.id

    def _select_random(self, candidates: List["CombatUnit"]) -> Optional[str]:
        """Select random enemy."""
        if not candidates:
            return None
        return self.rng.choice(candidates).id

    def is_in_range(
        self,
        attacker: "CombatUnit",
        target_id: str,
        attack_range: Optional[int] = None,
    ) -> bool:
        """
        Check if target is within attack range.

        Args:
            attacker: The attacking unit.
            target_id: Target unit ID.
            attack_range: Range to check (uses attacker's range if None).

        Returns:
            True if target is in range.
        """
        attacker_pos = self.context.get_unit_position(attacker.id)
        target_pos = self.context.get_unit_position(target_id)

        if attacker_pos is None or target_pos is None:
            return False

        range_ = (
            attack_range if attack_range is not None else attacker.stats.attack_range
        )
        return attacker_pos.distance_to(target_pos) <= range_

    def get_distance(self, attacker: "CombatUnit", target_id: str) -> Optional[int]:
        """
        Get distance between attacker and target.

        Args:
            attacker: The attacking unit.
            target_id: Target unit ID.

        Returns:
            Distance in hex units or None if positions unknown.
        """
        attacker_pos = self.context.get_unit_position(attacker.id)
        target_pos = self.context.get_unit_position(target_id)

        if attacker_pos is None or target_pos is None:
            return None

        return attacker_pos.distance_to(target_pos)

    def get_units_in_range(
        self,
        attacker: "CombatUnit",
        range_: int,
        enemies_only: bool = True,
        include_self: bool = False,
    ) -> List[str]:
        """
        Get all unit IDs within range.

        Args:
            attacker: Center unit.
            range_: Maximum range.
            enemies_only: Only include enemy units.
            include_self: Include the attacker itself.

        Returns:
            List of unit IDs within range.
        """
        attacker_pos = self.context.get_unit_position(attacker.id)
        if attacker_pos is None:
            return []

        result = []
        for unit_id, unit in self.context.all_units.items():
            # Skip self unless included
            if unit_id == attacker.id and not include_self:
                continue
            # Skip allies if enemies_only
            if enemies_only and unit.team == attacker.team:
                continue
            # Skip non-targetable
            if not unit.is_targetable:
                continue

            unit_pos = self.context.get_unit_position(unit_id)
            if unit_pos and attacker_pos.distance_to(unit_pos) <= range_:
                result.append(unit_id)

        return result

    def get_units_in_area(
        self,
        center: "HexPosition",
        range_: int,
        enemies_only: bool = False,
        attacker_team: Optional["Team"] = None,
    ) -> List[str]:
        """
        Get all unit IDs within range of a position.

        Args:
            center: Center position.
            range_: Maximum range.
            enemies_only: Only include enemies of attacker_team.
            attacker_team: Team to consider as "friendly" when enemies_only=True.

        Returns:
            List of unit IDs within range of position.
        """
        result = []
        for unit_id, unit in self.context.all_units.items():
            if not unit.is_targetable:
                continue
            if enemies_only and attacker_team and unit.team == attacker_team:
                continue

            unit_pos = self.context.get_unit_position(unit_id)
            if unit_pos and center.distance_to(unit_pos) <= range_:
                result.append(unit_id)

        return result

    def validate_current_target(
        self, attacker: "CombatUnit", attack_range: Optional[int] = None
    ) -> bool:
        """
        Check if current target is still valid.

        Args:
            attacker: The attacking unit.
            attack_range: Range to check (optional).

        Returns:
            True if current target is valid and in range.
        """
        if attacker.current_target_id is None:
            return False

        target = self.context.get_unit(attacker.current_target_id)
        if target is None or not target.is_targetable:
            return False

        # Check range if specified
        if attack_range is not None:
            return self.is_in_range(attacker, attacker.current_target_id, attack_range)

        return True

    def acquire_target(
        self,
        attacker: "CombatUnit",
        priority: TargetingPriority = TargetingPriority.NEAREST,
        keep_current: bool = True,
    ) -> Optional[str]:
        """
        Acquire a target, potentially keeping current if still valid.

        TFT units generally stick to their current target if it's alive
        and in range, only switching when necessary.

        Args:
            attacker: The attacking unit.
            priority: Targeting priority for new target.
            keep_current: If True, keep current target if valid.

        Returns:
            Target unit ID or None.
        """
        # Try to keep current target
        if keep_current and attacker.current_target_id is not None:
            target = self.context.get_unit(attacker.current_target_id)
            if target is not None and target.is_targetable:
                attacker.current_target_id = target.id
                return target.id

        # Find new target
        new_target = self.find_target(attacker, priority)
        attacker.current_target_id = new_target
        return new_target
