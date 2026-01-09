"""Player Units Management for TFT Set 16.

Manages a player's owned champions on bench and board.
"""

from typing import Optional, TYPE_CHECKING, Any
from dataclasses import dataclass, field

from src.data.models.champion import Champion
from src.data.models.item import Item
from src.core.constants import BENCH_SIZE, STAR_MULTIPLIER, COPIES_FOR_2_STAR

if TYPE_CHECKING:
    from src.core.synergy_calculator import SynergyCalculator, ActiveTrait
    from src.core.item_manager import ItemInstance


@dataclass
class ChampionInstance:
    """
    An instance of a champion owned by a player.
    Tracks star level, items, position, etc.
    """

    champion: Champion
    star_level: int = 1
    items: list[Any] = field(default_factory=list)  # List of ItemInstance
    position: Optional[tuple[int, int]] = None  # (x, y) on board, None if on bench

    @property
    def max_items(self) -> int:
        """Maximum number of items this unit can hold."""
        return 3

    def get_stats(self) -> dict:
        """
        Calculate current stats based on star level and items.

        Returns:
            Dictionary of current stats.
        """
        base_stats = self.champion.stats
        star_idx = self.star_level - 1

        # Base stats at star level
        stats = {
            "health": base_stats.health[star_idx],
            "attack_damage": base_stats.attack_damage[star_idx],
            "attack_speed": base_stats.attack_speed,
            "armor": base_stats.armor,
            "magic_resist": base_stats.magic_resist,
            "mana_start": base_stats.mana[0],
            "mana_max": base_stats.mana[1],
            "attack_range": base_stats.attack_range,
            "crit_chance": base_stats.crit_chance,
            "crit_damage": base_stats.crit_damage,
            "ability_power": 100,  # Base AP
            "omnivamp": 0,
        }

        # Apply item stats
        for item_inst in self.items:
            # Handle both ItemInstance and Item (for backwards compatibility)
            if hasattr(item_inst, 'item'):
                # ItemInstance
                item_stats = item_inst.item.stats
            else:
                # Raw Item
                item_stats = item_inst.stats
            stats["health"] += item_stats.health
            stats["attack_damage"] += item_stats.ad
            stats["ability_power"] += item_stats.ap
            stats["armor"] += item_stats.armor
            stats["magic_resist"] += item_stats.mr
            stats["attack_speed"] += item_stats.attack_speed / 100  # Convert %
            stats["mana_start"] += item_stats.mana
            stats["crit_chance"] += item_stats.crit_chance / 100
            stats["crit_damage"] += item_stats.crit_damage / 100
            stats["omnivamp"] += item_stats.omnivamp / 100

        return stats

    def can_add_item(self) -> bool:
        """Check if champion can hold more items."""
        return len(self.items) < self.max_items

    def add_item(self, item: Item) -> bool:
        """
        Add item if possible.

        Args:
            item: The item to add.

        Returns:
            True if item was added, False if full.
        """
        if not self.can_add_item():
            return False
        self.items.append(item)
        return True

    def remove_item(self, item_id: str) -> Optional[Item]:
        """
        Remove an item by ID.

        Args:
            item_id: ID of the item to remove.

        Returns:
            The removed item, or None if not found.
        """
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return self.items.pop(i)
        return None

    def get_sell_value(self) -> int:
        """
        Calculate sell value based on cost and star level.
        1-star: cost, 2-star: cost*3, 3-star: cost*9

        Returns:
            Gold value when sold.
        """
        cost = self.champion.cost
        if self.star_level == 1:
            return cost
        elif self.star_level == 2:
            return cost * 3
        else:  # 3-star
            return cost * 9

    def is_on_board(self) -> bool:
        """Check if unit is on the board."""
        return self.position is not None

    def get_item_traits(self) -> list[str]:
        """
        Get traits granted by equipped emblems.

        Returns:
            List of trait IDs from emblem items.
        """
        traits = []
        for item in self.items:
            # Handle both Item and ItemInstance
            item_obj = item.item if hasattr(item, "item") else item
            if item_obj.grants_trait:
                traits.append(item_obj.grants_trait)
        return traits

    def get_all_traits(self) -> list[str]:
        """
        Get all traits including base and from items.

        Returns:
            List of all trait IDs.
        """
        return list(self.champion.traits) + self.get_item_traits()

    def has_item(self, item_id: str) -> bool:
        """
        Check if champion has a specific item equipped.

        Args:
            item_id: The item ID to check.

        Returns:
            True if item is equipped.
        """
        for item in self.items:
            item_obj = item.item if hasattr(item, "item") else item
            if item_obj.id == item_id:
                return True
        return False

    def get_calculated_stats(
        self,
        trait_bonuses: Optional[dict] = None,
        active_traits: Optional[dict] = None,
    ):
        """
        Get stats using the StatCalculator.

        Args:
            trait_bonuses: Aggregated trait bonuses.
            active_traits: Active trait dict.

        Returns:
            CalculatedStats object.
        """
        from src.core.stat_calculator import StatCalculator
        calc = StatCalculator()
        return calc.calculate_stats(self, trait_bonuses, active_traits)

    def __repr__(self) -> str:
        stars = "★" * self.star_level
        items_str = f" [{len(self.items)} items]" if self.items else ""
        return f"{self.champion.name} {stars}{items_str}"


class PlayerUnits:
    """
    Manages a player's owned champions (bench + board).
    """

    BOARD_WIDTH = 7
    BOARD_HEIGHT = 4

    def __init__(self):
        """Initialize empty bench and board."""
        self.bench: list[Optional[ChampionInstance]] = [None] * BENCH_SIZE
        self.board: dict[tuple[int, int], ChampionInstance] = {}  # (x, y) -> unit
        self.champion_counts: dict[str, int] = {}  # champion_id -> total copies owned
        self._synergy_calculator: Optional["SynergyCalculator"] = None
        self._cached_synergies: Optional[dict[str, "ActiveTrait"]] = None

    def add_to_bench(self, champion: Champion) -> Optional[ChampionInstance]:
        """
        Add champion to first empty bench slot.

        Args:
            champion: The champion to add.

        Returns:
            The created ChampionInstance, or None if bench is full.
        """
        # Find first empty slot
        for i, slot in enumerate(self.bench):
            if slot is None:
                instance = ChampionInstance(champion=champion)
                self.bench[i] = instance
                self._update_count(champion.id, 1)

                # Check for auto-upgrade
                self._try_auto_upgrade(champion.id)

                return instance

        return None  # Bench full

    def _update_count(self, champion_id: str, delta: int) -> None:
        """Update the count of a champion."""
        if champion_id not in self.champion_counts:
            self.champion_counts[champion_id] = 0
        self.champion_counts[champion_id] += delta
        if self.champion_counts[champion_id] <= 0:
            del self.champion_counts[champion_id]

    def _try_auto_upgrade(self, champion_id: str) -> Optional[ChampionInstance]:
        """
        Try to auto-upgrade if player has 3 copies of same star level.
        Recursively checks for further upgrades (3 two-stars -> 3-star).

        Args:
            champion_id: The champion to try upgrading.

        Returns:
            The upgraded instance if upgrade happened, None otherwise.
        """
        # Find all instances of this champion
        instances_1star = []
        instances_2star = []

        # Check bench
        for i, instance in enumerate(self.bench):
            if instance and instance.champion.id == champion_id:
                if instance.star_level == 1:
                    instances_1star.append(("bench", i, instance))
                elif instance.star_level == 2:
                    instances_2star.append(("bench", i, instance))

        # Check board
        for pos, instance in self.board.items():
            if instance.champion.id == champion_id:
                if instance.star_level == 1:
                    instances_1star.append(("board", pos, instance))
                elif instance.star_level == 2:
                    instances_2star.append(("board", pos, instance))

        # Try 1-star -> 2-star upgrade
        if len(instances_1star) >= 3:
            upgraded = self._perform_upgrade(instances_1star[:3], 2)
            # Recursively check if we now have 3 two-stars
            self._try_auto_upgrade(champion_id)
            return upgraded

        # Try 2-star -> 3-star upgrade
        if len(instances_2star) >= 3:
            return self._perform_upgrade(instances_2star[:3], 3)

        return None

    def _perform_upgrade(
        self, instances: list[tuple], target_star: int
    ) -> ChampionInstance:
        """
        Perform the actual upgrade, combining 3 units into 1.

        Args:
            instances: List of (location, position, instance) tuples.
            target_star: The target star level.

        Returns:
            The upgraded instance.
        """
        # Keep the first instance as the upgraded one
        location, pos, upgraded = instances[0]
        upgraded.star_level = target_star

        # Combine items from all instances (keep up to 3)
        all_items = []
        for _, _, instance in instances:
            all_items.extend(instance.items)

        upgraded.items = all_items[:3]  # Keep first 3 items

        # Remove the other 2 instances
        for loc, p, instance in instances[1:]:
            if loc == "bench":
                self.bench[p] = None
            else:
                del self.board[p]

        return upgraded

    def can_upgrade(self, champion_id: str) -> bool:
        """
        Check if player has 3 copies of any star level to upgrade.

        Args:
            champion_id: The champion to check.

        Returns:
            True if upgrade is possible.
        """
        count_1star = 0
        count_2star = 0

        for instance in self.bench:
            if instance and instance.champion.id == champion_id:
                if instance.star_level == 1:
                    count_1star += 1
                elif instance.star_level == 2:
                    count_2star += 1

        for instance in self.board.values():
            if instance.champion.id == champion_id:
                if instance.star_level == 1:
                    count_1star += 1
                elif instance.star_level == 2:
                    count_2star += 1

        return count_1star >= 3 or count_2star >= 3

    def sell(self, instance: ChampionInstance) -> int:
        """
        Sell a champion from bench or board.

        Args:
            instance: The champion instance to sell.

        Returns:
            Gold value received.
        """
        gold = instance.get_sell_value()
        champion_id = instance.champion.id

        # Calculate copies being sold based on star level
        copies = 1 if instance.star_level == 1 else (3 if instance.star_level == 2 else 9)
        self._update_count(champion_id, -copies)

        # Remove from bench
        for i, bench_instance in enumerate(self.bench):
            if bench_instance is instance:
                self.bench[i] = None
                return gold

        # Remove from board
        for pos, board_instance in list(self.board.items()):
            if board_instance is instance:
                del self.board[pos]
                return gold

        return gold

    def sell_from_bench(self, slot_index: int) -> tuple[Optional[ChampionInstance], int]:
        """
        Sell a champion from a specific bench slot.

        Args:
            slot_index: Index of the bench slot.

        Returns:
            Tuple of (sold instance, gold received) or (None, 0) if empty.
        """
        if slot_index < 0 or slot_index >= BENCH_SIZE:
            return None, 0

        instance = self.bench[slot_index]
        if instance is None:
            return None, 0

        gold = self.sell(instance)
        return instance, gold

    def place_on_board(
        self, instance: ChampionInstance, x: int, y: int
    ) -> bool:
        """
        Place a unit on the board at position (x, y).

        Args:
            instance: The champion instance to place.
            x: X coordinate (0-6).
            y: Y coordinate (0-3).

        Returns:
            True if placed successfully.
        """
        if not (0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT):
            return False

        pos = (x, y)

        # Check if position is occupied
        if pos in self.board:
            return False

        # Remove from bench if there
        for i, bench_instance in enumerate(self.bench):
            if bench_instance is instance:
                self.bench[i] = None
                break

        # Place on board
        self.board[pos] = instance
        instance.position = pos
        self.invalidate_synergy_cache()
        return True

    def remove_from_board(self, x: int, y: int) -> Optional[ChampionInstance]:
        """
        Remove a unit from the board and return to bench.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            The removed instance, or None if position empty.
        """
        pos = (x, y)
        if pos not in self.board:
            return None

        instance = self.board[pos]
        del self.board[pos]
        instance.position = None
        self.invalidate_synergy_cache()

        # Try to add to bench
        for i, slot in enumerate(self.bench):
            if slot is None:
                self.bench[i] = instance
                return instance

        # If bench full, just return the instance (caller should handle)
        return instance

    def get_board_count(self) -> int:
        """Get number of units on board."""
        return len(self.board)

    def get_bench_count(self) -> int:
        """Get number of units on bench."""
        return sum(1 for slot in self.bench if slot is not None)

    def get_total_units(self) -> int:
        """Count all owned champions."""
        return self.get_board_count() + self.get_bench_count()

    def get_champion_count(self, champion_id: str) -> int:
        """
        Get count of copies of specific champion owned.
        This counts copies, not instances (3-star = 9 copies).

        Args:
            champion_id: The champion to count.

        Returns:
            Total copies owned.
        """
        return self.champion_counts.get(champion_id, 0)

    def get_all_instances(self) -> list[ChampionInstance]:
        """Get all champion instances (bench + board)."""
        instances = []
        for slot in self.bench:
            if slot is not None:
                instances.append(slot)
        instances.extend(self.board.values())
        return instances

    def get_board_units(self) -> list[ChampionInstance]:
        """Get all units currently on the board."""
        return list(self.board.values())

    def get_bench_units(self) -> list[Optional[ChampionInstance]]:
        """Get bench slots (including None for empty)."""
        return self.bench.copy()

    def has_bench_space(self) -> bool:
        """Check if there's space on the bench."""
        return any(slot is None for slot in self.bench)

    def clear(self) -> list[ChampionInstance]:
        """
        Clear all units. Returns list of all instances (for returning to pool).

        Returns:
            List of all instances that were owned.
        """
        instances = self.get_all_instances()
        self.bench = [None] * BENCH_SIZE
        self.board = {}
        self.champion_counts = {}
        return instances

    def __repr__(self) -> str:
        return f"PlayerUnits(bench={self.get_bench_count()}/9, board={self.get_board_count()})"

    # Synergy-related methods

    def _get_synergy_calculator(self) -> "SynergyCalculator":
        """Lazy-load synergy calculator."""
        if self._synergy_calculator is None:
            from src.core.synergy_calculator import SynergyCalculator
            self._synergy_calculator = SynergyCalculator()
        return self._synergy_calculator

    def get_active_synergies(self) -> dict[str, "ActiveTrait"]:
        """
        Get current active synergies from board champions.
        Uses caching - invalidate on board change.

        Returns:
            Dict mapping trait_id to ActiveTrait
        """
        if self._cached_synergies is None:
            from src.core.emblem_system import EmblemSystem

            board_champions = list(self.board.values())
            emblems = EmblemSystem.get_all_emblem_traits(board_champions)
            calc = self._get_synergy_calculator()
            self._cached_synergies = calc.calculate_synergies(board_champions, emblems)
        return self._cached_synergies

    def invalidate_synergy_cache(self) -> None:
        """Call when board changes."""
        self._cached_synergies = None
