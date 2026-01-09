"""Shop System for TFT Set 16.

Manages a player's shop with 5 slots for champion selection.
Implements accurate TFT shop mechanics including:
- Pool-weighted champion selection
- "Consecutive shops will not repeat unbought champions" rule
- Guaranteed unlock champion appearance
"""

import random
from typing import Optional, TYPE_CHECKING

from src.data.models.champion import Champion
from src.core.champion_pool import ChampionPool
from src.core.constants import SHOP_ODDS, REROLL_COST, SHOP_SIZE

if TYPE_CHECKING:
    from src.core.unlock_manager import UnlockManager


class Shop:
    """
    Manages a player's shop with 5 slots.

    Key TFT Shop Mechanics:
    1. Champion probability is weighted by remaining pool count
    2. Consecutive shops will not repeat unbought champions
    3. Unlocked champions appear in rightmost slot on next refresh
    """

    def __init__(self, pool: ChampionPool, player_level: int = 1):
        """
        Initialize shop for a player.

        Args:
            pool: The shared champion pool.
            player_level: The player's current level (affects odds).
        """
        self.pool = pool
        self.level = player_level
        self.slots: list[Optional[Champion]] = [None] * SHOP_SIZE
        self.locked = False  # Shop lock feature
        self._slot_taken_from_pool: list[bool] = [False] * SHOP_SIZE

        # Guaranteed unlock champion for next refresh
        self._pending_unlock_champion_id: Optional[str] = None

        # Track unbought champions from previous shop (won't appear in next refresh)
        # Key mechanic: "Consecutive shops will not repeat unbought champions"
        self._previous_unbought: set[str] = set()

    def refresh(self) -> list[Optional[Champion]]:
        """
        Refresh all shop slots based on current level odds.
        Returns the new shop contents.

        If shop is locked, returns current contents without changes.

        Algorithm:
        1. Champions to exclude = unbought from current shop (before refresh)
        2. Return unbought champions to pool
        3. If there's a pending unlock champion, place it in rightmost slot
        4. For each remaining slot:
           a. Roll cost tier based on level odds
           b. Pick random available champion (excluding previous unbought + current shop)
           c. Remove from pool temporarily
        """
        if self.locked:
            return self.slots.copy()

        # Champions to exclude from this refresh (unbought from current shop)
        # These are the champions currently in shop that weren't purchased
        excluded_champions: set[str] = set()
        for i, champion in enumerate(self.slots):
            if champion and self._slot_taken_from_pool[i]:
                excluded_champions.add(champion.id)

        # Return unbought champions to pool
        self._return_unbought_to_pool()

        # Track what we're putting in this shop (to avoid duplicates within same shop)
        current_shop_champions: set[str] = set()

        # Handle pending unlock champion (guaranteed in rightmost slot)
        rightmost_reserved = False
        if self._pending_unlock_champion_id:
            unlock_champion = self.pool.get_champion(self._pending_unlock_champion_id)
            if unlock_champion:
                # Place in rightmost slot (index 4)
                taken = self.pool.take(unlock_champion.id, 1)
                self.slots[SHOP_SIZE - 1] = unlock_champion
                self._slot_taken_from_pool[SHOP_SIZE - 1] = taken > 0
                rightmost_reserved = True
                current_shop_champions.add(unlock_champion.id)
            self._pending_unlock_champion_id = None

        # Fill remaining slots
        end_slot = SHOP_SIZE - 1 if rightmost_reserved else SHOP_SIZE
        for i in range(end_slot):
            cost = self._roll_cost_tier()
            champion = self._pick_champion_of_cost(
                cost,
                excluded=excluded_champions | current_shop_champions
            )

            if champion:
                # Take from pool
                taken = self.pool.take(champion.id, 1)
                self._slot_taken_from_pool[i] = taken > 0
                current_shop_champions.add(champion.id)
            else:
                self._slot_taken_from_pool[i] = False

            self.slots[i] = champion

        return self.slots.copy()

    def set_pending_unlock(self, champion_id: str) -> None:
        """
        Set a champion to appear in the next shop refresh.

        The champion will appear in the rightmost slot on the next refresh.
        This is used when a champion is newly unlocked.

        Args:
            champion_id: The champion ID to guarantee in next refresh.
        """
        self._pending_unlock_champion_id = champion_id

    def _return_unbought_to_pool(self) -> None:
        """Return all unbought champions in current shop to the pool."""
        for i, champion in enumerate(self.slots):
            if champion and self._slot_taken_from_pool[i]:
                self.pool.return_champion(champion.id, 1)
                self._slot_taken_from_pool[i] = False

    def _roll_cost_tier(self) -> int:
        """
        Roll which cost tier to show based on level odds.
        Uses weighted random selection.

        Returns:
            Cost tier (1-5).
        """
        level = max(1, min(10, self.level))
        odds = SHOP_ODDS.get(level, [100, 0, 0, 0, 0])

        # Weighted random selection
        roll = random.randint(1, 100)
        cumulative = 0

        for cost, chance in enumerate(odds, start=1):
            cumulative += chance
            if roll <= cumulative:
                return cost

        return 1  # Fallback to 1-cost

    def _pick_champion_of_cost(
        self, cost: int, excluded: Optional[set[str]] = None
    ) -> Optional[Champion]:
        """
        Pick a random available champion of given cost.
        Weighted by remaining pool count.

        TFT uses pool-weighted selection: champions with more copies
        remaining in the pool have higher chance of appearing.

        Args:
            cost: The cost tier to pick from.
            excluded: Set of champion IDs to exclude from selection.

        Returns:
            A champion of that cost, or None if none available.
        """
        available = self.pool.get_available_champions_of_cost(cost)

        if not available:
            return None

        # Filter out excluded champions
        if excluded:
            available = [(c, count) for c, count in available if c.id not in excluded]

        if not available:
            return None

        # Weight by available count (key TFT mechanic)
        # If there are 20 copies of Garen and 10 of Poppy in pool,
        # Garen has 2x the chance of appearing
        total_weight = sum(count for _, count in available)
        if total_weight == 0:
            return None

        roll = random.randint(1, total_weight)
        cumulative = 0

        for champion, count in available:
            cumulative += count
            if roll <= cumulative:
                return champion

        # Fallback
        return available[0][0] if available else None

    def purchase(self, slot_index: int) -> Optional[Champion]:
        """
        Purchase champion from shop slot.

        Args:
            slot_index: Index of the slot to purchase from (0-4).

        Returns:
            The champion if successful, None if slot empty or invalid.
        """
        if slot_index < 0 or slot_index >= SHOP_SIZE:
            return None

        champion = self.slots[slot_index]
        if champion is None:
            return None

        # Clear the slot
        self.slots[slot_index] = None
        # Mark as not taken from pool (already taken, now owned by player)
        self._slot_taken_from_pool[slot_index] = False

        return champion

    def toggle_lock(self) -> bool:
        """
        Toggle shop lock.

        Returns:
            New lock state (True = locked).
        """
        self.locked = not self.locked
        return self.locked

    def set_lock(self, locked: bool) -> None:
        """
        Set shop lock state directly.

        Args:
            locked: Whether shop should be locked.
        """
        self.locked = locked

    def set_level(self, level: int) -> None:
        """
        Update player level (affects shop odds).

        Args:
            level: New player level.
        """
        self.level = max(1, min(10, level))

    def get_odds(self) -> list[int]:
        """
        Get current shop odds based on level.

        Returns:
            List of odds [1-cost%, 2-cost%, 3-cost%, 4-cost%, 5-cost%].
        """
        level = max(1, min(10, self.level))
        return SHOP_ODDS.get(level, [100, 0, 0, 0, 0]).copy()

    def get_slot(self, slot_index: int) -> Optional[Champion]:
        """
        Get champion in a specific slot.

        Args:
            slot_index: Index of the slot (0-4).

        Returns:
            Champion in slot, or None if empty.
        """
        if 0 <= slot_index < SHOP_SIZE:
            return self.slots[slot_index]
        return None

    def get_all_slots(self) -> list[Optional[Champion]]:
        """
        Get all champions currently in shop.

        Returns:
            Copy of slots list.
        """
        return self.slots.copy()

    def is_slot_empty(self, slot_index: int) -> bool:
        """
        Check if a slot is empty.

        Args:
            slot_index: Index of the slot (0-4).

        Returns:
            True if empty, False otherwise.
        """
        if 0 <= slot_index < SHOP_SIZE:
            return self.slots[slot_index] is None
        return True

    def clear(self) -> None:
        """Clear shop and return all champions to pool."""
        self._return_unbought_to_pool()
        self.slots = [None] * SHOP_SIZE
        self._slot_taken_from_pool = [False] * SHOP_SIZE

    def __repr__(self) -> str:
        filled = sum(1 for s in self.slots if s is not None)
        lock_str = " (locked)" if self.locked else ""
        return f"Shop(level={self.level}, filled={filled}/5{lock_str})"
