"""Champion Pool Manager for TFT Set 16.

Manages the shared champion pool that all players draw from.

Key TFT Mechanics:
- All players share one champion pool
- Each champion has a fixed number of copies based on cost tier
- When a player buys a champion, it's removed from the pool
- When sold or player eliminated, champions return to the pool
- Probability of seeing a champion = (copies in pool) / (total copies of that cost)
"""

from typing import Optional

from src.data.models.champion import Champion
from src.core.constants import POOL_SIZE, SHOP_ODDS, CHAMPIONS_PER_TIER


class ChampionPool:
    """
    Manages the shared champion pool for all players.
    In TFT, all players draw from the same pool.
    """

    def __init__(self, champions: list[Champion], num_players: int = 8):
        """
        Initialize pool with all champions.
        Each champion has POOL_SIZE[cost] copies available.

        Args:
            champions: List of all champions in the game.
            num_players: Number of players in the game (affects nothing currently).
        """
        self.pool: dict[str, int] = {}  # champion_id -> remaining count
        self.champions: dict[str, Champion] = {}  # champion_id -> Champion
        self.champions_by_cost: dict[int, list[Champion]] = {}  # cost -> [champions]
        self.num_players = num_players
        self._initialize_pool(champions)

    def _initialize_pool(self, champions: list[Champion]) -> None:
        """Set up initial pool counts based on champion costs."""
        # Initialize cost buckets
        for cost in range(1, 8):
            self.champions_by_cost[cost] = []

        for champion in champions:
            # Skip unlockable champions - they start outside the pool
            if champion.is_unlockable:
                continue

            champion_id = champion.id
            cost = champion.cost

            # Set initial pool count based on cost
            self.pool[champion_id] = POOL_SIZE.get(cost, 0)
            self.champions[champion_id] = champion

            # Add to cost bucket
            if cost not in self.champions_by_cost:
                self.champions_by_cost[cost] = []
            self.champions_by_cost[cost].append(champion)

    def take(self, champion_id: str, count: int = 1) -> int:
        """
        Remove champions from pool (when purchased).

        Args:
            champion_id: The champion to take.
            count: Number of copies to take.

        Returns:
            Actual count taken (may be less if pool depleted).
        """
        if champion_id not in self.pool:
            return 0

        available = self.pool[champion_id]
        actual_taken = min(count, available)
        self.pool[champion_id] -= actual_taken
        return actual_taken

    def return_champion(self, champion_id: str, count: int = 1) -> None:
        """
        Return champions to pool (when sold or player eliminated).

        Args:
            champion_id: The champion to return.
            count: Number of copies to return.
        """
        if champion_id not in self.pool:
            # Champion might be unlockable or invalid
            return

        champion = self.champions.get(champion_id)
        if not champion:
            return

        max_pool = POOL_SIZE.get(champion.cost, 0)
        self.pool[champion_id] = min(self.pool[champion_id] + count, max_pool)

    def get_available(self, champion_id: str) -> int:
        """
        Get remaining count of a specific champion.

        Args:
            champion_id: The champion to check.

        Returns:
            Number of copies remaining in pool.
        """
        return self.pool.get(champion_id, 0)

    def get_champion(self, champion_id: str) -> Optional[Champion]:
        """
        Get champion data by ID.

        Args:
            champion_id: The champion ID.

        Returns:
            Champion object if found, None otherwise.
        """
        return self.champions.get(champion_id)

    def get_probability(self, champion_id: str, level: int) -> float:
        """
        Calculate probability of seeing this champion in a single shop slot.
        Considers: level odds, remaining pool, total pool for that cost.

        Args:
            champion_id: The champion to calculate probability for.
            level: Player's current level.

        Returns:
            Probability (0.0 to 1.0) of seeing this champion in one slot.
        """
        champion = self.champions.get(champion_id)
        if not champion:
            return 0.0

        cost = champion.cost
        if cost > 5:
            # 6+ cost champions don't appear in normal shop
            return 0.0

        # Get the odds of rolling this cost tier at this level
        level = max(1, min(10, level))  # Clamp to valid range
        cost_odds = SHOP_ODDS.get(level, [100, 0, 0, 0, 0])

        if cost > len(cost_odds):
            return 0.0

        tier_probability = cost_odds[cost - 1] / 100.0

        # Calculate champion's share of the cost pool
        available = self.pool.get(champion_id, 0)
        total_in_tier = self.get_total_available_by_cost(cost)

        if total_in_tier == 0:
            return 0.0

        champion_probability = available / total_in_tier

        return tier_probability * champion_probability

    def get_champions_by_cost(self, cost: int) -> list[Champion]:
        """
        Get all champions of a specific cost tier.

        Args:
            cost: The cost tier (1-7).

        Returns:
            List of champions at that cost.
        """
        return self.champions_by_cost.get(cost, [])

    def get_total_available_by_cost(self, cost: int) -> int:
        """
        Get total remaining champions of a cost tier.

        Args:
            cost: The cost tier (1-7).

        Returns:
            Total copies remaining across all champions of that cost.
        """
        total = 0
        for champion in self.champions_by_cost.get(cost, []):
            total += self.pool.get(champion.id, 0)
        return total

    def get_available_champions_of_cost(self, cost: int) -> list[tuple[Champion, int]]:
        """
        Get all champions of a cost tier with their available counts.

        Args:
            cost: The cost tier (1-7).

        Returns:
            List of (Champion, available_count) tuples for champions with count > 0.
        """
        result = []
        for champion in self.champions_by_cost.get(cost, []):
            count = self.pool.get(champion.id, 0)
            if count > 0:
                result.append((champion, count))
        return result

    def add_unlockable_to_pool(self, champion: Champion) -> None:
        """
        Add an unlockable champion to the pool when unlocked.

        Args:
            champion: The unlockable champion to add.
        """
        if champion.id in self.pool:
            return  # Already in pool

        cost = champion.cost
        self.pool[champion.id] = POOL_SIZE.get(cost, 0)
        self.champions[champion.id] = champion

        if cost not in self.champions_by_cost:
            self.champions_by_cost[cost] = []
        self.champions_by_cost[cost].append(champion)

    def get_pool_state(self) -> dict[str, int]:
        """
        Get a copy of the current pool state.

        Returns:
            Dictionary mapping champion_id to remaining count.
        """
        return self.pool.copy()

    def get_max_pool_size(self, cost: int) -> int:
        """
        Get the maximum pool size for a cost tier.

        Args:
            cost: The cost tier (1-7).

        Returns:
            Maximum copies per champion of that cost.
        """
        return POOL_SIZE.get(cost, 0)

    def get_champion_count_by_cost(self, cost: int) -> int:
        """
        Get the number of unique champions at a cost tier.

        Args:
            cost: The cost tier (1-7).

        Returns:
            Number of unique champions at that cost.
        """
        return len(self.champions_by_cost.get(cost, []))

    def get_expected_pool_total(self, cost: int) -> int:
        """
        Get expected total pool size for a cost tier (all copies of all champions).

        Args:
            cost: The cost tier (1-7).

        Returns:
            Expected total if pool was full.
        """
        num_champions = self.get_champion_count_by_cost(cost)
        copies_per_champion = POOL_SIZE.get(cost, 0)
        return num_champions * copies_per_champion

    def calculate_champion_odds(
        self,
        champion_id: str,
        player_level: int,
        shop_slot_count: int = 5,
    ) -> dict[str, float]:
        """
        Calculate detailed odds for finding a specific champion.

        This implements accurate TFT probability calculation:
        P(champion in any slot) = 1 - (1 - P(single slot))^5

        Where P(single slot) = P(roll cost tier) * (copies in pool / total copies of tier)

        Args:
            champion_id: The champion to calculate odds for.
            player_level: Player's current level.
            shop_slot_count: Number of shop slots (default 5).

        Returns:
            Dictionary with:
            - single_slot: Probability in one slot
            - any_slot: Probability in any of 5 slots
            - copies_remaining: Copies left in pool
            - tier_total: Total copies of this tier in pool
        """
        champion = self.champions.get(champion_id)
        if not champion:
            return {
                "single_slot": 0.0,
                "any_slot": 0.0,
                "copies_remaining": 0,
                "tier_total": 0,
            }

        cost = champion.cost
        if cost > 5:
            # 6+ cost champions have special rules
            return {
                "single_slot": 0.0,
                "any_slot": 0.0,
                "copies_remaining": self.pool.get(champion_id, 0),
                "tier_total": self.get_total_available_by_cost(cost),
            }

        # Get tier odds at this level
        level = max(1, min(10, player_level))
        odds = SHOP_ODDS.get(level, [100, 0, 0, 0, 0])
        tier_odds = odds[cost - 1] / 100.0

        if tier_odds == 0:
            return {
                "single_slot": 0.0,
                "any_slot": 0.0,
                "copies_remaining": self.pool.get(champion_id, 0),
                "tier_total": self.get_total_available_by_cost(cost),
            }

        # Get champion's share of the tier pool
        copies_remaining = self.pool.get(champion_id, 0)
        tier_total = self.get_total_available_by_cost(cost)

        if tier_total == 0 or copies_remaining == 0:
            return {
                "single_slot": 0.0,
                "any_slot": 0.0,
                "copies_remaining": copies_remaining,
                "tier_total": tier_total,
            }

        # P(champion | tier rolled) = copies / total
        champion_odds = copies_remaining / tier_total

        # P(champion in one slot) = P(tier) * P(champion | tier)
        single_slot = tier_odds * champion_odds

        # P(champion in any slot) = 1 - P(not in any slot)
        # P(not in any slot) = (1 - single_slot)^5
        any_slot = 1 - ((1 - single_slot) ** shop_slot_count)

        return {
            "single_slot": single_slot,
            "any_slot": any_slot,
            "copies_remaining": copies_remaining,
            "tier_total": tier_total,
        }

    def calculate_copies_needed(self, champion_id: str, target_star: int = 3) -> int:
        """
        Calculate how many more copies needed to reach target star level.

        Args:
            champion_id: The champion ID.
            target_star: Target star level (2 or 3).

        Returns:
            Copies needed (9 for 3-star, 3 for 2-star from 1-star).
        """
        if target_star == 3:
            return 9
        elif target_star == 2:
            return 3
        return 0

    def is_contested(self, champion_id: str, threshold: float = 0.5) -> bool:
        """
        Check if a champion is being contested (many copies taken from pool).

        Args:
            champion_id: The champion to check.
            threshold: Fraction of pool that must be gone (default 0.5 = 50%).

        Returns:
            True if contested (many copies taken).
        """
        champion = self.champions.get(champion_id)
        if not champion:
            return False

        max_copies = POOL_SIZE.get(champion.cost, 0)
        current_copies = self.pool.get(champion_id, 0)

        if max_copies == 0:
            return False

        fraction_remaining = current_copies / max_copies
        return fraction_remaining < (1 - threshold)

    def __repr__(self) -> str:
        total = sum(self.pool.values())
        return f"ChampionPool(total={total}, champions={len(self.champions)})"
