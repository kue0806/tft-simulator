"""Probability Calculator for TFT Set 16.

Calculate various probabilities for decision making.
"""

import math
from typing import Optional

from src.core.champion_pool import ChampionPool
from src.core.constants import SHOP_ODDS, REROLL_COST, SHOP_SIZE


class ProbabilityCalculator:
    """
    Calculate various probabilities for decision making.
    """

    @staticmethod
    def single_slot_probability(
        champion_id: str,
        pool: ChampionPool,
        level: int
    ) -> float:
        """
        Calculate probability of seeing a specific champion in a single shop slot.

        Args:
            champion_id: The champion to find.
            pool: The current champion pool state.
            level: Player's level.

        Returns:
            Probability (0.0 to 1.0).
        """
        return pool.get_probability(champion_id, level)

    @staticmethod
    def chance_to_see_in_shop(
        champion_id: str,
        pool: ChampionPool,
        level: int
    ) -> float:
        """
        Calculate probability of seeing at least one copy in a single shop refresh.

        Args:
            champion_id: The champion to find.
            pool: The current champion pool state.
            level: Player's level.

        Returns:
            Probability (0.0 to 1.0).
        """
        p_single = pool.get_probability(champion_id, level)
        # P(at least one) = 1 - P(none in 5 slots)
        return 1 - (1 - p_single) ** SHOP_SIZE

    @staticmethod
    def chance_to_hit(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int = 1,
        num_rolls: int = 1
    ) -> float:
        """
        Calculate probability of finding at least X copies in Y rolls.

        Args:
            champion_id: The champion to find.
            pool: The current champion pool state.
            level: Player's level.
            copies_needed: Number of copies needed.
            num_rolls: Number of shop refreshes.

        Returns:
            Probability (0.0 to 1.0).
        """
        if copies_needed <= 0:
            return 1.0
        if num_rolls <= 0:
            return 0.0

        p_single = pool.get_probability(champion_id, level)
        if p_single <= 0:
            return 0.0

        # Expected copies per roll = p_single * SHOP_SIZE
        # This is an approximation using Poisson distribution
        # for the probability of getting at least copies_needed in num_rolls * SHOP_SIZE trials

        expected_per_shop = p_single * SHOP_SIZE
        total_expected = expected_per_shop * num_rolls

        # Use Poisson approximation for small probabilities
        # P(X >= k) = 1 - P(X < k) = 1 - sum(P(X=i) for i in 0..k-1)
        # P(X=i) = (lambda^i * e^-lambda) / i!

        poisson_cdf = 0.0
        for i in range(copies_needed):
            poisson_cdf += (total_expected ** i) * math.exp(-total_expected) / math.factorial(i)

        return max(0.0, min(1.0, 1 - poisson_cdf))

    @staticmethod
    def expected_rolls_for_copies(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int
    ) -> float:
        """
        Calculate expected number of rolls to find X copies.

        Args:
            champion_id: The champion to find.
            pool: The current champion pool state.
            level: Player's level.
            copies_needed: Number of copies needed.

        Returns:
            Expected number of shop refreshes.
        """
        if copies_needed <= 0:
            return 0.0

        p_single = pool.get_probability(champion_id, level)
        if p_single <= 0:
            return float('inf')

        # Expected copies per roll
        expected_per_roll = p_single * SHOP_SIZE

        if expected_per_roll <= 0:
            return float('inf')

        # Simple expectation: copies_needed / expected_per_roll
        return copies_needed / expected_per_roll

    @staticmethod
    def expected_gold_for_copies(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        copies_needed: int
    ) -> float:
        """
        Calculate expected gold cost to find X copies.
        gold = expected_rolls * REROLL_COST

        Args:
            champion_id: The champion to find.
            pool: The current champion pool state.
            level: Player's level.
            copies_needed: Number of copies needed.

        Returns:
            Expected gold cost.
        """
        expected_rolls = ProbabilityCalculator.expected_rolls_for_copies(
            champion_id, pool, level, copies_needed
        )
        return expected_rolls * REROLL_COST

    @staticmethod
    def two_star_probability(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        current_copies: int,
        gold_budget: int
    ) -> float:
        """
        Probability of hitting 2-star (3 copies total) given budget.

        Args:
            champion_id: The champion to upgrade.
            pool: The current champion pool state.
            level: Player's level.
            current_copies: Copies already owned.
            gold_budget: Gold available for rolling.

        Returns:
            Probability (0.0 to 1.0).
        """
        copies_needed = max(0, 3 - current_copies)
        if copies_needed <= 0:
            return 1.0

        num_rolls = gold_budget // REROLL_COST

        return ProbabilityCalculator.chance_to_hit(
            champion_id, pool, level, copies_needed, num_rolls
        )

    @staticmethod
    def three_star_probability(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        current_copies: int,
        gold_budget: int
    ) -> float:
        """
        Probability of hitting 3-star (9 copies total) given budget.

        Args:
            champion_id: The champion to upgrade.
            pool: The current champion pool state.
            level: Player's level.
            current_copies: Copies already owned.
            gold_budget: Gold available for rolling.

        Returns:
            Probability (0.0 to 1.0).
        """
        copies_needed = max(0, 9 - current_copies)
        if copies_needed <= 0:
            return 1.0

        num_rolls = gold_budget // REROLL_COST

        return ProbabilityCalculator.chance_to_hit(
            champion_id, pool, level, copies_needed, num_rolls
        )

    @staticmethod
    def rolldown_analysis(
        targets: list[tuple[str, int]],  # [(champion_id, copies_needed), ...]
        pool: ChampionPool,
        level: int,
        gold_budget: int
    ) -> dict:
        """
        Analyze a rolldown scenario with multiple targets.

        Args:
            targets: List of (champion_id, copies_needed) tuples.
            pool: The current champion pool state.
            level: Player's level.
            gold_budget: Gold available for rolling.

        Returns:
            Dictionary with analysis results.
        """
        num_rolls = gold_budget // REROLL_COST
        results = {
            "gold_budget": gold_budget,
            "num_rolls": num_rolls,
            "targets": []
        }

        for champion_id, copies_needed in targets:
            target_result = {
                "champion_id": champion_id,
                "copies_needed": copies_needed,
                "single_slot_prob": ProbabilityCalculator.single_slot_probability(
                    champion_id, pool, level
                ),
                "prob_to_hit": ProbabilityCalculator.chance_to_hit(
                    champion_id, pool, level, copies_needed, num_rolls
                ),
                "expected_rolls": ProbabilityCalculator.expected_rolls_for_copies(
                    champion_id, pool, level, copies_needed
                ),
                "expected_gold": ProbabilityCalculator.expected_gold_for_copies(
                    champion_id, pool, level, copies_needed
                ),
            }
            results["targets"].append(target_result)

        return results

    @staticmethod
    def optimal_level_for_champion(
        champion_id: str,
        pool: ChampionPool
    ) -> int:
        """
        Find the optimal level to roll for a specific champion.

        Args:
            champion_id: The champion to target.
            pool: The current champion pool state.

        Returns:
            Best level to roll at.
        """
        champion = pool.get_champion(champion_id)
        if not champion:
            return 7  # Default

        cost = champion.cost

        # Generally, the optimal level is when your target cost has highest odds
        # while still being accessible
        if cost == 1:
            return 5  # 1-costs fall off at higher levels
        elif cost == 2:
            return 6  # Peak at level 6
        elif cost == 3:
            return 7  # Peak at level 7
        elif cost == 4:
            return 8  # Peak at level 8
        else:  # cost == 5
            return 9  # Best at 9-10

    @staticmethod
    def contested_impact(
        champion_id: str,
        pool: ChampionPool,
        level: int,
        opponents_with_copies: int
    ) -> float:
        """
        Calculate how being contested affects your hit rate.

        Args:
            champion_id: The champion being contested.
            pool: The current champion pool state.
            level: Your level.
            opponents_with_copies: Number of opponents also going for this champion.

        Returns:
            Ratio of current probability vs uncontested (1.0 = not impacted).
        """
        champion = pool.get_champion(champion_id)
        if not champion:
            return 1.0

        # Current probability
        current_prob = pool.get_probability(champion_id, level)

        # Estimate uncontested probability (assuming avg 3 copies taken per opponent)
        avg_copies_taken = opponents_with_copies * 3
        available = pool.get_available(champion_id)

        # What would probability be with those copies back?
        from src.core.constants import POOL_SIZE
        max_pool = POOL_SIZE.get(champion.cost, 0)
        uncontested_available = min(available + avg_copies_taken, max_pool)

        if uncontested_available == 0:
            return 1.0

        # Calculate ratio
        current_ratio = available / pool.get_total_available_by_cost(champion.cost)
        uncontested_ratio = uncontested_available / (
            pool.get_total_available_by_cost(champion.cost) + avg_copies_taken
        )

        if uncontested_ratio == 0:
            return 1.0

        return current_ratio / uncontested_ratio
