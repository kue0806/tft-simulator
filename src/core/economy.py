"""Economy System for TFT Set 16.

Handles gold income, interest, streaks, leveling, and economy simulation.
"""

from dataclasses import dataclass
from typing import Optional

from src.core.constants import (
    REROLL_COST,
    MAX_INTEREST,
    LEVEL_XP,
    SHOP_ODDS,
)


@dataclass
class IncomeBreakdown:
    """Breakdown of income sources for a round."""

    base_income: int = 5
    interest: int = 0
    streak_bonus: int = 0
    pvp_win_bonus: int = 0  # +1 gold for winning PvP round
    first_blood: int = 0  # Extra gold for first kill
    pve_bonus: int = 0  # PvE round drops
    total: int = 0

    def calculate_total(self) -> int:
        """Calculate and store total income."""
        self.total = (
            self.base_income
            + self.interest
            + self.streak_bonus
            + self.pvp_win_bonus
            + self.first_blood
            + self.pve_bonus
        )
        return self.total


@dataclass
class EconomyState:
    """Complete economy state for a player."""

    gold: int = 0
    level: int = 1
    xp: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    round_number: int = 0
    stage: str = "1-1"


class EconomyCalculator:
    """Calculate all economy-related values."""

    # Base income per round
    BASE_INCOME = 5

    # Interest: 1 gold per 10 gold, max 5
    INTEREST_RATE = 0.1
    MAX_INTEREST = 5

    # Streak bonuses (TFT Set 16)
    # 3-4 streak: +1, 5 streak: +2, 6+ streak: +3
    STREAK_BONUS = {
        0: 0,
        1: 0,
        2: 0,  # No bonus for 2 streak
        3: 1,
        4: 1,
        5: 2,
        6: 3,  # 6+ streak
    }

    # XP costs per level (uses constants but stored here for reference)
    XP_TO_LEVEL = LEVEL_XP

    # Gold cost to buy XP
    XP_PURCHASE_COST = 4
    XP_PER_PURCHASE = 4

    def calculate_interest(self, gold: int) -> int:
        """
        Calculate interest from current gold.
        1 gold per 10 gold held, max 5.

        Args:
            gold: Current gold amount.

        Returns:
            Interest earned.
        """
        interest = gold // 10
        return min(interest, self.MAX_INTEREST)

    def calculate_streak_bonus(self, streak: int) -> int:
        """
        Calculate bonus gold from win/loss streak.
        2-4: +1, 5: +2, 6+: +3 (TFT Set 16)

        Args:
            streak: Current streak count (win or loss).

        Returns:
            Bonus gold from streak.
        """
        if streak >= 6:
            return 3
        if streak == 5:
            return 2
        return self.STREAK_BONUS.get(streak, 0)

    def calculate_round_income(
        self,
        state: EconomyState,
        won_combat: bool = True,
        is_pve: bool = False,
        pve_gold: int = 0,
    ) -> IncomeBreakdown:
        """
        Calculate total income for end of round.

        Args:
            state: Current economy state.
            won_combat: Whether player won this round.
            is_pve: Whether this was a PvE round.
            pve_gold: Gold from PvE drops.

        Returns:
            IncomeBreakdown with all income components.
        """
        breakdown = IncomeBreakdown()

        # Base income
        breakdown.base_income = self.BASE_INCOME

        # Interest
        breakdown.interest = self.calculate_interest(state.gold)

        # Streak bonus
        streak = state.win_streak if won_combat else state.loss_streak
        breakdown.streak_bonus = self.calculate_streak_bonus(streak)

        # PvP win bonus: +1 gold for winning a PvP round
        if won_combat and not is_pve:
            breakdown.pvp_win_bonus = 1

        # PvE bonus
        if is_pve:
            breakdown.pve_bonus = pve_gold

        breakdown.calculate_total()
        return breakdown

    def calculate_xp_needed(self, current_level: int) -> int:
        """
        Get XP needed to reach next level.

        Args:
            current_level: Current player level.

        Returns:
            XP needed for next level, 0 if at max.
        """
        next_level = current_level + 1
        if next_level > 10:
            return 0  # Max level
        return self.XP_TO_LEVEL.get(next_level, 0)

    def calculate_gold_to_level(
        self, current_level: int, current_xp: int, target_level: int
    ) -> int:
        """
        Calculate gold needed to reach target level from current state.

        Args:
            current_level: Current player level.
            current_xp: Current XP towards next level.
            target_level: Target level to reach.

        Returns:
            Gold needed to reach target level.
        """
        if target_level <= current_level:
            return 0

        total_xp_needed = 0

        # Calculate XP needed for each level
        for lvl in range(current_level + 1, target_level + 1):
            total_xp_needed += self.XP_TO_LEVEL.get(lvl, 0)

        # Subtract current XP progress
        total_xp_needed -= current_xp

        # Convert to gold (4 gold = 4 XP)
        purchases_needed = (total_xp_needed + self.XP_PER_PURCHASE - 1) // self.XP_PER_PURCHASE
        return purchases_needed * self.XP_PURCHASE_COST

    def calculate_rounds_to_level(
        self,
        current_level: int,
        current_xp: int,
        target_level: int,
        passive_xp_per_round: int = 2,
    ) -> int:
        """
        Calculate rounds needed to reach target level with passive XP only.

        Args:
            current_level: Current player level.
            current_xp: Current XP towards next level.
            target_level: Target level to reach.
            passive_xp_per_round: XP gained per round (default 2).

        Returns:
            Number of rounds needed.
        """
        if target_level <= current_level:
            return 0

        total_xp_needed = 0
        for lvl in range(current_level + 1, target_level + 1):
            total_xp_needed += self.XP_TO_LEVEL.get(lvl, 0)

        total_xp_needed -= current_xp

        return (total_xp_needed + passive_xp_per_round - 1) // passive_xp_per_round

    def simulate_economy(
        self, initial_state: EconomyState, rounds: int, strategy: str = "standard"
    ) -> list[EconomyState]:
        """
        Simulate economy over N rounds.

        Strategies:
        - "standard": Normal play, level at standard timings
        - "econ": Max economy, slow level
        - "aggressive": Fast level, low economy

        Args:
            initial_state: Starting economy state.
            rounds: Number of rounds to simulate.
            strategy: Economy strategy to use.

        Returns:
            List of EconomyState for each round.
        """
        states = [initial_state]
        current = EconomyState(
            gold=initial_state.gold,
            level=initial_state.level,
            xp=initial_state.xp,
            win_streak=initial_state.win_streak,
            loss_streak=initial_state.loss_streak,
            round_number=initial_state.round_number,
            stage=initial_state.stage,
        )

        for _ in range(rounds):
            # Simulate one round
            current = self._simulate_round(current, strategy)
            states.append(
                EconomyState(
                    gold=current.gold,
                    level=current.level,
                    xp=current.xp,
                    win_streak=current.win_streak,
                    loss_streak=current.loss_streak,
                    round_number=current.round_number,
                    stage=current.stage,
                )
            )

        return states

    def _simulate_round(self, state: EconomyState, strategy: str) -> EconomyState:
        """
        Simulate a single round.

        Args:
            state: Current economy state.
            strategy: Economy strategy.

        Returns:
            Updated state after the round.
        """
        # Assume 50% win rate for simulation
        won = state.round_number % 2 == 0

        # Update streaks
        if won:
            state.win_streak += 1
            state.loss_streak = 0
        else:
            state.loss_streak += 1
            state.win_streak = 0

        # Calculate income
        income = self.calculate_round_income(state, won)
        state.gold += income.total

        # Add passive XP (2 per round after stage 1)
        if state.round_number > 3:
            state.xp += 2

        # Check level up
        xp_needed = self.calculate_xp_needed(state.level)
        while state.xp >= xp_needed and state.level < 10 and xp_needed > 0:
            state.xp -= xp_needed
            state.level += 1
            xp_needed = self.calculate_xp_needed(state.level)

        # Strategy-based spending
        if strategy == "aggressive":
            # Buy XP if above 10 gold
            while state.gold >= 14 and state.level < 9:
                state.gold -= 4
                state.xp += 4
                # Check for level up after buying XP
                xp_needed = self.calculate_xp_needed(state.level)
                while state.xp >= xp_needed and state.level < 10 and xp_needed > 0:
                    state.xp -= xp_needed
                    state.level += 1
                    xp_needed = self.calculate_xp_needed(state.level)
        elif strategy == "econ":
            # Only spend above 50 gold
            pass  # Do nothing, save gold
        else:  # standard
            # Level at key stages
            if state.level < 8 and state.gold > 50:
                state.gold -= 4
                state.xp += 4
                # Check for level up
                xp_needed = self.calculate_xp_needed(state.level)
                if state.xp >= xp_needed:
                    state.xp -= xp_needed
                    state.level += 1

        state.round_number += 1
        return state


class RolldownCalculator:
    """Calculate rolldown budgets and probabilities."""

    def __init__(self):
        """Initialize the rolldown calculator."""
        self.econ = EconomyCalculator()

    def calculate_rolldown_budget(
        self,
        current_gold: int,
        health: int,
        stage: str,
        target_level: Optional[int] = None,
    ) -> int:
        """
        Calculate how much gold to spend on rolling.

        Rules:
        - Keep 10 gold minimum for interest
        - Keep more if healthy (above 50 HP)
        - Roll everything if low HP (below 30)

        Args:
            current_gold: Current gold amount.
            health: Current player health.
            stage: Current stage string.
            target_level: Optional target level (unused but for future).

        Returns:
            Gold budget for rolling.
        """
        min_reserve = 10  # For 1 interest

        if health < 30:
            # Desperate - roll everything
            return current_gold - REROLL_COST
        elif health < 50:
            # Low - keep minimal reserve
            min_reserve = 10
        else:
            # Healthy - keep more reserve
            min_reserve = 20

        return max(0, current_gold - min_reserve)

    def expected_rolls(self, gold_budget: int) -> int:
        """
        Calculate number of rolls from gold budget.

        Args:
            gold_budget: Gold available for rolling.

        Returns:
            Number of shop refreshes possible.
        """
        return gold_budget // REROLL_COST

    def calculate_optimal_rolldown_stage(
        self,
        current_state: EconomyState,
        target_cost: int,  # Target champion cost (4 or 5)
        desired_copies: int = 6,  # For 2-star
    ) -> dict:
        """
        Determine optimal stage to rolldown.

        Args:
            current_state: Current economy state.
            target_cost: Cost tier to target (1-5).
            desired_copies: Number of copies needed.

        Returns:
            Dict with recommendation details.
        """
        recommendations = []

        # Evaluate different timing options
        timings = [
            ("3-2", 6, 30),  # Early level 6
            ("3-5", 7, 40),  # Standard level 7
            ("4-1", 7, 50),  # Late level 7
            ("4-2", 8, 50),  # Standard level 8
            ("4-5", 8, 60),  # Late level 8
            ("5-1", 9, 70),  # Level 9
        ]

        for stage, level, expected_gold in timings:
            # Get odds for target cost at this level
            odds = SHOP_ODDS.get(level, [0] * 5)
            target_odds = odds[target_cost - 1] / 100  # Convert to decimal

            # Calculate expected hits
            num_rolls = expected_gold // REROLL_COST
            champions_per_roll = 5
            expected_hits = num_rolls * champions_per_roll * target_odds * 0.1  # Rough estimate

            recommendations.append(
                {
                    "stage": stage,
                    "level": level,
                    "expected_gold": expected_gold,
                    "target_odds": target_odds,
                    "expected_rolls": num_rolls,
                    "estimated_hits": expected_hits,
                    "viable": expected_hits >= desired_copies * 0.5,
                }
            )

        # Find best viable option
        viable = [r for r in recommendations if r["viable"]]
        if viable:
            best = max(viable, key=lambda x: x["estimated_hits"])
        else:
            best = recommendations[-1]  # Default to latest

        return {
            "recommended_stage": best["stage"],
            "recommended_level": best["level"],
            "expected_gold": best["expected_gold"],
            "hit_probability": min(0.95, best["estimated_hits"] / desired_copies),
            "all_options": recommendations,
        }
