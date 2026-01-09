"""Rolldown Strategy Planner.

Provides recommendations for:
- When to rolldown (timing)
- How much to roll (budget)
- Target level and units
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum, auto
from functools import lru_cache

from src.core.constants import LEVEL_XP, SHOP_ODDS
from src.data.loaders.champion_loader import get_champion_by_id

if TYPE_CHECKING:
    from src.core.game_state import PlayerState, GameState


class RolldownStrategy(Enum):
    """Rolldown strategy types."""

    FAST_8 = "fast_8"  # Level to 8, then rolldown
    FAST_9 = "fast_9"  # Level to 9, then rolldown
    SLOW_ROLL_6 = "slow_roll_6"  # Slow roll at 6 (1-cost 3-stars)
    SLOW_ROLL_7 = "slow_roll_7"  # Slow roll at 7 (2-cost 3-stars)
    SLOW_ROLL_8 = "slow_roll_8"  # Slow roll at 8 (3-cost 3-stars)
    ALL_IN = "all_in"  # All-in (low HP emergency)
    SAVE = "save"  # Save economy


@dataclass
class RolldownTiming:
    """Recommended rolldown timing."""

    stage: str  # "4-1", "4-2", etc.
    level: int  # Target level
    gold_threshold: int  # Gold to start rolldown
    description: str


@dataclass
class RolldownPlan:
    """Complete rolldown plan."""

    strategy: RolldownStrategy
    current_phase: str  # "leveling", "rolling", "stabilized"

    # Timing
    recommended_timing: RolldownTiming
    is_rolldown_now: bool  # Should roll now

    # Budget allocation
    roll_budget: int  # Gold for rolling
    level_budget: int  # Gold for leveling
    save_amount: int  # Gold to save

    # Targets
    target_units: List[str]  # Units to find
    target_star_levels: Dict[str, int]  # Target star levels

    # Probability analysis
    hit_probability: float  # Chance to hit targets per roll
    expected_rolls: int  # Expected rolls needed

    # Advice
    advice: List[str]


class RolldownPlanner:
    """
    Rolldown strategy planner.

    Analyzes game state and provides rolldown timing/strategy recommendations.

    Usage:
        planner = RolldownPlanner()
        plan = planner.create_plan(player, game_state, target_units)
    """

    # Key rolldown timings
    KEY_TIMINGS = {
        "3-2": RolldownTiming("3-2", 6, 30, "Early stabilization, 2-cost carries"),
        "4-1": RolldownTiming("4-1", 7, 50, "Mid-game power spike"),
        "4-2": RolldownTiming("4-2", 8, 50, "4-cost carry rolldown"),
        "4-5": RolldownTiming("4-5", 8, 30, "4-cost completion/all-in"),
        "5-1": RolldownTiming("5-1", 8, 50, "Late rolldown"),
        "5-2": RolldownTiming("5-2", 9, 50, "5-cost carry rolldown"),
    }

    def __init__(self):
        """Initialize rolldown planner."""
        pass

    def create_plan(
        self,
        player: "PlayerState",
        game: "GameState",
        target_units: List[str],
        target_stars: Optional[Dict[str, int]] = None,
    ) -> RolldownPlan:
        """
        Create rolldown plan for player.

        Args:
            player: Player state.
            game: Game state.
            target_units: Champion IDs to find.
            target_stars: Target star levels (default 2 for all).

        Returns:
            Complete rolldown plan.
        """
        if target_stars is None:
            target_stars = {u: 2 for u in target_units}

        # Determine strategy
        strategy = self._determine_strategy(player, game, target_units)

        # Get current phase
        current_phase = self._get_current_phase(player, strategy)

        # Get timing recommendation
        timing = self._recommend_timing(player, game, strategy)

        # Should rolldown now?
        is_rolldown_now = self._should_rolldown_now(player, game, strategy)

        # Allocate budget
        roll_budget, level_budget, save_amount = self._allocate_budget(
            player, game, strategy, is_rolldown_now
        )

        # Calculate hit probability
        hit_prob, expected_rolls = self._calculate_hit_probability(
            player, target_units, target_stars
        )

        # Generate advice
        advice = self._generate_advice(
            player, game, strategy, is_rolldown_now, hit_prob
        )

        return RolldownPlan(
            strategy=strategy,
            current_phase=current_phase,
            recommended_timing=timing,
            is_rolldown_now=is_rolldown_now,
            roll_budget=roll_budget,
            level_budget=level_budget,
            save_amount=save_amount,
            target_units=target_units,
            target_star_levels=target_stars,
            hit_probability=hit_prob,
            expected_rolls=expected_rolls,
            advice=advice,
        )

    def _determine_strategy(
        self,
        player: "PlayerState",
        game: "GameState",
        target_units: List[str],
    ) -> RolldownStrategy:
        """
        Determine optimal rolldown strategy.

        Args:
            player: Player state.
            game: Game state.
            target_units: Target unit IDs.

        Returns:
            Recommended strategy.
        """
        # Critical HP - all in
        if player.health <= 30:
            return RolldownStrategy.ALL_IN

        # Analyze target unit costs
        avg_cost = self._get_average_target_cost(target_units)

        # Low cost units - slow roll at appropriate level
        if avg_cost <= 1.5:
            return RolldownStrategy.SLOW_ROLL_6

        if avg_cost <= 2.5:
            return RolldownStrategy.SLOW_ROLL_7

        # Mid cost units - slow roll 8 or fast 8
        if avg_cost <= 3.5:
            if player.health >= 70:
                return RolldownStrategy.FAST_8
            return RolldownStrategy.SLOW_ROLL_8

        # High cost units - fast 8 or fast 9
        if player.health >= 50 and player.gold >= 50:
            return RolldownStrategy.FAST_9

        return RolldownStrategy.FAST_8

    def _get_current_phase(
        self, player: "PlayerState", strategy: RolldownStrategy
    ) -> str:
        """
        Determine current phase of the strategy.

        Args:
            player: Player state.
            strategy: Current strategy.

        Returns:
            Phase string: "leveling", "rolling", or "stabilized".
        """
        target_level = self._get_target_level(strategy)

        if player.level < target_level:
            return "leveling"
        elif player.gold > 20:
            return "rolling"
        else:
            return "stabilized"

    def _recommend_timing(
        self,
        player: "PlayerState",
        game: "GameState",
        strategy: RolldownStrategy,
    ) -> RolldownTiming:
        """
        Recommend rolldown timing.

        Args:
            player: Player state.
            game: Game state.
            strategy: Current strategy.

        Returns:
            Recommended timing.
        """
        stage = game.stage_manager.get_stage_string()

        if strategy == RolldownStrategy.SLOW_ROLL_6:
            return RolldownTiming("3-2+", 6, 50, "Maintain 50+ gold while rolling")
        elif strategy == RolldownStrategy.SLOW_ROLL_7:
            return RolldownTiming("4-1+", 7, 50, "Maintain 50+ gold while rolling")
        elif strategy == RolldownStrategy.SLOW_ROLL_8:
            return RolldownTiming("4-5+", 8, 50, "Maintain 50+ gold while rolling")
        elif strategy == RolldownStrategy.FAST_8:
            return self.KEY_TIMINGS.get("4-2", self.KEY_TIMINGS["4-1"])
        elif strategy == RolldownStrategy.FAST_9:
            return self.KEY_TIMINGS.get("5-2", self.KEY_TIMINGS["5-1"])
        elif strategy == RolldownStrategy.ALL_IN:
            return RolldownTiming(stage, player.level, 0, "All-in now!")

        return self.KEY_TIMINGS.get("4-2", self.KEY_TIMINGS["4-1"])

    def _should_rolldown_now(
        self,
        player: "PlayerState",
        game: "GameState",
        strategy: RolldownStrategy,
    ) -> bool:
        """
        Determine if player should rolldown now.

        Args:
            player: Player state.
            game: Game state.
            strategy: Current strategy.

        Returns:
            True if should rolldown now.
        """
        if strategy == RolldownStrategy.ALL_IN:
            return True

        # Slow roll strategies - roll when over 50 gold at target level
        if strategy in [
            RolldownStrategy.SLOW_ROLL_6,
            RolldownStrategy.SLOW_ROLL_7,
            RolldownStrategy.SLOW_ROLL_8,
        ]:
            target_level = self._get_target_level(strategy)
            return player.level >= target_level and player.gold > 50

        stage = game.stage_manager.get_stage_string()
        target_level = self._get_target_level(strategy)

        # Check if at target level and key timing
        if player.level >= target_level:
            key_stages = ["4-1", "4-2", "4-5", "5-1", "5-2"]
            return stage in key_stages

        return False

    def _allocate_budget(
        self,
        player: "PlayerState",
        game: "GameState",
        strategy: RolldownStrategy,
        is_rolldown: bool,
    ) -> Tuple[int, int, int]:
        """
        Allocate gold budget for rolling, leveling, saving.

        Args:
            player: Player state.
            game: Game state.
            strategy: Current strategy.
            is_rolldown: Whether rolldown is happening.

        Returns:
            Tuple of (roll_budget, level_budget, save_amount).
        """
        gold = player.gold

        if strategy == RolldownStrategy.ALL_IN:
            return gold, 0, 0

        if strategy == RolldownStrategy.SAVE:
            return 0, 0, gold

        # Slow roll - maintain 50 gold
        if strategy in [
            RolldownStrategy.SLOW_ROLL_6,
            RolldownStrategy.SLOW_ROLL_7,
            RolldownStrategy.SLOW_ROLL_8,
        ]:
            roll_budget = max(0, gold - 50)
            return roll_budget, 0, 50

        # Calculate level cost
        target_level = self._get_target_level(strategy)
        level_cost = 0
        if player.level < target_level:
            level_cost = self._calculate_level_cost(player, target_level)

        if is_rolldown:
            # During rolldown: level up first, then roll remaining
            remaining = gold - level_cost
            save = min(10, remaining)  # Keep minimal gold
            roll_budget = max(0, remaining - save)
            return roll_budget, level_cost, save
        else:
            # Saving phase: maintain interest
            save = min(50, gold)
            level_budget = min(level_cost, gold - save)
            return 0, level_budget, save

    def _calculate_hit_probability(
        self,
        player: "PlayerState",
        target_units: List[str],
        target_stars: Dict[str, int],
    ) -> Tuple[float, int]:
        """
        Calculate probability of hitting target units.

        Args:
            player: Player state.
            target_units: Target unit IDs.
            target_stars: Target star levels.

        Returns:
            Tuple of (hit_probability_per_roll, expected_rolls_needed).
        """
        level = player.level
        odds_list = SHOP_ODDS.get(level, [100, 0, 0, 0, 0])

        # Convert to dict
        odds = {cost + 1: odds_list[cost] / 100 for cost in range(5)}

        # Calculate probability per unit
        total_prob = 0.0
        for unit in target_units:
            cost = self._get_unit_cost(unit)
            cost_odds = odds.get(cost, 0)
            # Approximate probability (depends on pool size and remaining copies)
            unit_prob = cost_odds * 0.1  # Rough estimate
            total_prob += unit_prob

        # 5 slots per roll
        per_roll_prob = 1 - (1 - total_prob) ** 5

        if per_roll_prob > 0:
            expected_rolls = int(1 / per_roll_prob)
        else:
            expected_rolls = 999

        return per_roll_prob, expected_rolls

    def _generate_advice(
        self,
        player: "PlayerState",
        game: "GameState",
        strategy: RolldownStrategy,
        is_rolldown: bool,
        hit_prob: float,
    ) -> List[str]:
        """
        Generate advice strings for the player.

        Args:
            player: Player state.
            game: Game state.
            strategy: Current strategy.
            is_rolldown: Whether rolldown is happening.
            hit_prob: Hit probability.

        Returns:
            List of advice strings.
        """
        advice = []

        # Strategy description
        strategy_desc = {
            RolldownStrategy.FAST_8: "Level to 8 and find 4-cost carries",
            RolldownStrategy.FAST_9: "Level to 9 and find 5-cost carries",
            RolldownStrategy.SLOW_ROLL_6: "Slow roll at 6 maintaining 50+ gold",
            RolldownStrategy.SLOW_ROLL_7: "Slow roll at 7 maintaining 50+ gold",
            RolldownStrategy.SLOW_ROLL_8: "Slow roll at 8 maintaining 50+ gold",
            RolldownStrategy.ALL_IN: "Spend all gold now to stabilize!",
            RolldownStrategy.SAVE: "Build economy",
        }
        advice.append(strategy_desc.get(strategy, ""))

        # HP warnings
        if player.health <= 30:
            advice.append("CRITICAL HP! Stabilize immediately")
        elif player.health <= 50:
            advice.append("Low HP - rolldown soon")

        # Hit probability feedback
        if hit_prob < 0.1:
            advice.append("Low hit rate - consider leveling first")
        elif hit_prob > 0.3:
            advice.append("Good hit rate!")

        # Economy advice
        if player.gold >= 50 and not is_rolldown:
            advice.append("Maintain 50 gold for interest")

        return advice

    def _get_target_level(self, strategy: RolldownStrategy) -> int:
        """Get target level for strategy."""
        return {
            RolldownStrategy.SLOW_ROLL_6: 6,
            RolldownStrategy.SLOW_ROLL_7: 7,
            RolldownStrategy.SLOW_ROLL_8: 8,
            RolldownStrategy.FAST_8: 8,
            RolldownStrategy.FAST_9: 9,
            RolldownStrategy.ALL_IN: 8,
            RolldownStrategy.SAVE: 8,
        }.get(strategy, 8)

    def _calculate_level_cost(self, player: "PlayerState", target: int) -> int:
        """Calculate gold needed to reach target level."""
        total = 0
        current_xp = player.xp

        for lvl in range(player.level, target):
            xp_for_level = LEVEL_XP.get(lvl + 1, 100)
            xp_needed = max(0, xp_for_level - current_xp)
            gold_needed = ((xp_needed + 3) // 4) * 4  # 4 gold per 4 XP
            total += gold_needed
            current_xp = 0  # Reset for next level calculation

        return total

    def _get_average_target_cost(self, target_units: List[str]) -> float:
        """Get average cost of target units."""
        if not target_units:
            return 3.0
        costs = [self._get_unit_cost(u) for u in target_units]
        return sum(costs) / len(costs)

    @lru_cache(maxsize=128)
    def _get_unit_cost(self, champion_id: str) -> int:
        """
        Get cost of a champion by ID.

        Args:
            champion_id: Champion ID.

        Returns:
            Champion cost (default 3 if not found).
        """
        champion = get_champion_by_id(champion_id)
        if champion is not None:
            return champion.cost
        return 3  # Default fallback

    def estimate_rolls_to_complete(
        self,
        player: "PlayerState",
        target_units: List[str],
        target_stars: Dict[str, int],
    ) -> int:
        """
        Estimate rolls needed to complete targets.

        Args:
            player: Player state.
            target_units: Target unit IDs.
            target_stars: Target star levels.

        Returns:
            Estimated roll count.
        """
        hit_prob, expected = self._calculate_hit_probability(
            player, target_units, target_stars
        )

        # Calculate copies still needed
        total_copies_needed = 0
        for unit_id in target_units:
            current = self._count_owned_copies(player, unit_id)
            target = 3 ** (target_stars.get(unit_id, 2) - 1) * 3
            needed = max(0, target - current)
            total_copies_needed += needed

        # Each hit gives approximately 1 copy
        return expected * total_copies_needed

    def _count_owned_copies(self, player: "PlayerState", champion_id: str) -> int:
        """Count copies of champion owned."""
        count = 0
        for inst in player.units.bench:
            if inst and inst.champion.id == champion_id:
                count += 3 ** (inst.star_level - 1)
        for inst in player.units.board.values():
            if inst.champion.id == champion_id:
                count += 3 ** (inst.star_level - 1)
        return count
