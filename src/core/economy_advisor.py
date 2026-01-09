"""Economy Advisor for TFT Set 16.

Provides strategic economy advice based on game state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EconomyStrategy(Enum):
    """Economy strategies for different game situations."""

    LOSS_STREAK = "loss_streak"  # Intentional losing for gold
    WIN_STREAK = "win_streak"  # Play strongest board
    STANDARD = "standard"  # Balanced approach
    FAST_8 = "fast_8"  # Rush level 8
    SLOW_ROLL = "slow_roll"  # Stay at level, roll for 3-stars
    ALL_IN = "all_in"  # Desperate rolldown


@dataclass
class EconomyAdvice:
    """Economy recommendation for current state."""

    strategy: EconomyStrategy
    action: str  # "level", "roll", "save", "all_in"
    gold_to_spend: int
    gold_to_keep: int
    reasoning: list[str]
    priority: str  # "high", "medium", "low"


class EconomyAdvisor:
    """Provides economy advice based on game state."""

    def __init__(self):
        """Initialize the advisor with economy calculators."""
        from src.core.economy import EconomyCalculator, RolldownCalculator

        self.econ = EconomyCalculator()
        self.rolldown = RolldownCalculator()

    def get_advice(
        self,
        gold: int,
        level: int,
        xp: int,
        health: int,
        stage: str,
        win_streak: int = 0,
        loss_streak: int = 0,
        board_strength: float = 0.5,  # 0-1 relative strength
    ) -> EconomyAdvice:
        """
        Get economy advice for current situation.

        Args:
            gold: Current gold.
            level: Current player level.
            xp: Current XP towards next level.
            health: Current player health.
            stage: Current stage string (e.g., "4-2").
            win_streak: Current win streak count.
            loss_streak: Current loss streak count.
            board_strength: Relative board strength (0-1).

        Returns:
            EconomyAdvice with recommendation.
        """
        reasons = []

        # Determine strategy based on state
        strategy = self._determine_strategy(
            health, stage, board_strength, win_streak, loss_streak
        )

        # Calculate recommended action
        if health < 20:
            # Critical HP - must stabilize
            action = "all_in"
            gold_to_spend = gold - 2  # Keep minimum for 1 roll
            gold_to_keep = 2
            reasons.append("Critical HP - must stabilize immediately")
            priority = "high"

        elif health < 40:
            # Low HP - aggressive play
            action = "roll"
            gold_to_keep = 10
            gold_to_spend = max(0, gold - gold_to_keep)
            reasons.append("Low HP - roll to stabilize")
            priority = "high"

        elif strategy == EconomyStrategy.FAST_8:
            # Fast 8 strategy
            if level < 8:
                action = "level"
                gold_to_spend = self._gold_to_level_up(level, xp)
                gold_to_keep = gold - gold_to_spend
                reasons.append("Fast 8 - leveling aggressively")
            else:
                action = "roll"
                gold_to_keep = 30
                gold_to_spend = max(0, gold - gold_to_keep)
                reasons.append("Level 8 reached - rolling for upgrades")
            priority = "medium"

        elif strategy == EconomyStrategy.SLOW_ROLL:
            # Slow roll at level 5/6/7
            if gold > 50:
                action = "roll"
                gold_to_spend = gold - 50
                gold_to_keep = 50
                reasons.append("Slow rolling above 50 gold")
            else:
                action = "save"
                gold_to_spend = 0
                gold_to_keep = gold
                reasons.append("Building to 50 gold")
            priority = "medium"

        elif strategy == EconomyStrategy.LOSS_STREAK:
            action = "save"
            gold_to_spend = 0
            gold_to_keep = gold
            reasons.append(f"Loss streaking ({loss_streak}) - maximizing econ")
            priority = "low"

        elif strategy == EconomyStrategy.WIN_STREAK:
            # Maintain win streak - invest in board
            action, gold_to_spend, gold_to_keep = self._win_streak_play(
                gold, level, xp, stage
            )
            reasons.append(f"Win streaking ({win_streak}) - maintain strength")
            priority = "medium"

        else:
            # Standard play
            action, gold_to_spend, gold_to_keep = self._standard_play(
                gold, level, xp, stage
            )
            reasons.append("Standard economy play")
            priority = "medium"

        return EconomyAdvice(
            strategy=strategy,
            action=action,
            gold_to_spend=gold_to_spend,
            gold_to_keep=gold_to_keep,
            reasoning=reasons,
            priority=priority,
        )

    def _determine_strategy(
        self,
        health: int,
        stage: str,
        board_strength: float,
        win_streak: int,
        loss_streak: int,
    ) -> EconomyStrategy:
        """
        Determine best strategy for current state.

        Args:
            health: Current player health.
            stage: Current stage string.
            board_strength: Relative board strength (0-1).
            win_streak: Current win streak count.
            loss_streak: Current loss streak count.

        Returns:
            Recommended EconomyStrategy.
        """
        stage_num = int(stage.split("-")[0])

        if health < 30:
            return EconomyStrategy.ALL_IN

        if loss_streak >= 4 and health > 60:
            return EconomyStrategy.LOSS_STREAK

        if win_streak >= 4 and board_strength > 0.7:
            return EconomyStrategy.WIN_STREAK

        if stage_num <= 3 and board_strength < 0.4:
            return EconomyStrategy.SLOW_ROLL

        if stage_num >= 4 and board_strength > 0.6:
            return EconomyStrategy.FAST_8

        return EconomyStrategy.STANDARD

    def _gold_to_level_up(self, level: int, xp: int) -> int:
        """
        Calculate gold needed to level up once.

        Args:
            level: Current level.
            xp: Current XP towards next level.

        Returns:
            Gold needed to level up.
        """
        xp_needed = self.econ.calculate_xp_needed(level) - xp
        if xp_needed <= 0:
            return 0
        purchases = (xp_needed + 3) // 4  # 4 XP per 4 gold
        return purchases * 4

    def _standard_play(
        self, gold: int, level: int, xp: int, stage: str
    ) -> tuple[str, int, int]:
        """
        Standard economy decisions.

        Args:
            gold: Current gold.
            level: Current level.
            xp: Current XP.
            stage: Current stage string.

        Returns:
            Tuple of (action, gold_to_spend, gold_to_keep).
        """
        stage_num = int(stage.split("-")[0])
        round_num = int(stage.split("-")[1])

        # Standard level timings
        level_timings = {
            (2, 1): 4,  # Level 4 at 2-1
            (2, 5): 5,  # Level 5 at 2-5
            (3, 2): 6,  # Level 6 at 3-2
            (4, 1): 7,  # Level 7 at 4-1
            (4, 2): 8,  # Level 8 at 4-2
        }

        target_level = level_timings.get((stage_num, round_num))

        if target_level and level < target_level:
            # Should level up
            cost = self._gold_to_level_up(level, xp)
            if gold >= cost + 10:  # Keep 10 for interest
                return "level", cost, gold - cost

        # Otherwise save for interest
        return "save", 0, gold

    def _win_streak_play(
        self, gold: int, level: int, xp: int, stage: str
    ) -> tuple[str, int, int]:
        """
        Win streak economy decisions - invest to maintain strength.

        Args:
            gold: Current gold.
            level: Current level.
            xp: Current XP.
            stage: Current stage string.

        Returns:
            Tuple of (action, gold_to_spend, gold_to_keep).
        """
        stage_num = int(stage.split("-")[0])

        # More aggressive leveling during win streak
        if level < 8 and gold > 30:
            cost = self._gold_to_level_up(level, xp)
            if cost <= gold - 20:  # Keep at least 20
                return "level", cost, gold - cost

        # Light rolling to maintain upgrades
        if gold > 50 and stage_num >= 3:
            spend = min(10, gold - 50)  # Roll a bit above 50
            return "roll", spend, gold - spend

        return "save", 0, gold

    def get_rolldown_recommendation(
        self,
        gold: int,
        level: int,
        health: int,
        target_champions: Optional[list[str]],
        stage: str,
    ) -> dict:
        """
        Get specific rolldown recommendation.

        Args:
            gold: Current gold.
            level: Current player level.
            health: Current player health.
            target_champions: List of champions to look for.
            stage: Current stage string.

        Returns:
            Dict with rolldown recommendation details.
        """
        # Calculate budget
        budget = self.rolldown.calculate_rolldown_budget(gold, health, stage)
        num_rolls = budget // 2

        # Determine if should rolldown now
        should_roll = False
        reason = ""

        if health < 30:
            should_roll = True
            reason = "Critical HP - must roll now"
        elif stage == "4-2" and level == 8:
            should_roll = True
            reason = "Standard 4-2 rolldown at level 8"
        elif stage == "5-1" and level == 9:
            should_roll = True
            reason = "Standard 5-1 rolldown at level 9"
        elif budget >= 30:
            should_roll = True
            reason = f"Good budget ({budget}g) available"

        return {
            "should_rolldown": should_roll,
            "budget": budget,
            "num_rolls": num_rolls,
            "reason": reason,
            "keep_gold": gold - budget,
        }

    def get_level_recommendation(
        self, gold: int, level: int, xp: int, stage: str, health: int
    ) -> dict:
        """
        Get leveling recommendation.

        Args:
            gold: Current gold.
            level: Current player level.
            xp: Current XP.
            stage: Current stage string.
            health: Current health.

        Returns:
            Dict with leveling recommendation.
        """
        stage_num = int(stage.split("-")[0])
        round_num = int(stage.split("-")[1])

        # Standard level targets by stage
        level_targets = {
            2: 5,  # Level 5 by end of stage 2
            3: 7,  # Level 7 by end of stage 3
            4: 8,  # Level 8 by end of stage 4
            5: 9,  # Level 9 by end of stage 5
        }

        target = level_targets.get(stage_num, level)
        if level >= target:
            return {
                "should_level": False,
                "current_level": level,
                "target_level": target,
                "gold_needed": 0,
                "reason": "Already at or above target level",
            }

        gold_needed = self._gold_to_level_up(level, xp)
        can_afford = gold >= gold_needed + 10  # Keep 10 for interest

        should_level = can_afford and (health < 60 or gold > 50)

        return {
            "should_level": should_level,
            "current_level": level,
            "target_level": target,
            "gold_needed": gold_needed,
            "reason": (
                "Level up to hit power spike"
                if should_level
                else "Save gold for economy"
            ),
        }
