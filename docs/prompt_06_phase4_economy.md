# TFT Simulator - Phase 4: Economy System

## Overview
Implement the complete economy system including gold income, interest, streaks, leveling, and economic strategy simulation.

## Part 1: Economy Calculator

Create `src/core/economy.py`:

```python
from dataclasses import dataclass
from typing import Optional
from src.core.constants import (
    REROLL_COST, INTEREST_PER_10_GOLD, MAX_INTEREST,
    LEVEL_XP, XP_PER_PURCHASE, STREAK_BONUS
)

@dataclass
class IncomeBreakdown:
    """Breakdown of income sources for a round."""
    base_income: int = 5
    interest: int = 0
    streak_bonus: int = 0
    first_blood: int = 0  # Extra gold for first kill
    pve_bonus: int = 0    # PvE round drops
    total: int = 0
    
    def calculate_total(self) -> int:
        self.total = (
            self.base_income + 
            self.interest + 
            self.streak_bonus + 
            self.first_blood +
            self.pve_bonus
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
    """
    Calculate all economy-related values.
    """
    
    # Base income per round
    BASE_INCOME = 5
    
    # Interest: 1 gold per 10 gold, max 5
    INTEREST_RATE = 0.1
    MAX_INTEREST = 5
    
    # Streak bonuses
    STREAK_BONUS = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 3,  # 5+ streak
    }
    
    # XP costs per level
    XP_TO_LEVEL = {
        2: 2,
        3: 6,
        4: 10,
        5: 20,
        6: 36,
        7: 56,
        8: 80,
        9: 84,
        10: 100,
    }
    
    # Gold cost to buy XP
    XP_PURCHASE_COST = 4
    XP_PER_PURCHASE = 4
    
    def calculate_interest(self, gold: int) -> int:
        """
        Calculate interest from current gold.
        1 gold per 10 gold held, max 5.
        """
        interest = gold // 10
        return min(interest, self.MAX_INTEREST)
    
    def calculate_streak_bonus(self, streak: int) -> int:
        """
        Calculate bonus gold from win/loss streak.
        2-3: +1, 4: +2, 5+: +3
        """
        if streak >= 5:
            return 3
        return self.STREAK_BONUS.get(streak, 0)
    
    def calculate_round_income(
        self, 
        state: EconomyState,
        won_combat: bool = True,
        is_pve: bool = False,
        pve_gold: int = 0
    ) -> IncomeBreakdown:
        """
        Calculate total income for end of round.
        
        Args:
            state: Current economy state
            won_combat: Whether player won this round
            is_pve: Whether this was a PvE round
            pve_gold: Gold from PvE drops
        """
        breakdown = IncomeBreakdown()
        
        # Base income
        breakdown.base_income = self.BASE_INCOME
        
        # Interest
        breakdown.interest = self.calculate_interest(state.gold)
        
        # Streak bonus
        streak = state.win_streak if won_combat else state.loss_streak
        breakdown.streak_bonus = self.calculate_streak_bonus(streak)
        
        # PvE bonus
        if is_pve:
            breakdown.pve_bonus = pve_gold
        
        breakdown.calculate_total()
        return breakdown
    
    def calculate_xp_needed(self, current_level: int) -> int:
        """Get XP needed to reach next level."""
        next_level = current_level + 1
        if next_level > 10:
            return 0  # Max level
        return self.XP_TO_LEVEL.get(next_level, 0)
    
    def calculate_gold_to_level(
        self, 
        current_level: int, 
        current_xp: int,
        target_level: int
    ) -> int:
        """
        Calculate gold needed to reach target level from current state.
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
        passive_xp_per_round: int = 2
    ) -> int:
        """
        Calculate rounds needed to reach target level with passive XP only.
        """
        if target_level <= current_level:
            return 0
        
        total_xp_needed = 0
        for lvl in range(current_level + 1, target_level + 1):
            total_xp_needed += self.XP_TO_LEVEL.get(lvl, 0)
        
        total_xp_needed -= current_xp
        
        return (total_xp_needed + passive_xp_per_round - 1) // passive_xp_per_round
    
    def simulate_economy(
        self,
        initial_state: EconomyState,
        rounds: int,
        strategy: str = "standard"
    ) -> list[EconomyState]:
        """
        Simulate economy over N rounds.
        
        Strategies:
        - "standard": Normal play, level at standard timings
        - "econ": Max economy, slow level
        - "aggressive": Fast level, low economy
        """
        states = [initial_state]
        current = EconomyState(**vars(initial_state))
        
        for _ in range(rounds):
            # Simulate one round
            current = self._simulate_round(current, strategy)
            states.append(EconomyState(**vars(current)))
        
        return states
    
    def _simulate_round(
        self, 
        state: EconomyState, 
        strategy: str
    ) -> EconomyState:
        """Simulate a single round."""
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
        while state.xp >= xp_needed and state.level < 10:
            state.xp -= xp_needed
            state.level += 1
            xp_needed = self.calculate_xp_needed(state.level)
        
        # Strategy-based spending
        if strategy == "aggressive":
            # Buy XP if above 10 gold
            while state.gold >= 14 and state.level < 9:
                state.gold -= 4
                state.xp += 4
        elif strategy == "econ":
            # Only spend above 50 gold
            pass  # Do nothing, save gold
        else:  # standard
            # Level at key stages
            if state.level < 8 and state.gold > 50:
                state.gold -= 4
                state.xp += 4
        
        state.round_number += 1
        return state


class RolldownCalculator:
    """
    Calculate rolldown budgets and probabilities.
    """
    
    def __init__(self):
        self.econ = EconomyCalculator()
    
    def calculate_rolldown_budget(
        self,
        current_gold: int,
        health: int,
        stage: str,
        target_level: int = None
    ) -> int:
        """
        Calculate how much gold to spend on rolling.
        
        Rules:
        - Keep 10 gold minimum for interest
        - Keep more if healthy (above 50 HP)
        - Roll everything if low HP (below 30)
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
        """Calculate number of rolls from gold budget."""
        return gold_budget // REROLL_COST
    
    def calculate_optimal_rolldown_stage(
        self,
        current_state: EconomyState,
        target_cost: int,  # Target champion cost (4 or 5)
        desired_copies: int = 6  # For 2-star
    ) -> dict:
        """
        Determine optimal stage to rolldown.
        
        Returns:
            {
                "recommended_stage": "4-2",
                "recommended_level": 8,
                "expected_gold": 50,
                "hit_probability": 0.65,
            }
        """
        from src.core.constants import SHOP_ODDS
        
        recommendations = []
        
        # Evaluate different timing options
        timings = [
            ("3-2", 6, 30),   # Early level 6
            ("3-5", 7, 40),   # Standard level 7
            ("4-1", 7, 50),   # Late level 7
            ("4-2", 8, 50),   # Standard level 8
            ("4-5", 8, 60),   # Late level 8
            ("5-1", 9, 70),   # Level 9
        ]
        
        for stage, level, expected_gold in timings:
            # Get odds for target cost at this level
            odds = SHOP_ODDS.get(level, [0]*5)
            target_odds = odds[target_cost - 1] / 100  # Convert to decimal
            
            # Calculate expected hits
            num_rolls = expected_gold // REROLL_COST
            champions_per_roll = 5
            expected_hits = num_rolls * champions_per_roll * target_odds * 0.1  # Rough estimate
            
            recommendations.append({
                "stage": stage,
                "level": level,
                "expected_gold": expected_gold,
                "target_odds": target_odds,
                "expected_rolls": num_rolls,
                "estimated_hits": expected_hits,
                "viable": expected_hits >= desired_copies * 0.5
            })
        
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
            "all_options": recommendations
        }
```

## Part 2: Stage Manager

Create `src/core/stage_manager.py`:

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class RoundType(Enum):
    PVE = "pve"
    PVP = "pvp"
    CAROUSEL = "carousel"
    AUGMENT = "augment"


@dataclass
class RoundInfo:
    """Information about a specific round."""
    stage: str          # e.g., "2-1"
    stage_num: int      # e.g., 2
    round_num: int      # e.g., 1
    round_type: RoundType
    passive_xp: int     # XP gained this round
    is_carousel: bool = False
    is_augment: bool = False


class StageManager:
    """
    Manage game stages and rounds.
    """
    
    # Round types by stage-round
    ROUND_TYPES = {
        # Stage 1: PvE
        "1-1": RoundType.PVE,
        "1-2": RoundType.PVE,
        "1-3": RoundType.PVE,
        "1-4": RoundType.PVE,
        
        # Stage 2+: First round is carousel or PvP
        "2-1": RoundType.PVP,
        "2-2": RoundType.PVP,
        "2-3": RoundType.PVP,
        "2-4": RoundType.AUGMENT,  # First augment
        "2-5": RoundType.PVP,
        "2-6": RoundType.PVP,
        "2-7": RoundType.PVP,
        
        # Pattern continues...
    }
    
    # Carousel rounds
    CAROUSEL_ROUNDS = ["2-4", "3-4", "4-4", "5-4", "6-4"]
    
    # Augment rounds
    AUGMENT_ROUNDS = ["2-1", "3-2", "4-2"]
    
    # Passive XP per round (starts stage 2)
    PASSIVE_XP = 2
    
    def __init__(self):
        self.current_stage = 1
        self.current_round = 1
        self.total_rounds = 0
    
    def get_stage_string(self) -> str:
        """Get current stage as string (e.g., '2-3')."""
        return f"{self.current_stage}-{self.current_round}"
    
    def advance_round(self) -> RoundInfo:
        """
        Advance to next round and return info.
        """
        self.current_round += 1
        self.total_rounds += 1
        
        # Check for stage advancement
        max_rounds = self._get_max_rounds_in_stage()
        if self.current_round > max_rounds:
            self.current_stage += 1
            self.current_round = 1
        
        return self.get_current_round_info()
    
    def _get_max_rounds_in_stage(self) -> int:
        """Get number of rounds in current stage."""
        if self.current_stage == 1:
            return 4  # Stage 1 has 4 PvE rounds
        return 7  # Other stages have 7 rounds
    
    def get_current_round_info(self) -> RoundInfo:
        """Get info about current round."""
        stage_str = self.get_stage_string()
        
        # Determine round type
        round_type = self.ROUND_TYPES.get(stage_str, RoundType.PVP)
        
        # Passive XP (none in stage 1)
        passive_xp = self.PASSIVE_XP if self.current_stage > 1 else 0
        
        return RoundInfo(
            stage=stage_str,
            stage_num=self.current_stage,
            round_num=self.current_round,
            round_type=round_type,
            passive_xp=passive_xp,
            is_carousel=stage_str in self.CAROUSEL_ROUNDS,
            is_augment=stage_str in self.AUGMENT_ROUNDS
        )
    
    def get_rounds_until(self, target_stage: str) -> int:
        """Calculate rounds until target stage."""
        target_parts = target_stage.split("-")
        target_stage_num = int(target_parts[0])
        target_round_num = int(target_parts[1])
        
        rounds = 0
        
        # Rounds remaining in current stage
        if self.current_stage == target_stage_num:
            return target_round_num - self.current_round
        
        # Complete current stage
        rounds += self._get_max_rounds_in_stage() - self.current_round
        
        # Add full stages in between
        for stage in range(self.current_stage + 1, target_stage_num):
            rounds += 7  # Each stage has 7 rounds
        
        # Add rounds in target stage
        rounds += target_round_num
        
        return rounds
    
    def is_rolldown_timing(self) -> bool:
        """Check if current round is a common rolldown timing."""
        stage_str = self.get_stage_string()
        rolldown_timings = ["3-2", "4-1", "4-2", "4-5", "5-1"]
        return stage_str in rolldown_timings
    
    def is_level_timing(self) -> bool:
        """Check if current round is a standard leveling timing."""
        timings = {
            "2-1": 4,   # Level 4
            "2-5": 5,   # Level 5
            "3-2": 6,   # Level 6
            "4-1": 7,   # Level 7
            "4-2": 8,   # Level 8
            "5-1": 9,   # Level 9
        }
        return self.get_stage_string() in timings
```

## Part 3: Economy Strategy Advisor

Create `src/core/economy_advisor.py`:

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class EconomyStrategy(Enum):
    LOSS_STREAK = "loss_streak"    # Intentional losing for gold
    WIN_STREAK = "win_streak"      # Play strongest board
    STANDARD = "standard"          # Balanced approach
    FAST_8 = "fast_8"              # Rush level 8
    SLOW_ROLL = "slow_roll"        # Stay at level, roll for 3-stars
    ALL_IN = "all_in"              # Desperate rolldown


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
    """
    Provides economy advice based on game state.
    """
    
    def __init__(self):
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
        board_strength: float = 0.5  # 0-1 relative strength
    ) -> EconomyAdvice:
        """
        Get economy advice for current situation.
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
            priority=priority
        )
    
    def _determine_strategy(
        self,
        health: int,
        stage: str,
        board_strength: float,
        win_streak: int,
        loss_streak: int
    ) -> EconomyStrategy:
        """Determine best strategy for current state."""
        stage_num = int(stage.split("-")[0])
        
        if health < 30:
            return EconomyStrategy.ALL_IN
        
        if loss_streak >= 4 and health > 60:
            return EconomyStrategy.LOSS_STREAK
        
        if win_streak >= 4 and board_strength > 0.7:
            return EconomyStrategy.WIN_STREAK
        
        if stage_num <= 3 and board_strength < 0.4:
            return EconomyStrategy.SLOW_ROLL
        
        return EconomyStrategy.STANDARD
    
    def _gold_to_level_up(self, level: int, xp: int) -> int:
        """Calculate gold needed to level up once."""
        xp_needed = self.econ.calculate_xp_needed(level) - xp
        purchases = (xp_needed + 3) // 4  # 4 XP per 4 gold
        return purchases * 4
    
    def _standard_play(
        self,
        gold: int,
        level: int,
        xp: int,
        stage: str
    ) -> tuple[str, int, int]:
        """Standard economy decisions."""
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
    
    def get_rolldown_recommendation(
        self,
        gold: int,
        level: int,
        health: int,
        target_champions: list[str],
        stage: str
    ) -> dict:
        """
        Get specific rolldown recommendation.
        """
        # Calculate budget
        budget = self.rolldown.calculate_rolldown_budget(gold, health, stage)
        num_rolls = budget // 2
        
        # Determine if should rolldown now
        stage_num = int(stage.split("-")[0])
        round_num = int(stage.split("-")[1])
        
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
            "keep_gold": gold - budget
        }
```

## Part 4: Integration with GameState

Update `src/core/game_state.py`:

```python
# Add to PlayerState class

class PlayerState:
    def __init__(self, player_id: int, pool: ChampionPool):
        # ... existing code ...
        self.economy = EconomyCalculator()
        self.stage_manager = StageManager()
    
    def end_round(self, won: bool, pve_gold: int = 0) -> IncomeBreakdown:
        """
        Process end of round economy.
        """
        # Update streaks
        if won:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        
        # Create economy state
        state = EconomyState(
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            stage=self.stage_manager.get_stage_string()
        )
        
        # Calculate income
        is_pve = self.stage_manager.get_current_round_info().round_type == RoundType.PVE
        income = self.economy.calculate_round_income(
            state, won, is_pve, pve_gold
        )
        
        # Add income
        self.gold += income.total
        
        # Add passive XP
        round_info = self.stage_manager.get_current_round_info()
        self.xp += round_info.passive_xp
        
        # Check level up
        self._check_level_up()
        
        # Advance round
        self.stage_manager.advance_round()
        
        return income
    
    def _check_level_up(self) -> bool:
        """Check and process level up."""
        xp_needed = self.economy.calculate_xp_needed(self.level)
        if self.xp >= xp_needed and self.level < 10:
            self.xp -= xp_needed
            self.level += 1
            self.shop.set_level(self.level)
            return True
        return False
    
    def buy_xp(self) -> bool:
        """Spend 4 gold for 4 XP."""
        if self.gold < 4:
            return False
        self.gold -= 4
        self.xp += 4
        self._check_level_up()
        return True
    
    def get_economy_advice(self) -> EconomyAdvice:
        """Get economy advice for current state."""
        advisor = EconomyAdvisor()
        return advisor.get_advice(
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            health=self.health,
            stage=self.stage_manager.get_stage_string(),
            win_streak=self.win_streak,
            loss_streak=self.loss_streak
        )
```

## Part 5: Tests

Create `tests/test_economy.py`:

```python
import pytest
from src.core.economy import EconomyCalculator, EconomyState, RolldownCalculator
from src.core.stage_manager import StageManager, RoundType
from src.core.economy_advisor import EconomyAdvisor, EconomyStrategy

class TestEconomyCalculator:
    
    def test_interest_calculation(self):
        """10g = 1 interest, 50g = 5 interest (max)."""
        calc = EconomyCalculator()
        assert calc.calculate_interest(10) == 1
        assert calc.calculate_interest(25) == 2
        assert calc.calculate_interest(50) == 5
        assert calc.calculate_interest(100) == 5  # Max 5
    
    def test_streak_bonus(self):
        """Streak bonuses: 2-3=1, 4=2, 5+=3."""
        calc = EconomyCalculator()
        assert calc.calculate_streak_bonus(0) == 0
        assert calc.calculate_streak_bonus(2) == 1
        assert calc.calculate_streak_bonus(4) == 2
        assert calc.calculate_streak_bonus(5) == 3
        assert calc.calculate_streak_bonus(10) == 3
    
    def test_round_income(self):
        """Total income = base + interest + streak."""
        calc = EconomyCalculator()
        state = EconomyState(gold=50, win_streak=5)
        income = calc.calculate_round_income(state, won_combat=True)
        
        assert income.base_income == 5
        assert income.interest == 5
        assert income.streak_bonus == 3
        assert income.total == 13
    
    def test_gold_to_level(self):
        """Calculate gold needed to reach target level."""
        calc = EconomyCalculator()
        # Level 1 to 4: 2 + 6 + 10 = 18 XP = 20 gold (5 purchases)
        gold = calc.calculate_gold_to_level(1, 0, 4)
        assert gold == 20
    
    def test_simulate_economy(self):
        """Economy simulation over rounds."""
        calc = EconomyCalculator()
        initial = EconomyState(gold=10, level=1)
        states = calc.simulate_economy(initial, rounds=10)
        
        assert len(states) == 11  # Initial + 10 rounds
        assert states[-1].gold > states[0].gold


class TestRolldownCalculator:
    
    def test_rolldown_budget_healthy(self):
        """Healthy player keeps reserve."""
        calc = RolldownCalculator()
        budget = calc.calculate_rolldown_budget(50, health=80, stage="4-2")
        assert budget == 30  # Keep 20 reserve
    
    def test_rolldown_budget_low_hp(self):
        """Low HP player keeps minimal reserve."""
        calc = RolldownCalculator()
        budget = calc.calculate_rolldown_budget(50, health=40, stage="4-2")
        assert budget == 40  # Keep 10 reserve
    
    def test_rolldown_budget_critical(self):
        """Critical HP player rolls everything."""
        calc = RolldownCalculator()
        budget = calc.calculate_rolldown_budget(50, health=20, stage="4-2")
        assert budget == 48  # Almost everything
    
    def test_optimal_rolldown_stage(self):
        """Calculate best rolldown timing."""
        calc = RolldownCalculator()
        state = EconomyState(gold=30, level=6)
        result = calc.calculate_optimal_rolldown_stage(state, target_cost=4)
        
        assert "recommended_stage" in result
        assert "recommended_level" in result


class TestStageManager:
    
    def test_stage_string(self):
        """Stage string format."""
        manager = StageManager()
        assert manager.get_stage_string() == "1-1"
    
    def test_advance_round(self):
        """Advancing rounds and stages."""
        manager = StageManager()
        
        # Advance through stage 1
        for _ in range(4):
            manager.advance_round()
        
        assert manager.current_stage == 2
        assert manager.current_round == 1
    
    def test_round_types(self):
        """Correct round types."""
        manager = StageManager()
        info = manager.get_current_round_info()
        assert info.round_type == RoundType.PVE  # Stage 1 is PvE
    
    def test_rounds_until(self):
        """Calculate rounds until target."""
        manager = StageManager()
        rounds = manager.get_rounds_until("2-1")
        assert rounds == 4  # 4 rounds in stage 1


class TestEconomyAdvisor:
    
    def test_critical_hp_advice(self):
        """Critical HP recommends all-in."""
        advisor = EconomyAdvisor()
        advice = advisor.get_advice(
            gold=50, level=7, xp=0, health=15, stage="4-2"
        )
        assert advice.strategy == EconomyStrategy.ALL_IN
        assert advice.action == "all_in"
    
    def test_loss_streak_advice(self):
        """Long loss streak recommends saving."""
        advisor = EconomyAdvisor()
        advice = advisor.get_advice(
            gold=30, level=5, xp=0, health=80, stage="2-5",
            loss_streak=5
        )
        assert advice.strategy == EconomyStrategy.LOSS_STREAK
        assert advice.action == "save"
    
    def test_standard_level_timing(self):
        """Standard timing recommends leveling."""
        advisor = EconomyAdvisor()
        advice = advisor.get_advice(
            gold=20, level=5, xp=0, health=70, stage="3-2"
        )
        # At 3-2 with level 5, should recommend leveling to 6
        assert advice.action in ["level", "save"]
```

## Expected Output

```
src/core/
├── economy.py           # Economy calculations
├── stage_manager.py     # Stage/round management
└── economy_advisor.py   # Economy recommendations

tests/
└── test_economy.py      # Economy tests
```

## Verification

```python
# Quick test
from src.core.economy import EconomyCalculator, EconomyState
from src.core.economy_advisor import EconomyAdvisor

# Test interest
calc = EconomyCalculator()
assert calc.calculate_interest(50) == 5

# Test income
state = EconomyState(gold=50, win_streak=3)
income = calc.calculate_round_income(state, won_combat=True)
print(f"Income: {income.total} (base={income.base_income}, interest={income.interest}, streak={income.streak_bonus})")

# Test advisor
advisor = EconomyAdvisor()
advice = advisor.get_advice(
    gold=40, level=7, xp=10, health=60, stage="4-1"
)
print(f"Strategy: {advice.strategy}, Action: {advice.action}")
```

## Priority

1. **Must**: EconomyCalculator with interest/streak/income
2. **Must**: StageManager with round progression
3. **Must**: Level-up cost calculations
4. **Should**: RolldownCalculator with budget
5. **Should**: EconomyAdvisor with strategies
6. **Must**: Tests for core functionality
