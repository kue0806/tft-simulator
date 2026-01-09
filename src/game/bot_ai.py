"""AI Bot logic for TFT Set 16.

Simple AI strategies for bot players.
"""

import random
from enum import Enum
from typing import Optional
from src.core.game_state import PlayerState


class BotDifficulty(Enum):
    """Bot difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BotStrategy(Enum):
    """Bot playstyle strategies."""
    AGGRESSIVE = "aggressive"  # Fast level, roll early
    ECONOMY = "economy"        # Slow roll, max interest
    STANDARD = "standard"      # Balanced approach


class BotAI:
    """AI controller for a bot player."""

    def __init__(
        self,
        player: PlayerState,
        difficulty: BotDifficulty = BotDifficulty.MEDIUM,
        strategy: Optional[BotStrategy] = None,
    ):
        """
        Initialize bot AI.

        Args:
            player: The player state this bot controls.
            difficulty: Bot difficulty level.
            strategy: Playstyle strategy (random if None).
        """
        self.player = player
        self.difficulty = difficulty
        self.strategy = strategy or random.choice(list(BotStrategy))

        # Track what traits bot is trying to build
        self.target_traits: list[str] = []
        self.priority_champions: list[str] = []

    def take_turn(self, stage: str) -> list[str]:
        """
        Execute bot's turn during planning phase.

        Args:
            stage: Current stage string (e.g., "2-3").

        Returns:
            List of actions taken (for logging).
        """
        actions = []

        # 1. Decide on leveling
        if self._should_buy_xp(stage):
            while self.player.can_afford_xp() and self._should_buy_xp(stage):
                self.player.buy_xp()
                actions.append(f"Bought XP (now level {self.player.level})")

        # 2. Decide on rolling
        rolls = self._get_roll_count(stage)
        for _ in range(rolls):
            if self.player.can_afford_reroll():
                self.player.reroll()
                actions.append("Rerolled shop")

                # Try to buy after each roll
                buy_actions = self._try_buy_champions()
                actions.extend(buy_actions)

        # 3. Try to buy champions from current shop
        buy_actions = self._try_buy_champions()
        actions.extend(buy_actions)

        # 4. Place units on board
        place_actions = self._place_units()
        actions.extend(place_actions)

        return actions

    def _should_buy_xp(self, stage: str) -> bool:
        """Determine if bot should buy XP."""
        stage_num = int(stage.split("-")[0])
        round_num = int(stage.split("-")[1])

        # Don't level past 9
        if self.player.level >= 9:
            return False

        if self.strategy == BotStrategy.AGGRESSIVE:
            # Level aggressively
            if stage == "2-1" and self.player.level < 4:
                return self.player.gold >= 8
            if stage == "2-5" and self.player.level < 5:
                return self.player.gold >= 10
            if stage == "3-2" and self.player.level < 6:
                return self.player.gold >= 20
            if stage == "4-1" and self.player.level < 7:
                return self.player.gold >= 30
            if stage == "4-2" and self.player.level < 8:
                return self.player.gold >= 40
            return False

        elif self.strategy == BotStrategy.ECONOMY:
            # Only level when above 50 gold
            if self.player.gold < 50:
                return False
            # Standard level timings but with econ
            if stage_num >= 3 and self.player.level < 6:
                return True
            if stage_num >= 4 and self.player.level < 7:
                return True
            if stage_num >= 5 and self.player.level < 8:
                return True
            return False

        else:  # STANDARD
            # Standard level timings
            if stage == "2-1" and self.player.level < 4:
                return self.player.gold >= 4
            if stage == "2-5" and self.player.level < 5:
                return self.player.gold >= 8
            if stage == "3-2" and self.player.level < 6:
                return self.player.gold >= 12
            if stage == "4-1" and self.player.level < 7:
                return self.player.gold >= 20
            if stage == "4-2" and self.player.level < 8:
                return self.player.gold >= 30
            if stage_num >= 5 and self.player.level < 9:
                return self.player.gold >= 50
            return False

    def _get_roll_count(self, stage: str) -> int:
        """Determine how many times to roll."""
        stage_num = int(stage.split("-")[0])

        if self.strategy == BotStrategy.ECONOMY:
            # Rarely roll, save gold
            if self.player.gold > 50 and self.player.health < 50:
                return random.randint(1, 3)
            return 0

        elif self.strategy == BotStrategy.AGGRESSIVE:
            # Roll more often
            if self.player.gold > 20:
                return random.randint(2, 5)
            elif self.player.gold > 10:
                return random.randint(1, 3)
            return 0

        else:  # STANDARD
            # Roll at key stages or when low HP
            if self.player.health < 40:
                return min(self.player.gold // 2, 10)
            if stage_num >= 4 and self.player.gold > 30:
                return random.randint(2, 5)
            if self.player.gold > 50:
                return random.randint(1, 3)
            return 0

    def _try_buy_champions(self) -> list[str]:
        """Try to buy champions from shop."""
        actions = []

        # Check each shop slot
        for slot in range(5):
            champion = self.player.shop.get_slot(slot)
            if champion is None:
                continue

            # Decide if we want this champion
            if self._should_buy_champion(champion):
                if self.player.units.has_bench_space():
                    result = self.player.buy_champion(slot)
                    if result:
                        actions.append(f"Bought {champion.name}")

        return actions

    def _should_buy_champion(self, champion) -> bool:
        """Determine if bot should buy a specific champion."""
        # Always buy if we can upgrade
        owned_count = self._count_owned_champion(champion.id)
        if owned_count >= 2:
            return True  # Can make 2-star
        if owned_count == 1 and self.difficulty != BotDifficulty.EASY:
            return True  # Working towards 2-star

        # Buy based on traits we're building
        if any(trait in self.target_traits for trait in champion.traits):
            return True

        # Buy based on cost and gold
        if self.difficulty == BotDifficulty.EASY:
            # Easy bots buy randomly
            return random.random() < 0.3
        elif self.difficulty == BotDifficulty.MEDIUM:
            # Medium bots are more selective
            if champion.cost <= 2:
                return random.random() < 0.5
            return random.random() < 0.3
        else:  # HARD
            # Hard bots are very selective
            if champion.cost <= self.player.level - 1:
                return random.random() < 0.4
            return False

    def _count_owned_champion(self, champion_id: str) -> int:
        """Count how many copies of a champion the bot owns."""
        count = 0
        # Check bench
        for unit in self.player.units.get_bench_units():
            if unit and unit.champion.id == champion_id:
                count += (1 if unit.star_level == 1 else 3)
        # Check board
        for unit in self.player.units.get_board_units():
            if unit.champion.id == champion_id:
                count += (1 if unit.star_level == 1 else 3)
        return count

    def _place_units(self) -> list[str]:
        """Place units from bench to board."""
        actions = []

        # Get board limit
        board_limit = self.player.get_board_size_limit()
        current_board = self.player.units.get_board_count()

        # Get bench units sorted by star level (prioritize 2-stars)
        bench_units = [u for u in self.player.units.get_bench_units() if u is not None]
        bench_units.sort(key=lambda u: (-u.star_level, -u.champion.cost))

        for unit in bench_units:
            if current_board >= board_limit:
                break

            # Find empty board position (simple: first available)
            placed = False
            for row in range(4):
                if placed:
                    break
                for col in range(7):
                    if (col, row) not in self.player.units.board:
                        # Move from bench to board
                        bench_idx = self.player.units.bench.index(unit)
                        self.player.units.bench[bench_idx] = None
                        self.player.units.board[(col, row)] = unit
                        unit.position = (col, row)
                        actions.append(f"Placed {unit.champion.name} at ({col}, {row})")
                        current_board += 1
                        placed = True
                        break

        return actions

    def choose_carousel_pick(self, carousel: list) -> int:
        """
        Choose which carousel champion to pick.

        Args:
            carousel: List of available CarouselChampion objects.

        Returns:
            Index of champion to pick.
        """
        available = [(i, c) for i, c in enumerate(carousel) if c.is_available]
        if not available:
            return 0

        if self.difficulty == BotDifficulty.EASY:
            # Random pick
            return random.choice(available)[0]

        # Prefer items we need or champions we're building
        # For now, just pick the highest cost champion
        available.sort(key=lambda x: -x[1].cost)
        return available[0][0]

    def choose_augment(self, choices) -> int:
        """
        Choose which augment to pick.

        Args:
            choices: AugmentChoice object with options.

        Returns:
            Index of augment to pick.
        """
        if not choices.options:
            return 0

        if self.difficulty == BotDifficulty.EASY:
            return random.randint(0, len(choices.options) - 1)

        # Prefer economy augments when low on gold
        if self.player.gold < 20:
            for i, aug in enumerate(choices.options):
                if aug.category.value == "economy":
                    return i

        # Prefer combat augments otherwise
        for i, aug in enumerate(choices.options):
            if aug.category.value == "combat":
                return i

        return 0

    def update_target_traits(self) -> None:
        """Update target traits based on current board."""
        traits = self.player.get_active_traits()
        # Target the traits we already have the most of
        sorted_traits = sorted(traits.items(), key=lambda x: -x[1])
        self.target_traits = [t[0] for t in sorted_traits[:3]]
