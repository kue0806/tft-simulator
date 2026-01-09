"""Unit Unlock Manager for TFT Set 16.

Manages the unlock conditions and state for unlockable champions.
Uses structured unlock_type and unlock_params from JSON data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

from src.data.models.champion import Champion
from src.data.loaders import get_unlockable_champions

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from src.core.champion_pool import ChampionPool


@dataclass
class PlayerUnlockState:
    """Tracks unlock progress for a player."""
    unlocked_champions: set[str] = field(default_factory=set)

    # Resource tracking
    souls_collected: int = 0
    sunshards_collected: int = 0
    serpents_spent: int = 0

    # Combat tracking
    trait_combat_counts: dict[str, int] = field(default_factory=dict)
    reroll_count: int = 0
    gold_drops: dict[str, int] = field(default_factory=dict)  # champion_id -> drops

    # Win/loss streak tracking (with unit)
    current_streak: int = 0  # positive = wins, negative = losses
    streak_with_unit: dict[str, tuple[int, int]] = field(default_factory=dict)  # champion_id -> (win_streak, loss_streak)
    alternate_count: dict[str, int] = field(default_factory=dict)  # champion_id -> alternate count
    last_result_with_unit: dict[str, bool] = field(default_factory=dict)  # champion_id -> last_was_win

    # HP tracking
    hp_lost: int = 0

    # Sold units tracking
    sold_units: dict[str, int] = field(default_factory=dict)  # champion_id:star_level -> count

    # Stage tracking
    current_stage: str = "1-1"

    # Item break tracking
    item_broke_on_unit: dict[str, bool] = field(default_factory=dict)  # champion_id -> broke


class UnlockConditionEvaluator(ABC):
    """Base class for condition evaluators."""

    @abstractmethod
    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        """Check if condition is met."""
        pass


class RerollEvaluator(UnlockConditionEvaluator):
    """Evaluates reroll count before a stage."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        required = params.get("reroll_count", 0)
        before_stage = params.get("before_stage", "99-99")

        # Check if current stage is before required stage
        if not self._is_before_stage(unlock_state.current_stage, before_stage):
            return False

        return unlock_state.reroll_count >= required

    def _is_before_stage(self, current: str, target: str) -> bool:
        """Check if current stage is before target stage."""
        try:
            c_stage, c_round = map(int, current.split("-"))
            t_stage, t_round = map(int, target.split("-"))
            return (c_stage, c_round) < (t_stage, t_round)
        except ValueError:
            return False


class UnitItemsEvaluator(UnlockConditionEvaluator):
    """Evaluates unit with X items equipped."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        item_count = params.get("item_count", 0)

        if not champion_id:
            return False

        for pos, unit in player.units.board.items():
            if unit.champion.id == champion_id:
                if len(unit.items) >= item_count:
                    return True

        return False


class TraitUnitsEvaluator(UnlockConditionEvaluator):
    """Evaluates unique units with trait(s)."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        # Check level requirement
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        # Get trait(s) to check
        trait_id = params.get("trait_id")
        traits = params.get("traits", [])
        if trait_id:
            traits = [trait_id]

        if not traits:
            return False

        required = params.get("unit_count", 0)
        unique_ids = set()

        for pos, unit in player.units.board.items():
            for trait in traits:
                if trait in unit.champion.traits:
                    unique_ids.add(unit.champion.id)
                    break

        return len(unique_ids) >= required


class TraitUnitItemsEvaluator(UnlockConditionEvaluator):
    """Evaluates trait unit with X items equipped."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        # Check level requirement
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        traits = params.get("traits", [])
        item_count = params.get("item_count", 0)

        if not traits:
            return False

        for pos, unit in player.units.board.items():
            for trait in traits:
                if trait in unit.champion.traits:
                    if len(unit.items) >= item_count:
                        return True

        return False


class UnitStarItemsEvaluator(UnlockConditionEvaluator):
    """Evaluates unit at star level with X items."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        # Check level requirement
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        champion_id = params.get("champion_id")
        star_level = params.get("star_level", 1)
        item_count = params.get("item_count", 0)

        if not champion_id:
            return False

        for pos, unit in player.units.board.items():
            if unit.champion.id == champion_id:
                if unit.star_level >= star_level and len(unit.items) >= item_count:
                    return True

        return False


class GoldDropEvaluator(UnlockConditionEvaluator):
    """Evaluates champion dropping gold."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        drop_count = params.get("drop_count", 1)

        if not champion_id:
            return False

        return unlock_state.gold_drops.get(champion_id, 0) >= drop_count


class SoulsEvaluator(UnlockConditionEvaluator):
    """Evaluates souls collected."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        required = params.get("soul_count", 0)
        return unlock_state.souls_collected >= required


class TraitStarLevelsEvaluator(UnlockConditionEvaluator):
    """Evaluates total star levels of trait units."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        # Check level requirement
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        traits = params.get("traits", [])
        required = params.get("star_levels", 0)

        if not traits:
            return False

        total_stars = 0
        for pos, unit in player.units.board.items():
            for trait in traits:
                if trait in unit.champion.traits:
                    total_stars += unit.star_level
                    break

        return total_stars >= required


class LossStreakWithUnitEvaluator(UnlockConditionEvaluator):
    """Evaluates loss streak with unit fielded."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        streak_counts = params.get("streak_counts", [])

        if not champion_id or not streak_counts:
            return False

        win_streak, loss_streak = unlock_state.streak_with_unit.get(champion_id, (0, 0))

        # Check if loss streak matches any required count
        return loss_streak in streak_counts


class WinStreakWithUnitEvaluator(UnlockConditionEvaluator):
    """Evaluates win streak with unit fielded."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        streak_counts = params.get("streak_counts", [])

        if not champion_id or not streak_counts:
            return False

        win_streak, loss_streak = unlock_state.streak_with_unit.get(champion_id, (0, 0))

        # Check if win streak matches any required count
        return win_streak in streak_counts


class TraitCombatCountEvaluator(UnlockConditionEvaluator):
    """Evaluates trait active for X combats."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        trait_id = params.get("trait_id")
        combat_count = params.get("combat_count", 0)

        if not trait_id:
            return False

        return unlock_state.trait_combat_counts.get(trait_id, 0) >= combat_count


class TraitUnitsAndHpLostEvaluator(UnlockConditionEvaluator):
    """Evaluates trait units fielded AND HP lost."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        traits = params.get("traits", [])
        unit_count = params.get("unit_count", 0)
        hp_lost = params.get("hp_lost", 0)

        if not traits:
            return False

        # Check HP lost
        if unlock_state.hp_lost < hp_lost:
            return False

        # Check unique trait units
        unique_ids = set()
        for pos, unit in player.units.board.items():
            for trait in traits:
                if trait in unit.champion.traits:
                    unique_ids.add(unit.champion.id)
                    break

        return len(unique_ids) >= unit_count


class ItemOnNonRoleEvaluator(UnlockConditionEvaluator):
    """Evaluates item equipped on unit without a specific role."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        item_id = params.get("item_id")
        excluded_role = params.get("excluded_role")

        if not item_id or not excluded_role:
            return False

        for pos, unit in player.units.board.items():
            # Check if unit doesn't have excluded role
            unit_role = getattr(unit.champion, 'role', None)
            if unit_role != excluded_role:
                # Check if unit has the item
                for item in unit.items:
                    if item.id == item_id:
                        return True

        return False


class DuplicateItemEvaluator(UnlockConditionEvaluator):
    """Evaluates multiple copies of same item on one unit."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        item_id = params.get("item_id")
        item_count = params.get("item_count", 2)

        if not item_id:
            return False

        for pos, unit in player.units.board.items():
            count = sum(1 for item in unit.items if item.id == item_id)
            if count >= item_count:
                return True

        return False


class MultipleUnitStarsEvaluator(UnlockConditionEvaluator):
    """Evaluates multiple units at specific star levels."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        requirements = params.get("requirements", [])

        if not requirements:
            return False

        for req in requirements:
            champion_id = req.get("champion_id")
            star_level = req.get("star_level", 1)

            found = False
            for pos, unit in player.units.board.items():
                if unit.champion.id == champion_id and unit.star_level >= star_level:
                    found = True
                    break

            if not found:
                return False

        return True


class UnitStarEvaluator(UnlockConditionEvaluator):
    """Evaluates specific unit at star level."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        star_level = params.get("star_level", 1)

        if not champion_id:
            return False

        for pos, unit in player.units.board.items():
            if unit.champion.id == champion_id and unit.star_level >= star_level:
                return True

        return False


class UnitStarCountEvaluator(UnlockConditionEvaluator):
    """Evaluates count of units at specific star level."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        star_level = params.get("star_level", 1)
        unit_count = params.get("unit_count", 1)

        if not champion_id:
            return False

        count = 0
        for pos, unit in player.units.board.items():
            if unit.champion.id == champion_id and unit.star_level >= star_level:
                count += 1

        return count >= unit_count


class StatThresholdEvaluator(UnlockConditionEvaluator):
    """Evaluates unit stat threshold."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        stat = params.get("stat")
        threshold = params.get("threshold", 0)

        if not stat:
            return False

        for pos, unit in player.units.board.items():
            # Get stat value from unit
            if stat == "health":
                value = unit.get_total_stats().get("max_health", 0)
            elif stat == "omnivamp":
                value = unit.get_total_stats().get("omnivamp", 0)
            else:
                value = unit.get_total_stats().get(stat, 0)

            if value >= threshold:
                return True

        return False


class FrontRowCountEvaluator(UnlockConditionEvaluator):
    """Evaluates unit count in front rows."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        max_units = params.get("max_units", 1)

        # Front rows are rows 0 and 1 (or 3 and 4 depending on team)
        front_row_count = 0
        for pos, unit in player.units.board.items():
            # Assuming positions are (x, y) tuples and front rows are y=0,1
            if isinstance(pos, tuple) and len(pos) >= 2:
                if pos[1] <= 1:  # Front two rows
                    front_row_count += 1
            elif hasattr(pos, 'row') and pos.row <= 1:
                front_row_count += 1

        return front_row_count <= max_units


class SerpentsSpentEvaluator(UnlockConditionEvaluator):
    """Evaluates Bilgewater Silver Serpents spent."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        required = params.get("serpent_count", 0)
        return unlock_state.serpents_spent >= required


class SunshardsEvaluator(UnlockConditionEvaluator):
    """Evaluates Ixtal Sunshards collected."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        required = params.get("sunshard_count", 0)
        return unlock_state.sunshards_collected >= required


class RegionCountEvaluator(UnlockConditionEvaluator):
    """Evaluates number of different region traits active."""

    # Region traits in Set 16
    REGION_TRAITS = {
        "bilgewater", "demacia", "freljord", "ionia", "ixtal",
        "noxus", "piltover", "shadow_isles", "shurima", "targon",
        "void", "zaun"
    }

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        min_level = params.get("min_level", 0)
        if player.level < min_level:
            return False

        required = params.get("region_count", 0)

        # Count unique regions on board
        regions = set()
        for pos, unit in player.units.board.items():
            for trait in unit.champion.traits:
                if trait in self.REGION_TRAITS:
                    regions.add(trait)

        return len(regions) >= required


class AlternateResultsWithUnitEvaluator(UnlockConditionEvaluator):
    """Evaluates alternating win/loss pattern with unit."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        alternate_counts = params.get("alternate_counts", [])

        if not champion_id or not alternate_counts:
            return False

        count = unlock_state.alternate_count.get(champion_id, 0)
        return count in alternate_counts


class SellUnitsEvaluator(UnlockConditionEvaluator):
    """Evaluates selling specific units."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        requirements = params.get("requirements", [])

        if not requirements:
            return False

        for req in requirements:
            champion_id = req.get("champion_id")
            star_level = req.get("star_level", 1)

            key = f"{champion_id}:{star_level}"
            if unlock_state.sold_units.get(key, 0) < 1:
                return False

        return True


class UnitItemBreakEvaluator(UnlockConditionEvaluator):
    """Evaluates item breaking on a unit."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        champion_id = params.get("champion_id")
        star_level = params.get("star_level", 1)

        if not champion_id:
            return False

        # Check if item broke on this unit
        if not unlock_state.item_broke_on_unit.get(champion_id, False):
            return False

        # Check if unit is at required star level on board
        for pos, unit in player.units.board.items():
            if unit.champion.id == champion_id and unit.star_level >= star_level:
                return True

        return False


class AugmentAndUnitEvaluator(UnlockConditionEvaluator):
    """Evaluates having augment and unit for X combats."""

    def evaluate(
        self,
        player: "PlayerState",
        unlock_state: PlayerUnlockState,
        params: dict[str, Any],
    ) -> bool:
        # This is a complex condition that would need augment tracking
        # For now, return False as augment system isn't implemented yet
        return False


class UnlockManager:
    """
    Manages champion unlocking for all players.

    Tracks unlock conditions and state, adding champions to pool when unlocked.
    Uses structured unlock_type and unlock_params from JSON data.
    """

    # Evaluators for each unlock type
    EVALUATORS: dict[str, UnlockConditionEvaluator] = {
        "reroll": RerollEvaluator(),
        "unit_items": UnitItemsEvaluator(),
        "trait_units": TraitUnitsEvaluator(),
        "trait_unit_items": TraitUnitItemsEvaluator(),
        "unit_star_items": UnitStarItemsEvaluator(),
        "gold_drop": GoldDropEvaluator(),
        "souls": SoulsEvaluator(),
        "trait_star_levels": TraitStarLevelsEvaluator(),
        "loss_streak_with_unit": LossStreakWithUnitEvaluator(),
        "win_streak_with_unit": WinStreakWithUnitEvaluator(),
        "trait_combat_count": TraitCombatCountEvaluator(),
        "trait_units_and_hp_lost": TraitUnitsAndHpLostEvaluator(),
        "item_on_non_role": ItemOnNonRoleEvaluator(),
        "duplicate_item": DuplicateItemEvaluator(),
        "multiple_unit_stars": MultipleUnitStarsEvaluator(),
        "unit_star": UnitStarEvaluator(),
        "unit_star_count": UnitStarCountEvaluator(),
        "stat_threshold": StatThresholdEvaluator(),
        "front_row_count": FrontRowCountEvaluator(),
        "serpents_spent": SerpentsSpentEvaluator(),
        "sunshards": SunshardsEvaluator(),
        "region_count": RegionCountEvaluator(),
        "alternate_results_with_unit": AlternateResultsWithUnitEvaluator(),
        "sell_units": SellUnitsEvaluator(),
        "unit_item_break": UnitItemBreakEvaluator(),
        "augment_and_unit": AugmentAndUnitEvaluator(),
    }

    def __init__(self, pool: "ChampionPool"):
        """
        Initialize the unlock manager.

        Args:
            pool: The shared champion pool.
        """
        self.pool = pool
        self.player_states: dict[int, PlayerUnlockState] = {}
        self.unlockable_champions: dict[str, Champion] = {}
        self.champion_unlock_types: dict[str, str] = {}
        self.champion_unlock_params: dict[str, dict[str, Any]] = {}

        # Track newly unlocked champions per player for shop guarantee
        self.pending_shop_unlocks: dict[int, list[str]] = {}

        self._load_unlockables()

    def _load_unlockables(self) -> None:
        """Load all unlockable champions and their conditions."""
        unlockables = get_unlockable_champions()

        for champion in unlockables:
            self.unlockable_champions[champion.id] = champion

            # Store unlock type and params from champion data
            self.champion_unlock_types[champion.id] = getattr(
                champion, 'unlock_type', None
            ) or ""
            self.champion_unlock_params[champion.id] = getattr(
                champion, 'unlock_params', None
            ) or {}

    def get_player_state(self, player_id: int) -> PlayerUnlockState:
        """Get or create unlock state for a player."""
        if player_id not in self.player_states:
            self.player_states[player_id] = PlayerUnlockState()
        return self.player_states[player_id]

    def check_unlocks(self, player: "PlayerState") -> list[Champion]:
        """
        Check all unlock conditions for a player.

        Args:
            player: The player to check.

        Returns:
            List of newly unlocked champions.
        """
        unlock_state = self.get_player_state(player.player_id)
        newly_unlocked = []

        for champion_id, champion in self.unlockable_champions.items():
            # Skip if already unlocked
            if champion_id in unlock_state.unlocked_champions:
                continue

            unlock_type = self.champion_unlock_types.get(champion_id, "")
            unlock_params = self.champion_unlock_params.get(champion_id, {})

            if not unlock_type:
                continue

            # Get evaluator for this unlock type
            evaluator = self.EVALUATORS.get(unlock_type)
            if not evaluator:
                continue

            # Evaluate condition
            if evaluator.evaluate(player, unlock_state, unlock_params):
                self._unlock_champion(player.player_id, champion)
                newly_unlocked.append(champion)

        return newly_unlocked

    def _unlock_champion(self, player_id: int, champion: Champion) -> None:
        """Mark a champion as unlocked and add to pool."""
        unlock_state = self.get_player_state(player_id)
        unlock_state.unlocked_champions.add(champion.id)

        # Add to shared pool
        self.pool.add_unlockable_to_pool(champion)

        # Add to pending shop unlocks for guaranteed appearance
        if player_id not in self.pending_shop_unlocks:
            self.pending_shop_unlocks[player_id] = []
        self.pending_shop_unlocks[player_id].append(champion.id)

    def get_pending_shop_unlock(self, player_id: int) -> Optional[str]:
        """Get and remove a pending unlock for shop guarantee."""
        if player_id in self.pending_shop_unlocks:
            pending = self.pending_shop_unlocks[player_id]
            if pending:
                return pending.pop(0)
        return None

    def add_souls(self, player_id: int, count: int) -> None:
        """Add souls for Shadow Isles tracking."""
        state = self.get_player_state(player_id)
        state.souls_collected += count

    def add_sunshards(self, player_id: int, count: int) -> None:
        """Add sunshards for Ixtal tracking."""
        state = self.get_player_state(player_id)
        state.sunshards_collected += count

    def spend_serpents(self, player_id: int, count: int) -> None:
        """Record serpents spent for Bilgewater tracking."""
        state = self.get_player_state(player_id)
        state.serpents_spent += count

    def record_trait_combat(self, player_id: int, trait_id: str) -> None:
        """Record that a trait was active during combat."""
        state = self.get_player_state(player_id)
        state.trait_combat_counts[trait_id] = (
            state.trait_combat_counts.get(trait_id, 0) + 1
        )

    def record_reroll(self, player_id: int) -> None:
        """Record a shop reroll."""
        state = self.get_player_state(player_id)
        state.reroll_count += 1

    def record_gold_drop(self, player_id: int, champion_id: str) -> None:
        """Record a champion dropping gold."""
        state = self.get_player_state(player_id)
        state.gold_drops[champion_id] = state.gold_drops.get(champion_id, 0) + 1

    def record_hp_lost(self, player_id: int, amount: int) -> None:
        """Record HP lost by player."""
        state = self.get_player_state(player_id)
        state.hp_lost += amount

    def record_combat_result(
        self,
        player_id: int,
        won: bool,
        units_on_board: list[str],
    ) -> None:
        """Record combat result for streak tracking."""
        state = self.get_player_state(player_id)

        for champion_id in units_on_board:
            # Get current streaks
            win_streak, loss_streak = state.streak_with_unit.get(champion_id, (0, 0))

            if won:
                win_streak += 1
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0

            state.streak_with_unit[champion_id] = (win_streak, loss_streak)

            # Track alternating results
            last_was_win = state.last_result_with_unit.get(champion_id)
            if last_was_win is not None and last_was_win != won:
                state.alternate_count[champion_id] = (
                    state.alternate_count.get(champion_id, 0) + 1
                )
            state.last_result_with_unit[champion_id] = won

    def record_unit_sold(
        self,
        player_id: int,
        champion_id: str,
        star_level: int,
    ) -> None:
        """Record selling a unit."""
        state = self.get_player_state(player_id)
        key = f"{champion_id}:{star_level}"
        state.sold_units[key] = state.sold_units.get(key, 0) + 1

    def record_item_break(self, player_id: int, champion_id: str) -> None:
        """Record an item breaking on a unit."""
        state = self.get_player_state(player_id)
        state.item_broke_on_unit[champion_id] = True

    def update_stage(self, player_id: int, stage: str) -> None:
        """Update current stage for a player."""
        state = self.get_player_state(player_id)
        state.current_stage = stage

    def is_unlocked(self, player_id: int, champion_id: str) -> bool:
        """Check if a champion is unlocked for a player."""
        state = self.get_player_state(player_id)
        return champion_id in state.unlocked_champions

    def get_unlocked_champions(self, player_id: int) -> list[Champion]:
        """Get all unlocked champions for a player."""
        state = self.get_player_state(player_id)
        return [
            self.unlockable_champions[cid]
            for cid in state.unlocked_champions
            if cid in self.unlockable_champions
        ]

    def get_unlock_progress(
        self,
        player: "PlayerState",
        champion_id: str,
    ) -> dict[str, Any]:
        """
        Get progress toward unlocking a specific champion.

        Args:
            player: The player to check.
            champion_id: The champion to check progress for.

        Returns:
            Dictionary with condition and current progress.
        """
        if champion_id not in self.unlockable_champions:
            return {"error": "Not an unlockable champion"}

        champion = self.unlockable_champions[champion_id]
        unlock_type = self.champion_unlock_types.get(champion_id, "")
        unlock_params = self.champion_unlock_params.get(champion_id, {})
        unlock_state = self.get_player_state(player.player_id)

        if champion_id in unlock_state.unlocked_champions:
            return {
                "champion": champion.name,
                "unlocked": True,
                "condition": champion.unlock_condition,
            }

        progress = {
            "champion": champion.name,
            "unlocked": False,
            "condition": champion.unlock_condition,
            "unlock_type": unlock_type,
        }

        # Add type-specific progress info
        if unlock_type == "souls":
            progress["souls_required"] = unlock_params.get("soul_count", 0)
            progress["souls_current"] = unlock_state.souls_collected
        elif unlock_type == "sunshards":
            progress["sunshards_required"] = unlock_params.get("sunshard_count", 0)
            progress["sunshards_current"] = unlock_state.sunshards_collected
        elif unlock_type == "serpents_spent":
            progress["serpents_required"] = unlock_params.get("serpent_count", 0)
            progress["serpents_current"] = unlock_state.serpents_spent
        elif unlock_type == "trait_combat_count":
            trait = unlock_params.get("trait_id", "")
            progress["combats_required"] = unlock_params.get("combat_count", 0)
            progress["combats_current"] = unlock_state.trait_combat_counts.get(trait, 0)
        elif unlock_type == "reroll":
            progress["rerolls_required"] = unlock_params.get("reroll_count", 0)
            progress["rerolls_current"] = unlock_state.reroll_count
            progress["before_stage"] = unlock_params.get("before_stage", "")

        # Add level requirement if present
        min_level = unlock_params.get("min_level", 0)
        if min_level > 0:
            progress["level_required"] = min_level
            progress["current_level"] = player.level

        return progress

    def reset_player(self, player_id: int) -> None:
        """Reset unlock state for a player (new game)."""
        self.player_states[player_id] = PlayerUnlockState()
        self.pending_shop_unlocks[player_id] = []

    def get_all_unlockables(self) -> list[Champion]:
        """Get all unlockable champions."""
        return list(self.unlockable_champions.values())


# Singleton instance
_unlock_manager: Optional[UnlockManager] = None


def get_unlock_manager(pool: Optional["ChampionPool"] = None) -> UnlockManager:
    """Get or create the unlock manager singleton."""
    global _unlock_manager
    if _unlock_manager is None:
        if pool is None:
            raise ValueError("Pool required for first initialization")
        _unlock_manager = UnlockManager(pool)
    return _unlock_manager


def reset_unlock_manager() -> None:
    """Reset the unlock manager (for testing)."""
    global _unlock_manager
    _unlock_manager = None
