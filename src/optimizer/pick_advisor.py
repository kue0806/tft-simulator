"""Shop Purchase Recommendation System.

Recommends which champions to buy from the shop based on:
- Upgrade potential (2-star/3-star)
- Synergy activation/upgrades
- Core composition units
- Economy considerations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum, auto

from src.core.synergy_calculator import SynergyCalculator

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from src.core.shop import Shop
    from src.data.models.champion import Champion


class PickReason(Enum):
    """Reasons for recommending a champion purchase."""

    UPGRADE_2STAR = auto()  # Can upgrade to 2-star
    UPGRADE_3STAR = auto()  # Can upgrade to 3-star
    SYNERGY_ACTIVATE = auto()  # Activates a new synergy
    SYNERGY_UPGRADE = auto()  # Upgrades existing synergy tier
    CORE_CARRY = auto()  # Core carry unit for target composition
    STRONG_UNIT = auto()  # Strong unit for tempo
    ECONOMY_PAIR = auto()  # Holds pair for future upgrade
    PIVOT_OPTION = auto()  # Option for pivoting to different comp


@dataclass
class PickRecommendation:
    """A single purchase recommendation."""

    champion_id: str
    champion_name: str
    shop_index: int  # Shop slot (0-4)
    score: float  # Recommendation score (higher is better)
    reasons: List[PickReason]
    synergy_delta: Dict[str, int]  # Synergy changes {trait_id: count_change}
    cost: int

    # Upgrade information
    copies_owned: int  # Current copies held
    copies_needed: int  # Copies needed for next upgrade


@dataclass
class PickAdvice:
    """Complete purchase advice for current shop."""

    recommendations: List[PickRecommendation]
    should_refresh: bool  # Whether to reroll
    refresh_reason: Optional[str]
    gold_to_save: int  # Gold to maintain for interest


class PickAdvisor:
    """
    Shop purchase recommendation system.

    Analyzes current shop and provides purchase recommendations based on
    player's current board, bench, and target composition.

    Usage:
        advisor = PickAdvisor()
        advice = advisor.analyze(player_state, shop)
        for rec in advice.recommendations:
            print(f"{rec.champion_name}: {rec.score:.1f} - {rec.reasons}")
    """

    def __init__(self, synergy_calculator: Optional[SynergyCalculator] = None):
        """
        Initialize pick advisor.

        Args:
            synergy_calculator: SynergyCalculator instance for synergy analysis.
        """
        self.synergy_calc = synergy_calculator or SynergyCalculator()

        # Scoring weights for different pick reasons
        self.weights = {
            PickReason.UPGRADE_3STAR: 100,
            PickReason.UPGRADE_2STAR: 50,
            PickReason.CORE_CARRY: 40,
            PickReason.SYNERGY_ACTIVATE: 30,
            PickReason.SYNERGY_UPGRADE: 25,
            PickReason.STRONG_UNIT: 15,
            PickReason.ECONOMY_PAIR: 10,
            PickReason.PIVOT_OPTION: 5,
        }

    def analyze(
        self,
        player: "PlayerState",
        target_comp: Optional[List[str]] = None,
    ) -> PickAdvice:
        """
        Analyze current shop and provide recommendations.

        Args:
            player: Player state with shop, units, gold, etc.
            target_comp: Target composition champion IDs (optional).

        Returns:
            PickAdvice with ranked recommendations.
        """
        recommendations = []
        shop = player.shop

        for idx, champion in enumerate(shop.slots):
            if champion is None:
                continue

            rec = self._evaluate_champion(champion, idx, player, target_comp)
            if rec:
                recommendations.append(rec)

        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Determine if should refresh
        should_refresh, refresh_reason = self._should_refresh(recommendations, player)

        # Calculate gold to maintain for interest
        gold_to_save = self._calculate_gold_to_save(player)

        return PickAdvice(
            recommendations=recommendations,
            should_refresh=should_refresh,
            refresh_reason=refresh_reason,
            gold_to_save=gold_to_save,
        )

    def _evaluate_champion(
        self,
        champion: "Champion",
        shop_index: int,
        player: "PlayerState",
        target_comp: Optional[List[str]],
    ) -> Optional[PickRecommendation]:
        """
        Evaluate a single champion for purchase recommendation.

        Args:
            champion: Champion in shop slot.
            shop_index: Index of shop slot (0-4).
            player: Player state.
            target_comp: Target composition champion IDs.

        Returns:
            PickRecommendation if score > 0, else None.
        """
        reasons = []
        score = 0.0

        # Count copies owned (bench + board)
        copies_owned = self._count_copies(player, champion.id)

        # 1. Upgrade checks
        if copies_owned >= 6:  # Can make 3-star
            reasons.append(PickReason.UPGRADE_3STAR)
            score += self.weights[PickReason.UPGRADE_3STAR]
        elif copies_owned >= 2:  # Can make 2-star
            reasons.append(PickReason.UPGRADE_2STAR)
            score += self.weights[PickReason.UPGRADE_2STAR]
        elif copies_owned >= 1:  # Has pair
            reasons.append(PickReason.ECONOMY_PAIR)
            score += self.weights[PickReason.ECONOMY_PAIR]

        # 2. Synergy analysis
        synergy_delta = self._calculate_synergy_delta(player, champion)
        for trait_id, delta in synergy_delta.items():
            if delta > 0:
                # Check if new activation or tier upgrade
                if self._is_new_activation(player, trait_id, champion):
                    reasons.append(PickReason.SYNERGY_ACTIVATE)
                    score += self.weights[PickReason.SYNERGY_ACTIVATE]
                else:
                    reasons.append(PickReason.SYNERGY_UPGRADE)
                    score += self.weights[PickReason.SYNERGY_UPGRADE]

        # 3. Target composition check
        if target_comp and champion.id in target_comp:
            reasons.append(PickReason.CORE_CARRY)
            score += self.weights[PickReason.CORE_CARRY]

        # 4. Strong unit check (high cost = strong tempo)
        if champion.cost >= 4:
            reasons.append(PickReason.STRONG_UNIT)
            score += self.weights[PickReason.STRONG_UNIT] * (champion.cost / 5)

        # No score means no recommendation
        if score <= 0:
            return None

        # Adjust score for cost efficiency
        score = score / champion.cost

        # Calculate copies needed for next upgrade
        if copies_owned < 9:
            copies_needed = 3 - (copies_owned % 3)
        else:
            copies_needed = 0

        return PickRecommendation(
            champion_id=champion.id,
            champion_name=champion.name,
            shop_index=shop_index,
            score=score,
            reasons=reasons,
            synergy_delta=synergy_delta,
            cost=champion.cost,
            copies_owned=copies_owned,
            copies_needed=copies_needed,
        )

    def _count_copies(self, player: "PlayerState", champion_id: str) -> int:
        """
        Count total copies of a champion (bench + board).

        Accounts for star levels:
        - 1-star = 1 copy
        - 2-star = 3 copies
        - 3-star = 9 copies

        Args:
            player: Player state.
            champion_id: Champion ID to count.

        Returns:
            Total copies owned.
        """
        count = 0

        # Count bench
        for instance in player.units.bench:
            if instance and instance.champion.id == champion_id:
                count += 3 ** (instance.star_level - 1)

        # Count board
        for instance in player.units.board.values():
            if instance.champion.id == champion_id:
                count += 3 ** (instance.star_level - 1)

        return count

    def _calculate_synergy_delta(
        self, player: "PlayerState", champion: "Champion"
    ) -> Dict[str, int]:
        """
        Calculate synergy count changes from adding champion.

        Args:
            player: Player state.
            champion: Champion to potentially add.

        Returns:
            Dict mapping trait_id to count change.
        """
        # Get current synergies from board units
        board_units = list(player.units.board.values())
        current_synergies = self.synergy_calc.calculate_synergies(board_units)

        # Preview adding the champion
        deltas = self.synergy_calc.preview_add_champion(board_units, champion)

        # Convert to simple count delta
        result = {}
        for trait_id, delta in deltas.items():
            count_change = delta.new_count - delta.old_count
            if count_change != 0:
                result[trait_id] = count_change

        return result

    def _is_new_activation(
        self, player: "PlayerState", trait_id: str, champion: "Champion"
    ) -> bool:
        """
        Check if adding champion would activate a new synergy.

        Args:
            player: Player state.
            trait_id: Trait to check.
            champion: Champion being added.

        Returns:
            True if this would be a new activation.
        """
        board_units = list(player.units.board.values())
        current_synergies = self.synergy_calc.calculate_synergies(board_units)

        if trait_id not in current_synergies:
            return True

        return not current_synergies[trait_id].is_active

    def _should_refresh(
        self, recommendations: List[PickRecommendation], player: "PlayerState"
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if player should refresh shop.

        Args:
            recommendations: Current recommendations.
            player: Player state.

        Returns:
            Tuple of (should_refresh, reason).
        """
        # No good recommendations
        if not recommendations:
            return True, "No useful champions in shop"

        # Best recommendation score is too low
        if recommendations[0].score < 10:
            return True, "Low recommendation scores"

        # Can't afford to refresh
        if player.gold < 2:
            return False, None

        return False, None

    def _calculate_gold_to_save(self, player: "PlayerState") -> int:
        """
        Calculate gold to maintain for interest.

        Args:
            player: Player state.

        Returns:
            Gold threshold to maintain.
        """
        current_gold = player.gold

        # Calculate current interest threshold (10, 20, 30, 40, 50)
        interest_threshold = (current_gold // 10) * 10

        # Check if close to next threshold
        next_threshold = ((current_gold // 10) + 1) * 10
        if next_threshold - current_gold <= 3:
            return next_threshold

        return interest_threshold

    def get_affordable_recommendations(
        self, advice: PickAdvice, gold: int
    ) -> List[PickRecommendation]:
        """
        Filter recommendations to only affordable ones.

        Args:
            advice: Full pick advice.
            gold: Available gold.

        Returns:
            List of affordable recommendations.
        """
        affordable = []
        remaining_gold = gold

        for rec in advice.recommendations:
            if rec.cost <= remaining_gold:
                affordable.append(rec)
                # Don't subtract - just filter by affordability
                # User decides actual purchases

        return affordable

    def get_priority_buys(
        self, advice: PickAdvice, gold: int, preserve_interest: bool = True
    ) -> List[PickRecommendation]:
        """
        Get recommended buys respecting gold constraints.

        Args:
            advice: Full pick advice.
            gold: Available gold.
            preserve_interest: Whether to preserve interest gold.

        Returns:
            List of recommended purchases in priority order.
        """
        priority_buys = []
        available_gold = gold

        if preserve_interest:
            available_gold = max(0, gold - advice.gold_to_save)

        for rec in advice.recommendations:
            if rec.cost <= available_gold:
                priority_buys.append(rec)
                available_gold -= rec.cost

        return priority_buys
