"""Pivot Analyzer.

Analyzes when and how to pivot to a different composition:
- Pivot timing
- Pivot options
- Pivot cost analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum, auto

from .comp_builder import CompBuilder, CompTemplate, CompRecommendation

if TYPE_CHECKING:
    from src.core.game_state import PlayerState, GameState


class PivotReason(Enum):
    """Reasons for pivoting."""

    CONTESTED = auto()  # Contested by other players
    LOW_ROLLS = auto()  # Bad roll luck
    HP_CRITICAL = auto()  # HP too low
    BETTER_ITEMS = auto()  # Items better for different carry
    HIGHROLL = auto()  # Hit better units
    LOBBY_READ = auto()  # Lobby meta read


@dataclass
class PivotOption:
    """A pivot option."""

    target_comp: CompTemplate
    from_comp: Optional[CompTemplate]

    # Transition analysis
    shared_units: List[str]  # Units to keep
    units_to_sell: List[str]  # Units to sell
    units_to_buy: List[str]  # Units to acquire

    # Cost analysis
    gold_loss: int  # Gold lost from selling
    roll_cost: int  # Expected roll cost
    total_cost: int  # Total transition cost

    # Risk assessment
    success_probability: float  # Chance of successful pivot
    risk_level: str  # "low", "medium", "high"

    reasons: List[PivotReason]


@dataclass
class PivotAdvice:
    """Complete pivot advice."""

    should_pivot: bool
    urgency: str  # "immediate", "soon", "optional"
    current_comp_health: float  # 0-100
    options: List[PivotOption]
    recommendation: Optional[PivotOption]
    explanation: str


class PivotAnalyzer:
    """
    Pivot analysis and recommendations.

    Analyzes when pivoting is beneficial and provides
    options for transitioning to different compositions.

    Usage:
        analyzer = PivotAnalyzer()
        advice = analyzer.analyze(player, game, current_comp)
    """

    def __init__(self):
        """Initialize pivot analyzer."""
        self.comp_builder = CompBuilder()

    def analyze(
        self,
        player: "PlayerState",
        game: "GameState",
        current_comp: Optional[CompTemplate] = None,
        contested_units: Optional[List[str]] = None,
    ) -> PivotAdvice:
        """
        Analyze pivot situation.

        Args:
            player: Player state.
            game: Game state.
            current_comp: Current target composition.
            contested_units: Units contested by other players.

        Returns:
            PivotAdvice with recommendations.
        """
        # Evaluate current composition health
        comp_health = self._evaluate_comp_health(
            player, current_comp, contested_units
        )

        # Determine if pivot is needed
        should_pivot, urgency, reasons = self._should_pivot(
            player, comp_health, contested_units
        )

        # Generate pivot options if needed
        options: List[PivotOption] = []
        if should_pivot:
            options = self._generate_pivot_options(player, current_comp, reasons)

        # Get best recommendation
        recommendation = options[0] if options else None

        # Generate explanation
        explanation = self._generate_explanation(
            should_pivot, urgency, comp_health, reasons
        )

        return PivotAdvice(
            should_pivot=should_pivot,
            urgency=urgency,
            current_comp_health=comp_health,
            options=options,
            recommendation=recommendation,
            explanation=explanation,
        )

    def calculate_pivot_cost(
        self,
        player: "PlayerState",
        from_comp: Optional[CompTemplate],
        to_comp: CompTemplate,
    ) -> PivotOption:
        """
        Calculate cost of pivoting between compositions.

        Args:
            player: Player state.
            from_comp: Source composition (None if unknown).
            to_comp: Target composition.

        Returns:
            PivotOption with cost analysis.
        """
        # Get current units
        current_units = set()
        for inst in player.units.board.values():
            current_units.add(inst.champion.id)
        for inst in player.units.bench:
            if inst:
                current_units.add(inst.champion.id)

        # Calculate shared, to sell, to buy
        target_units = set(to_comp.core_units)
        shared = current_units & target_units
        to_sell = current_units - target_units
        to_buy = target_units - current_units

        # Calculate gold loss
        gold_loss = self._calculate_sell_loss(player, list(to_sell))

        # Estimate roll cost
        roll_cost = len(to_buy) * 15  # ~15 gold per unit

        # Estimate success probability
        success_prob = self._estimate_success_probability(player, list(to_buy))

        # Determine risk level
        if success_prob > 0.7:
            risk = "low"
        elif success_prob > 0.4:
            risk = "medium"
        else:
            risk = "high"

        return PivotOption(
            target_comp=to_comp,
            from_comp=from_comp,
            shared_units=list(shared),
            units_to_sell=list(to_sell),
            units_to_buy=list(to_buy),
            gold_loss=gold_loss,
            roll_cost=roll_cost,
            total_cost=gold_loss + roll_cost,
            success_probability=success_prob,
            risk_level=risk,
            reasons=[],
        )

    def _evaluate_comp_health(
        self,
        player: "PlayerState",
        current_comp: Optional[CompTemplate],
        contested_units: Optional[List[str]],
    ) -> float:
        """
        Evaluate current composition health.

        Args:
            player: Player state.
            current_comp: Current target composition.
            contested_units: Contested unit IDs.

        Returns:
            Health score (0-100).
        """
        if current_comp is None:
            return 50.0

        health = 100.0

        # Core unit ownership ratio
        owned_cores = sum(
            1 for u in current_comp.core_units if self._owns_unit(player, u)
        )
        core_ratio = owned_cores / len(current_comp.core_units)
        health *= core_ratio

        # Contested penalty
        if contested_units:
            contested_cores = sum(
                1 for u in current_comp.core_units if u in contested_units
            )
            if contested_cores > 0:
                health -= contested_cores * 15

        # Star level progress bonus
        upgrade_progress = self._calculate_upgrade_progress(player)
        health += upgrade_progress * 20

        return max(0, min(100, health))

    def _should_pivot(
        self,
        player: "PlayerState",
        comp_health: float,
        contested_units: Optional[List[str]],
    ) -> Tuple[bool, str, List[PivotReason]]:
        """
        Determine if pivot is needed.

        Args:
            player: Player state.
            comp_health: Current composition health.
            contested_units: Contested unit IDs.

        Returns:
            Tuple of (should_pivot, urgency, reasons).
        """
        reasons: List[PivotReason] = []

        # Critical HP
        if player.health <= 30:
            reasons.append(PivotReason.HP_CRITICAL)
            return True, "immediate", reasons

        # Very poor composition health
        if comp_health < 40:
            reasons.append(PivotReason.LOW_ROLLS)
            return True, "soon", reasons

        # Heavily contested
        if contested_units and len(contested_units) >= 3:
            reasons.append(PivotReason.CONTESTED)
            return True, "soon", reasons

        # Healthy - no pivot needed
        if comp_health >= 70:
            return False, "none", []

        return False, "optional", reasons

    def _generate_pivot_options(
        self,
        player: "PlayerState",
        current_comp: Optional[CompTemplate],
        reasons: List[PivotReason],
    ) -> List[PivotOption]:
        """
        Generate pivot options.

        Args:
            player: Player state.
            current_comp: Current composition.
            reasons: Pivot reasons.

        Returns:
            List of pivot options sorted by viability.
        """
        # Get composition recommendations
        recommendations = self.comp_builder.recommend(player, top_n=5)

        options: List[PivotOption] = []
        for rec in recommendations:
            # Skip current composition
            if current_comp and rec.template.name == current_comp.name:
                continue

            option = self.calculate_pivot_cost(player, current_comp, rec.template)
            option.reasons = reasons
            options.append(option)

        # Sort by success probability / cost ratio
        options.sort(
            key=lambda o: o.success_probability / (o.total_cost + 1),
            reverse=True,
        )

        return options[:3]

    def _owns_unit(self, player: "PlayerState", champion_id: str) -> bool:
        """Check if player owns a unit."""
        for inst in player.units.board.values():
            if inst.champion.id == champion_id:
                return True
        for inst in player.units.bench:
            if inst and inst.champion.id == champion_id:
                return True
        return False

    def _calculate_sell_loss(
        self, player: "PlayerState", units_to_sell: List[str]
    ) -> int:
        """
        Calculate gold loss from selling units.

        Args:
            player: Player state.
            units_to_sell: Unit IDs to sell.

        Returns:
            Estimated gold loss.
        """
        loss = 0
        for unit_id in units_to_sell:
            # Find the instance
            instance = None
            for inst in player.units.board.values():
                if inst.champion.id == unit_id:
                    instance = inst
                    break
            if instance is None:
                for inst in player.units.bench:
                    if inst and inst.champion.id == unit_id:
                        instance = inst
                        break

            if instance:
                # Sell value is full cost * star copies
                sell_value = instance.get_sell_value()
                # Loss is negligible if 1-star
                if instance.star_level > 1:
                    # Lose some value on invested gold
                    loss += instance.champion.cost

        return loss

    def _estimate_success_probability(
        self, player: "PlayerState", units_to_buy: List[str]
    ) -> float:
        """
        Estimate pivot success probability.

        Args:
            player: Player state.
            units_to_buy: Units to acquire.

        Returns:
            Probability (0-1).
        """
        if not units_to_buy:
            return 1.0

        gold = player.gold
        rolls_available = gold // 2

        # Estimate rolls needed per unit
        needed_rolls = len(units_to_buy) * 10

        if rolls_available >= needed_rolls:
            return 0.8
        elif rolls_available >= needed_rolls * 0.5:
            return 0.5
        else:
            return 0.2

    def _calculate_upgrade_progress(self, player: "PlayerState") -> float:
        """
        Calculate average star level progress.

        Args:
            player: Player state.

        Returns:
            Progress ratio (0-1).
        """
        total_stars = 0
        unit_count = 0

        for inst in player.units.board.values():
            total_stars += inst.star_level
            unit_count += 1

        if unit_count == 0:
            return 0.0

        avg_stars = total_stars / unit_count
        return (avg_stars - 1) / 2  # 1-star=0, 2-star=0.5, 3-star=1

    def _generate_explanation(
        self,
        should_pivot: bool,
        urgency: str,
        comp_health: float,
        reasons: List[PivotReason],
    ) -> str:
        """
        Generate explanation string.

        Args:
            should_pivot: Whether to pivot.
            urgency: Urgency level.
            comp_health: Composition health.
            reasons: Pivot reasons.

        Returns:
            Explanation string.
        """
        if not should_pivot:
            return f"Composition is healthy ({comp_health:.0f}%). Continue current plan."

        reason_texts = {
            PivotReason.CONTESTED: "Units contested by other players",
            PivotReason.LOW_ROLLS: "Not finding core units",
            PivotReason.HP_CRITICAL: "HP critical",
            PivotReason.BETTER_ITEMS: "Items better for different carry",
            PivotReason.HIGHROLL: "Found better units",
            PivotReason.LOBBY_READ: "Better meta read",
        }

        reason_str = ", ".join(reason_texts.get(r, str(r)) for r in reasons)

        urgency_text = {
            "immediate": "Immediately",
            "soon": "Soon",
            "optional": "Optionally",
        }

        return f"{urgency_text[urgency]} consider pivoting. Reasons: {reason_str}"

    def find_natural_pivots(
        self, player: "PlayerState", current_comp: CompTemplate
    ) -> List[CompTemplate]:
        """
        Find compositions that naturally transition from current.

        Args:
            player: Player state.
            current_comp: Current composition.

        Returns:
            List of compatible compositions.
        """
        compatible = []
        current_units = set(current_comp.core_units)

        for template in self.comp_builder.templates:
            if template.name == current_comp.name:
                continue

            target_units = set(template.core_units)
            overlap = len(current_units & target_units)

            # At least 2 shared units
            if overlap >= 2:
                compatible.append(template)

        return compatible

    def get_transition_path(
        self,
        player: "PlayerState",
        from_comp: CompTemplate,
        to_comp: CompTemplate,
    ) -> List[str]:
        """
        Get step-by-step transition path.

        Args:
            player: Player state.
            from_comp: Starting composition.
            to_comp: Target composition.

        Returns:
            List of transition steps.
        """
        option = self.calculate_pivot_cost(player, from_comp, to_comp)

        steps = []

        # Step 1: Identify units to keep
        if option.shared_units:
            steps.append(f"Keep: {', '.join(option.shared_units)}")

        # Step 2: Sell unneeded units
        if option.units_to_sell:
            steps.append(f"Sell: {', '.join(option.units_to_sell)}")

        # Step 3: Acquire new units
        if option.units_to_buy:
            steps.append(f"Find: {', '.join(option.units_to_buy)}")

        # Step 4: Cost summary
        steps.append(f"Estimated cost: {option.total_cost} gold")
        steps.append(f"Success chance: {option.success_probability * 100:.0f}%")

        return steps
