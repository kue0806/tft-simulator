"""Synergy Calculator for TFT Set 16.

Calculates active traits and their bonuses based on fielded champions.
"""

from typing import Optional
from dataclasses import dataclass

from src.data.models.trait import Trait, TraitBreakpoint
from src.data.models.champion import Champion
from src.data.loaders import load_traits, load_origins, load_classes
from src.core.player_units import ChampionInstance


@dataclass
class ActiveTrait:
    """Represents an active trait with its current state."""

    trait: Trait
    count: int  # Number of champions with this trait
    active_breakpoint: Optional[TraitBreakpoint]  # Current active tier
    next_breakpoint: Optional[TraitBreakpoint]  # Next tier to reach
    is_active: bool  # Whether any breakpoint is met

    @property
    def style(self) -> str:
        """Return visual style: bronze/silver/gold/chromatic based on tier."""
        if not self.is_active:
            return "inactive"

        breakpoints = self.trait.breakpoints
        if not breakpoints:
            return "unique"

        # Find index of current breakpoint
        try:
            idx = breakpoints.index(self.active_breakpoint)
        except ValueError:
            return "bronze"

        total = len(breakpoints)

        if total == 1:
            return "gold"
        elif idx == 0:
            return "bronze"
        elif idx == total - 1:
            return "chromatic" if total >= 4 else "gold"
        elif idx == total - 2:
            return "gold"
        else:
            return "silver"


@dataclass
class SynergyDelta:
    """Represents change in a synergy."""

    trait_id: str
    old_count: int
    new_count: int
    was_active: bool
    will_be_active: bool
    breakpoint_change: str  # "upgrade", "downgrade", "none"


class SynergyCalculator:
    """
    Calculates active synergies from a list of champions.
    """

    def __init__(self):
        """Initialize with all traits loaded."""
        self.all_traits = {t.id: t for t in load_traits()}
        self.origins = {t.id: t for t in load_origins()}
        self.classes = {t.id: t for t in load_classes()}

    def calculate_synergies(
        self,
        champions: list[ChampionInstance],
        emblems: list[str] | None = None,
    ) -> dict[str, ActiveTrait]:
        """
        Calculate all active synergies for given champions.

        Args:
            champions: List of ChampionInstance on board
            emblems: List of trait IDs from equipped emblems

        Returns:
            Dict mapping trait_id to ActiveTrait
        """
        if emblems is None:
            emblems = []

        # Count traits
        trait_counts = self._count_traits(champions, emblems)

        # Build ActiveTrait for each trait with at least 1 count
        result = {}
        for trait_id, count in trait_counts.items():
            trait = self.all_traits.get(trait_id)
            if trait is None:
                continue

            active_bp = self._get_active_breakpoint(trait, count)
            next_bp = self._get_next_breakpoint(trait, count)
            is_active = active_bp is not None

            result[trait_id] = ActiveTrait(
                trait=trait,
                count=count,
                active_breakpoint=active_bp,
                next_breakpoint=next_bp,
                is_active=is_active,
            )

        return result

    def _count_traits(
        self,
        champions: list[ChampionInstance],
        emblems: list[str],
    ) -> dict[str, int]:
        """
        Count occurrences of each trait.
        Each unique champion counts once per trait they have.
        Emblems add +1 to their trait.

        Args:
            champions: List of ChampionInstance
            emblems: List of trait IDs from emblems

        Returns:
            Dict mapping trait_id to count
        """
        counts: dict[str, int] = {}

        # Track unique champion IDs to avoid double counting (e.g., 2-star same champ)
        seen_champions: set[str] = set()

        for instance in champions:
            champ_id = instance.champion.id

            # Each unique champion counts once
            if champ_id in seen_champions:
                continue
            seen_champions.add(champ_id)

            # Count each trait the champion has
            for trait_id in instance.champion.traits:
                counts[trait_id] = counts.get(trait_id, 0) + 1

        # Add emblem traits
        for trait_id in emblems:
            counts[trait_id] = counts.get(trait_id, 0) + 1

        return counts

    def _get_active_breakpoint(
        self,
        trait: Trait,
        count: int,
    ) -> Optional[TraitBreakpoint]:
        """
        Get the highest breakpoint that is met.
        Example: Demacia (2/4/6/8), count=5 -> returns 4-breakpoint

        Args:
            trait: The trait to check
            count: Current unit count

        Returns:
            The active breakpoint, or None if none met
        """
        # Use the built-in method from Trait model
        return trait.get_active_breakpoint(count)

    def _get_next_breakpoint(
        self,
        trait: Trait,
        count: int,
    ) -> Optional[TraitBreakpoint]:
        """
        Get the next breakpoint to reach.
        Example: Demacia (2/4/6/8), count=5 -> returns 6-breakpoint

        Args:
            trait: The trait to check
            count: Current unit count

        Returns:
            The next breakpoint to reach, or None if at max
        """
        sorted_breakpoints = sorted(trait.breakpoints, key=lambda x: x.count)

        for bp in sorted_breakpoints:
            if bp.count > count:
                return bp

        return None  # Already at max

    def get_trait_bonuses(
        self,
        active_traits: dict[str, ActiveTrait],
    ) -> dict[str, float]:
        """
        Aggregate all stat bonuses from active traits.

        Args:
            active_traits: Dict of active traits

        Returns:
            Combined bonuses like {"armor": 40, "magic_resist": 40, ...}
        """
        bonuses: dict[str, float] = {}

        for trait_id, active_trait in active_traits.items():
            if not active_trait.is_active or active_trait.active_breakpoint is None:
                continue

            # Get stats from the active breakpoint
            for stat, value in active_trait.active_breakpoint.stats.items():
                bonuses[stat] = bonuses.get(stat, 0) + value

        return bonuses

    def preview_add_champion(
        self,
        current_champions: list[ChampionInstance],
        new_champion: Champion,
        emblems: list[str] | None = None,
    ) -> dict[str, SynergyDelta]:
        """
        Preview how synergies would change if adding a champion.
        Useful for shop recommendations.

        Args:
            current_champions: Current board champions
            new_champion: Champion to potentially add
            emblems: Current emblem traits

        Returns:
            Dict of trait_id to SynergyDelta
        """
        if emblems is None:
            emblems = []

        # Current synergies
        current_synergies = self.calculate_synergies(current_champions, emblems)

        # Create temporary instance for the new champion
        temp_instance = ChampionInstance(champion=new_champion)
        new_champions = current_champions + [temp_instance]

        # New synergies
        new_synergies = self.calculate_synergies(new_champions, emblems)

        # Calculate deltas
        deltas = {}

        # Check all traits that would exist after adding
        all_traits = set(current_synergies.keys()) | set(new_synergies.keys())

        for trait_id in all_traits:
            old_count = current_synergies.get(trait_id, ActiveTrait(
                trait=self.all_traits.get(trait_id),
                count=0,
                active_breakpoint=None,
                next_breakpoint=None,
                is_active=False,
            )).count if trait_id in current_synergies else 0

            new = new_synergies.get(trait_id)
            new_count = new.count if new else 0

            was_active = trait_id in current_synergies and current_synergies[trait_id].is_active
            will_be_active = trait_id in new_synergies and new_synergies[trait_id].is_active

            # Determine breakpoint change
            old_bp = current_synergies[trait_id].active_breakpoint if trait_id in current_synergies else None
            new_bp = new_synergies[trait_id].active_breakpoint if trait_id in new_synergies else None

            if new_bp and not old_bp:
                bp_change = "upgrade"
            elif old_bp and new_bp and new_bp.count > old_bp.count:
                bp_change = "upgrade"
            elif old_bp and new_bp and new_bp.count < old_bp.count:
                bp_change = "downgrade"
            elif old_bp and not new_bp:
                bp_change = "downgrade"
            else:
                bp_change = "none"

            deltas[trait_id] = SynergyDelta(
                trait_id=trait_id,
                old_count=old_count,
                new_count=new_count,
                was_active=was_active,
                will_be_active=will_be_active,
                breakpoint_change=bp_change,
            )

        return deltas

    def preview_remove_champion(
        self,
        current_champions: list[ChampionInstance],
        champion_to_remove: ChampionInstance,
        emblems: list[str] | None = None,
    ) -> dict[str, SynergyDelta]:
        """
        Preview how synergies would change if removing a champion.
        Useful for sell decisions.

        Args:
            current_champions: Current board champions
            champion_to_remove: Champion instance to potentially remove
            emblems: Current emblem traits

        Returns:
            Dict of trait_id to SynergyDelta
        """
        if emblems is None:
            emblems = []

        # Current synergies
        current_synergies = self.calculate_synergies(current_champions, emblems)

        # Remove the champion
        new_champions = [c for c in current_champions if c is not champion_to_remove]

        # New synergies
        new_synergies = self.calculate_synergies(new_champions, emblems)

        # Calculate deltas (same logic as preview_add)
        deltas = {}
        all_traits = set(current_synergies.keys()) | set(new_synergies.keys())

        for trait_id in all_traits:
            old = current_synergies.get(trait_id)
            old_count = old.count if old else 0

            new = new_synergies.get(trait_id)
            new_count = new.count if new else 0

            was_active = old and old.is_active
            will_be_active = new and new.is_active

            old_bp = old.active_breakpoint if old else None
            new_bp = new.active_breakpoint if new else None

            if new_bp and not old_bp:
                bp_change = "upgrade"
            elif old_bp and new_bp and new_bp.count > old_bp.count:
                bp_change = "upgrade"
            elif old_bp and new_bp and new_bp.count < old_bp.count:
                bp_change = "downgrade"
            elif old_bp and not new_bp:
                bp_change = "downgrade"
            else:
                bp_change = "none"

            deltas[trait_id] = SynergyDelta(
                trait_id=trait_id,
                old_count=old_count,
                new_count=new_count,
                was_active=was_active,
                will_be_active=will_be_active,
                breakpoint_change=bp_change,
            )

        return deltas

    def get_contributing_champions(
        self,
        champions: list[ChampionInstance],
        trait_id: str,
    ) -> list[ChampionInstance]:
        """
        Get list of champions contributing to a specific trait.

        Args:
            champions: List of ChampionInstance
            trait_id: The trait to check

        Returns:
            List of ChampionInstance with this trait
        """
        seen: set[str] = set()
        result: list[ChampionInstance] = []

        for instance in champions:
            if instance.champion.id in seen:
                continue

            if trait_id in instance.champion.traits:
                result.append(instance)
                seen.add(instance.champion.id)

        return result
