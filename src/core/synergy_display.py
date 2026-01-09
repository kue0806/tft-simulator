"""Synergy Display for TFT Set 16.

Format synergies for display in UI.
"""

from dataclasses import dataclass

from src.data.models.trait import Trait
from src.core.synergy_calculator import ActiveTrait
from src.core.player_units import ChampionInstance


@dataclass
class SynergyDisplay:
    """Formatted synergy for UI display."""

    name: str
    trait_id: str
    count: int
    breakpoint_text: str  # e.g., "4/6" or "2/4/6"
    style: str  # bronze/silver/gold/chromatic
    is_active: bool
    effect_description: str
    champions: list[str]  # Names of champions contributing


class SynergyFormatter:
    """Format synergies for display."""

    # Priority order for sorting styles
    STYLE_PRIORITY = {
        "chromatic": 0,
        "gold": 1,
        "silver": 2,
        "bronze": 3,
        "unique": 4,
        "inactive": 5,
    }

    @staticmethod
    def format_for_display(
        active_traits: dict[str, ActiveTrait],
        champions: list[ChampionInstance],
    ) -> list[SynergyDisplay]:
        """
        Convert active traits to display format.
        Sorted by: active first, then by tier (chromatic > gold > silver > bronze)

        Args:
            active_traits: Dict of trait_id to ActiveTrait
            champions: List of champions on board

        Returns:
            List of SynergyDisplay objects, sorted by priority
        """
        displays = []

        for trait_id, active_trait in active_traits.items():
            # Get champions contributing to this trait
            contributing = SynergyFormatter._get_contributing_champions(
                champions, trait_id
            )
            champion_names = [c.champion.name for c in contributing]

            # Get breakpoint text
            breakpoint_text = SynergyFormatter.get_breakpoint_text(
                active_trait.trait, active_trait.count
            )

            # Get effect description
            effect_desc = ""
            if active_trait.is_active and active_trait.active_breakpoint:
                effect_desc = active_trait.active_breakpoint.effect
            elif active_trait.next_breakpoint:
                effect_desc = f"Need {active_trait.next_breakpoint.count} for: {active_trait.next_breakpoint.effect}"

            display = SynergyDisplay(
                name=active_trait.trait.name,
                trait_id=trait_id,
                count=active_trait.count,
                breakpoint_text=breakpoint_text,
                style=active_trait.style,
                is_active=active_trait.is_active,
                effect_description=effect_desc,
                champions=champion_names,
            )
            displays.append(display)

        # Sort by priority
        return SynergyFormatter.sort_by_priority(displays)

    @staticmethod
    def get_breakpoint_text(trait: Trait, current_count: int) -> str:
        """
        Generate breakpoint text like "4/6" where active is highlighted.

        Args:
            trait: The trait
            current_count: Current unit count

        Returns:
            Breakpoint string like "2/4/6" with current count
        """
        breakpoint_counts = sorted([bp.count for bp in trait.breakpoints])

        # Format with current count indicator
        parts = []
        for bp_count in breakpoint_counts:
            if current_count >= bp_count:
                parts.append(f"({bp_count})")  # Active breakpoint
            else:
                parts.append(str(bp_count))

        return f"{current_count}: " + "/".join(parts)

    @staticmethod
    def sort_by_priority(synergies: list[SynergyDisplay]) -> list[SynergyDisplay]:
        """
        Sort synergies by visual priority:
        1. Active before inactive
        2. Higher tier before lower
        3. More champions before fewer

        Args:
            synergies: List of SynergyDisplay objects

        Returns:
            Sorted list
        """

        def sort_key(synergy: SynergyDisplay) -> tuple:
            style_priority = SynergyFormatter.STYLE_PRIORITY.get(synergy.style, 99)
            # Negative count so higher counts come first
            return (
                0 if synergy.is_active else 1,
                style_priority,
                -synergy.count,
                synergy.name,
            )

        return sorted(synergies, key=sort_key)

    @staticmethod
    def _get_contributing_champions(
        champions: list[ChampionInstance],
        trait_id: str,
    ) -> list[ChampionInstance]:
        """
        Get champions contributing to a trait.

        Args:
            champions: All champions
            trait_id: The trait to check

        Returns:
            Champions with this trait
        """
        seen: set[str] = set()
        result: list[ChampionInstance] = []

        for champ in champions:
            if champ.champion.id in seen:
                continue

            if trait_id in champ.champion.traits:
                result.append(champ)
                seen.add(champ.champion.id)

        return result

    @staticmethod
    def format_compact(synergies: list[SynergyDisplay]) -> str:
        """
        Format synergies as a compact string for terminal display.

        Args:
            synergies: List of SynergyDisplay objects

        Returns:
            Compact string representation
        """
        lines = []
        for syn in synergies:
            if syn.is_active:
                style_indicator = {
                    "chromatic": "★★★★",
                    "gold": "★★★",
                    "silver": "★★",
                    "bronze": "★",
                    "unique": "◆",
                }.get(syn.style, "")
                lines.append(f"{style_indicator} {syn.name} ({syn.count})")
            else:
                lines.append(f"  {syn.name} ({syn.count})")

        return "\n".join(lines)
