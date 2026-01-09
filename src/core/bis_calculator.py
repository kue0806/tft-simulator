"""Best in Slot Calculator for TFT Set 16.

Calculate optimal item recommendations for champions.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from src.data.models.item import Item
from src.data.loaders import load_combined_items, load_components
from src.core.stat_calculator import StatCalculator

if TYPE_CHECKING:
    from src.core.player_units import ChampionInstance


@dataclass
class ItemRecommendation:
    """A recommended item with reasoning."""

    item: Item
    score: float
    reasons: list[str]
    priority: int  # 1 = highest


class BiSCalculator:
    """
    Calculate Best in Slot items for champions.
    """

    # Item categories by champion role
    CARRY_ITEMS = [
        "infinity_edge",
        "jeweled_gauntlet",
        "giant_slayer",
        "rageblade",
        "rabadons_deathcap",
        "hand_of_justice",
        "bloodthirster",
        "last_whisper",
        "blue_buff",
    ]

    TANK_ITEMS = [
        "warmogs_armor",
        "dragons_claw",
        "bramble_vest",
        "gargoyle_stoneplate",
        "sunfire_cape",
        "redemption",
        "titans_resolve",
    ]

    AP_CARRY_ITEMS = [
        "rabadons_deathcap",
        "jeweled_gauntlet",
        "blue_buff",
        "spear_of_shojin",
        "archangels_staff",
        "hextech_gunblade",
        "morellonomicon",
    ]

    AD_CARRY_ITEMS = [
        "infinity_edge",
        "last_whisper",
        "giant_slayer",
        "rageblade",
        "bloodthirster",
        "runaans_hurricane",
        "statikk_shiv",
        "rapid_firecannon",
    ]

    UTILITY_ITEMS = [
        "zekes_herald",
        "locket_of_the_iron_solari",
        "chalice_of_power",
        "shroud_of_stillness",
        "zephyr",
    ]

    def __init__(self):
        """Initialize the BiS calculator."""
        self.stat_calculator = StatCalculator()
        self._load_items()

    def _load_items(self) -> None:
        """Load all items for reference."""
        self.all_combined = {i.id: i for i in load_combined_items()}
        self.all_components = {i.id: i for i in load_components()}

    def get_bis(
        self,
        champion: "ChampionInstance",
        available_items: Optional[list[Item]] = None,
        team_context: Optional[dict] = None,
    ) -> list[ItemRecommendation]:
        """
        Get best in slot items for a champion.

        Args:
            champion: The champion to itemize
            available_items: Items available to choose from (optional)
            team_context: Team composition context for decisions

        Returns:
            List of 3 recommended items in priority order
        """
        recommendations = []

        # Determine champion role
        role = self._determine_role(champion)

        # Get candidate items based on role
        candidates = self._get_candidates(role, available_items)

        # Score each item
        for item in candidates:
            score, reasons = self._score_item(champion, item, role)
            recommendations.append(
                ItemRecommendation(
                    item=item,
                    score=score,
                    reasons=reasons,
                    priority=0,
                )
            )

        # Sort by score and assign priority
        recommendations.sort(key=lambda x: x.score, reverse=True)
        for i, rec in enumerate(recommendations[:3]):
            rec.priority = i + 1

        return recommendations[:3]

    def _determine_role(self, champion: "ChampionInstance") -> str:
        """
        Determine champion's role based on traits and stats.

        Args:
            champion: The champion instance.

        Returns:
            Role string: "ad_carry", "ap_carry", "tank", "support", "assassin"
        """
        traits = [t.lower() for t in champion.champion.traits]
        base = champion.champion

        # Check traits for role hints
        tank_traits = ["warden", "bruiser", "juggernaut", "defender"]
        ad_carry_traits = ["slayer", "gunslinger", "longshot", "quickstriker"]
        ap_traits = ["arcanist", "invoker"]
        assassin_traits = ["slayer", "quickstriker"]
        support_traits = ["invoker", "caretaker"]

        # Priority check
        if any(t in assassin_traits for t in traits) and base.stats.attack_range <= 2:
            return "assassin"
        if any(t in tank_traits for t in traits):
            return "tank"
        if any(t in ap_traits for t in traits):
            return "ap_carry"
        if any(t in ad_carry_traits for t in traits):
            return "ad_carry"
        if any(t in support_traits for t in traits):
            return "support"

        # Check base stats
        if base.stats.attack_range >= 4:
            return "ad_carry"
        if base.stats.armor > 40 or base.stats.health[0] > 800:
            return "tank"

        return "ad_carry"  # Default

    def _get_candidates(
        self,
        role: str,
        available_items: Optional[list[Item]] = None,
    ) -> list[Item]:
        """
        Get candidate items for role.

        Args:
            role: The champion role.
            available_items: Available items to filter from.

        Returns:
            List of candidate items.
        """
        role_items = {
            "ad_carry": self.AD_CARRY_ITEMS,
            "ap_carry": self.AP_CARRY_ITEMS,
            "tank": self.TANK_ITEMS,
            "support": self.UTILITY_ITEMS + self.TANK_ITEMS,
            "assassin": self.AD_CARRY_ITEMS,
        }

        preferred = role_items.get(role, self.AD_CARRY_ITEMS)

        if available_items:
            return [i for i in available_items if i.id in preferred]

        # Return all items in preferred list
        return [
            self.all_combined[item_id]
            for item_id in preferred
            if item_id in self.all_combined
        ]

    def _score_item(
        self,
        champion: "ChampionInstance",
        item: Item,
        role: str,
    ) -> tuple[float, list[str]]:
        """
        Score an item for a champion.

        Args:
            champion: The champion instance.
            item: The item to score.
            role: The champion's role.

        Returns:
            Tuple of (score, list of reasons).
        """
        score = 0.0
        reasons = []

        stats = item.stats
        if not stats:
            return score, reasons

        # Role-specific scoring
        if role in ["ad_carry", "assassin"]:
            if stats.ad > 0:
                score += stats.ad * 2
                reasons.append(f"+{stats.ad} AD")
            if stats.crit_chance > 0:
                score += stats.crit_chance * 1.5
                reasons.append(f"+{stats.crit_chance}% Crit")
            if stats.crit_damage > 0:
                score += stats.crit_damage * 1.0
                reasons.append(f"+{stats.crit_damage}% Crit Dmg")
            if stats.attack_speed > 0:
                score += stats.attack_speed * 1.2
                reasons.append(f"+{stats.attack_speed}% AS")

        elif role == "ap_carry":
            if stats.ap > 0:
                score += stats.ap * 2
                reasons.append(f"+{stats.ap} AP")
            if stats.mana > 0:
                score += stats.mana * 0.5
                reasons.append(f"+{stats.mana} Mana")

        elif role in ["tank", "support"]:
            if stats.health > 0:
                score += stats.health * 0.1
                reasons.append(f"+{stats.health} HP")
            if stats.armor > 0:
                score += stats.armor * 1.5
                reasons.append(f"+{stats.armor} Armor")
            if stats.mr > 0:
                score += stats.mr * 1.5
                reasons.append(f"+{stats.mr} MR")

        # Universal stats
        if stats.omnivamp > 0:
            score += stats.omnivamp * 0.5
            reasons.append(f"+{stats.omnivamp}% Omnivamp")

        # Item effect bonus
        if item.effect:
            score += 10
            reasons.append("Has special effect")

        return score, reasons

    def suggest_components(
        self,
        target_item: Item,
        available_components: list[Item],
    ) -> Optional[tuple[Item, Item]]:
        """
        Suggest which components to pick for a target item.

        Args:
            target_item: The item to build.
            available_components: Components available.

        Returns:
            Tuple of two components if both available, None otherwise.
        """
        if not target_item.components:
            return None

        comp1_id, comp2_id = target_item.components
        available_ids = [c.id for c in available_components]

        has_comp1 = comp1_id in available_ids
        has_comp2 = comp2_id in available_ids

        if has_comp1 and has_comp2:
            comp1 = next(c for c in available_components if c.id == comp1_id)
            comp2 = next(c for c in available_components if c.id == comp2_id)
            return (comp1, comp2)

        return None

    def get_missing_components(
        self,
        target_item: Item,
        available_components: list[Item],
    ) -> list[str]:
        """
        Get which components are missing to build an item.

        Args:
            target_item: The item to build.
            available_components: Components available.

        Returns:
            List of missing component IDs.
        """
        if not target_item.components:
            return []

        comp1_id, comp2_id = target_item.components
        available_ids = [c.id for c in available_components]
        missing = []

        if comp1_id not in available_ids:
            missing.append(comp1_id)
        if comp2_id not in available_ids and comp2_id != comp1_id:
            missing.append(comp2_id)
        elif comp2_id == comp1_id and available_ids.count(comp1_id) < 2:
            missing.append(comp2_id)

        return missing

    def get_buildable_bis(
        self,
        champion: "ChampionInstance",
        available_components: list[Item],
    ) -> list[ItemRecommendation]:
        """
        Get BiS items that can be built from available components.

        Args:
            champion: The champion to itemize.
            available_components: Available components.

        Returns:
            List of buildable BiS recommendations.
        """
        # Get full BiS
        full_bis = self.get_bis(champion)

        # Filter to only buildable items
        buildable = []
        for rec in full_bis:
            if self.suggest_components(rec.item, available_components):
                buildable.append(rec)

        return buildable
