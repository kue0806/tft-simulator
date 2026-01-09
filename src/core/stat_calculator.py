"""Stat Calculator for TFT Set 16.

Calculate total stats for champions from base stats, items, and traits.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.player_units import ChampionInstance
    from src.core.item_manager import ItemInstance


@dataclass
class CalculatedStats:
    """Complete calculated stats for a champion instance."""

    # Core stats
    health: int = 0
    max_health: int = 0
    mana: int = 0
    max_mana: int = 0
    attack_damage: int = 0
    ability_power: int = 100  # Base 100 AP
    armor: int = 0
    magic_resist: int = 0
    attack_speed: float = 0.0
    crit_chance: float = 0.25  # Base 25% crit
    crit_damage: float = 1.4  # Base 140% crit damage
    dodge_chance: float = 0.0
    attack_range: int = 1

    # Derived stats
    omnivamp: float = 0.0
    damage_amp: float = 0.0
    durability: float = 0.0
    mana_regen: float = 0.0
    mana_cost_reduction: float = 0.0  # Reduces mana cost by percentage
    mana_start: int = 0  # Starting mana

    def __post_init__(self):
        """Ensure health is set to max_health."""
        if self.health == 0 and self.max_health > 0:
            self.health = self.max_health


class StatCalculator:
    """
    Calculate total stats for a champion from:
    - Base stats (scaled by star level)
    - Item stats
    - Trait bonuses
    """

    def calculate_stats(
        self,
        champion: "ChampionInstance",
        trait_bonuses: Optional[dict] = None,
        active_traits: Optional[dict] = None,
    ) -> CalculatedStats:
        """
        Calculate complete stats for a champion.

        Args:
            champion: The champion instance
            trait_bonuses: Aggregated bonuses from traits
            active_traits: Active trait dict for special effects

        Returns:
            CalculatedStats with all bonuses applied
        """
        stats = CalculatedStats()

        # 1. Apply base stats with star scaling
        self._apply_base_stats(stats, champion)

        # 2. Apply item stats
        self._apply_item_stats(stats, champion)

        # 3. Apply trait bonuses
        if trait_bonuses:
            self._apply_trait_bonuses(stats, trait_bonuses)

        # 4. Apply special trait effects
        if active_traits:
            self._apply_special_effects(stats, champion, active_traits)

        # Ensure health = max_health after all bonuses
        stats.health = stats.max_health

        return stats

    def _apply_base_stats(
        self,
        stats: CalculatedStats,
        champion: "ChampionInstance",
    ) -> None:
        """Apply base champion stats with star level scaling."""
        base = champion.champion.stats
        star = champion.star_level
        star_idx = min(star - 1, 2)  # 0, 1, 2 for star levels 1, 2, 3

        # Health and AD use star-indexed arrays
        stats.max_health = base.health[star_idx]
        stats.health = stats.max_health
        stats.attack_damage = base.attack_damage[star_idx]

        # These don't scale with stars
        stats.armor = base.armor
        stats.magic_resist = base.magic_resist
        stats.attack_speed = base.attack_speed
        stats.mana = base.mana[0]  # Starting mana
        stats.max_mana = base.mana[1]  # Max mana
        stats.attack_range = base.attack_range
        stats.crit_chance = base.crit_chance
        stats.crit_damage = base.crit_damage

    def _apply_item_stats(
        self,
        stats: CalculatedStats,
        champion: "ChampionInstance",
    ) -> None:
        """Apply stats from all equipped items."""
        for item in champion.items:
            # Handle both Item and ItemInstance
            item_obj = item.item if hasattr(item, "item") else item
            item_stats = item_obj.stats
            if not item_stats:
                continue

            # Add flat stats
            stats.attack_damage += item_stats.ad
            stats.ability_power += item_stats.ap
            stats.armor += item_stats.armor
            stats.magic_resist += item_stats.mr
            stats.max_health += item_stats.health
            stats.max_mana += item_stats.mana

            # Add percentage stats (convert from percentage to decimal)
            stats.attack_speed += item_stats.attack_speed / 100
            stats.crit_chance += item_stats.crit_chance / 100
            stats.crit_damage += item_stats.crit_damage / 100
            stats.omnivamp += item_stats.omnivamp / 100
            stats.durability += item_stats.durability / 100

        # Update current health to match max health after items
        stats.health = stats.max_health

    def _apply_trait_bonuses(
        self,
        stats: CalculatedStats,
        bonuses: dict,
    ) -> None:
        """Apply aggregated trait bonuses."""
        # Flat bonus mapping
        flat_bonus_mapping = {
            "armor": "armor",
            "magic_resist": "magic_resist",
            "mr": "magic_resist",
            "ability_power": "ability_power",
            "ap": "ability_power",
            "attack_damage": "attack_damage",
            "ad": "attack_damage",
            "health": "max_health",
            "max_health": "max_health",
            "hp": "max_health",
            "crit_chance": "crit_chance",
            "crit": "crit_chance",
            "omnivamp": "omnivamp",
            "mana_regen": "mana_regen",
            "durability": "durability",
            "damage_amp": "damage_amp",
            "mana": "mana_start",
        }

        # Percentage bonus mapping - these multiply the base stat
        percent_bonus_mapping = {
            "health_percent": "max_health",
            "hp_percent": "max_health",
            "ad_percent": "attack_damage",
            "attack_damage_percent": "attack_damage",
            "ap_percent": "ability_power",
            "ability_power_percent": "ability_power",
            "armor_percent": "armor",
            "mr_percent": "magic_resist",
            "magic_resist_percent": "magic_resist",
            "attack_speed_percent": "attack_speed",
            "as_percent": "attack_speed",
        }

        # Apply flat bonuses first
        for key, value in bonuses.items():
            if key in flat_bonus_mapping:
                attr = flat_bonus_mapping[key]
                current = getattr(stats, attr)
                setattr(stats, attr, current + value)

        # Apply percentage bonuses (multiply base stat by percentage)
        for key, value in bonuses.items():
            if key in percent_bonus_mapping:
                attr = percent_bonus_mapping[key]
                current = getattr(stats, attr)
                # value is in percent (e.g., 25 = 25%)
                bonus = current * (value / 100)
                setattr(stats, attr, current + bonus)

        # Handle attack_speed specially (it's already a multiplier in some contexts)
        if "attack_speed" in bonuses:
            # attack_speed from traits is usually a percentage (e.g., 10 = 10%)
            stats.attack_speed *= (1 + bonuses["attack_speed"] / 100)
        if "as" in bonuses:
            stats.attack_speed *= (1 + bonuses["as"] / 100)

        # Update health after trait bonuses
        stats.health = stats.max_health

    def _apply_special_effects(
        self,
        stats: CalculatedStats,
        champion: "ChampionInstance",
        active_traits: dict,
    ) -> None:
        """Apply special trait effects that depend on conditions."""
        from src.core.unique_traits import get_handler

        for trait_id, active_trait in active_traits.items():
            if not active_trait.is_active:
                continue

            handler = get_handler(trait_id)
            if handler is None:
                continue

            # Get bonuses from unique trait handler
            try:
                handler_bonuses = handler.get_bonuses(champion, active_trait)
                if handler_bonuses:
                    self._apply_handler_bonuses(stats, handler_bonuses)
            except Exception:
                # Silently skip if handler fails
                pass

    def _apply_handler_bonuses(
        self,
        stats: CalculatedStats,
        bonuses: dict,
    ) -> None:
        """Apply bonuses from unique trait handlers."""
        bonus_mapping = {
            "health": "max_health",
            "max_health": "max_health",
            "attack_damage": "attack_damage",
            "ad": "attack_damage",
            "ability_power": "ability_power",
            "ap": "ability_power",
            "armor": "armor",
            "magic_resist": "magic_resist",
            "mr": "magic_resist",
            "attack_speed": "attack_speed",
            "as": "attack_speed",
            "crit_chance": "crit_chance",
            "omnivamp": "omnivamp",
            "durability": "durability",
            "damage_amp": "damage_amp",
            "mana_cost_reduction": "mana_cost_reduction",
        }

        for key, value in bonuses.items():
            if key in bonus_mapping:
                attr = bonus_mapping[key]
                if hasattr(stats, attr):
                    current = getattr(stats, attr)
                    if key in ["attack_speed", "as"] and value < 1:
                        # Attack speed multiplier (e.g., 0.1 = +10%)
                        setattr(stats, attr, current * (1 + value))
                    else:
                        setattr(stats, attr, current + value)

    def get_effective_health(self, stats: CalculatedStats) -> float:
        """
        Calculate effective health considering armor/MR.
        EHP = HP * (1 + Armor/100) for physical
        EHP = HP * (1 + MR/100) for magic

        Args:
            stats: The calculated stats.

        Returns:
            Average effective health against physical and magic damage.
        """
        armor_multiplier = 1 + (stats.armor / 100)
        mr_multiplier = 1 + (stats.magic_resist / 100)

        # Average of physical and magic EHP
        physical_ehp = stats.max_health * armor_multiplier
        magic_ehp = stats.max_health * mr_multiplier

        return (physical_ehp + magic_ehp) / 2

    def get_physical_ehp(self, stats: CalculatedStats) -> float:
        """Get effective health against physical damage."""
        return stats.max_health * (1 + stats.armor / 100)

    def get_magic_ehp(self, stats: CalculatedStats) -> float:
        """Get effective health against magic damage."""
        return stats.max_health * (1 + stats.magic_resist / 100)

    def get_dps(self, stats: CalculatedStats) -> float:
        """
        Calculate theoretical DPS (damage per second).
        DPS = AD * AS * (1 + CritChance * (CritDamage - 1))

        Args:
            stats: The calculated stats.

        Returns:
            Theoretical DPS value.
        """
        crit_multiplier = 1 + (stats.crit_chance * (stats.crit_damage - 1))
        return stats.attack_damage * stats.attack_speed * crit_multiplier

    def get_burst_damage(self, stats: CalculatedStats) -> float:
        """
        Estimate ability burst damage.
        This is a rough estimate based on AP.

        Args:
            stats: The calculated stats.

        Returns:
            Estimated burst damage.
        """
        # Rough estimate: base ability damage scales with AP
        # Most abilities have ~100% AP scaling
        return stats.ability_power * 3  # Assumes 3 ability casts in a fight

    def compare_stats(
        self,
        stats1: CalculatedStats,
        stats2: CalculatedStats,
    ) -> dict[str, float]:
        """
        Compare two stat blocks.

        Args:
            stats1: First stat block.
            stats2: Second stat block.

        Returns:
            Dict of stat name to difference (positive = stats2 is higher).
        """
        differences = {}

        for attr in [
            "max_health",
            "attack_damage",
            "ability_power",
            "armor",
            "magic_resist",
            "attack_speed",
            "crit_chance",
            "crit_damage",
            "omnivamp",
        ]:
            val1 = getattr(stats1, attr)
            val2 = getattr(stats2, attr)
            if val1 != val2:
                differences[attr] = val2 - val1

        return differences
