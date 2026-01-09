# TFT Simulator - Phase 3B: Item System

## Overview
Implement the complete item system including equipping, combining, stat calculation, and item recommendations.

## Part 1: Item Manager

Create `src/core/item_manager.py`:

```python
from typing import Optional
from dataclasses import dataclass, field
from src.data.models.item import Item, ItemType
from src.data.loaders.item_loader import (
    load_components, load_combined, get_recipe, build_recipe_matrix
)

@dataclass
class ItemInstance:
    """An instance of an item that can be equipped."""
    item: Item
    equipped_to: Optional[str] = None  # champion_id if equipped
    
    @property
    def is_component(self) -> bool:
        return self.item.type == ItemType.COMPONENT
    
    @property
    def is_combined(self) -> bool:
        return self.item.type in [ItemType.COMBINED, ItemType.RADIANT, ItemType.ARTIFACT]
    
    @property
    def is_emblem(self) -> bool:
        return self.item.type == ItemType.EMBLEM or self.item.grants_trait is not None


class ItemManager:
    """
    Manages a player's items (inventory + equipped).
    """
    MAX_ITEMS_PER_CHAMPION = 3
    
    def __init__(self):
        self.inventory: list[ItemInstance] = []  # Unequipped items
        self.recipe_matrix = build_recipe_matrix()
        self._load_item_data()
    
    def _load_item_data(self):
        """Load all item data for lookups."""
        self.components = {i.id: i for i in load_components()}
        self.combined = {i.id: i for i in load_combined()}
        self.all_items = {**self.components, **self.combined}
    
    def add_to_inventory(self, item: Item) -> ItemInstance:
        """Add a new item to inventory."""
        instance = ItemInstance(item=item)
        self.inventory.append(instance)
        return instance
    
    def remove_from_inventory(self, item_instance: ItemInstance) -> bool:
        """Remove item from inventory."""
        if item_instance in self.inventory:
            self.inventory.remove(item_instance)
            return True
        return False
    
    def equip_item(
        self, 
        item_instance: ItemInstance, 
        champion: 'ChampionInstance'
    ) -> bool:
        """
        Equip item to champion.
        
        Rules:
        - Champion can hold max 3 items
        - Components auto-combine if possible
        - Cannot equip emblem if champion has that trait
        
        Returns True if successful.
        """
        pass
    
    def unequip_item(
        self, 
        item_instance: ItemInstance, 
        champion: 'ChampionInstance'
    ) -> bool:
        """
        Unequip item from champion back to inventory.
        Note: In real TFT, items cannot be freely unequipped.
        This is for simulation/planning purposes.
        """
        pass
    
    def try_combine(
        self, 
        component1: ItemInstance, 
        component2: ItemInstance
    ) -> Optional[ItemInstance]:
        """
        Try to combine two components into a completed item.
        Returns the new item if successful, None otherwise.
        """
        if not component1.is_component or not component2.is_component:
            return None
        
        recipe = get_recipe(component1.item.id, component2.item.id)
        if recipe:
            # Remove components from inventory
            self.remove_from_inventory(component1)
            self.remove_from_inventory(component2)
            # Create combined item
            return self.add_to_inventory(recipe)
        return None
    
    def auto_combine_on_equip(
        self, 
        component: ItemInstance, 
        champion: 'ChampionInstance'
    ) -> Optional[ItemInstance]:
        """
        When equipping a component to a champion that has another component,
        automatically combine them.
        """
        champion_components = [
            item for item in champion.items 
            if item.is_component
        ]
        
        for existing in champion_components:
            recipe = get_recipe(component.item.id, existing.item.id)
            if recipe:
                # Remove both components, add combined
                champion.items.remove(existing)
                combined = ItemInstance(item=recipe)
                champion.items.append(combined)
                return combined
        
        return None
    
    def get_available_recipes(self) -> list[tuple[Item, Item, Item]]:
        """
        Get all possible recipes from current inventory.
        Returns list of (component1, component2, result).
        """
        recipes = []
        components = [i for i in self.inventory if i.is_component]
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                recipe = get_recipe(comp1.item.id, comp2.item.id)
                if recipe:
                    recipes.append((comp1.item, comp2.item, recipe))
        
        return recipes
    
    def get_components_for_item(self, item_id: str) -> Optional[tuple[str, str]]:
        """Get required components for a combined item."""
        item = self.all_items.get(item_id)
        if item and item.components:
            return item.components
        return None
    
    def can_build_item(self, item_id: str) -> bool:
        """Check if item can be built from current inventory."""
        components_needed = self.get_components_for_item(item_id)
        if not components_needed:
            return False
        
        comp1, comp2 = components_needed
        inventory_ids = [i.item.id for i in self.inventory if i.is_component]
        
        # Check if we have both components
        if comp1 == comp2:
            return inventory_ids.count(comp1) >= 2
        else:
            return comp1 in inventory_ids and comp2 in inventory_ids
```

## Part 2: Stat Calculator

Create `src/core/stat_calculator.py`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChampionStats:
    """Complete stats for a champion instance."""
    # Base stats
    health: int = 0
    max_health: int = 0
    mana: int = 0
    max_mana: int = 0
    attack_damage: int = 0
    ability_power: int = 0
    armor: int = 0
    magic_resist: int = 0
    attack_speed: float = 0.0
    crit_chance: float = 0.0
    crit_damage: float = 1.5  # 150% default
    dodge_chance: float = 0.0
    
    # Derived stats
    omnivamp: float = 0.0
    damage_amp: float = 0.0
    durability: float = 0.0
    mana_regen: float = 0.0


class StatCalculator:
    """
    Calculate total stats for a champion from:
    - Base stats (scaled by star level)
    - Item stats
    - Trait bonuses
    """
    
    # Star level multipliers
    STAR_MULTIPLIERS = {
        1: 1.0,
        2: 1.8,   # 2-star = 180% of 1-star
        3: 3.24,  # 3-star = 324% of 1-star (180% * 180%)
    }
    
    def calculate_stats(
        self,
        champion: 'ChampionInstance',
        trait_bonuses: dict = None,
        active_traits: dict = None
    ) -> ChampionStats:
        """
        Calculate complete stats for a champion.
        
        Args:
            champion: The champion instance
            trait_bonuses: Aggregated bonuses from traits
            active_traits: Active trait dict for special effects
        """
        stats = ChampionStats()
        
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
        
        return stats
    
    def _apply_base_stats(
        self, 
        stats: ChampionStats, 
        champion: 'ChampionInstance'
    ) -> None:
        """Apply base champion stats with star level scaling."""
        base = champion.champion
        star = champion.star_level
        multiplier = self.STAR_MULTIPLIERS.get(star, 1.0)
        
        # Health and AD scale with stars
        star_idx = star - 1  # 0, 1, 2 for star levels 1, 2, 3
        stats.max_health = base.health[star_idx] if star_idx < len(base.health) else int(base.health[0] * multiplier)
        stats.health = stats.max_health
        stats.attack_damage = base.attack_damage[star_idx] if star_idx < len(base.attack_damage) else int(base.attack_damage[0] * multiplier)
        
        # These don't scale with stars
        stats.armor = base.armor
        stats.magic_resist = base.magic_resist
        stats.attack_speed = base.attack_speed
        stats.mana = base.mana[0]  # Starting mana
        stats.max_mana = base.mana[1]  # Max mana
    
    def _apply_item_stats(
        self, 
        stats: ChampionStats, 
        champion: 'ChampionInstance'
    ) -> None:
        """Apply stats from all equipped items."""
        for item_instance in champion.items:
            item_stats = item_instance.item.stats
            if not item_stats:
                continue
            
            # Map item stat keys to ChampionStats fields
            stat_mapping = {
                "ad": "attack_damage",
                "ap": "ability_power",
                "as": "attack_speed",  # Percentage, needs conversion
                "armor": "armor",
                "mr": "magic_resist",
                "health": "max_health",
                "hp": "max_health",
                "mana": "max_mana",
                "crit": "crit_chance",
                "crit_chance": "crit_chance",
                "crit_damage": "crit_damage",
                "omnivamp": "omnivamp",
                "dodge": "dodge_chance",
            }
            
            for key, value in item_stats.items():
                if key in stat_mapping:
                    attr = stat_mapping[key]
                    current = getattr(stats, attr)
                    
                    # Handle percentage stats
                    if key == "as":
                        # Attack speed is additive percentage
                        setattr(stats, attr, current + (value / 100))
                    elif key in ["crit", "crit_chance", "omnivamp", "dodge"]:
                        # These are percentages stored as decimals
                        setattr(stats, attr, current + (value / 100))
                    else:
                        setattr(stats, attr, current + value)
        
        # Update current health to match max health after items
        stats.health = stats.max_health
    
    def _apply_trait_bonuses(
        self, 
        stats: ChampionStats, 
        bonuses: dict
    ) -> None:
        """Apply aggregated trait bonuses."""
        bonus_mapping = {
            "armor": "armor",
            "magic_resist": "magic_resist",
            "ability_power": "ability_power",
            "attack_damage": "attack_damage",
            "attack_speed": "attack_speed",
            "health": "max_health",
            "max_health": "max_health",
            "crit_chance": "crit_chance",
            "omnivamp": "omnivamp",
            "mana_regen": "mana_regen",
            "durability": "durability",
            "damage_amp": "damage_amp",
        }
        
        for key, value in bonuses.items():
            if key in bonus_mapping:
                attr = bonus_mapping[key]
                current = getattr(stats, attr)
                setattr(stats, attr, current + value)
    
    def _apply_special_effects(
        self,
        stats: ChampionStats,
        champion: 'ChampionInstance',
        active_traits: dict
    ) -> None:
        """Apply special trait effects that depend on conditions."""
        # Example: Juggernaut durability bonus when above 50% HP
        # Example: Yordle scaling based on count
        # These would reference the unique trait handlers
        pass
    
    def get_effective_health(self, stats: ChampionStats) -> float:
        """
        Calculate effective health considering armor/MR.
        EHP = HP * (1 + Armor/100) for physical
        """
        armor_multiplier = 1 + (stats.armor / 100)
        mr_multiplier = 1 + (stats.magic_resist / 100)
        
        # Average of physical and magic EHP
        physical_ehp = stats.max_health * armor_multiplier
        magic_ehp = stats.max_health * mr_multiplier
        
        return (physical_ehp + magic_ehp) / 2
    
    def get_dps(self, stats: ChampionStats) -> float:
        """
        Calculate theoretical DPS (damage per second).
        DPS = AD * AS * (1 + CritChance * (CritDamage - 1))
        """
        crit_multiplier = 1 + (stats.crit_chance * (stats.crit_damage - 1))
        return stats.attack_damage * stats.attack_speed * crit_multiplier
```

## Part 3: Best in Slot (BiS) Calculator

Create `src/core/bis_calculator.py`:

```python
from dataclasses import dataclass
from typing import Optional

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
        "infinity_edge", "jeweled_gauntlet", "giant_slayer",
        "guinsoos_rageblade", "rabadons_deathcap", "hand_of_justice",
        "bloodthirster", "last_whisper", "blue_buff"
    ]
    
    TANK_ITEMS = [
        "warmogs_armor", "dragons_claw", "bramble_vest",
        "gargoyle_stoneplate", "sunfire_cape", "redemption",
        "titans_resolve"
    ]
    
    AP_CARRY_ITEMS = [
        "rabadons_deathcap", "jeweled_gauntlet", "blue_buff",
        "spear_of_shojin", "archangels_staff", "hextech_gunblade"
    ]
    
    AD_CARRY_ITEMS = [
        "infinity_edge", "last_whisper", "giant_slayer",
        "guinsoos_rageblade", "bloodthirster", "runaans_hurricane"
    ]
    
    def __init__(self):
        self.stat_calculator = StatCalculator()
    
    def get_bis(
        self, 
        champion: 'ChampionInstance',
        available_items: list[Item] = None,
        team_context: dict = None
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
            recommendations.append(ItemRecommendation(
                item=item,
                score=score,
                reasons=reasons,
                priority=0
            ))
        
        # Sort by score and assign priority
        recommendations.sort(key=lambda x: x.score, reverse=True)
        for i, rec in enumerate(recommendations[:3]):
            rec.priority = i + 1
        
        return recommendations[:3]
    
    def _determine_role(self, champion: 'ChampionInstance') -> str:
        """
        Determine champion's role based on traits and stats.
        Returns: "ad_carry", "ap_carry", "tank", "support", "assassin"
        """
        traits = champion.champion.traits
        base = champion.champion
        
        # Check traits for role hints
        tank_traits = ["warden", "bruiser", "juggernaut", "defender"]
        carry_traits = ["slayer", "gunslinger", "longshot", "quickstriker"]
        ap_traits = ["arcanist", "invoker"]
        
        if any(t.lower() in tank_traits for t in traits):
            return "tank"
        if any(t.lower() in ap_traits for t in traits):
            return "ap_carry"
        if any(t.lower() in carry_traits for t in traits):
            return "ad_carry"
        
        # Check base stats
        if base.attack_range >= 4:
            return "ad_carry"
        if base.armor > 40 or base.health[0] > 800:
            return "tank"
        
        return "ad_carry"  # Default
    
    def _get_candidates(
        self, 
        role: str, 
        available_items: list[Item] = None
    ) -> list[Item]:
        """Get candidate items for role."""
        role_items = {
            "ad_carry": self.AD_CARRY_ITEMS,
            "ap_carry": self.AP_CARRY_ITEMS,
            "tank": self.TANK_ITEMS,
            "support": self.TANK_ITEMS,
            "assassin": self.AD_CARRY_ITEMS
        }
        
        preferred = role_items.get(role, self.AD_CARRY_ITEMS)
        
        if available_items:
            return [i for i in available_items if i.id in preferred]
        
        # Return all items in preferred list
        from src.data.loaders.item_loader import load_combined
        all_combined = load_combined()
        return [i for i in all_combined if i.id in preferred]
    
    def _score_item(
        self, 
        champion: 'ChampionInstance', 
        item: Item,
        role: str
    ) -> tuple[float, list[str]]:
        """
        Score an item for a champion.
        Returns (score, list of reasons).
        """
        score = 0.0
        reasons = []
        
        stats = item.stats or {}
        
        # Role-specific scoring
        if role in ["ad_carry", "assassin"]:
            if "ad" in stats:
                score += stats["ad"] * 2
                reasons.append(f"+{stats['ad']} AD")
            if "crit" in stats or "crit_chance" in stats:
                crit = stats.get("crit", stats.get("crit_chance", 0))
                score += crit * 1.5
                reasons.append(f"+{crit}% Crit")
            if "as" in stats:
                score += stats["as"] * 1.2
                reasons.append(f"+{stats['as']}% AS")
        
        elif role == "ap_carry":
            if "ap" in stats:
                score += stats["ap"] * 2
                reasons.append(f"+{stats['ap']} AP")
            if "mana" in stats:
                score += stats["mana"] * 0.5
                reasons.append(f"+{stats['mana']} Mana")
        
        elif role == "tank":
            if "health" in stats or "hp" in stats:
                hp = stats.get("health", stats.get("hp", 0))
                score += hp * 0.1
                reasons.append(f"+{hp} HP")
            if "armor" in stats:
                score += stats["armor"] * 1.5
                reasons.append(f"+{stats['armor']} Armor")
            if "mr" in stats:
                score += stats["mr"] * 1.5
                reasons.append(f"+{stats['mr']} MR")
        
        # Item effect bonus
        if item.effect:
            score += 10  # Bonus for having an effect
            reasons.append("Has special effect")
        
        return score, reasons
    
    def suggest_components(
        self, 
        target_item: Item,
        available_components: list[Item]
    ) -> Optional[tuple[Item, Item]]:
        """
        Suggest which components to pick for a target item.
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
```

## Part 4: Update ChampionInstance

Update `src/core/player_units.py` to integrate items:

```python
class ChampionInstance:
    def __init__(self, champion: Champion, star_level: int = 1):
        self.champion = champion
        self.star_level = star_level
        self.items: list[ItemInstance] = []
        self.position: Optional[tuple[int, int]] = None
        self._cached_stats: Optional[ChampionStats] = None
    
    def get_stats(
        self, 
        trait_bonuses: dict = None,
        active_traits: dict = None
    ) -> ChampionStats:
        """Get current stats with items and traits applied."""
        # Could add caching here
        calculator = StatCalculator()
        return calculator.calculate_stats(
            self, 
            trait_bonuses, 
            active_traits
        )
    
    def can_add_item(self) -> bool:
        return len(self.items) < 3
    
    def add_item(self, item: ItemInstance) -> bool:
        if not self.can_add_item():
            return False
        self.items.append(item)
        item.equipped_to = self.champion.id
        self._cached_stats = None  # Invalidate cache
        return True
    
    def remove_item(self, item: ItemInstance) -> bool:
        if item in self.items:
            self.items.remove(item)
            item.equipped_to = None
            self._cached_stats = None
            return True
        return False
    
    def get_sell_value(self) -> int:
        """Sell value based on cost and star level."""
        base_cost = self.champion.cost
        if self.star_level == 1:
            return base_cost
        elif self.star_level == 2:
            return base_cost * 3
        else:  # 3-star
            return base_cost * 9
    
    def get_item_traits(self) -> list[str]:
        """Get traits granted by equipped emblems."""
        traits = []
        for item in self.items:
            if item.item.grants_trait:
                traits.append(item.item.grants_trait)
        return traits
```

## Part 5: Tests

Create `tests/test_item_system.py`:

```python
import pytest
from src.core.item_manager import ItemManager, ItemInstance
from src.core.stat_calculator import StatCalculator, ChampionStats
from src.core.bis_calculator import BiSCalculator

class TestItemManager:
    
    def test_add_to_inventory(self):
        """Add item to inventory."""
        pass
    
    def test_equip_item(self):
        """Equip item to champion."""
        pass
    
    def test_max_items_limit(self):
        """Cannot equip more than 3 items."""
        pass
    
    def test_auto_combine(self):
        """Components auto-combine when equipped."""
        pass
    
    def test_try_combine_manual(self):
        """Manually combine two components."""
        pass
    
    def test_get_available_recipes(self):
        """Get all buildable recipes from inventory."""
        pass
    
    def test_can_build_item(self):
        """Check if item can be built."""
        pass


class TestStatCalculator:
    
    def test_base_stats_1_star(self):
        """Calculate 1-star base stats."""
        pass
    
    def test_base_stats_2_star(self):
        """2-star has 180% stats."""
        pass
    
    def test_base_stats_3_star(self):
        """3-star has 324% stats."""
        pass
    
    def test_item_stats_applied(self):
        """Item stats add to champion stats."""
        pass
    
    def test_multiple_items_stack(self):
        """Multiple item stats stack correctly."""
        pass
    
    def test_trait_bonuses_applied(self):
        """Trait bonuses add to stats."""
        pass
    
    def test_effective_health(self):
        """EHP calculation with armor/MR."""
        pass
    
    def test_dps_calculation(self):
        """DPS calculation with crit."""
        pass


class TestBiSCalculator:
    
    def test_determine_role_tank(self):
        """Tank traits → tank role."""
        pass
    
    def test_determine_role_carry(self):
        """Carry traits → carry role."""
        pass
    
    def test_bis_ad_carry(self):
        """AD carry gets AD/crit items."""
        pass
    
    def test_bis_ap_carry(self):
        """AP carry gets AP/mana items."""
        pass
    
    def test_bis_tank(self):
        """Tank gets defensive items."""
        pass
    
    def test_suggest_components(self):
        """Suggest components for target item."""
        pass
```

## Expected Output

```
src/core/
├── item_manager.py      # Item inventory & equipping
├── stat_calculator.py   # Stat calculation
└── bis_calculator.py    # Item recommendations

tests/
└── test_item_system.py  # All item tests
```

## Verification

```python
# Quick test
from src.core.item_manager import ItemManager
from src.core.stat_calculator import StatCalculator
from src.core.bis_calculator import BiSCalculator

# Create manager
manager = ItemManager()

# Add some components
bf = manager.add_to_inventory(manager.components["bf_sword"])
bow = manager.add_to_inventory(manager.components["recurve_bow"])

# Get available recipes
recipes = manager.get_available_recipes()
# Should include Giant Slayer (BF + Bow)

# Combine into Giant Slayer
giant_slayer = manager.try_combine(bf, bow)
assert giant_slayer.item.id == "giant_slayer"

# Calculate stats
calc = StatCalculator()
champion = ChampionInstance(get_champion("jinx"), star_level=2)
champion.add_item(giant_slayer)

stats = calc.calculate_stats(champion)
assert stats.attack_damage > champion.champion.attack_damage[1]

# Get BiS
bis = BiSCalculator()
recommendations = bis.get_bis(champion)
assert len(recommendations) == 3
```

## Priority

1. **Must**: ItemManager with equip/combine
2. **Must**: StatCalculator with star scaling + item stats
3. **Must**: Auto-combine on equip
4. **Should**: BiSCalculator with role detection
5. **Should**: Available recipes from inventory
6. **Must**: Tests for core functionality
