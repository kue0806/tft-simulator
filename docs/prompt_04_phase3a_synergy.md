# TFT Simulator - Phase 3A: Synergy Calculator

## Overview
Implement the synergy/trait system that calculates active traits and their bonuses based on fielded champions.

## Part 1: Synergy Calculator Core

Create `src/core/synergy_calculator.py`:

```python
from typing import Optional
from dataclasses import dataclass
from src.data.models.trait import Trait, TraitBreakpoint
from src.data.loaders.trait_loader import load_traits, load_origins, load_classes

@dataclass
class ActiveTrait:
    """Represents an active trait with its current state."""
    trait: Trait
    count: int                    # Number of champions with this trait
    active_breakpoint: Optional[TraitBreakpoint]  # Current active tier
    next_breakpoint: Optional[TraitBreakpoint]    # Next tier to reach
    is_active: bool              # Whether any breakpoint is met
    
    @property
    def style(self) -> str:
        """Return visual style: bronze/silver/gold/chromatic based on tier."""
        if not self.is_active:
            return "inactive"
        breakpoints = self.trait.breakpoints
        if not breakpoints:
            return "unique"
        
        idx = breakpoints.index(self.active_breakpoint)
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


class SynergyCalculator:
    """
    Calculates active synergies from a list of champions.
    """
    
    def __init__(self):
        self.all_traits = {t.id: t for t in load_traits()}
        self.origins = {t.id: t for t in load_origins()}
        self.classes = {t.id: t for t in load_classes()}
    
    def calculate_synergies(
        self, 
        champions: list[ChampionInstance],
        emblems: list[str] = []  # Additional traits from emblems
    ) -> dict[str, ActiveTrait]:
        """
        Calculate all active synergies for given champions.
        
        Args:
            champions: List of ChampionInstance on board
            emblems: List of trait IDs from equipped emblems
            
        Returns:
            Dict mapping trait_id to ActiveTrait
        """
        pass
    
    def _count_traits(
        self, 
        champions: list[ChampionInstance],
        emblems: list[str]
    ) -> dict[str, int]:
        """
        Count occurrences of each trait.
        Each unique champion counts once per trait they have.
        Emblems add +1 to their trait.
        """
        pass
    
    def _get_active_breakpoint(
        self, 
        trait: Trait, 
        count: int
    ) -> Optional[TraitBreakpoint]:
        """
        Get the highest breakpoint that is met.
        Example: Demacia (2/4/6/8), count=5 → returns 4-breakpoint
        """
        pass
    
    def _get_next_breakpoint(
        self, 
        trait: Trait, 
        count: int
    ) -> Optional[TraitBreakpoint]:
        """
        Get the next breakpoint to reach.
        Example: Demacia (2/4/6/8), count=5 → returns 6-breakpoint
        """
        pass
    
    def get_trait_bonuses(
        self, 
        active_traits: dict[str, ActiveTrait]
    ) -> dict[str, any]:
        """
        Aggregate all stat bonuses from active traits.
        Returns combined bonuses like:
        {
            "armor": 40,
            "magic_resist": 40,
            "ability_power": 25,
            ...
        }
        """
        pass
    
    def preview_add_champion(
        self,
        current_champions: list[ChampionInstance],
        new_champion: Champion,
        emblems: list[str] = []
    ) -> dict[str, SynergyDelta]:
        """
        Preview how synergies would change if adding a champion.
        Useful for shop recommendations.
        """
        pass
    
    def preview_remove_champion(
        self,
        current_champions: list[ChampionInstance],
        champion_to_remove: ChampionInstance,
        emblems: list[str] = []
    ) -> dict[str, SynergyDelta]:
        """
        Preview how synergies would change if removing a champion.
        Useful for sell decisions.
        """
        pass


@dataclass
class SynergyDelta:
    """Represents change in a synergy."""
    trait_id: str
    old_count: int
    new_count: int
    was_active: bool
    will_be_active: bool
    breakpoint_change: str  # "upgrade", "downgrade", "none"
```

## Part 2: Unique Trait Handlers

Create `src/core/unique_traits.py`:

```python
"""
Handle special/unique traits that have complex mechanics.
"""

class UniqueTraitHandler:
    """Base class for unique trait handlers."""
    
    def apply(self, game_state, champions: list[ChampionInstance]) -> None:
        raise NotImplementedError
    
    def get_bonus(self, game_state, champions: list[ChampionInstance]) -> dict:
        raise NotImplementedError


class DemaciaHandler(UniqueTraitHandler):
    """
    Demacia: Rally when team loses 25% HP.
    Each Rally reduces ability costs by 10%.
    """
    def __init__(self):
        self.rally_count = 0
    
    def on_health_lost(self, health_lost_percent: float) -> bool:
        """Check if rally triggers."""
        pass
    
    def get_bonus(self, champions) -> dict:
        """Return mana cost reduction."""
        pass


class IoniaHandler(UniqueTraitHandler):
    """
    Ionia: Random path each game.
    Paths: Spirit, Generosity, Enlightenment, Transcendence, Precision
    """
    PATHS = {
        "spirit": {"health": True, "stacking_ad_ap": True},
        "generosity": {"gold_on_takedown": True, "ad_ap_per_gold": True},
        "enlightenment": {"ad_ap_by_level": True},
        "transcendence": {"health": True, "bonus_magic_damage": True},
        "precision": {"crit_abilities": True, "stacking_ad_ap": True}
    }
    
    def __init__(self):
        self.current_path = None
    
    def roll_path(self) -> str:
        """Randomly select path at game start."""
        pass


class NoxusHandler(UniqueTraitHandler):
    """
    Noxus: Summon Atakhan when enemies lose 15% HP.
    Atakhan power scales with Noxian star levels.
    """
    def calculate_atakhan_power(self, noxian_champions: list) -> dict:
        pass


class VoidHandler(UniqueTraitHandler):
    """
    Void: Grants mutations to void champions.
    """
    MUTATIONS = [
        "vampiric",      # Omnivamp
        "voracious",     # Attack speed on kill
        "riftborn",      # Damage amp
        "adaptive",      # Armor/MR
        # ... etc
    ]
    
    def assign_mutation(self, champion: ChampionInstance, mutation: str) -> None:
        pass


class YordleHandler(UniqueTraitHandler):
    """
    Yordle: Scaling bonuses per Yordle fielded.
    3-stars grant 50% more!
    """
    def calculate_bonus(self, yordle_champions: list) -> dict:
        base_hp = 50
        base_as = 0.07
        
        total_hp = 0
        total_as = 0
        
        for champ in yordle_champions:
            multiplier = 1.5 if champ.star_level == 3 else 1.0
            total_hp += base_hp * multiplier
            total_as += base_as * multiplier
        
        return {"health": total_hp, "attack_speed": total_as}


# Registry of unique trait handlers
UNIQUE_HANDLERS = {
    "demacia": DemaciaHandler,
    "ionia": IoniaHandler,
    "noxus": NoxusHandler,
    "void": VoidHandler,
    "yordle": YordleHandler,
    # Add more as needed
}
```

## Part 3: Emblem System

Create `src/core/emblem_system.py`:

```python
"""
Handle emblem items that grant additional traits.
"""

class EmblemSystem:
    """
    Manages emblem effects on champions.
    """
    
    # Emblem item IDs mapped to trait IDs
    EMBLEM_TRAITS = {
        "noxus_emblem": "noxus",
        "slayer_emblem": "slayer",
        "arcanist_emblem": "arcanist",
        "bruiser_emblem": "bruiser",
        "warden_emblem": "warden",
        # ... etc (from combined.json)
    }
    
    @staticmethod
    def get_emblem_traits(champion: ChampionInstance) -> list[str]:
        """
        Get list of trait IDs granted by equipped emblems.
        """
        emblem_traits = []
        for item in champion.items:
            if item.id in EmblemSystem.EMBLEM_TRAITS:
                emblem_traits.append(EmblemSystem.EMBLEM_TRAITS[item.id])
        return emblem_traits
    
    @staticmethod
    def get_all_emblem_traits(champions: list[ChampionInstance]) -> list[str]:
        """
        Get all emblem traits from all champions.
        """
        all_emblems = []
        for champ in champions:
            all_emblems.extend(EmblemSystem.get_emblem_traits(champ))
        return all_emblems
    
    @staticmethod
    def can_equip_emblem(champion: ChampionInstance, emblem_trait: str) -> bool:
        """
        Check if champion can equip this emblem.
        Cannot equip if champion already has the trait naturally.
        """
        return emblem_trait not in champion.champion.traits
```

## Part 4: Synergy Display & Formatting

Create `src/core/synergy_display.py`:

```python
"""
Format synergies for display.
"""

@dataclass
class SynergyDisplay:
    """Formatted synergy for UI display."""
    name: str
    count: int
    breakpoint_text: str  # e.g., "4/6" or "2/4/6"
    style: str            # bronze/silver/gold/chromatic
    is_active: bool
    effect_description: str
    champions: list[str]  # Names of champions contributing


class SynergyFormatter:
    """Format synergies for display."""
    
    @staticmethod
    def format_for_display(
        active_traits: dict[str, ActiveTrait],
        champions: list[ChampionInstance]
    ) -> list[SynergyDisplay]:
        """
        Convert active traits to display format.
        Sorted by: active first, then by tier (chromatic > gold > silver > bronze)
        """
        pass
    
    @staticmethod
    def get_breakpoint_text(trait: Trait, current_count: int) -> str:
        """
        Generate breakpoint text like "4/6" where 4 is active.
        """
        pass
    
    @staticmethod
    def sort_by_priority(synergies: list[SynergyDisplay]) -> list[SynergyDisplay]:
        """
        Sort synergies by visual priority:
        1. Active before inactive
        2. Higher tier before lower
        3. More champions before fewer
        """
        STYLE_PRIORITY = {
            "chromatic": 0,
            "gold": 1,
            "silver": 2,
            "bronze": 3,
            "unique": 4,
            "inactive": 5
        }
        pass
```

## Part 5: Integration with PlayerUnits

Update `src/core/player_units.py` to include synergy calculation:

```python
class PlayerUnits:
    def __init__(self):
        # ... existing code ...
        self.synergy_calculator = SynergyCalculator()
        self._cached_synergies: Optional[dict[str, ActiveTrait]] = None
    
    def get_active_synergies(self) -> dict[str, ActiveTrait]:
        """
        Get current active synergies from board champions.
        Uses caching - invalidate on board change.
        """
        if self._cached_synergies is None:
            board_champions = list(self.board.values())
            emblems = EmblemSystem.get_all_emblem_traits(board_champions)
            self._cached_synergies = self.synergy_calculator.calculate_synergies(
                board_champions, emblems
            )
        return self._cached_synergies
    
    def invalidate_synergy_cache(self) -> None:
        """Call when board changes."""
        self._cached_synergies = None
    
    # Override methods that change board to invalidate cache
    def place_on_board(self, champion: ChampionInstance, position: tuple) -> bool:
        result = super().place_on_board(champion, position)
        if result:
            self.invalidate_synergy_cache()
        return result
```

## Part 6: Tests

Create `tests/test_synergy_calculator.py`:

```python
import pytest
from src.core.synergy_calculator import SynergyCalculator, ActiveTrait

class TestSynergyCalculator:
    
    def test_single_trait_activation(self):
        """Test basic trait activation with 2 champions."""
        # 2 Demacia champions should activate Demacia (2)
        pass
    
    def test_multiple_breakpoints(self):
        """Test trait with multiple breakpoints."""
        # Demacia: 2/4/6/8
        # With 5 Demacia, should show 4 active, 6 as next
        pass
    
    def test_multiple_traits(self):
        """Test champion with multiple traits."""
        # Vi has Piltover, Zaun, Bruiser
        pass
    
    def test_emblem_adds_trait(self):
        """Test emblem granting additional trait."""
        pass
    
    def test_unique_champions_only(self):
        """Same champion multiple times counts once."""
        # 3 copies of same champion = 1 trait count
        pass
    
    def test_style_calculation(self):
        """Test bronze/silver/gold/chromatic assignment."""
        pass
    
    def test_preview_add_champion(self):
        """Test synergy preview when adding champion."""
        pass
    
    def test_preview_remove_champion(self):
        """Test synergy preview when removing champion."""
        pass
    
    def test_trait_bonuses_aggregation(self):
        """Test combining bonuses from multiple traits."""
        pass
    
    def test_inactive_traits_shown(self):
        """Traits with 1 champion show but inactive."""
        pass


class TestUniqueTraits:
    
    def test_yordle_scaling(self):
        """Yordle bonus scales with count and star level."""
        pass
    
    def test_ionia_path_selection(self):
        """Ionia randomly selects path."""
        pass
    
    def test_void_mutations(self):
        """Void assigns mutations to champions."""
        pass


class TestEmblemSystem:
    
    def test_emblem_detection(self):
        """Detect emblem items on champions."""
        pass
    
    def test_cannot_double_trait(self):
        """Cannot equip emblem for existing trait."""
        pass
    
    def test_emblem_counts_for_trait(self):
        """Emblem adds to trait count."""
        pass
```

## Expected Output

```
src/core/
├── synergy_calculator.py   # Core synergy calculation
├── unique_traits.py        # Special trait handlers
├── emblem_system.py        # Emblem management
└── synergy_display.py      # Display formatting

tests/
└── test_synergy_calculator.py  # Comprehensive tests
```

## Verification

```python
# Quick test
calc = SynergyCalculator()

# Create some champions on board
champions = [
    ChampionInstance(get_champion("jarvan_iv")),  # Demacia, Warden
    ChampionInstance(get_champion("sona")),        # Demacia, Invoker
    ChampionInstance(get_champion("garen")),       # Demacia, Defender
    ChampionInstance(get_champion("vayne")),       # Demacia, Longshot
]

synergies = calc.calculate_synergies(champions)

# Should have Demacia (4) active
assert synergies["demacia"].is_active
assert synergies["demacia"].count == 4
assert synergies["demacia"].style == "silver"  # 4 is second breakpoint

# Should have various 1-count inactive traits
assert synergies["warden"].count == 1
assert not synergies["warden"].is_active
```

## Priority

1. **Must**: SynergyCalculator with calculate_synergies
2. **Must**: Breakpoint detection (active/next)
3. **Must**: Style calculation (bronze/silver/gold/chromatic)
4. **Should**: Preview add/remove champion
5. **Should**: Emblem system integration
6. **Nice**: Unique trait handlers (complex mechanics)
7. **Must**: Tests for core functionality
