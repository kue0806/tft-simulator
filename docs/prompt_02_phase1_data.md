# TFT Simulator - Phase 1: Data Models & Data Collection

## Task
Create the core data models and populate them with real TFT Set 16 "Lore & Legends" data.

## Part 1: Data Models

Create Pydantic models in `src/data/models/`:

### 1.1 Champion Model (`champion.py`)
```python
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ChampionCost(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6    # Unlockable only
    SEVEN = 7  # Unlockable only (e.g., Baron Nashor)

class Champion(BaseModel):
    id: str
    name: str
    cost: ChampionCost
    traits: list[str]  # e.g., ["Demacia", "Warden"]
    health: list[int]  # [1-star, 2-star, 3-star]
    mana: tuple[int, int]  # (start, max)
    armor: int
    magic_resist: int
    attack_damage: list[int]
    attack_speed: float
    attack_range: int
    ability_name: str
    ability_description: str
    is_unlockable: bool = False
    unlock_condition: Optional[str] = None
```

### 1.2 Trait Model (`trait.py`)
```python
class TraitType(str, Enum):
    ORIGIN = "origin"
    CLASS = "class"
    UNIQUE = "unique"  # Single champion traits

class TraitBreakpoint(BaseModel):
    count: int
    effect: str
    stats: dict  # e.g., {"armor": 20, "magic_resist": 20}

class Trait(BaseModel):
    id: str
    name: str
    type: TraitType
    description: str
    breakpoints: list[TraitBreakpoint]
    champions: list[str]  # champion ids
```

### 1.3 Item Model (`item.py`)
```python
class ItemType(str, Enum):
    COMPONENT = "component"
    COMBINED = "combined"
    RADIANT = "radiant"
    ARTIFACT = "artifact"
    EMBLEM = "emblem"

class Item(BaseModel):
    id: str
    name: str
    type: ItemType
    stats: dict  # e.g., {"ad": 10, "as": 10}
    effect: Optional[str] = None
    components: Optional[tuple[str, str]] = None  # For combined items
    grants_trait: Optional[str] = None  # For emblems
```

## Part 2: Data Collection

Scrape/collect data from these sources and save as JSON files in `data/`:

### 2.1 Champions Data
Source: https://tftactics.gg/champions/

Collect ALL 100 champions. Here's a partial list to get started:

**1-Cost Champions (Base):**
- Blitzcrank (Zaun, Juggernaut)
- Jarvan IV (Demacia, Warden)
- Kog'Maw (Void, Longshot)
- Lulu (Yordle, Arcanist)
- Rumble (Yordle, Piltover, Bruiser)
- Sona (Demacia, Invoker)

**2-Cost Champions:**
- Cho'Gath (Void, Bruiser)
- Ekko (Zaun, Quickstriker)
- Poppy (Yordle, Demacia, Defender) [Unlockable]
- Rek'Sai (Void, Bruiser)
- Teemo (Yordle, Quickstriker)
- Tristana (Yordle, Gunslinger)
- Vi (Piltover, Zaun, Bruiser)
- Xin Zhao (Demacia, Ionia, Vanquisher)

**3-Cost Champions:**
- Loris (Ixtal, Vanquisher)
- Malzahar (Void, Arcanist)
- Nautilus (Bilgewater, Vanquisher)
- Neeko (Ionia, Arcanist)
- Qiyana (Ixtal, Slayer)
- Vayne (Demacia, Longshot)

**4-Cost Champions:**
- Ambessa (Noxus, Bruiser)
- Bel'Veth (Void, Riftscourge)
- Braum (Freljord, Vanquisher)
- Garen (Demacia, Defender)
- Jinx (Zaun, Gunslinger)
- Kai'Sa (Void, Assimilator) [Unlockable]
- Lux (Demacia, Arcanist)

**5-Cost Champions:**
- Azir (Shurima, Emperor)
- Ornn (Freljord, Blacksmith)
- Swain (Noxus, Arcanist)
- Viego (Shadow Isles, Harvester)
- Zoe (Ionia, Arcanist)

**6-Cost Champions (Unlockable):**
- Aurelion Sol (Targon, Star Forger) [Unlockable]
- Galio (Demacia, Heroic) [Unlockable]
- T-Hex (Piltover, HexMech) [Unlockable]

**7-Cost Champions (Unlockable):**
- Baron Nashor (Void, Riftscourge) [Unlockable] - Takes 2 slots

### 2.2 Traits Data

**Origins (Region traits):**
- Bilgewater (2/4/6) - Silver Serpents, Black Market
- Demacia (2/4/6/8) - Rally mechanic, Armor/MR
- Freljord (2/4/6)
- Ionia (2/3/4/5) - Multiple paths (Spirit, Generosity, Enlightenment, etc.)
- Ixtal (3/5/7) - Quest system, Sunshards
- Noxus (3/5/7/9) - Summons Atakhan
- Piltover (2/4/6) - Invention system
- Shadow Isles (2/4/6) - Soul collection
- Shurima (2/4/6) - Attack speed, healing, Ascension
- Targon (2/3/4) - No synergy bonus, but stronger base stats
- Void (3/5/7) - Mutations
- Yordle (3/5/7/9) - Scaling per Yordle
- Zaun (2/4/6) - Shimmer-Fused

**Classes:**
- Arcanist (2/4/6) - AP bonus
- Ascendant (1) - Charms in shop
- Bruiser (2/4/6) - Max Health
- Defender (2/4/6) - Armor/MR
- Disruptor (2/4) - Dazzle enemies
- Gunslinger (2/4) - AD, bonus damage
- Heroic (1) - Galio unique
- Huntress (2/3) - Neeko/Nidalee
- Invoker (2/4/6) - Mana regen
- Juggernaut (2/4/6) - Durability
- Longshot (2/4) - Damage Amp at range
- Quickstriker (2/4) - Attack Speed
- Slayer (2/4/6) - Omnivamp, AD
- Vanquisher (2/4/6) - Crit abilities
- Warden (2/4/6) - Shield on low HP

### 2.3 Items Data

**Base Components (9):**
- B.F. Sword (+10 AD)
- Recurve Bow (+10% AS)
- Needlessly Large Rod (+10 AP)
- Tear of the Goddess (+15 Mana)
- Chain Vest (+20 Armor)
- Negatron Cloak (+20 MR)
- Giant's Belt (+150 HP)
- Sparring Gloves (+10% Crit)
- Spatula (Emblem creation)

**Key Combined Items:**
- Infinity Edge (BF + Gloves): Crit abilities
- Rabadon's Deathcap (Rod + Rod): +50 AP
- Guinsoo's Rageblade (Rod + Bow): Stacking AS
- Bloodthirster (BF + Cloak): Omnivamp + Shield
- etc.

## Part 3: Create Data Loaders

Create `src/data/loaders/` with:

### 3.1 `champion_loader.py`
```python
def load_champions() -> list[Champion]:
    """Load all champions from JSON"""
    pass

def get_champion_by_id(champion_id: str) -> Champion:
    pass

def get_champions_by_cost(cost: int) -> list[Champion]:
    pass

def get_champions_by_trait(trait: str) -> list[Champion]:
    pass
```

### 3.2 `trait_loader.py`
```python
def load_traits() -> list[Trait]:
    pass

def get_trait_by_id(trait_id: str) -> Trait:
    pass

def get_origins() -> list[Trait]:
    pass

def get_classes() -> list[Trait]:
    pass
```

### 3.3 `item_loader.py`
```python
def load_items() -> list[Item]:
    pass

def get_item_by_id(item_id: str) -> Item:
    pass

def get_recipe(component1: str, component2: str) -> Optional[Item]:
    pass
```

## Part 4: Data Files Structure

Create JSON files:
```
data/
├── champions/
│   └── set16_champions.json
├── traits/
│   ├── origins.json
│   └── classes.json
├── items/
│   ├── components.json
│   ├── combined.json
│   ├── radiants.json
│   └── artifacts.json
└── constants/
    ├── pool_sizes.json      # Units per cost tier
    ├── shop_odds.json       # Level-based shop odds
    └── level_costs.json     # XP requirements
```

## Part 5: Game Constants

Create `src/core/constants.py`:
```python
# Champion pool sizes (per champion at that cost)
POOL_SIZE = {
    1: 30,
    2: 25,
    3: 18,
    4: 10,
    5: 9,
    6: 8,  # Unlockables
    7: 6,  # Baron Nashor etc.
}

# Shop odds by level [1-cost, 2-cost, 3-cost, 4-cost, 5-cost]
SHOP_ODDS = {
    1:  [100, 0,   0,   0,   0],
    2:  [100, 0,   0,   0,   0],
    3:  [75,  25,  0,   0,   0],
    4:  [55,  30,  15,  0,   0],
    5:  [45,  33,  20,  2,   0],
    6:  [30,  40,  25,  5,   0],
    7:  [19,  30,  40,  10,  1],
    8:  [18,  25,  32,  22,  3],
    9:  [10,  20,  25,  35,  10],
    10: [5,   10,  20,  40,  25],
}

# Level up XP costs
LEVEL_XP = {
    2: 2,
    3: 6,
    4: 10,
    5: 20,
    6: 36,
    7: 56,
    8: 80,
    9: 84,
    10: 100,
}

# Economy
REROLL_COST = 2
INTEREST_PER_10_GOLD = 1
MAX_INTEREST = 5

# Streak bonuses
STREAK_BONUS = {
    2: 1,
    3: 1,
    4: 2,
    5: 3,  # 5+ streak
}
```

## Expected Output

1. All model files created in `src/data/models/`
2. JSON data files with at least 30 champions, all traits, and all items
3. Loader functions working and tested
4. Constants file with accurate Set 16 values

## Data Sources

Use these URLs for reference:
- Champions: https://tftactics.gg/champions/
- Traits: https://tftactics.gg/db/origins/ and https://tftactics.gg/db/classes/
- Items: https://mobalytics.gg/tft/items
- Set Update: https://tftactics.gg/set-update/

## Priority

1. **Must have**: All 1-5 cost base champions (60 total)
2. **Should have**: Unlockable champions (40 total)
3. **Must have**: All origins and classes with breakpoints
4. **Must have**: All base components and common combined items
5. **Nice to have**: Radiant and Artifact items

Start by creating the data models, then populate with real data. Test loaders with simple queries.
