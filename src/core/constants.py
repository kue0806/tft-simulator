"""TFT Set 16 Game Constants."""

from typing import Final

# =============================================================================
# CHAMPION POOL SIZES (Set 16.1 - Lore & Legends)
# =============================================================================
# Number of copies of each champion available in the shared pool (8-player game)
# Source: https://www.esportstales.com/teamfight-tactics/champion-pool-size-and-draw-chances
POOL_SIZE: Final[dict[int, int]] = {
    1: 30,  # 30 copies of each 1-cost champion
    2: 25,  # 25 copies of each 2-cost champion
    3: 18,  # 18 copies of each 3-cost champion
    4: 10,  # 10 copies of each 4-cost champion
    5: 9,   # 9 copies of each 5-cost champion
    6: 9,   # 9 copies of each 6-cost unlockable (same as 5-cost)
    7: 9,   # 9 copies of each 7-cost unlockable (same as legendary)
}

# Number of unique champions at each cost tier (Set 16)
# Source: https://tftactics.gg/db/rolling/
CHAMPIONS_PER_TIER: Final[dict[int, int]] = {
    1: 22,  # 22 different 1-cost champions
    2: 20,  # 20 different 2-cost champions
    3: 17,  # 17 different 3-cost champions
    4: 10,  # 10 different 4-cost champions
    5: 9,   # 9 different 5-cost champions
}

# =============================================================================
# SHOP ODDS (Set 16.1 - Patch 16.1)
# =============================================================================
# Probability of each cost tier appearing in shop by player level
# Format: [1-cost%, 2-cost%, 3-cost%, 4-cost%, 5-cost%]
# Source: https://tftactics.gg/db/rolling/ and https://www.esportstales.com/
SHOP_ODDS: Final[dict[int, list[int]]] = {
    1:  [100, 0,   0,   0,   0],
    2:  [100, 0,   0,   0,   0],
    3:  [75,  25,  0,   0,   0],
    4:  [55,  30,  15,  0,   0],
    5:  [45,  33,  20,  2,   0],   # 4-cost starts at level 5
    6:  [30,  40,  25,  5,   0],
    7:  [19,  30,  40,  10,  1],   # 5-cost starts at level 7
    8:  [15,  20,  32,  30,  3],   # Set 16.1 updated odds
    9:  [10,  17,  25,  33,  15],  # Set 16.1 updated odds
    10: [10,  17,  25,  33,  15],  # Same as level 9 in Set 16.1
}

# =============================================================================
# LEVELING
# =============================================================================
# XP required to reach each level (cumulative from level 1)
LEVEL_XP: Final[dict[int, int]] = {
    2: 0,    # Starts at level 2 with 0 XP needed
    3: 2,    # 2 XP to reach level 3
    4: 6,    # 6 XP to reach level 4
    5: 10,   # 10 XP to reach level 5
    6: 20,   # 20 XP to reach level 6
    7: 36,   # 36 XP to reach level 7
    8: 60,   # 60 XP to reach level 8
    9: 68,   # 68 XP to reach level 9
    10: 68,  # 68 XP to reach level 10
}

# Passive XP per round (2 XP starting from 1-2)
PASSIVE_XP_PER_ROUND: Final[int] = 2

# Max board size at each level
BOARD_SIZE: Final[dict[int, int]] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
}

# =============================================================================
# ECONOMY
# =============================================================================
REROLL_COST: Final[int] = 2
INTEREST_PER_10_GOLD: Final[int] = 1
MAX_INTEREST: Final[int] = 5
BASE_INCOME: Final[int] = 5
WIN_GOLD: Final[int] = 1  # Gold for winning PvP round

# Passive gold income per round (different in early stages)
ROUND_PASSIVE_GOLD: Final[dict[str, int]] = {
    "1-1": 0,   # No gold at start
    "1-2": 2,   # +2 gold
    "1-3": 2,   # +2 gold
    "1-4": 3,   # +3 gold
    "2-1": 4,   # +4 gold
    # 2-2 onwards: BASE_INCOME (5 gold)
}

def get_round_passive_gold(stage: str) -> int:
    """Get passive gold income for a specific round."""
    if stage in ROUND_PASSIVE_GOLD:
        return ROUND_PASSIVE_GOLD[stage]
    return BASE_INCOME  # Default 5 gold after 2-1

# Streak bonuses (applies to both win and loss streaks)
STREAK_BONUS: Final[dict[int, int]] = {
    2: 1,   # 2-3 streak: +1 gold
    3: 1,
    4: 1,   # 4 streak: +1 gold (corrected)
    5: 2,   # 5 streak: +2 gold (corrected)
    6: 3,   # 6+ streak: +3 gold
}

def get_streak_bonus(streak: int) -> int:
    """Calculate bonus gold from win/loss streak."""
    if streak < 2:
        return 0
    if streak >= 6:
        return STREAK_BONUS[6]
    return STREAK_BONUS.get(streak, 0)

def calculate_interest(gold: int) -> int:
    """Calculate interest earned (1 gold per 10, max 5)."""
    return min(gold // 10, MAX_INTEREST)

# =============================================================================
# STAR LEVELS
# =============================================================================
# Copies needed to upgrade
COPIES_FOR_2_STAR: Final[int] = 3
COPIES_FOR_3_STAR: Final[int] = 9  # 3 x 2-star = 9 copies total

# Stat multipliers per star level
STAR_MULTIPLIER: Final[dict[int, dict[str, float]]] = {
    1: {"health": 1.0, "ad": 1.0, "ability": 1.0},
    2: {"health": 1.8, "ad": 1.5, "ability": 1.5},
    3: {"health": 3.24, "ad": 2.25, "ability": 2.25},
}

# =============================================================================
# COMBAT
# =============================================================================
BOARD_WIDTH: Final[int] = 7
BOARD_HEIGHT: Final[int] = 4

# Combat timing
TICK_RATE: Final[int] = 30  # Updates per second
TICK_DURATION: Final[float] = 1.0 / TICK_RATE
MAX_COMBAT_DURATION: Final[float] = 60.0  # 60 second timeout
BASE_MOVEMENT_SPEED: Final[float] = 550.0  # Base movement speed
MAX_ATTACK_SPEED: Final[float] = 5.0  # Max attacks per second

# Role-based mana per attack (TFT Set 15+ Roles Revamped)
# Assassin, Marksman, Fighter: 10 mana per attack
# Caster: 7 mana per attack
# Tank: 5 mana per attack
class UnitRole:
    TANK = "tank"
    FIGHTER = "fighter"
    MARKSMAN = "marksman"
    CASTER = "caster"
    ASSASSIN = "assassin"
    SPECIALIST = "specialist"

MANA_PER_ATTACK_BY_ROLE: Final[dict[str, int]] = {
    UnitRole.ASSASSIN: 10,
    UnitRole.MARKSMAN: 10,
    UnitRole.FIGHTER: 10,
    UnitRole.CASTER: 7,
    UnitRole.TANK: 5,
    UnitRole.SPECIALIST: 10,  # Default to fighter-like
}

# Default mana per attack (for units without defined role)
MANA_PER_ATTACK_DEFAULT: Final[int] = 10

# Mana gained from taking damage (based on pre-mitigation damage)
# Formula: min(42, damage * 0.07 + flat_mana)
MANA_ON_DAMAGE_PERCENT: Final[float] = 0.07  # 7% of pre-mitigation damage
MANA_ON_DAMAGE_FLAT: Final[float] = 0.0  # Flat mana per damage instance
MANA_ON_DAMAGE_CAP: Final[float] = 42.0  # Maximum mana per damage instance

# Critical Strike defaults
BASE_CRIT_CHANCE: Final[float] = 0.25  # 25% base crit chance
BASE_CRIT_DAMAGE: Final[float] = 1.40  # 140% crit damage (40% bonus)

# Targeting priority for tiebreaker (same distance)
# Lower number = higher priority target
TARGET_PRIORITY_BY_ROLE: Final[dict[str, int]] = {
    UnitRole.TANK: 1,       # Tanks targeted first
    UnitRole.FIGHTER: 2,
    UnitRole.MARKSMAN: 2,
    UnitRole.CASTER: 2,
    UnitRole.SPECIALIST: 2,
    UnitRole.ASSASSIN: 3,   # Assassins targeted last
}

# =============================================================================
# PLAYER DAMAGE (per combat loss)
# =============================================================================
# Base stage damage (taken regardless of surviving units)
BASE_STAGE_DAMAGE: Final[dict[int, int]] = {
    1: 0,
    2: 2,
    3: 6,
    4: 7,
    5: 10,
    6: 12,
    7: 17,
    8: 150,  # Stage 8+ (if exists)
}

# Unit damage by cost tier and star level
# Format: UNIT_DAMAGE[star_level][cost_tier]
UNIT_DAMAGE_BY_STAR_AND_COST: Final[dict[int, dict[int, int]]] = {
    1: {  # 1-star
        1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    },
    2: {  # 2-star
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8,
    },
    3: {  # 3-star
        1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10,
    },
}

def calculate_player_damage(stage: int, surviving_units: list[tuple[int, int]]) -> int:
    """
    Calculate damage dealt to losing player.

    Args:
        stage: Current stage number (1-7).
        surviving_units: List of (cost_tier, star_level) for surviving enemy units.

    Returns:
        Total damage dealt to player.
    """
    # Base stage damage
    base_damage = BASE_STAGE_DAMAGE.get(stage, BASE_STAGE_DAMAGE[7])

    # Sum unit damage
    unit_damage = 0
    for cost_tier, star_level in surviving_units:
        star_damage = UNIT_DAMAGE_BY_STAR_AND_COST.get(star_level, UNIT_DAMAGE_BY_STAR_AND_COST[1])
        unit_damage += star_damage.get(cost_tier, cost_tier)  # Fallback to cost tier

    return base_damage + unit_damage

# Legacy constants (kept for backward compatibility)
MANA_PER_ATTACK: Final[int] = 10
MANA_PER_DAMAGE_TAKEN: Final[float] = 0.07
PLAYER_DAMAGE_BASE: Final[int] = 0
PLAYER_DAMAGE_PER_UNIT: Final[dict[int, int]] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8,
}
STAGE_DAMAGE_BONUS: Final[dict[int, int]] = {
    1: 0, 2: 0, 3: 1, 4: 2, 5: 5, 6: 8, 7: 15,
}

# =============================================================================
# SHOP
# =============================================================================
SHOP_SIZE: Final[int] = 5  # Number of champions shown in shop
BENCH_SIZE: Final[int] = 9  # Number of bench slots

# =============================================================================
# STAGES AND ROUND TYPES
# =============================================================================
# PvE rounds - monster encounters
PVE_ROUNDS: Final[list[str]] = [
    "1-2", "1-3", "1-4",  # Stage 1 PvE (after opening carousel)
    "2-7",  # Krugs
    "3-7",  # Wolves
    "4-7",  # Raptors
    "5-7",  # Dragon/Rift Herald
    "6-7",  # Elder Dragon
    "7-7",  # Baron Nashor
]

# Carousel rounds - shared draft
CAROUSEL_ROUNDS: Final[list[str]] = [
    "1-1",  # Opening carousel (all players pick simultaneously, 1-cost only)
    "2-4",  # Stage 2 carousel
    "3-4",  # Stage 3 carousel
    "4-4",  # Stage 4 carousel
    "5-4",  # Stage 5 carousel (combined items)
    "6-4",  # Stage 6 carousel (combined items + components, 5-costs appear)
    "7-4",  # Stage 7 carousel
]

# Augment selection rounds
AUGMENT_ROUNDS: Final[list[str]] = [
    "2-1",  # First augment choice (43 seconds)
    "3-2",  # Second augment choice (58 seconds)
    "4-2",  # Third augment choice (58 seconds)
]

# Augment timer in seconds
AUGMENT_TIMER: Final[dict[str, int]] = {
    "2-1": 43,
    "3-2": 58,
    "4-2": 58,
}

# PvE Monster Types by round
PVE_MONSTERS: Final[dict[str, str]] = {
    "1-2": "minions",      # Easy minions
    "1-3": "minions",      # Easy minions
    "1-4": "minions",      # Easy minions
    "2-7": "krugs",        # Krugs
    "3-7": "wolves",       # Wolves
    "4-7": "raptors",      # Raptors (last component drops)
    "5-7": "rift_herald",  # Dragon/Rift Herald
    "6-7": "elder_dragon", # Elder Dragon
    "7-7": "baron_nashor", # Baron Nashor
}

# =============================================================================
# SET 16 SPECIFIC
# =============================================================================
SET_NUMBER: Final[int] = 16
SET_NAME: Final[str] = "Lore & Legends"

# Unlockable champion costs that can appear in shop after unlocking
UNLOCKABLE_COSTS: Final[list[int]] = [6, 7]

# Maximum trait breakpoints for emblem consideration
MAX_TRAIT_UNITS: Final[int] = 9
