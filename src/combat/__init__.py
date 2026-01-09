"""Combat simulation module for TFT.

This module provides a complete combat simulation system including:
- Hex grid positioning and movement
- Unit targeting and pathfinding
- Attack and ability execution
- Status effects and crowd control
- Monte Carlo win rate simulation
"""

# Hex Grid
from .hex_grid import HexPosition, HexGrid, Team

# Combat Units
from .combat_unit import CombatUnit, CombatStats, CombatResult, UnitState

# Targeting
from .targeting import (
    TargetSelector,
    TargetingContext,
    TargetingPriority,
)

# Pathfinding
from .pathfinding import (
    PathFinder,
    PathNode,
    get_blocked_positions,
    get_walkable_neighbors,
)

# Movement
from .movement import (
    MovementSystem,
    MovementState,
    MOVE_TIME_PER_HEX,
)

# Attack
from .attack import (
    AttackSystem,
    AttackResult,
    DamageEvent,
    DamageType,
    MANA_ON_ATTACK,
    MANA_ON_DAMAGE_TAKEN,
    calculate_effective_hp,
    calculate_dps,
    get_mana_per_attack,
    get_mana_on_damage,
)

# Abilities
from .ability import (
    AbilitySystem,
    AbilityData,
    AbilityCast,
    AbilityResult,
    AbilityTargetType,
    create_simple_damage_ability,
)

# Status Effects
from .status_effects import (
    StatusEffectSystem,
    StatusEffect,
    StatusEffectType,
    create_stun,
    create_burn,
    create_shield,
    create_attack_speed_buff,
    create_grievous_wounds,
)

# Combat Engine
from .combat_engine import (
    CombatEngine,
    CombatState,
    CombatPhase,
    TICK_RATE,
    TICK_DURATION,
    MAX_COMBAT_TICKS,
)

# Import player damage constants from core
from src.core.constants import (
    BASE_STAGE_DAMAGE,
    UNIT_DAMAGE_BY_STAR_AND_COST,
    calculate_player_damage,
)

# Legacy alias
PLAYER_DAMAGE_BASE = 0  # Deprecated, use BASE_STAGE_DAMAGE instead

# Simulation
from .simulation import (
    CombatSimulator,
    SimulationResult,
    PositioningAnalysis,
    quick_simulate,
    estimate_board_strength,
)

__all__ = [
    # Hex Grid
    "HexPosition",
    "HexGrid",
    "Team",
    # Combat Unit
    "CombatUnit",
    "CombatStats",
    "CombatResult",
    "UnitState",
    # Targeting
    "TargetSelector",
    "TargetingContext",
    "TargetingPriority",
    # Pathfinding
    "PathFinder",
    "PathNode",
    "get_blocked_positions",
    "get_walkable_neighbors",
    # Movement
    "MovementSystem",
    "MovementState",
    "MOVE_TIME_PER_HEX",
    # Attack
    "AttackSystem",
    "AttackResult",
    "DamageEvent",
    "DamageType",
    "MANA_ON_ATTACK",
    "MANA_ON_DAMAGE_TAKEN",
    "calculate_effective_hp",
    "calculate_dps",
    "get_mana_per_attack",
    "get_mana_on_damage",
    # Abilities
    "AbilitySystem",
    "AbilityData",
    "AbilityCast",
    "AbilityResult",
    "AbilityTargetType",
    "create_simple_damage_ability",
    # Status Effects
    "StatusEffectSystem",
    "StatusEffect",
    "StatusEffectType",
    "create_stun",
    "create_burn",
    "create_shield",
    "create_attack_speed_buff",
    "create_grievous_wounds",
    # Combat Engine
    "CombatEngine",
    "CombatState",
    "CombatPhase",
    "TICK_RATE",
    "TICK_DURATION",
    "MAX_COMBAT_TICKS",
    "PLAYER_DAMAGE_BASE",  # Deprecated alias
    "BASE_STAGE_DAMAGE",
    "UNIT_DAMAGE_BY_STAR_AND_COST",
    "calculate_player_damage",
    # Simulation
    "CombatSimulator",
    "SimulationResult",
    "PositioningAnalysis",
    "quick_simulate",
    "estimate_board_strength",
]
