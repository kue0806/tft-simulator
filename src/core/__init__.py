# Core simulation modules
from .constants import (
    POOL_SIZE,
    SHOP_ODDS,
    LEVEL_XP,
    BOARD_SIZE,
    REROLL_COST,
    INTEREST_PER_10_GOLD,
    MAX_INTEREST,
    BASE_INCOME,
    STREAK_BONUS,
    get_streak_bonus,
    calculate_interest,
    COPIES_FOR_2_STAR,
    COPIES_FOR_3_STAR,
    STAR_MULTIPLIER,
    SHOP_SIZE,
    BENCH_SIZE,
)

from .champion_pool import ChampionPool
from .shop import Shop
from .player_units import ChampionInstance, PlayerUnits
from .probability import ProbabilityCalculator
from .game_state import PlayerState, GameState
from .synergy_calculator import SynergyCalculator, ActiveTrait, SynergyDelta
from .emblem_system import EmblemSystem
from .synergy_display import SynergyDisplay, SynergyFormatter
from .unique_traits import (
    UniqueTraitHandler,
    DemaciaHandler,
    IoniaHandler,
    NoxusHandler,
    VoidHandler,
    YordleHandler,
    BilgewaterHandler,
    UNIQUE_HANDLERS,
    get_handler,
)
from .item_manager import ItemManager, ItemInstance
from .stat_calculator import StatCalculator, CalculatedStats
from .bis_calculator import BiSCalculator, ItemRecommendation
from .economy import EconomyCalculator, EconomyState, IncomeBreakdown, RolldownCalculator
from .stage_manager import StageManager, RoundType, RoundInfo
from .economy_advisor import EconomyAdvisor, EconomyStrategy, EconomyAdvice

__all__ = [
    # Constants
    "POOL_SIZE",
    "SHOP_ODDS",
    "LEVEL_XP",
    "BOARD_SIZE",
    "REROLL_COST",
    "INTEREST_PER_10_GOLD",
    "MAX_INTEREST",
    "BASE_INCOME",
    "STREAK_BONUS",
    "get_streak_bonus",
    "calculate_interest",
    "COPIES_FOR_2_STAR",
    "COPIES_FOR_3_STAR",
    "STAR_MULTIPLIER",
    "SHOP_SIZE",
    "BENCH_SIZE",
    # Classes
    "ChampionPool",
    "Shop",
    "ChampionInstance",
    "PlayerUnits",
    "ProbabilityCalculator",
    "PlayerState",
    "GameState",
    # Synergy System
    "SynergyCalculator",
    "ActiveTrait",
    "SynergyDelta",
    "EmblemSystem",
    "SynergyDisplay",
    "SynergyFormatter",
    # Unique Trait Handlers
    "UniqueTraitHandler",
    "DemaciaHandler",
    "IoniaHandler",
    "NoxusHandler",
    "VoidHandler",
    "YordleHandler",
    "BilgewaterHandler",
    "UNIQUE_HANDLERS",
    "get_handler",
    # Item System
    "ItemManager",
    "ItemInstance",
    "StatCalculator",
    "CalculatedStats",
    "BiSCalculator",
    "ItemRecommendation",
    # Economy System
    "EconomyCalculator",
    "EconomyState",
    "IncomeBreakdown",
    "RolldownCalculator",
    "StageManager",
    "RoundType",
    "RoundInfo",
    "EconomyAdvisor",
    "EconomyStrategy",
    "EconomyAdvice",
]
