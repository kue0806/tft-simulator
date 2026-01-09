"""
Optimizer-related API schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# === Pick Advisor ===


class PickReasonEnum(str, Enum):
    """Pick reason enum."""

    UPGRADE_2STAR = "upgrade_2star"
    UPGRADE_3STAR = "upgrade_3star"
    SYNERGY_ACTIVATE = "synergy_activate"
    SYNERGY_UPGRADE = "synergy_upgrade"
    CORE_CARRY = "core_carry"
    STRONG_UNIT = "strong_unit"
    ECONOMY_PAIR = "economy_pair"


class PickRecommendationSchema(BaseModel):
    """Pick recommendation schema."""

    champion_id: str
    champion_name: str
    shop_index: int
    score: float
    reasons: List[PickReasonEnum]
    cost: int
    copies_owned: int
    copies_needed: int


class PickAdviceRequest(BaseModel):
    """Pick advice request."""

    player_id: int
    target_comp: Optional[List[str]] = None


class PickAdviceResponse(BaseModel):
    """Pick advice response."""

    recommendations: List[PickRecommendationSchema]
    should_refresh: bool
    refresh_reason: Optional[str] = None
    gold_to_save: int


# === Rolldown Planner ===


class RolldownStrategyEnum(str, Enum):
    """Rolldown strategy enum."""

    FAST_8 = "fast_8"
    FAST_9 = "fast_9"
    SLOW_ROLL_6 = "slow_roll_6"
    SLOW_ROLL_7 = "slow_roll_7"
    SLOW_ROLL_8 = "slow_roll_8"
    ALL_IN = "all_in"
    SAVE = "save"


class RolldownPlanRequest(BaseModel):
    """Rolldown plan request."""

    player_id: int
    target_units: List[str]
    target_stars: Optional[Dict[str, int]] = None


class RolldownPlanResponse(BaseModel):
    """Rolldown plan response."""

    strategy: RolldownStrategyEnum
    current_phase: str
    is_rolldown_now: bool
    roll_budget: int
    level_budget: int
    save_amount: int
    hit_probability: float
    expected_rolls: int
    advice: List[str]


# === Comp Builder ===


class CompStyleEnum(str, Enum):
    """Composition style enum."""

    REROLL = "reroll"
    STANDARD = "standard"
    FAST_9 = "fast_9"
    FLEX = "flex"


class CompTemplateSchema(BaseModel):
    """Composition template schema."""

    name: str
    style: CompStyleEnum
    core_units: List[str]
    carry: str
    tier: str
    difficulty: str
    description: str


class CompRecommendationSchema(BaseModel):
    """Composition recommendation schema."""

    template: CompTemplateSchema
    match_score: float
    missing_units: List[str]
    current_units: List[str]
    transition_cost: int


class CompRecommendRequest(BaseModel):
    """Composition recommendation request."""

    player_id: int
    style_filter: Optional[CompStyleEnum] = None
    top_n: int = Field(default=3, ge=1, le=10)


# === Pivot Analyzer ===


class PivotReasonEnum(str, Enum):
    """Pivot reason enum."""

    CONTESTED = "contested"
    LOW_ROLLS = "low_rolls"
    HP_CRITICAL = "hp_critical"
    BETTER_ITEMS = "better_items"
    HIGHROLL = "highroll"


class PivotOptionSchema(BaseModel):
    """Pivot option schema."""

    target_comp_name: str
    shared_units: List[str]
    units_to_sell: List[str]
    units_to_buy: List[str]
    total_cost: int
    success_probability: float
    risk_level: str


class PivotAdviceRequest(BaseModel):
    """Pivot advice request."""

    player_id: int
    current_comp_name: Optional[str] = None
    contested_units: Optional[List[str]] = None


class PivotAdviceResponse(BaseModel):
    """Pivot advice response."""

    should_pivot: bool
    urgency: str
    current_comp_health: float
    options: List[PivotOptionSchema]
    explanation: str


# === Board Optimizer ===


class PositionSchema(BaseModel):
    """Position schema."""

    row: int
    col: int


class PositionRecommendationSchema(BaseModel):
    """Position recommendation schema."""

    unit_id: str
    position: PositionSchema
    score: float
    reasons: List[str]


class BoardLayoutSchema(BaseModel):
    """Board layout schema."""

    positions: Dict[str, PositionSchema]
    total_score: float
    win_rate: float
    description: str


class OptimizeBoardRequest(BaseModel):
    """Board optimization request."""

    player_id: int
    iterations: int = Field(default=100, ge=10, le=500)


class OptimizeBoardResponse(BaseModel):
    """Board optimization response."""

    layout: BoardLayoutSchema
    unit_recommendations: List[PositionRecommendationSchema]
