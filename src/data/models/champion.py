"""Champion data model for TFT Set 16."""

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


class ChampionCost(IntEnum):
    """Champion cost tiers."""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6    # Unlockable only
    SEVEN = 7  # Unlockable only (e.g., Baron Nashor)


class ChampionStats(BaseModel):
    """Champion base statistics."""
    health: list[int] = Field(..., min_length=3, max_length=3, description="HP at [1-star, 2-star, 3-star]")
    mana: tuple[int, int] = Field(..., description="(starting mana, max mana)")
    armor: int = Field(..., ge=0)
    magic_resist: int = Field(..., ge=0)
    attack_damage: list[int] = Field(..., min_length=3, max_length=3, description="AD at [1-star, 2-star, 3-star]")
    attack_speed: float = Field(..., gt=0)
    attack_range: int = Field(..., ge=1, le=5, description="Attack range in hexes")
    crit_chance: float = Field(default=0.25, description="Base crit chance")
    crit_damage: float = Field(default=1.4, description="Base crit damage multiplier")


class AbilityScaling(BaseModel):
    """Ability damage/effect scaling per star level."""
    values: list[float] = Field(..., min_length=3, max_length=3, description="Values at [1-star, 2-star, 3-star]")
    stat: str = Field(default="damage", description="What this scaling represents")


class Ability(BaseModel):
    """Champion ability details."""
    name: str
    description: str
    mana_cost: Optional[int] = None
    damage_type: Optional[str] = Field(default=None, description="physical, magic, or true")
    scalings: list[AbilityScaling] = Field(default_factory=list)


class Champion(BaseModel):
    """TFT Champion model."""
    id: str = Field(..., description="Unique identifier (lowercase, no spaces)")
    name: str = Field(..., description="Display name")
    cost: ChampionCost
    traits: list[str] = Field(..., min_length=1, description="List of trait IDs")
    stats: ChampionStats
    ability: Ability
    is_unlockable: bool = Field(default=False, description="Whether this champion requires unlocking")
    unlock_condition: Optional[str] = Field(default=None, description="Condition to unlock this champion")
    unlock_type: Optional[str] = Field(default=None, description="Type of unlock condition")
    unlock_params: Optional[dict] = Field(default=None, description="Parameters for unlock condition")
    slots: int = Field(default=1, description="Board slots this unit takes (2 for Baron Nashor)")

    model_config = {"use_enum_values": True}

    @property
    def health_1star(self) -> int:
        return self.stats.health[0]

    @property
    def health_2star(self) -> int:
        return self.stats.health[1]

    @property
    def health_3star(self) -> int:
        return self.stats.health[2]
