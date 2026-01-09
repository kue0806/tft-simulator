"""Item data model for TFT Set 16."""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class ItemType(StrEnum):
    """Item classification."""
    COMPONENT = "component"
    COMBINED = "combined"
    RADIANT = "radiant"
    ARTIFACT = "artifact"
    EMBLEM = "emblem"
    SUPPORT = "support"
    CONSUMABLE = "consumable"


class ItemStats(BaseModel):
    """Item stat bonuses."""
    ad: int = Field(default=0, description="Attack Damage")
    ap: int = Field(default=0, description="Ability Power")
    armor: int = Field(default=0, description="Armor")
    mr: int = Field(default=0, description="Magic Resist")
    health: int = Field(default=0, description="Health")
    mana: int = Field(default=0, description="Mana")
    attack_speed: float = Field(default=0.0, description="Attack Speed %")
    crit_chance: float = Field(default=0.0, description="Crit Chance %")
    crit_damage: float = Field(default=0.0, description="Crit Damage %")
    omnivamp: float = Field(default=0.0, description="Omnivamp %")
    durability: float = Field(default=0.0, description="Damage reduction %")


class Item(BaseModel):
    """TFT Item model."""
    id: str = Field(..., description="Unique identifier (lowercase, underscores)")
    name: str = Field(..., description="Display name")
    type: ItemType
    stats: ItemStats = Field(default_factory=ItemStats)
    effect: Optional[str] = Field(default=None, description="Special effect description")
    components: Optional[tuple[str, str]] = Field(default=None, description="Component IDs for combined items")
    grants_trait: Optional[str] = Field(default=None, description="Trait ID for emblems")
    is_unique: bool = Field(default=False, description="Can only equip one of this item")
    is_radiant: bool = Field(default=False, description="Radiant version of an item")

    model_config = {"use_enum_values": True}

    @property
    def is_component(self) -> bool:
        return self.type == ItemType.COMPONENT

    @property
    def is_combined(self) -> bool:
        return self.type == ItemType.COMBINED

    @property
    def is_emblem(self) -> bool:
        return self.type == ItemType.EMBLEM
