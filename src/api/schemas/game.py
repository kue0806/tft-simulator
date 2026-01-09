"""
Game-related API schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# === Request Schemas ===


class CreateGameRequest(BaseModel):
    """Game creation request."""

    player_count: int = Field(default=8, ge=2, le=8)


class PlayerSetupRequest(BaseModel):
    """Player setup request."""

    level: int = Field(default=1, ge=1, le=10)
    gold: int = Field(default=0, ge=0)
    hp: int = Field(default=100, ge=0, le=100)


class PlaceUnitRequest(BaseModel):
    """Unit placement request."""

    champion_id: str
    position: Dict[str, int]  # {"row": 0, "col": 3}


class MoveUnitRequest(BaseModel):
    """Unit move request."""

    unit_id: str
    new_position: Dict[str, int]


class EquipItemRequest(BaseModel):
    """Item equip request."""

    unit_id: str
    item_id: str


# === Response Schemas ===


class ChampionInstanceSchema(BaseModel):
    """Champion instance schema."""

    id: str
    champion_id: str
    name: str
    cost: int
    star_level: int
    items: List[str]
    traits: List[str]
    position: Optional[Dict[str, int]] = None

    class Config:
        from_attributes = True


class PlayerStateSchema(BaseModel):
    """Player state schema."""

    player_id: int
    hp: int
    gold: int
    level: int
    xp: int
    streak: int
    board: List[ChampionInstanceSchema]
    bench: List[Optional[ChampionInstanceSchema]]

    class Config:
        from_attributes = True


class GameStateSchema(BaseModel):
    """Game state schema."""

    game_id: str
    stage: str
    round_type: str
    players: List[PlayerStateSchema]

    class Config:
        from_attributes = True


class SynergySchema(BaseModel):
    """Synergy info schema."""

    trait_id: str
    name: str
    count: int
    breakpoints: List[int]
    active_breakpoint: Optional[int] = None
    is_active: bool
    style: str  # bronze, silver, gold, chromatic


class PlayerSynergiesResponse(BaseModel):
    """Player synergies response."""

    synergies: List[SynergySchema]
    total_active: int


class ShopSlotSchema(BaseModel):
    """Shop slot schema."""

    index: int
    champion_id: Optional[str] = None
    champion_name: Optional[str] = None
    cost: Optional[int] = None
    is_purchased: bool


class ShopStateSchema(BaseModel):
    """Shop state schema."""

    slots: List[ShopSlotSchema]
    is_locked: bool
    refresh_cost: int
