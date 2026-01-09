"""
Combat-related API schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class TeamSetup(BaseModel):
    """Team setup schema."""

    units: List[Dict[str, Any]]  # [{champion_id, star_level, items, position}]


class SimulateCombatRequest(BaseModel):
    """Combat simulation request."""

    team_blue: TeamSetup
    team_red: TeamSetup
    iterations: int = Field(default=100, ge=1, le=1000)


class UnitCombatStats(BaseModel):
    """Unit combat statistics."""

    unit_id: str
    champion_name: str
    damage_dealt: float
    damage_taken: float
    healing_done: float
    kills: int
    survived: bool


class CombatResultSchema(BaseModel):
    """Combat result schema."""

    winner: str  # "blue" or "red"
    units_remaining: int
    damage_to_loser: int
    rounds_taken: int
    unit_stats: List[UnitCombatStats]


class SimulationResultSchema(BaseModel):
    """Simulation result schema."""

    iterations: int
    blue_wins: int
    red_wins: int
    blue_win_rate: float
    average_rounds: float
    average_damage: float

    # Detailed stats
    blue_unit_stats: List[Dict[str, float]]
    red_unit_stats: List[Dict[str, float]]


class QuickCombatRequest(BaseModel):
    """Quick combat request (game state based)."""

    player_id: int
    opponent_id: int
    iterations: int = Field(default=50, ge=1, le=500)
