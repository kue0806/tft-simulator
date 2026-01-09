"""
Combat simulation API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from ..schemas.combat import (
    SimulateCombatRequest,
    SimulationResultSchema,
    QuickCombatRequest,
    CombatResultSchema,
)
from ..services.combat_service import CombatService
from ..dependencies import get_combat_service

router = APIRouter()


@router.post("/simulate", response_model=SimulationResultSchema)
async def simulate_combat(
    request: SimulateCombatRequest,
    service: CombatService = Depends(get_combat_service),
):
    """
    Simulate combat between custom teams.

    Run N iterations and return aggregate statistics.
    """
    result = service.simulate(
        team_blue=request.team_blue,
        team_red=request.team_red,
        iterations=request.iterations,
    )
    return result


@router.post("/quick/{game_id}", response_model=SimulationResultSchema)
async def quick_combat(
    game_id: str,
    request: QuickCombatRequest,
    service: CombatService = Depends(get_combat_service),
):
    """
    Quick combat simulation between game players.

    Uses current game state for both players.
    """
    try:
        result = service.simulate_players(
            game_id=game_id,
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            iterations=request.iterations,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/single", response_model=CombatResultSchema)
async def single_combat(
    request: SimulateCombatRequest,
    service: CombatService = Depends(get_combat_service),
):
    """Run a single combat (1 iteration)."""
    service.simulate(
        team_blue=request.team_blue,
        team_red=request.team_red,
        iterations=1,
    )
    result = service.get_last_combat_result()
    if not result:
        raise HTTPException(status_code=500, detail="Combat failed")
    return result


@router.get("/stats/{game_id}/player/{player_id}")
async def get_combat_stats(
    game_id: str,
    player_id: int,
    service: CombatService = Depends(get_combat_service),
) -> Dict[str, Any]:
    """Get player combat statistics."""
    stats = service.get_player_combat_stats(game_id, player_id)
    return stats
