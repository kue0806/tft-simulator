"""
Game state management API routes.
"""

from fastapi import APIRouter, HTTPException, Depends

from ..schemas.game import (
    CreateGameRequest,
    PlayerSetupRequest,
    PlaceUnitRequest,
    MoveUnitRequest,
    EquipItemRequest,
    GameStateSchema,
    PlayerStateSchema,
    PlayerSynergiesResponse,
)
from ..schemas.common import BaseResponse
from ..services.game_service import GameService
from ..dependencies import get_game_service

router = APIRouter()


@router.post("/create", response_model=GameStateSchema)
async def create_game(
    request: CreateGameRequest,
    service: GameService = Depends(get_game_service),
):
    """Create a new game."""
    game = service.create_game(request.player_count)
    return game


@router.get("/{game_id}", response_model=GameStateSchema)
async def get_game(
    game_id: str,
    service: GameService = Depends(get_game_service),
):
    """Get game state."""
    game = service.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@router.get("/{game_id}/player/{player_id}", response_model=PlayerStateSchema)
async def get_player(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Get player state."""
    player = service.get_player(game_id, player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


@router.post("/{game_id}/player/{player_id}/setup", response_model=PlayerStateSchema)
async def setup_player(
    game_id: str,
    player_id: int,
    request: PlayerSetupRequest,
    service: GameService = Depends(get_game_service),
):
    """Setup player."""
    try:
        player = service.setup_player(game_id, player_id, request)
        return player
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{game_id}/player/{player_id}/place", response_model=BaseResponse)
async def place_unit(
    game_id: str,
    player_id: int,
    request: PlaceUnitRequest,
    service: GameService = Depends(get_game_service),
):
    """Place unit on board."""
    success = service.place_unit(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to place unit")
    return BaseResponse(message="Unit placed successfully")


@router.post("/{game_id}/player/{player_id}/move", response_model=BaseResponse)
async def move_unit(
    game_id: str,
    player_id: int,
    request: MoveUnitRequest,
    service: GameService = Depends(get_game_service),
):
    """Move unit."""
    success = service.move_unit(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to move unit")
    return BaseResponse(message="Unit moved successfully")


@router.post("/{game_id}/player/{player_id}/equip", response_model=BaseResponse)
async def equip_item(
    game_id: str,
    player_id: int,
    request: EquipItemRequest,
    service: GameService = Depends(get_game_service),
):
    """Equip item."""
    success = service.equip_item(game_id, player_id, request)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to equip item")
    return BaseResponse(message="Item equipped successfully")


@router.get(
    "/{game_id}/player/{player_id}/synergies", response_model=PlayerSynergiesResponse
)
async def get_synergies(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Get player synergies."""
    synergies = service.get_player_synergies(game_id, player_id)
    return synergies


@router.post("/{game_id}/next-round", response_model=GameStateSchema)
async def next_round(
    game_id: str,
    service: GameService = Depends(get_game_service),
):
    """Advance to next round."""
    try:
        game = service.advance_round(game_id)
        return game
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{game_id}", response_model=BaseResponse)
async def delete_game(
    game_id: str,
    service: GameService = Depends(get_game_service),
):
    """Delete game."""
    service.delete_game(game_id)
    return BaseResponse(message="Game deleted")
