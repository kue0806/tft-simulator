"""
Shop API routes.
"""

from fastapi import APIRouter, HTTPException, Depends

from ..schemas.game import ShopStateSchema
from ..schemas.common import BaseResponse
from ..services.game_service import GameService
from ..dependencies import get_game_service

router = APIRouter()


@router.get("/{game_id}/player/{player_id}", response_model=ShopStateSchema)
async def get_shop(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Get shop state."""
    shop = service.get_shop(game_id, player_id)
    return shop


@router.post("/{game_id}/player/{player_id}/refresh", response_model=ShopStateSchema)
async def refresh_shop(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Refresh shop (costs 2 gold)."""
    try:
        shop = service.refresh_shop(game_id, player_id)
        return shop
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/{game_id}/player/{player_id}/buy/{slot_index}", response_model=BaseResponse
)
async def buy_champion(
    game_id: str,
    player_id: int,
    slot_index: int,
    service: GameService = Depends(get_game_service),
):
    """Buy champion from shop slot."""
    success = service.buy_champion(game_id, player_id, slot_index)
    if not success:
        raise HTTPException(status_code=400, detail="Purchase failed")
    return BaseResponse(message="Champion purchased")


@router.post(
    "/{game_id}/player/{player_id}/sell/{unit_id}", response_model=BaseResponse
)
async def sell_unit(
    game_id: str,
    player_id: int,
    unit_id: str,
    service: GameService = Depends(get_game_service),
):
    """Sell unit."""
    success = service.sell_unit(game_id, player_id, unit_id)
    if not success:
        raise HTTPException(status_code=400, detail="Sell failed")
    return BaseResponse(message="Unit sold")


@router.post("/{game_id}/player/{player_id}/lock", response_model=BaseResponse)
async def toggle_lock(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Toggle shop lock."""
    is_locked = service.toggle_shop_lock(game_id, player_id)
    return BaseResponse(message=f"Shop {'locked' if is_locked else 'unlocked'}")


@router.post("/{game_id}/player/{player_id}/levelup", response_model=BaseResponse)
async def buy_xp(
    game_id: str,
    player_id: int,
    service: GameService = Depends(get_game_service),
):
    """Buy XP (4 gold = 4 XP)."""
    success = service.buy_xp(game_id, player_id)
    if not success:
        raise HTTPException(status_code=400, detail="Not enough gold")
    return BaseResponse(message="XP purchased")
