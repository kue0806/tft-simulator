"""
Optimizer recommendation API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..schemas.optimizer import (
    PickAdviceRequest,
    PickAdviceResponse,
    RolldownPlanRequest,
    RolldownPlanResponse,
    CompRecommendRequest,
    CompRecommendationSchema,
    CompTemplateSchema,
    PivotAdviceRequest,
    PivotAdviceResponse,
    OptimizeBoardRequest,
    OptimizeBoardResponse,
    PositionRecommendationSchema,
)
from ..services.optimizer_service import OptimizerService
from ..dependencies import get_optimizer_service

router = APIRouter()


# === Pick Advisor ===


@router.post("/pick/{game_id}", response_model=PickAdviceResponse)
async def get_pick_advice(
    game_id: str,
    request: PickAdviceRequest,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Get shop purchase recommendations."""
    try:
        advice = service.get_pick_advice(
            game_id=game_id,
            player_id=request.player_id,
            target_comp=request.target_comp,
        )
        return advice
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Rolldown Planner ===


@router.post("/rolldown/{game_id}", response_model=RolldownPlanResponse)
async def get_rolldown_plan(
    game_id: str,
    request: RolldownPlanRequest,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Get rolldown strategy plan."""
    try:
        plan = service.get_rolldown_plan(
            game_id=game_id,
            player_id=request.player_id,
            target_units=request.target_units,
            target_stars=request.target_stars,
        )
        return plan
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Comp Builder ===


@router.post("/comp/{game_id}", response_model=List[CompRecommendationSchema])
async def get_comp_recommendations(
    game_id: str,
    request: CompRecommendRequest,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Get composition recommendations."""
    try:
        recommendations = service.get_comp_recommendations(
            game_id=game_id,
            player_id=request.player_id,
            style_filter=request.style_filter,
            top_n=request.top_n,
        )
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/comp/templates", response_model=List[CompTemplateSchema])
async def get_comp_templates(
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Get all composition templates."""
    return service.get_all_templates()


# === Pivot Analyzer ===


@router.post("/pivot/{game_id}", response_model=PivotAdviceResponse)
async def get_pivot_advice(
    game_id: str,
    request: PivotAdviceRequest,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Get pivot analysis."""
    try:
        advice = service.get_pivot_advice(
            game_id=game_id,
            player_id=request.player_id,
            current_comp_name=request.current_comp_name,
            contested_units=request.contested_units,
        )
        return advice
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Board Optimizer ===


@router.post("/board/{game_id}", response_model=OptimizeBoardResponse)
async def optimize_board(
    game_id: str,
    request: OptimizeBoardRequest,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Optimize board positioning."""
    try:
        result = service.optimize_board(
            game_id=game_id,
            player_id=request.player_id,
            iterations=request.iterations,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/board/{game_id}/player/{player_id}/suggest/{unit_id}",
    response_model=List[PositionRecommendationSchema],
)
async def suggest_position(
    game_id: str,
    player_id: int,
    unit_id: str,
    service: OptimizerService = Depends(get_optimizer_service),
):
    """Suggest positions for a specific unit."""
    try:
        suggestions = service.suggest_unit_position(
            game_id=game_id,
            player_id=player_id,
            unit_id=unit_id,
        )
        return suggestions
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
