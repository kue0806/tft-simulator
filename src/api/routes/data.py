"""
Static data API routes.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

from src.data.loaders import load_champions, load_traits, load_items

router = APIRouter()


# === Champions ===


@router.get("/champions")
async def get_all_champions(cost: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all champions, optionally filtered by cost."""
    champions = load_champions()
    if cost is not None:
        champions = [c for c in champions if c.cost == cost]
    return [c.model_dump() for c in champions]


@router.get("/champions/{champion_id}")
async def get_champion(champion_id: str) -> Dict[str, Any]:
    """Get specific champion by ID."""
    champions = load_champions()
    for champ in champions:
        if champ.id == champion_id:
            return champ.model_dump()
    raise HTTPException(status_code=404, detail="Champion not found")


@router.get("/champions/by-trait/{trait_id}")
async def get_champions_by_trait(trait_id: str) -> List[Dict[str, Any]]:
    """Get champions with specific trait."""
    champions = load_champions()
    filtered = [c for c in champions if trait_id in c.traits]
    return [c.model_dump() for c in filtered]


# === Traits ===


@router.get("/traits")
async def get_all_traits() -> List[Dict[str, Any]]:
    """Get all traits."""
    traits = load_traits()
    return [t.model_dump() for t in traits]


@router.get("/traits/{trait_id}")
async def get_trait(trait_id: str) -> Dict[str, Any]:
    """Get specific trait by ID."""
    traits = load_traits()
    for trait in traits:
        if trait.id == trait_id:
            return trait.model_dump()
    raise HTTPException(status_code=404, detail="Trait not found")


@router.get("/traits/origins")
async def get_origins() -> List[Dict[str, Any]]:
    """Get origin traits."""
    traits = load_traits()
    origins = [t for t in traits if t.type == "origin"]
    return [t.model_dump() for t in origins]


@router.get("/traits/classes")
async def get_classes() -> List[Dict[str, Any]]:
    """Get class traits."""
    traits = load_traits()
    classes = [t for t in traits if t.type == "class"]
    return [t.model_dump() for t in classes]


# === Items ===


@router.get("/items")
async def get_all_items() -> List[Dict[str, Any]]:
    """Get all items."""
    items = load_items()
    return [i.model_dump() for i in items]


@router.get("/items/{item_id}")
async def get_item(item_id: str) -> Dict[str, Any]:
    """Get specific item by ID."""
    items = load_items()
    for item in items:
        if item.id == item_id:
            return item.model_dump()
    raise HTTPException(status_code=404, detail="Item not found")


@router.get("/items/components")
async def get_components() -> List[Dict[str, Any]]:
    """Get base component items."""
    items = load_items()
    components = [i for i in items if i.is_component]
    return [i.model_dump() for i in components]


@router.get("/items/combined")
async def get_combined_items() -> List[Dict[str, Any]]:
    """Get combined items."""
    items = load_items()
    combined = [i for i in items if not i.is_component and i.recipe]
    return [i.model_dump() for i in combined]


@router.get("/items/recipe/{item_id}")
async def get_recipe(item_id: str) -> Dict[str, Any]:
    """Get item recipe."""
    items = load_items()
    for item in items:
        if item.id == item_id:
            if item.recipe:
                return {"item_id": item_id, "components": item.recipe}
            raise HTTPException(status_code=404, detail="Recipe not found")
    raise HTTPException(status_code=404, detail="Item not found")


# === Constants ===


@router.get("/constants/shop-odds")
async def get_shop_odds() -> Dict[int, Dict[int, float]]:
    """Get shop odds by level."""
    return {
        1: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        2: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        3: {1: 0.75, 2: 0.25, 3: 0.0, 4: 0.0, 5: 0.0},
        4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.0, 5: 0.0},
        5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.0},
        6: {1: 0.30, 2: 0.40, 3: 0.25, 4: 0.05, 5: 0.0},
        7: {1: 0.19, 2: 0.35, 3: 0.35, 4: 0.10, 5: 0.01},
        8: {1: 0.18, 2: 0.25, 3: 0.32, 4: 0.22, 5: 0.03},
        9: {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.35, 5: 0.10},
        10: {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.40, 5: 0.25},
    }


@router.get("/constants/pool-sizes")
async def get_pool_sizes() -> Dict[int, int]:
    """Get champion pool sizes by cost."""
    return {
        1: 30,
        2: 25,
        3: 18,
        4: 12,
        5: 10,
    }


@router.get("/constants/level-costs")
async def get_level_costs() -> Dict[int, int]:
    """Get XP required per level."""
    return {
        1: 2,
        2: 2,
        3: 6,
        4: 10,
        5: 20,
        6: 36,
        7: 56,
        8: 80,
        9: 100,
    }
