"""Trait data loader for TFT Set 16."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..models.trait import Trait, TraitBreakpoint, TraitType


# Get the data directory path
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
ORIGINS_FILE = DATA_DIR / "traits" / "origins.json"
CLASSES_FILE = DATA_DIR / "traits" / "classes.json"


def _parse_trait(trait_data: dict, trait_type: TraitType) -> Trait:
    """Parse a trait from JSON data.

    Args:
        trait_data: Dictionary containing trait data.
        trait_type: The type of trait (origin or class).

    Returns:
        Trait object.
    """
    breakpoints = [
        TraitBreakpoint(
            count=bp["count"],
            effect=bp["effect"],
            stats=bp.get("stats", {}),
        )
        for bp in trait_data["breakpoints"]
    ]

    return Trait(
        id=trait_data["id"],
        name=trait_data["name"],
        type=trait_type,
        description=trait_data.get("description", ""),
        breakpoints=breakpoints,
        champions=trait_data.get("champions", []),
        mechanic=trait_data.get("mechanic"),
    )


@lru_cache(maxsize=1)
def load_origins() -> list[Trait]:
    """Load all origin traits from JSON file.

    Returns:
        List of origin Trait objects.
    """
    with open(ORIGINS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [_parse_trait(t, TraitType.ORIGIN) for t in data["origins"]]


@lru_cache(maxsize=1)
def load_classes() -> list[Trait]:
    """Load all class traits from JSON file.

    Returns:
        List of class Trait objects.
    """
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [_parse_trait(t, TraitType.CLASS) for t in data["classes"]]


def load_traits() -> list[Trait]:
    """Load all traits (both origins and classes).

    Returns:
        List of all Trait objects.
    """
    return load_origins() + load_classes()


def get_trait_by_id(trait_id: str) -> Optional[Trait]:
    """Get a trait by its ID.

    Args:
        trait_id: The unique trait identifier.

    Returns:
        Trait object if found, None otherwise.
    """
    traits = load_traits()
    for trait in traits:
        if trait.id == trait_id:
            return trait
    return None


def get_origins() -> list[Trait]:
    """Get all origin traits.

    Returns:
        List of origin traits.
    """
    return load_origins()


def get_classes() -> list[Trait]:
    """Get all class traits.

    Returns:
        List of class traits.
    """
    return load_classes()


def get_unique_traits() -> list[Trait]:
    """Get all unique (single-champion) traits.

    Returns:
        List of unique traits.
    """
    traits = load_traits()
    return [t for t in traits if len(t.breakpoints) == 1 and t.breakpoints[0].count == 1]


def get_traits_by_type(trait_type: TraitType) -> list[Trait]:
    """Get traits by type.

    Args:
        trait_type: The type of traits to get.

    Returns:
        List of traits of that type.
    """
    traits = load_traits()
    return [t for t in traits if t.type == trait_type]


def get_trait_champions(trait_id: str) -> list[str]:
    """Get all champion IDs that have a specific trait.

    Args:
        trait_id: The trait ID to search for.

    Returns:
        List of champion IDs.
    """
    trait = get_trait_by_id(trait_id)
    if trait:
        return trait.champions
    return []


def clear_cache() -> None:
    """Clear the trait cache. Useful for testing or hot-reloading data."""
    load_origins.cache_clear()
    load_classes.cache_clear()
