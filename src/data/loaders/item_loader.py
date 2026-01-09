"""Item data loader for TFT Set 16."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..models.item import Item, ItemType, ItemStats


# Get the data directory path
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
COMPONENTS_FILE = DATA_DIR / "items" / "components.json"
COMBINED_FILE = DATA_DIR / "items" / "combined.json"


def _parse_item(item_data: dict) -> Item:
    """Parse an item from JSON data.

    Args:
        item_data: Dictionary containing item data.

    Returns:
        Item object.
    """
    stats_data = item_data.get("stats", {})
    stats = ItemStats(
        ad=stats_data.get("ad", 0),
        ap=stats_data.get("ap", 0),
        armor=stats_data.get("armor", 0),
        mr=stats_data.get("mr", 0),
        health=stats_data.get("health", 0),
        mana=stats_data.get("mana", 0),
        attack_speed=stats_data.get("attack_speed", 0.0),
        crit_chance=stats_data.get("crit_chance", 0.0),
        crit_damage=stats_data.get("crit_damage", 0.0),
        omnivamp=stats_data.get("omnivamp", 0.0),
        durability=stats_data.get("durability", 0.0),
    )

    # Parse components tuple if present
    components = None
    if "components" in item_data and item_data["components"]:
        components = tuple(item_data["components"])

    return Item(
        id=item_data["id"],
        name=item_data["name"],
        type=ItemType(item_data["type"]),
        stats=stats,
        effect=item_data.get("effect"),
        components=components,
        grants_trait=item_data.get("grants_trait"),
        is_unique=item_data.get("is_unique", False),
        is_radiant=item_data.get("is_radiant", False),
    )


@lru_cache(maxsize=1)
def load_components() -> list[Item]:
    """Load all component items from JSON file.

    Returns:
        List of component Item objects.
    """
    with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [_parse_item(item) for item in data["components"]]


@lru_cache(maxsize=1)
def load_combined_items() -> list[Item]:
    """Load all combined items from JSON file.

    Returns:
        List of combined Item objects.
    """
    with open(COMBINED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [_parse_item(item) for item in data["items"]]


def load_items() -> list[Item]:
    """Load all items (components and combined).

    Returns:
        List of all Item objects.
    """
    return load_components() + load_combined_items()


def get_item_by_id(item_id: str) -> Optional[Item]:
    """Get an item by its ID.

    Args:
        item_id: The unique item identifier.

    Returns:
        Item object if found, None otherwise.
    """
    items = load_items()
    for item in items:
        if item.id == item_id:
            return item
    return None


def get_recipe(component1: str, component2: str) -> Optional[Item]:
    """Get the combined item from two components.

    Args:
        component1: First component ID.
        component2: Second component ID.

    Returns:
        Combined Item if recipe exists, None otherwise.
    """
    combined_items = load_combined_items()

    for item in combined_items:
        if item.components:
            # Check both orderings
            if (item.components == (component1, component2) or
                item.components == (component2, component1)):
                return item
    return None


def get_components_for_item(item_id: str) -> Optional[tuple[str, str]]:
    """Get the component IDs needed to craft an item.

    Args:
        item_id: The combined item ID.

    Returns:
        Tuple of component IDs if item is craftable, None otherwise.
    """
    item = get_item_by_id(item_id)
    if item and item.components:
        return item.components
    return None


def get_items_by_type(item_type: ItemType) -> list[Item]:
    """Get all items of a specific type.

    Args:
        item_type: The type of items to get.

    Returns:
        List of items of that type.
    """
    items = load_items()
    return [item for item in items if item.type == item_type]


def get_items_with_component(component_id: str) -> list[Item]:
    """Get all combined items that use a specific component.

    Args:
        component_id: The component item ID.

    Returns:
        List of combined items using that component.
    """
    combined_items = load_combined_items()
    return [
        item for item in combined_items
        if item.components and component_id in item.components
    ]


def get_emblem_items() -> list[Item]:
    """Get all emblem items.

    Returns:
        List of emblem items.
    """
    items = load_items()
    return [item for item in items if item.type == ItemType.EMBLEM]


def build_recipe_matrix() -> dict[str, dict[str, str]]:
    """Build a complete recipe matrix mapping component pairs to items.

    Returns:
        Nested dict where recipe_matrix[comp1][comp2] = item_id.
    """
    combined_items = load_combined_items()
    matrix: dict[str, dict[str, str]] = {}

    for item in combined_items:
        if item.components:
            c1, c2 = item.components

            # Add both orderings
            if c1 not in matrix:
                matrix[c1] = {}
            if c2 not in matrix:
                matrix[c2] = {}

            matrix[c1][c2] = item.id
            matrix[c2][c1] = item.id

    return matrix


def clear_cache() -> None:
    """Clear the item cache. Useful for testing or hot-reloading data."""
    load_components.cache_clear()
    load_combined_items.cache_clear()
