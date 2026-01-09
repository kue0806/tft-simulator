"""Champion data loader for TFT Set 16."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..models.champion import Champion, ChampionStats, Ability


# Get the data directory path
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
CHAMPIONS_FILE = DATA_DIR / "champions" / "set16_champions.json"
UNLOCKABLES_FILE = DATA_DIR / "champions" / "set16_unlockables.json"


def _parse_champion(champ_data: dict) -> Champion:
    """Parse a champion from JSON data."""
    stats = ChampionStats(
        health=champ_data["stats"]["health"],
        mana=tuple(champ_data["stats"]["mana"]),
        armor=champ_data["stats"]["armor"],
        magic_resist=champ_data["stats"]["magic_resist"],
        attack_damage=champ_data["stats"]["attack_damage"],
        attack_speed=champ_data["stats"]["attack_speed"],
        attack_range=champ_data["stats"]["attack_range"],
    )

    ability = Ability(
        name=champ_data["ability"]["name"],
        description=champ_data["ability"]["description"],
        damage_type=champ_data["ability"].get("damage_type"),
    )

    return Champion(
        id=champ_data["id"],
        name=champ_data["name"],
        cost=champ_data["cost"],
        traits=champ_data["traits"],
        stats=stats,
        ability=ability,
        is_unlockable=champ_data.get("is_unlockable", False),
        unlock_condition=champ_data.get("unlock_condition"),
        slots=champ_data.get("slots", 1),
    )


@lru_cache(maxsize=1)
def load_champions() -> list[Champion]:
    """Load all champions from JSON files (base + unlockables).

    Returns:
        List of all Champion objects.
    """
    champions = []

    # Load base champions
    with open(CHAMPIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for champ_data in data["champions"]:
        champions.append(_parse_champion(champ_data))

    # Load unlockable champions
    if UNLOCKABLES_FILE.exists():
        with open(UNLOCKABLES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for champ_data in data["champions"]:
            champions.append(_parse_champion(champ_data))

    return champions


def get_champion_by_id(champion_id: str) -> Optional[Champion]:
    """Get a champion by their ID.

    Args:
        champion_id: The unique champion identifier.

    Returns:
        Champion object if found, None otherwise.
    """
    champions = load_champions()
    for champion in champions:
        if champion.id == champion_id:
            return champion
    return None


def get_champions_by_cost(cost: int) -> list[Champion]:
    """Get all champions of a specific cost.

    Args:
        cost: The champion cost tier (1-7).

    Returns:
        List of champions at that cost.
    """
    champions = load_champions()
    return [c for c in champions if c.cost == cost]


def get_champions_by_trait(trait: str) -> list[Champion]:
    """Get all champions with a specific trait.

    Args:
        trait: The trait ID to search for.

    Returns:
        List of champions with that trait.
    """
    champions = load_champions()
    return [c for c in champions if trait in c.traits]


def get_base_champions() -> list[Champion]:
    """Get all base (non-unlockable) champions.

    Returns:
        List of base champions.
    """
    champions = load_champions()
    return [c for c in champions if not c.is_unlockable]


def get_unlockable_champions() -> list[Champion]:
    """Get all unlockable champions.

    Returns:
        List of unlockable champions.
    """
    champions = load_champions()
    return [c for c in champions if c.is_unlockable]


def get_champions_count_by_cost() -> dict[int, int]:
    """Get count of champions at each cost tier.

    Returns:
        Dictionary mapping cost to champion count.
    """
    champions = load_champions()
    counts = {}
    for champion in champions:
        cost = champion.cost
        counts[cost] = counts.get(cost, 0) + 1
    return counts


def clear_cache() -> None:
    """Clear the champion cache. Useful for testing or hot-reloading data."""
    load_champions.cache_clear()
