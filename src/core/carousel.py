"""Carousel (Shared Draft) System for TFT Set 16.

Manages carousel rounds where players pick champions with items.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum

from src.core.constants import CAROUSEL_ROUNDS

if TYPE_CHECKING:
    from src.core.champion_pool import ChampionPool


class CarouselItemType(Enum):
    """Types of items on carousel champions."""
    COMPONENT = "component"
    COMBINED = "combined"
    SPATULA = "spatula"


@dataclass
class CarouselChampion:
    """A champion on the carousel with an equipped item."""
    champion_id: str
    champion_name: str
    cost: int
    item_id: str
    item_name: str
    item_type: CarouselItemType
    is_available: bool = True

    def __repr__(self) -> str:
        return f"{self.champion_name} ({self.cost}-cost) + {self.item_name}"


@dataclass
class CarouselPickOrder:
    """Order in which players pick from carousel."""
    player_id: int
    health: int
    pick_round: int  # Which round of picks (0 = first wave, 1 = second wave, etc.)


@dataclass
class CarouselResult:
    """Result of a carousel round for a player."""
    player_id: int
    champion: Optional[CarouselChampion]
    pick_order: int
    was_random: bool = False  # True if auto-assigned due to timeout


# Component items that appear on carousel
CAROUSEL_COMPONENTS = [
    ("bf_sword", "B.F. Sword"),
    ("recurve_bow", "Recurve Bow"),
    ("needlessly_large_rod", "Needlessly Large Rod"),
    ("tear_of_the_goddess", "Tear of the Goddess"),
    ("chain_vest", "Chain Vest"),
    ("negatron_cloak", "Negatron Cloak"),
    ("giants_belt", "Giant's Belt"),
    ("sparring_gloves", "Sparring Gloves"),
    ("spatula", "Spatula"),
]

# Combined items for late-game carousels
CAROUSEL_COMBINED_ITEMS = [
    ("infinity_edge", "Infinity Edge"),
    ("bloodthirster", "Bloodthirster"),
    ("deathblade", "Deathblade"),
    ("giant_slayer", "Giant Slayer"),
    ("rabadons_deathcap", "Rabadon's Deathcap"),
    ("jeweled_gauntlet", "Jeweled Gauntlet"),
    ("guinsoos_rageblade", "Guinsoo's Rageblade"),
    ("statikk_shiv", "Statikk Shiv"),
    ("titans_resolve", "Titan's Resolve"),
    ("warmogs_armor", "Warmog's Armor"),
    ("dragons_claw", "Dragon's Claw"),
    ("gargoyle_stoneplate", "Gargoyle Stoneplate"),
    ("redemption", "Redemption"),
    ("thiefs_gloves", "Thief's Gloves"),
]

# Champion pools by cost for carousel
CAROUSEL_CHAMPION_POOLS = {
    1: [
        ("aatrox", "Aatrox"),
        ("ahri", "Ahri"),
        ("ashe", "Ashe"),
        ("annie", "Annie"),
        ("blitzcrank", "Blitzcrank"),
        ("elise", "Elise"),
        ("jax", "Jax"),
        ("malphite", "Malphite"),
    ],
    2: [
        ("braum", "Braum"),
        ("cassiopeia", "Cassiopeia"),
        ("draven", "Draven"),
        ("ezreal", "Ezreal"),
        ("jarvan_iv", "Jarvan IV"),
        ("karma", "Karma"),
        ("kog_maw", "Kog'Maw"),
        ("syndra", "Syndra"),
    ],
    3: [
        ("akali", "Akali"),
        ("diana", "Diana"),
        ("gnar", "Gnar"),
        ("katarina", "Katarina"),
        ("leblanc", "LeBlanc"),
        ("lux", "Lux"),
        ("morgana", "Morgana"),
        ("vex", "Vex"),
    ],
    4: [
        ("graves", "Graves"),
        ("hecarim", "Hecarim"),
        ("kayn", "Kayn"),
        ("neeko", "Neeko"),
        ("olaf", "Olaf"),
        ("shen", "Shen"),
        ("taric", "Taric"),
        ("viego", "Viego"),
    ],
    5: [
        ("aphelios", "Aphelios"),
        ("bel_veth", "Bel'Veth"),
        ("galio", "Galio"),
        ("kaisa", "Kai'Sa"),
        ("mordekaiser", "Mordekaiser"),
        ("sion", "Sion"),
        ("urgot", "Urgot"),
        ("xayah", "Xayah"),
    ],
}


class CarouselSystem:
    """Manages carousel rounds and pick order."""

    # Carousel configuration by stage
    CAROUSEL_CONFIG = {
        "1-1": {
            "max_cost": 1,      # Only 1-cost champions
            "item_type": "component",
            "all_pick_simultaneously": True,
            "spatula_chance": 0.15,  # Increased in Set 16
        },
        "2-4": {
            "max_cost": 2,
            "item_type": "component",
            "all_pick_simultaneously": False,
            "spatula_chance": 0.15,
        },
        "3-4": {
            "max_cost": 3,
            "item_type": "component",
            "all_pick_simultaneously": False,
            "spatula_chance": 0.15,
        },
        "4-4": {
            "max_cost": 3,
            "item_type": "component",
            "all_pick_simultaneously": False,
            "spatula_chance": 0.15,
        },
        "5-4": {
            "max_cost": 4,
            "item_type": "combined",  # Combined items start here
            "all_pick_simultaneously": False,
            "spatula_chance": 0.0,
        },
        "6-4": {
            "max_cost": 5,  # 5-costs can appear
            "item_type": "mixed",  # Both combined and components
            "all_pick_simultaneously": False,
            "spatula_chance": 0.0,
        },
        "7-4": {
            "max_cost": 5,
            "item_type": "combined",
            "all_pick_simultaneously": False,
            "spatula_chance": 0.0,
        },
    }

    CAROUSEL_SIZE = 9  # Number of champions on carousel

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize carousel system.

        Args:
            seed: Random seed for reproducible results.
        """
        self.rng = random.Random(seed)

    def is_carousel_round(self, stage: str) -> bool:
        """Check if a stage is a carousel round."""
        return stage in CAROUSEL_ROUNDS

    def get_carousel_config(self, stage: str) -> dict:
        """Get carousel configuration for a stage."""
        return self.CAROUSEL_CONFIG.get(stage, {})

    def calculate_pick_order(
        self,
        player_healths: dict[int, int],
        stage: str,
    ) -> list[CarouselPickOrder]:
        """
        Calculate the order in which players pick from carousel.

        Args:
            player_healths: Dictionary mapping player_id to current health.
            stage: Current stage string.

        Returns:
            List of CarouselPickOrder sorted by pick order.
        """
        config = self.get_carousel_config(stage)

        # First carousel - all pick simultaneously
        if config.get("all_pick_simultaneously", False):
            return [
                CarouselPickOrder(player_id=pid, health=health, pick_round=0)
                for pid, health in player_healths.items()
            ]

        # Sort players by health (lowest first)
        sorted_players = sorted(player_healths.items(), key=lambda x: x[1])

        pick_orders = []
        for i, (player_id, health) in enumerate(sorted_players):
            # Players pick in pairs (2 at a time)
            pick_round = i // 2
            pick_orders.append(
                CarouselPickOrder(
                    player_id=player_id,
                    health=health,
                    pick_round=pick_round,
                )
            )

        return pick_orders

    def generate_carousel(
        self, stage: str, champion_pool: Optional["ChampionPool"] = None
    ) -> list[CarouselChampion]:
        """
        Generate the carousel with 9 champion+item pairs.

        Args:
            stage: Current stage string.
            champion_pool: The game's ChampionPool to use for selecting champions.
                          If None, falls back to hardcoded CAROUSEL_CHAMPION_POOLS.

        Returns:
            List of 9 CarouselChampion objects.
        """
        config = self.get_carousel_config(stage)
        if not config:
            config = self.CAROUSEL_CONFIG["2-4"]  # Default config

        max_cost = config.get("max_cost", 2)
        item_type = config.get("item_type", "component")
        spatula_chance = config.get("spatula_chance", 0.0)

        carousel = []

        # Build champion pool based on max cost
        available_champions = []

        if champion_pool is not None:
            # Use actual ChampionPool data
            for cost in range(1, max_cost + 1):
                champions = champion_pool.get_champions_by_cost(cost)
                for champ in champions:
                    available_champions.append((champ.id, champ.name, champ.cost))
        else:
            # Fallback to hardcoded data (deprecated)
            for cost in range(1, max_cost + 1):
                if cost in CAROUSEL_CHAMPION_POOLS:
                    available_champions.extend(
                        [(cid, cname, cost) for cid, cname in CAROUSEL_CHAMPION_POOLS[cost]]
                    )

        # Select 9 random champions
        selected_champions = self.rng.sample(
            available_champions,
            min(self.CAROUSEL_SIZE, len(available_champions)),
        )

        # Assign items to each champion
        for champ_id, champ_name, cost in selected_champions:
            # Determine item
            if item_type == "component":
                item = self._select_component_item(spatula_chance)
                carousel_item_type = (
                    CarouselItemType.SPATULA
                    if item[0] == "spatula"
                    else CarouselItemType.COMPONENT
                )
            elif item_type == "combined":
                item = self._select_combined_item()
                carousel_item_type = CarouselItemType.COMBINED
            else:  # mixed
                if self.rng.random() < 0.5:
                    item = self._select_component_item(spatula_chance)
                    carousel_item_type = (
                        CarouselItemType.SPATULA
                        if item[0] == "spatula"
                        else CarouselItemType.COMPONENT
                    )
                else:
                    item = self._select_combined_item()
                    carousel_item_type = CarouselItemType.COMBINED

            carousel.append(
                CarouselChampion(
                    champion_id=champ_id,
                    champion_name=champ_name,
                    cost=cost,
                    item_id=item[0],
                    item_name=item[1],
                    item_type=carousel_item_type,
                )
            )

        return carousel

    def _select_component_item(self, spatula_chance: float) -> tuple[str, str]:
        """Select a component item for carousel."""
        if self.rng.random() < spatula_chance:
            return ("spatula", "Spatula")

        # Filter out spatula from regular selection
        components = [c for c in CAROUSEL_COMPONENTS if c[0] != "spatula"]
        return self.rng.choice(components)

    def _select_combined_item(self) -> tuple[str, str]:
        """Select a combined item for carousel."""
        return self.rng.choice(CAROUSEL_COMBINED_ITEMS)

    def process_pick(
        self,
        carousel: list[CarouselChampion],
        player_id: int,
        pick_index: int,
    ) -> Optional[CarouselChampion]:
        """
        Process a player's carousel pick.

        Args:
            carousel: Current carousel state.
            player_id: Player making the pick.
            pick_index: Index of the champion to pick (0-8).

        Returns:
            The picked CarouselChampion, or None if invalid.
        """
        if pick_index < 0 or pick_index >= len(carousel):
            return None

        champion = carousel[pick_index]
        if not champion.is_available:
            return None

        champion.is_available = False
        return champion

    def auto_pick(
        self,
        carousel: list[CarouselChampion],
        player_id: int,
    ) -> Optional[CarouselChampion]:
        """
        Automatically pick a random available champion (timeout fallback).

        Args:
            carousel: Current carousel state.
            player_id: Player to auto-pick for.

        Returns:
            A random available CarouselChampion.
        """
        available = [c for c in carousel if c.is_available]
        if not available:
            return None

        champion = self.rng.choice(available)
        champion.is_available = False
        return champion

    def simulate_carousel_round(
        self,
        stage: str,
        player_healths: dict[int, int],
        player_preferences: Optional[dict[int, str]] = None,
        champion_pool: Optional["ChampionPool"] = None,
    ) -> dict[int, CarouselResult]:
        """
        Simulate an entire carousel round.

        Args:
            stage: Current stage string.
            player_healths: Dictionary mapping player_id to health.
            player_preferences: Optional dict of player_id to preferred item_id.
            champion_pool: The game's ChampionPool to use for selecting champions.

        Returns:
            Dictionary mapping player_id to their CarouselResult.
        """
        carousel = self.generate_carousel(stage, champion_pool)
        pick_order = self.calculate_pick_order(player_healths, stage)

        results: dict[int, CarouselResult] = {}

        # Group players by pick round
        rounds: dict[int, list[CarouselPickOrder]] = {}
        for po in pick_order:
            if po.pick_round not in rounds:
                rounds[po.pick_round] = []
            rounds[po.pick_round].append(po)

        pick_counter = 0
        for round_num in sorted(rounds.keys()):
            players_in_round = rounds[round_num]

            for player_order in players_in_round:
                player_id = player_order.player_id
                picked = None

                # Try to pick based on preference
                if player_preferences and player_id in player_preferences:
                    preferred_item = player_preferences[player_id]
                    for i, champ in enumerate(carousel):
                        if champ.is_available and champ.item_id == preferred_item:
                            picked = self.process_pick(carousel, player_id, i)
                            break

                # If no preference or preferred not available, auto-pick
                if picked is None:
                    picked = self.auto_pick(carousel, player_id)

                results[player_id] = CarouselResult(
                    player_id=player_id,
                    champion=picked,
                    pick_order=pick_counter,
                    was_random=player_preferences is None
                    or player_id not in player_preferences,
                )
                pick_counter += 1

        return results


# Singleton instance
_carousel_system: Optional[CarouselSystem] = None


def get_carousel_system(seed: Optional[int] = None) -> CarouselSystem:
    """Get or create the carousel system singleton."""
    global _carousel_system
    if _carousel_system is None or seed is not None:
        _carousel_system = CarouselSystem(seed)
    return _carousel_system
