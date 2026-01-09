"""Emblem System for TFT Set 16.

Handle emblem items that grant additional traits.
"""

from src.core.player_units import ChampionInstance


class EmblemSystem:
    """
    Manages emblem effects on champions.
    """

    # Emblem item IDs mapped to trait IDs
    # These are special items crafted with Spatula that grant a trait
    EMBLEM_TRAITS = {
        # Origin emblems
        "bilgewater_emblem": "bilgewater",
        "demacia_emblem": "demacia",
        "freljord_emblem": "freljord",
        "ionia_emblem": "ionia",
        "ixtal_emblem": "ixtal",
        "noxus_emblem": "noxus",
        "piltover_emblem": "piltover",
        "shadow_isles_emblem": "shadow_isles",
        "shurima_emblem": "shurima",
        "targon_emblem": "targon",
        "void_emblem": "void",
        "yordle_emblem": "yordle",
        "zaun_emblem": "zaun",
        # Class emblems
        "arcanist_emblem": "arcanist",
        "bruiser_emblem": "bruiser",
        "defender_emblem": "defender",
        "disruptor_emblem": "disruptor",
        "gunslinger_emblem": "gunslinger",
        "invoker_emblem": "invoker",
        "juggernaut_emblem": "juggernaut",
        "longshot_emblem": "longshot",
        "quickstriker_emblem": "quickstriker",
        "slayer_emblem": "slayer",
        "vanquisher_emblem": "vanquisher",
        "warden_emblem": "warden",
    }

    @staticmethod
    def get_emblem_traits(champion: ChampionInstance) -> list[str]:
        """
        Get list of trait IDs granted by equipped emblems.

        Args:
            champion: The champion instance to check

        Returns:
            List of trait IDs from emblems
        """
        emblem_traits = []
        for item in champion.items:
            if item.id in EmblemSystem.EMBLEM_TRAITS:
                emblem_traits.append(EmblemSystem.EMBLEM_TRAITS[item.id])
        return emblem_traits

    @staticmethod
    def get_all_emblem_traits(champions: list[ChampionInstance]) -> list[str]:
        """
        Get all emblem traits from all champions.

        Args:
            champions: List of champion instances

        Returns:
            List of all emblem trait IDs
        """
        all_emblems = []
        for champ in champions:
            all_emblems.extend(EmblemSystem.get_emblem_traits(champ))
        return all_emblems

    @staticmethod
    def can_equip_emblem(champion: ChampionInstance, emblem_trait: str) -> bool:
        """
        Check if champion can equip this emblem.
        Cannot equip if champion already has the trait naturally.

        Args:
            champion: The champion instance
            emblem_trait: The trait ID the emblem would grant

        Returns:
            True if the emblem can be equipped
        """
        return emblem_trait not in champion.champion.traits

    @staticmethod
    def is_emblem(item_id: str) -> bool:
        """
        Check if an item is an emblem.

        Args:
            item_id: The item ID to check

        Returns:
            True if the item is an emblem
        """
        return item_id in EmblemSystem.EMBLEM_TRAITS

    @staticmethod
    def get_trait_for_emblem(item_id: str) -> str | None:
        """
        Get the trait ID granted by an emblem item.

        Args:
            item_id: The emblem item ID

        Returns:
            The trait ID, or None if not an emblem
        """
        return EmblemSystem.EMBLEM_TRAITS.get(item_id)
