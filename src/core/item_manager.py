"""Item Manager for TFT Set 16.

Manages a player's items (inventory + equipped).
"""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from src.data.models.item import Item, ItemType
from src.data.loaders import (
    load_components,
    load_combined_items,
    get_recipe,
    build_recipe_matrix,
    get_item_by_id,
)

if TYPE_CHECKING:
    from src.core.player_units import ChampionInstance


@dataclass
class ItemInstance:
    """An instance of an item that can be equipped."""

    item: Item
    equipped_to: Optional[str] = None  # champion_id if equipped

    @property
    def is_component(self) -> bool:
        """Check if this is a component item."""
        return self.item.type == ItemType.COMPONENT

    @property
    def is_combined(self) -> bool:
        """Check if this is a combined item."""
        return self.item.type in [
            ItemType.COMBINED,
            ItemType.RADIANT,
            ItemType.ARTIFACT,
        ]

    @property
    def is_emblem(self) -> bool:
        """Check if this item grants a trait."""
        return (
            self.item.type == ItemType.EMBLEM or self.item.grants_trait is not None
        )

    def __repr__(self) -> str:
        equipped = f" (on {self.equipped_to})" if self.equipped_to else ""
        return f"ItemInstance({self.item.name}{equipped})"


class ItemManager:
    """
    Manages a player's items (inventory + equipped).
    """

    MAX_ITEMS_PER_CHAMPION = 3

    def __init__(self):
        """Initialize the item manager."""
        self.inventory: list[ItemInstance] = []  # Unequipped items
        self.recipe_matrix = build_recipe_matrix()
        self._load_item_data()

    def _load_item_data(self) -> None:
        """Load all item data for lookups."""
        self.components = {i.id: i for i in load_components()}
        self.combined = {i.id: i for i in load_combined_items()}
        self.all_items = {**self.components, **self.combined}

    def add_to_inventory(self, item: Item) -> ItemInstance:
        """
        Add a new item to inventory.

        Args:
            item: The item to add.

        Returns:
            The created ItemInstance.
        """
        instance = ItemInstance(item=item)
        self.inventory.append(instance)
        return instance

    def remove_from_inventory(self, item_instance: ItemInstance) -> bool:
        """
        Remove item from inventory.

        Args:
            item_instance: The item instance to remove.

        Returns:
            True if removed, False if not found.
        """
        if item_instance in self.inventory:
            self.inventory.remove(item_instance)
            return True
        return False

    def equip_item(
        self,
        item_instance: ItemInstance,
        champion: "ChampionInstance",
    ) -> bool:
        """
        Equip item to champion.

        Rules:
        - Champion can hold max 3 items
        - Components auto-combine if possible
        - Cannot equip emblem if champion has that trait

        Args:
            item_instance: The item to equip.
            champion: The champion to equip to.

        Returns:
            True if successful.
        """
        # Check if champion can hold more items
        if len(champion.items) >= self.MAX_ITEMS_PER_CHAMPION:
            return False

        # Check emblem restriction
        if item_instance.is_emblem and item_instance.item.grants_trait:
            if item_instance.item.grants_trait in champion.champion.traits:
                return False  # Cannot equip emblem for trait champion already has

        # Remove from inventory
        self.remove_from_inventory(item_instance)

        # If this is a component, try to auto-combine with existing components
        if item_instance.is_component:
            combined = self._auto_combine_on_equip(item_instance, champion)
            if combined:
                # Item was combined, don't add the component separately
                return True

        # Add item to champion
        champion.items.append(item_instance)
        item_instance.equipped_to = champion.champion.id
        return True

    def unequip_item(
        self,
        item_instance: ItemInstance,
        champion: "ChampionInstance",
    ) -> bool:
        """
        Unequip item from champion back to inventory.
        Note: In real TFT, items cannot be freely unequipped.
        This is for simulation/planning purposes.

        Args:
            item_instance: The item to unequip.
            champion: The champion to unequip from.

        Returns:
            True if successful.
        """
        if item_instance not in champion.items:
            return False

        champion.items.remove(item_instance)
        item_instance.equipped_to = None
        self.inventory.append(item_instance)
        return True

    def try_combine(
        self,
        component1: ItemInstance,
        component2: ItemInstance,
    ) -> Optional[ItemInstance]:
        """
        Try to combine two components into a completed item.

        Args:
            component1: First component.
            component2: Second component.

        Returns:
            The new combined item if successful, None otherwise.
        """
        if not component1.is_component or not component2.is_component:
            return None

        recipe = get_recipe(component1.item.id, component2.item.id)
        if recipe:
            # Remove components from inventory
            self.remove_from_inventory(component1)
            self.remove_from_inventory(component2)
            # Create combined item
            return self.add_to_inventory(recipe)
        return None

    def _auto_combine_on_equip(
        self,
        component: ItemInstance,
        champion: "ChampionInstance",
    ) -> Optional[ItemInstance]:
        """
        When equipping a component to a champion that has another component,
        automatically combine them.

        Args:
            component: The component being equipped.
            champion: The champion to equip to.

        Returns:
            The combined item if combination happened, None otherwise.
        """
        # Find existing components on the champion
        champion_components = [
            item for item in champion.items if item.is_component
        ]

        for existing in champion_components:
            recipe = get_recipe(component.item.id, existing.item.id)
            if recipe:
                # Remove the existing component
                champion.items.remove(existing)
                existing.equipped_to = None

                # Create combined item and add to champion
                combined = ItemInstance(item=recipe, equipped_to=champion.champion.id)
                champion.items.append(combined)
                return combined

        return None

    def get_available_recipes(self) -> list[tuple[Item, Item, Item]]:
        """
        Get all possible recipes from current inventory.

        Returns:
            List of (component1, component2, result).
        """
        recipes = []
        components = [i for i in self.inventory if i.is_component]

        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1 :]:
                recipe = get_recipe(comp1.item.id, comp2.item.id)
                if recipe:
                    recipes.append((comp1.item, comp2.item, recipe))

        return recipes

    def get_components_for_item(self, item_id: str) -> Optional[tuple[str, str]]:
        """
        Get required components for a combined item.

        Args:
            item_id: The combined item ID.

        Returns:
            Tuple of component IDs, or None if not craftable.
        """
        item = self.all_items.get(item_id)
        if item and item.components:
            return item.components
        return None

    def can_build_item(self, item_id: str) -> bool:
        """
        Check if item can be built from current inventory.

        Args:
            item_id: The item ID to check.

        Returns:
            True if item can be built.
        """
        components_needed = self.get_components_for_item(item_id)
        if not components_needed:
            return False

        comp1, comp2 = components_needed
        inventory_ids = [i.item.id for i in self.inventory if i.is_component]

        # Check if we have both components
        if comp1 == comp2:
            return inventory_ids.count(comp1) >= 2
        else:
            return comp1 in inventory_ids and comp2 in inventory_ids

    def get_item(self, item_id: str) -> Optional[Item]:
        """
        Get an item by ID.

        Args:
            item_id: The item ID.

        Returns:
            The Item, or None if not found.
        """
        return self.all_items.get(item_id) or get_item_by_id(item_id)

    def get_inventory_count(self) -> int:
        """Get number of items in inventory."""
        return len(self.inventory)

    def get_components_in_inventory(self) -> list[ItemInstance]:
        """Get all component items in inventory."""
        return [i for i in self.inventory if i.is_component]

    def get_combined_in_inventory(self) -> list[ItemInstance]:
        """Get all combined items in inventory."""
        return [i for i in self.inventory if i.is_combined]

    def clear_inventory(self) -> list[ItemInstance]:
        """
        Clear all items from inventory.

        Returns:
            List of removed items.
        """
        items = self.inventory.copy()
        self.inventory.clear()
        return items
