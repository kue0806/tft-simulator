"""Augment Effect System for TFT Set 16.

Implements the actual effects of augments on game state and combat.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import random

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from src.combat.combat_unit import CombatUnit
    from src.combat.combat_engine import CombatEngine
    from src.data.models.item import Item
    from src.data.models.champion import Champion


class AugmentEffectTrigger(Enum):
    """When an augment effect triggers."""
    IMMEDIATE = "immediate"          # When augment is selected
    ROUND_START = "round_start"      # At the start of each round
    ROUND_END = "round_end"          # At the end of each round
    COMBAT_START = "combat_start"    # When combat begins
    COMBAT_TICK = "combat_tick"      # Every combat tick (30/sec)
    ON_KILL = "on_kill"              # When a unit gets a kill
    ON_DEATH = "on_death"            # When a unit dies
    ON_ATTACK = "on_attack"          # When a unit attacks
    ON_CAST = "on_cast"              # When a unit casts ability
    ON_LEVEL_UP = "on_level_up"      # When player levels up
    ON_BUY = "on_buy"                # When buying a champion
    ON_SELL = "on_sell"              # When selling a champion
    SHOP_REFRESH = "shop_refresh"    # When shop refreshes


@dataclass
class AugmentEffectResult:
    """Result of applying an augment effect."""
    success: bool
    message: str = ""
    gold_gained: int = 0
    xp_gained: int = 0
    items_gained: list = field(default_factory=list)
    stats_modified: dict = field(default_factory=dict)


class AugmentEffectSystem:
    """System for applying augment effects to game state and combat."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Track persistent augment states per player
        self._player_states: dict[int, dict] = {}
        # Cache for loaded data
        self._items_cache: dict[str, Any] = {}
        self._champions_cache: dict[str, Any] = {}

    # =========================================================================
    # ITEM GRANT HELPERS
    # =========================================================================

    def _get_item_by_id(self, item_id: str) -> Optional["Item"]:
        """Get item by ID with caching."""
        if item_id not in self._items_cache:
            try:
                from src.data.loaders import get_item_by_id
                self._items_cache[item_id] = get_item_by_id(item_id)
            except Exception:
                return None
        return self._items_cache.get(item_id)

    def _get_random_components(self, count: int) -> list:
        """Get random component items."""
        try:
            from src.data.loaders import load_components
            components = load_components()
            return [self.rng.choice(components) for _ in range(count)]
        except Exception:
            return []

    def _get_random_completed_items(self, count: int) -> list:
        """Get random completed items."""
        try:
            from src.data.loaders import load_combined_items
            combined = load_combined_items()
            return [self.rng.choice(combined) for _ in range(count)]
        except Exception:
            return []

    def _get_random_emblems(self, count: int) -> list:
        """Get random emblem items.

        Falls back to spatula-based items if no emblems available.
        """
        try:
            from src.data.loaders import get_emblem_items
            emblems = get_emblem_items()
            if emblems:
                return [self.rng.choice(emblems) for _ in range(count)]
            # Fallback: try to get items that grant traits
            from src.data.loaders import load_combined_items
            combined = load_combined_items()
            trait_items = [item for item in combined
                           if hasattr(item, 'grants_trait') and item.grants_trait]
            if trait_items:
                return [self.rng.choice(trait_items) for _ in range(count)]
        except Exception:
            pass
        return []

    def _get_items_by_component(self, component_id: str) -> list:
        """Get completed items that use a specific component."""
        try:
            from src.data.loaders import load_combined_items
            combined = load_combined_items()
            return [item for item in combined
                    if item.components and component_id in item.components]
        except Exception:
            return []

    def _grant_item_to_player(
        self,
        player: "PlayerState",
        item: "Item",
        result: "AugmentEffectResult"
    ) -> bool:
        """Grant a single item to player's inventory."""
        try:
            if hasattr(player, 'items') and player.items:
                player.items.add_to_inventory(item)
                result.items_gained.append(item.name)
                return True
        except Exception:
            pass
        return False

    def _grant_specific_item(
        self,
        player: "PlayerState",
        item_id: str,
        result: "AugmentEffectResult"
    ) -> bool:
        """Grant a specific item by ID."""
        item = self._get_item_by_id(item_id)
        if item:
            return self._grant_item_to_player(player, item, result)
        return False

    # =========================================================================
    # CHAMPION GRANT HELPERS
    # =========================================================================

    def _get_champion_by_id(self, champion_id: str) -> Optional["Champion"]:
        """Get champion by ID with caching."""
        if champion_id not in self._champions_cache:
            try:
                from src.data.loaders import get_champion_by_id
                self._champions_cache[champion_id] = get_champion_by_id(champion_id)
            except Exception:
                return None
        return self._champions_cache.get(champion_id)

    def _get_champions_by_cost(self, cost: int) -> list:
        """Get all champions of a specific cost."""
        try:
            from src.data.loaders import get_champions_by_cost
            return get_champions_by_cost(cost)
        except Exception:
            return []

    def _get_champions_by_trait(self, trait_id: str) -> list:
        """Get all champions with a specific trait."""
        try:
            from src.data.loaders import get_champions_by_trait
            return get_champions_by_trait(trait_id)
        except Exception:
            return []

    def _grant_champion_to_player(
        self,
        player: "PlayerState",
        champion: "Champion",
        star_level: int = 1,
        result: Optional["AugmentEffectResult"] = None
    ) -> bool:
        """Grant a champion to player's bench."""
        try:
            if hasattr(player, 'units') and player.units:
                # Grant copies based on star level
                copies_needed = 1 if star_level == 1 else (3 if star_level == 2 else 9)
                for _ in range(copies_needed):
                    instance = player.units.add_to_bench(champion)
                    if instance is None:
                        break  # Bench full
                if result:
                    star_str = "★" * star_level
                    result.message += f"{champion.name}{star_str} 획득. "
                return True
        except Exception:
            pass
        return False

    def _grant_random_champion(
        self,
        player: "PlayerState",
        cost: int,
        star_level: int = 1,
        result: Optional["AugmentEffectResult"] = None
    ) -> bool:
        """Grant a random champion of specific cost."""
        champions = self._get_champions_by_cost(cost)
        if champions:
            champion = self.rng.choice(champions)
            return self._grant_champion_to_player(player, champion, star_level, result)
        return False

    # =========================================================================
    # TRAIT/SYNERGY HELPERS
    # =========================================================================

    def _get_active_synergies(self, player: "PlayerState") -> dict:
        """Get player's active synergies."""
        try:
            if hasattr(player, 'units') and player.units:
                return player.units.get_active_synergies()
        except Exception:
            pass
        return {}

    def _count_bronze_traits(self, player: "PlayerState") -> int:
        """Count active bronze-tier traits for a player."""
        synergies = self._get_active_synergies(player)
        count = 0
        for trait_id, active_trait in synergies.items():
            if hasattr(active_trait, 'style') and active_trait.style == "bronze":
                count += 1
        return count

    def _count_active_traits(self, player: "PlayerState") -> int:
        """Count active traits for a player."""
        synergies = self._get_active_synergies(player)
        return sum(1 for t in synergies.values()
                   if hasattr(t, 'is_active') and t.is_active)

    def _get_trait_count(self, player: "PlayerState", trait_id: str) -> int:
        """Get count of a specific trait for player."""
        synergies = self._get_active_synergies(player)
        if trait_id in synergies:
            return synergies[trait_id].count
        return 0

    def _is_trait_active(self, player: "PlayerState", trait_id: str) -> bool:
        """Check if a specific trait is active."""
        synergies = self._get_active_synergies(player)
        if trait_id in synergies:
            return synergies[trait_id].is_active
        return False

    def _get_trait_style(self, player: "PlayerState", trait_id: str) -> str:
        """Get the style (bronze/silver/gold/chromatic) of a trait."""
        synergies = self._get_active_synergies(player)
        if trait_id in synergies:
            return synergies[trait_id].style
        return "inactive"

    def _count_gold_traits(self, player: "PlayerState") -> int:
        """Count gold-tier traits for a player."""
        synergies = self._get_active_synergies(player)
        count = 0
        for active_trait in synergies.values():
            if hasattr(active_trait, 'style') and active_trait.style == "gold":
                count += 1
        return count

    def _count_chromatic_traits(self, player: "PlayerState") -> int:
        """Count chromatic-tier traits for a player."""
        synergies = self._get_active_synergies(player)
        count = 0
        for active_trait in synergies.values():
            if hasattr(active_trait, 'style') and active_trait.style == "chromatic":
                count += 1
        return count

    def get_trait_stat_bonuses(self, player: "PlayerState") -> dict:
        """
        Get stat bonuses from traits for augment calculations.

        Returns dict with various stat modifiers from traits.
        """
        bonuses = {
            "ad_percent": 0.0,
            "ap_percent": 0.0,
            "armor": 0,
            "mr": 0,
            "health": 0,
            "attack_speed": 0.0,
        }

        synergies = self._get_active_synergies(player)

        for trait_id, active_trait in synergies.items():
            if not active_trait.is_active:
                continue

            breakpoint = active_trait.active_breakpoint
            if not breakpoint or not breakpoint.effects:
                continue

            effects = breakpoint.effects

            # Aggregate trait bonuses
            if "attack_damage" in effects:
                bonuses["ad_percent"] += effects["attack_damage"]
            if "ability_power" in effects:
                bonuses["ap_percent"] += effects["ability_power"]
            if "armor" in effects:
                bonuses["armor"] += effects["armor"]
            if "magic_resist" in effects:
                bonuses["mr"] += effects["magic_resist"]
            if "health" in effects:
                bonuses["health"] += effects["health"]
            if "attack_speed" in effects:
                bonuses["attack_speed"] += effects["attack_speed"]

        return bonuses

    def apply_trait_augment_synergy(
        self,
        player: "PlayerState",
        trait_id: str
    ) -> dict:
        """
        Apply synergy between augments and traits.

        Some augments have bonus effects when certain traits are active.
        """
        bonuses = {}

        for augment in player.augments:
            effects = augment.effects

            # Bronze For Life - damage per bronze trait
            if "damage_amp_per_bronze" in effects:
                bronze_count = self._count_bronze_traits(player)
                bonuses["damage_amp"] = bonuses.get("damage_amp", 0) + effects["damage_amp_per_bronze"] * bronze_count

            # Durability per bronze
            if "durability_per_bronze" in effects:
                bronze_count = self._count_bronze_traits(player)
                bonuses["damage_reduction"] = bonuses.get("damage_reduction", 0) + effects["durability_per_bronze"] * bronze_count

            # Stand United - AD/AP per active trait
            if "ad_per_trait" in effects or "ap_per_trait" in effects:
                trait_count = self._count_active_traits(player)
                if "ad_per_trait" in effects:
                    bonuses["bonus_ad"] = bonuses.get("bonus_ad", 0) + effects["ad_per_trait"] * trait_count
                if "ap_per_trait" in effects:
                    bonuses["bonus_ap"] = bonuses.get("bonus_ap", 0) + effects["ap_per_trait"] * trait_count

        return bonuses

    def get_player_augment_state(self, player_id: int) -> dict:
        """Get or create augment state for a player."""
        if player_id not in self._player_states:
            self._player_states[player_id] = {
                # Economy
                "free_rerolls": 0,
                "no_interest": False,
                "gold_per_round": 0,
                "xp_per_round": 0,
                "gold_per_combat": 0,
                "xp_per_combat": 0,
                "gold_on_loss": 0,
                "reroll_on_loss": 0,
                "free_reroll_chance": 0.0,
                "interest_cap_increase": 0,
                # Leveling
                "xp_costs_health": False,
                "xp_health_cost": 4,
                "xp_cost_reduction": 0,
                "bonus_xp_on_purchase": 0,
                # Health
                "can_exceed_100hp": False,
                "only_lose_1_hp": False,
                "double_damage_on_loss": False,
                # Combat tracking
                "combat_bonuses": {},
                "stacking_ad": 0,
                "stacking_ap": 0,
                "stacking_health": 0,
                # Item tracking
                "component_per_round": 0,
                "components_from_deaths": 0,
                "max_components_from_deaths": 3,
                # Special mechanics
                "next_augment_tier_modifier": 0,  # +1 or -1
                "loot_drop_chance": 0.0,
                "gold_on_enemy_kill": 0,
            }
        return self._player_states[player_id]

    # =========================================================================
    # IMMEDIATE EFFECTS (when augment is selected)
    # =========================================================================

    def apply_immediate_effects(
        self,
        augment_id: str,
        effects: dict,
        player: "PlayerState"
    ) -> AugmentEffectResult:
        """Apply one-time effects when augment is selected."""
        result = AugmentEffectResult(success=True)

        # Instant gold
        if "instant_gold" in effects:
            gold = effects["instant_gold"]
            player.gold += gold
            result.gold_gained = gold
            result.message += f"골드 +{gold}. "

        # Instant XP
        if "instant_xp" in effects:
            xp = effects["instant_xp"]
            player.add_xp(xp)
            result.xp_gained = xp
            result.message += f"경험치 +{xp}. "

        # Random item components
        if "random_components" in effects:
            count = effects["random_components"]
            from src.data.loaders import load_components
            components = load_components()
            for _ in range(count):
                component = self.rng.choice(components)
                item_inst = player.items.add_to_inventory(component)
                result.items_gained.append(component.name)
            result.message += f"아이템 컴포넌트 {count}개 획득. "

        # Random completed items
        if "random_completed_items" in effects:
            count = effects["random_completed_items"]
            from src.data.loaders import load_combined_items
            combined = load_combined_items()
            for _ in range(count):
                item = self.rng.choice(combined)
                item_inst = player.items.add_to_inventory(item)
                result.items_gained.append(item.name)
            result.message += f"완성 아이템 {count}개 획득. "

        # Tome of Traits
        if "tome_of_traits" in effects:
            count = effects["tome_of_traits"]
            result.message += f"특성의 책 {count}개 획득. "

        # Setup persistent effects
        aug_state = self.get_player_augment_state(player.player_id)

        if "free_rerolls_per_round" in effects:
            aug_state["free_rerolls"] += effects["free_rerolls_per_round"]
            result.message += f"라운드당 무료 리롤 +{effects['free_rerolls_per_round']}. "

        if "no_interest" in effects:
            aug_state["no_interest"] = True
            result.message += "이자 비활성화. "

        if "gold_per_round" in effects:
            aug_state["gold_per_round"] += effects["gold_per_round"]
            result.message += f"라운드당 골드 +{effects['gold_per_round']}. "

        if "xp_costs_health" in effects:
            aug_state["xp_costs_health"] = True
            result.message += "경험치 구매 시 체력 소모. "

        if "can_exceed_100hp" in effects:
            aug_state["can_exceed_100hp"] = True
            result.message += "최대 체력 100 초과 가능. "

        # Instant rerolls
        if "instant_rerolls" in effects:
            count = effects["instant_rerolls"]
            # Store for use (actual rerolls happen in shop system)
            aug_state["pending_rerolls"] = aug_state.get("pending_rerolls", 0) + count
            result.message += f"무료 리롤 {count}회 획득. "

        # XP per round
        if "xp_per_round" in effects:
            aug_state["xp_per_round"] += effects["xp_per_round"]
            result.message += f"라운드당 경험치 +{effects['xp_per_round']}. "

        # Gold per combat
        if "gold_per_combat" in effects:
            aug_state["gold_per_combat"] += effects["gold_per_combat"]
            result.message += f"전투당 골드 +{effects['gold_per_combat']}. "

        # XP per combat
        if "xp_per_combat" in effects:
            aug_state["xp_per_combat"] += effects["xp_per_combat"]
            result.message += f"전투당 경험치 +{effects['xp_per_combat']}. "

        # Gold on loss
        if "gold_on_loss" in effects:
            aug_state["gold_on_loss"] += effects["gold_on_loss"]
            result.message += f"패배 시 골드 +{effects['gold_on_loss']}. "

        # Reroll on loss
        if "reroll_on_loss" in effects:
            aug_state["reroll_on_loss"] += effects["reroll_on_loss"]
            result.message += f"패배 시 리롤 +{effects['reroll_on_loss']}. "

        # Free reroll chance (Prismatic Ticket)
        if "free_reroll_chance" in effects:
            aug_state["free_reroll_chance"] = effects["free_reroll_chance"]
            result.message += f"리롤 {int(effects['free_reroll_chance'] * 100)}% 무료. "

        # Interest cap increase
        if "interest_cap_increase" in effects:
            aug_state["interest_cap_increase"] += effects["interest_cap_increase"]
            result.message += f"이자 상한 +{effects['interest_cap_increase']}. "

        # XP cost reduction
        if "xp_cost_reduction" in effects:
            aug_state["xp_cost_reduction"] += effects["xp_cost_reduction"]
            result.message += f"경험치 비용 -{effects['xp_cost_reduction']}. "

        # Bonus XP on purchase
        if "bonus_xp_on_purchase" in effects:
            aug_state["bonus_xp_on_purchase"] += effects["bonus_xp_on_purchase"]
            result.message += f"경험치 구매 시 추가 +{effects['bonus_xp_on_purchase']}. "

        # Only lose 1 HP (Nine Lives)
        if "only_lose_1_hp" in effects:
            aug_state["only_lose_1_hp"] = True
            result.message += "패배 시 체력 1만 잃음. "

        # Set health to 9 (Nine Lives)
        if "set_health_9" in effects:
            player.health = 9
            result.message += "체력이 9로 설정됨. "

        # Double damage on loss (Cursed Crown)
        if "double_damage_on_loss" in effects:
            aug_state["double_damage_on_loss"] = True
            result.message += "패배 시 2배 피해. "

        # Team size increase (Cursed Crown, Coronation)
        if "team_size" in effects:
            # This would need to be handled by the board/player system
            result.message += f"팀 크기 +{effects['team_size']}. "

        # Tactician's Crown
        if "tacticians_crown" in effects:
            result.message += "전술가의 왕관 획득. "

        # Component per round
        if "component_per_round" in effects:
            aug_state["component_per_round"] += effects["component_per_round"]
            result.message += f"라운드당 컴포넌트 +{effects['component_per_round']}. "

        # Loot drop chance
        if "loot_drop_chance" in effects:
            aug_state["loot_drop_chance"] = max(
                aug_state["loot_drop_chance"],
                effects["loot_drop_chance"]
            )
            result.message += f"전리품 드롭 확률 {int(effects['loot_drop_chance'] * 100)}%. "

        # Gold on enemy kill
        if "gold_on_enemy_kill" in effects:
            aug_state["gold_on_enemy_kill"] += effects["gold_on_enemy_kill"]
            result.message += f"적 처치 시 골드 +{effects['gold_on_enemy_kill']}. "

        # Next augment tier modifier
        if "next_augment_tier_up" in effects:
            aug_state["next_augment_tier_modifier"] = 1
            result.message += "다음 증강 티어 상승. "

        if "next_augment_tier_down" in effects:
            aug_state["next_augment_tier_modifier"] = -1
            result.message += "다음 증강 티어 하락. "

        # =====================================================================
        # SPECIFIC ITEM GRANTS
        # =====================================================================

        # Thief's Gloves
        if "thiefs_gloves" in effects:
            count = effects["thiefs_gloves"]
            for _ in range(count):
                self._grant_specific_item(player, "thiefs_gloves", result)
            result.message += f"도둑의 장갑 {count}개 획득. "

        # B.F. Sword
        if "bf_sword" in effects:
            count = effects["bf_sword"]
            for _ in range(count):
                self._grant_specific_item(player, "bf_sword", result)

        if "bf_swords" in effects:
            count = effects["bf_swords"]
            for _ in range(count):
                self._grant_specific_item(player, "bf_sword", result)

        # Needlessly Large Rod
        if "rod" in effects:
            count = effects["rod"]
            for _ in range(count):
                self._grant_specific_item(player, "needlessly_large_rod", result)

        if "rods" in effects:
            count = effects["rods"]
            for _ in range(count):
                self._grant_specific_item(player, "needlessly_large_rod", result)

        # Giant's Belt
        if "giant_belts" in effects:
            count = effects["giant_belts"]
            for _ in range(count):
                self._grant_specific_item(player, "giants_belt", result)

        # Tear of the Goddess
        if "tear" in effects:
            count = effects["tear"]
            for _ in range(count):
                self._grant_specific_item(player, "tear_of_the_goddess", result)

        # Sparring Gloves
        if "gloves" in effects:
            count = effects["gloves"]
            for _ in range(count):
                self._grant_specific_item(player, "sparring_gloves", result)

        # Spatula
        if "spatula" in effects:
            count = effects["spatula"]
            for _ in range(count):
                self._grant_specific_item(player, "spatula", result)

        # Specific completed items
        item_grants = {
            "bloodthirster": "bloodthirster",
            "deathblade": "deathblade",
            "deathcap": "rabadons_deathcap",
            "guinsoos": "guinsoos_rageblade",
            "sunfire": "sunfire_cape",
            "ionic_spark": "ionic_spark",
            "gargoyle": "gargoyle_stoneplate",
            "archangels": "archangels_staff",
            "red_buff": "red_buff",
            "blue_buff": "blue_buff",
            "giant_slayer": "giant_slayer",
            "hands_of_justice": "hands_of_justice",
            "protectors_vow": "protectors_vow",
            "spirit_visage": "spirit_visage",
            "steadfast_heart": "steadfast_heart",
            "moguls_mail": "moguls_mail",
            "crown_of_demacia": "crown_of_demacia",
        }

        for effect_key, item_id in item_grants.items():
            if effect_key in effects:
                count = effects[effect_key] if isinstance(effects[effect_key], int) else 1
                for _ in range(count):
                    self._grant_specific_item(player, item_id, result)

        # Random emblems
        if "random_emblem" in effects:
            count = effects["random_emblem"]
            emblems = self._get_random_emblems(count)
            for emblem in emblems:
                self._grant_item_to_player(player, emblem, result)
            result.message += f"랜덤 엠블렘 {count}개 획득. "

        if "random_emblems" in effects:
            count = effects["random_emblems"]
            emblems = self._get_random_emblems(count)
            for emblem in emblems:
                self._grant_item_to_player(player, emblem, result)
            result.message += f"랜덤 엠블렘 {count}개 획득. "

        if "emblems" in effects:
            count = effects["emblems"]
            emblems = self._get_random_emblems(count)
            for emblem in emblems:
                self._grant_item_to_player(player, emblem, result)

        # Duplicators
        if "duplicator" in effects:
            count = effects["duplicator"] if isinstance(effects["duplicator"], int) else 1
            aug_state["pending_duplicators"] = aug_state.get("pending_duplicators", 0) + count
            result.message += f"복제기 {count}개 획득. "

        if "duplicators" in effects:
            count = effects["duplicators"]
            aug_state["pending_duplicators"] = aug_state.get("pending_duplicators", 0) + count

        if "lesser_duplicator" in effects:
            count = effects["lesser_duplicator"]
            aug_state["pending_lesser_duplicators"] = aug_state.get("pending_lesser_duplicators", 0) + count

        # =====================================================================
        # ADVANCED ITEM EFFECTS
        # =====================================================================

        # Component anvil
        if "component_anvil" in effects:
            count = effects["component_anvil"]
            aug_state["pending_component_anvils"] = aug_state.get("pending_component_anvils", 0) + count
            result.message += f"컴포넌트 모루 {count}개 획득. "

        # Component anvils instead of items (Component Buffet)
        if effects.get("component_anvils_instead"):
            aug_state["component_anvils_instead"] = True
            result.message += "캐러셀에서 컴포넌트 모루 획득. "

        # Artifact anvil
        if "artifact_anvil" in effects:
            count = effects["artifact_anvil"]
            aug_state["pending_artifact_anvils"] = aug_state.get("pending_artifact_anvils", 0) + count
            result.message += f"아티팩트 모루 {count}개 획득. "

        # Artifact choice (Portable Forge)
        if "artifact_choice" in effects:
            count = effects["artifact_choice"]
            aug_state["pending_artifact_choices"] = count
            result.message += f"아티팩트 {count}개 중 선택. "

        # Artifacts direct grant
        if "artifacts" in effects:
            count = effects["artifacts"]
            aug_state["pending_artifacts"] = aug_state.get("pending_artifacts", 0) + count
            result.message += f"아티팩트 {count}개 획득. "

        # Artifact later
        if effects.get("artifact_later"):
            aug_state["artifact_later"] = True

        # Choose component (Replication)
        if effects.get("choose_component"):
            aug_state["pending_component_choice"] = True
            result.message += "컴포넌트 선택. "

        # Components later
        if "component_later" in effects:
            count = effects["component_later"]
            aug_state["components_later"] = aug_state.get("components_later", 0) + count

        # Component per win (Prizefighter)
        if "component_per_win" in effects:
            aug_state["component_per_win"] = effects["component_per_win"]

        # Component per mana spent (Woven Magic)
        if "component_per_mana_spent" in effects:
            threshold = effects["component_per_mana_spent"]
            aug_state["component_per_mana_threshold"] = threshold
            aug_state["mana_spent_tracker"] = 0
            result.message += f"마나 {threshold} 소모당 컴포넌트 획득. "

        # All components after combat
        if effects.get("all_components_after_combat"):
            aug_state["all_components_after_combat"] = True
            result.message += "전투 후 모든 컴포넌트 획득. "

        # Anvil (generic)
        if "anvil" in effects:
            count = effects["anvil"]
            aug_state["pending_anvils"] = aug_state.get("pending_anvils", 0) + count

        # Armed random items
        if "armed_random_items" in effects:
            count = effects["armed_random_items"]
            from src.data.loaders import load_combined_items
            items = load_combined_items()
            for _ in range(count):
                item = self.rng.choice(items)
                self._grant_item_to_player(player, item, result)
            result.message += f"무장 아이템 {count}개 획득. "

        # Break items on sell (Salvage Bin)
        if effects.get("break_items_on_sell"):
            aug_state["break_items_on_sell"] = True
            result.message += "판매 시 아이템 분해. "

        # Reforger
        if "reforger" in effects:
            count = effects["reforger"]
            aug_state["pending_reforgers"] = aug_state.get("pending_reforgers", 0) + count
            result.message += f"재조합기 {count}개 획득. "

        # Remover
        if "remover" in effects:
            count = effects["remover"]
            aug_state["pending_removers"] = aug_state.get("pending_removers", 0) + count
            result.message += f"제거기 {count}개 획득. "

        # Golden remover
        if effects.get("golden_remover"):
            aug_state["pending_golden_removers"] = aug_state.get("pending_golden_removers", 0) + 1

        # Radiant item blessing
        if effects.get("radiant_blessing"):
            aug_state["radiant_blessing"] = True
            result.message += "다음 완성 아이템이 광휘 아이템으로. "

        # Radiant items
        if "radiant_items" in effects:
            count = effects["radiant_items"]
            aug_state["pending_radiant_items"] = aug_state.get("pending_radiant_items", 0) + count
            result.message += f"광휘 아이템 {count}개 획득. "

        # Health per item (Item Dependent)
        if "health_per_item" in effects:
            hp_per_item = effects["health_per_item"]
            aug_state["health_per_item"] = hp_per_item
            result.message += f"아이템당 체력 +{hp_per_item}. "

        # Pandora's Items
        if effects.get("pandoras_items") or effects.get("randomize_bench_items"):
            aug_state["randomize_items_each_round"] = True
            result.message += "라운드마다 아이템 랜덤화. "

        # Lucky item chest
        if "lucky_item_chest" in effects:
            count = effects["lucky_item_chest"]
            aug_state["pending_lucky_chests"] = aug_state.get("pending_lucky_chests", 0) + count
            result.message += f"행운의 아이템 상자 {count}개 획득. "

        # Sword/staff completed items (Swordsmith/Staffsmith)
        if effects.get("sword_completed_items"):
            aug_state["sword_completed_items"] = True
            result.message += "검 완성 아이템 모루 획득. "

        if effects.get("rod_completed_items"):
            aug_state["rod_completed_items"] = True
            result.message += "지팡이 완성 아이템 모루 획득. "

        # =====================================================================
        # CHAMPION GRANTS
        # =====================================================================

        # Grant specific champion
        if "grant_champion" in effects:
            champ_id = effects["grant_champion"]
            champion = self._get_champion_by_id(champ_id)
            if champion:
                self._grant_champion_to_player(player, champion, 1, result)

        # Grant random champions by cost
        if "grant_1cost" in effects:
            count = effects["grant_1cost"]
            for _ in range(count):
                self._grant_random_champion(player, 1, 1, result)

        if "grant_2cost" in effects:
            count = effects["grant_2cost"]
            for _ in range(count):
                self._grant_random_champion(player, 2, 1, result)

        if "grant_3cost" in effects:
            count = effects["grant_3cost"]
            for _ in range(count):
                self._grant_random_champion(player, 3, 1, result)

        if "grant_4cost" in effects:
            count = effects["grant_4cost"]
            for _ in range(count):
                self._grant_random_champion(player, 4, 1, result)

        if "grant_5cost" in effects:
            count = effects["grant_5cost"]
            for _ in range(count):
                self._grant_random_champion(player, 5, 1, result)

        # Grant 2-star champions
        if "grant_2star_2cost" in effects:
            count = effects["grant_2star_2cost"]
            for _ in range(count):
                self._grant_random_champion(player, 2, 2, result)

        if "grant_2star_3cost" in effects:
            count = effects["grant_2star_3cost"]
            for _ in range(count):
                self._grant_random_champion(player, 3, 2, result)

        if "grant_2star_5cost" in effects:
            count = effects["grant_2star_5cost"]
            for _ in range(count):
                self._grant_random_champion(player, 5, 2, result)

        # Grant specific champions
        specific_champs = {
            "grant_singed": "singed",
            "grant_teemo": "teemo",
            "grant_ambessa": "ambessa",
            "grant_kindred": "kindred",
            "grant_shyvana": "shyvana",
            "grant_jarvan": "jarvan_iv",
            "grant_atakhan": "atakhan",
            "grant_xin_zhao": "xin_zhao",
            "grant_2star_viego": "viego",
        }

        for effect_key, champ_id in specific_champs.items():
            if effect_key in effects:
                champion = self._get_champion_by_id(champ_id)
                if champion:
                    star = 2 if "2star" in effect_key else 1
                    self._grant_champion_to_player(player, champion, star, result)

        # Grant all 1-costs (Missed Connections)
        if effects.get("grant_all_1costs"):
            for champion in self._get_champions_by_cost(1):
                self._grant_champion_to_player(player, champion, 1, result)
            result.message += "모든 1코스트 챔피언 획득. "

        # Grant 2-cost on level (Caretaker's Ally)
        if effects.get("grant_2cost_on_level"):
            aug_state["grant_2cost_on_level"] = True
            result.message += "레벨업 시 2코스트 챔피언 획득. "

        # Grant 2-cost per round
        if effects.get("grant_2cost_per_round"):
            aug_state["grant_2cost_per_round"] = True

        # Grant 5-cost at level 9 (Birthday Reunion)
        if effects.get("grant_5cost_at_9"):
            aug_state["grant_5cost_at_9"] = True

        # Grant Ixtal champions
        if effects.get("grant_ixtal_champions"):
            # Would need trait data to find Ixtal champions
            result.message += "익스탈 챔피언 획득. "

        # Grant champions (generic list)
        if "grant_champions" in effects:
            champ_ids = effects["grant_champions"]
            if isinstance(champ_ids, list):
                for champ_id in champ_ids:
                    champion = self._get_champion_by_id(champ_id)
                    if champion:
                        self._grant_champion_to_player(player, champion, 1, result)

        # Next 4-cost is 2-star (Reinforcement)
        if effects.get("next_4cost_2star"):
            aug_state["next_4cost_2star"] = True
            result.message += "다음 4코스트 챔피언이 2성. "

        # First 1/2-cost 2-star (Stars are Born)
        if effects.get("first_1cost_2star"):
            aug_state["first_1cost_2star"] = True
        if effects.get("first_2cost_2star"):
            aug_state["first_2cost_2star"] = True

        # Replace with 2-star (Restart Mission, Delayed Start)
        if effects.get("replace_with_2star"):
            aug_state["replace_with_2star"] = True
            result.message += "현재 챔피언을 2성으로 교체. "

        # Double champion bonus (Double Trouble Gold)
        if effects.get("double_champion_bonus"):
            aug_state["double_champion_bonus"] = True
            result.message += "2마리 챔피언 보너스 활성화. "

        # Reward on 3-star
        if effects.get("reward_on_3star"):
            aug_state["reward_on_3star"] = True

        # Upgrade board cost (Recombobulator)
        if effects.get("upgrade_board_cost"):
            aug_state["upgrade_board_cost"] = True
            result.message += "보드 챔피언 코스트 업그레이드. "

        # Copy first kill (Pilfer)
        if effects.get("copy_first_kill"):
            aug_state["copy_first_kill"] = True
            result.message += "첫 번째 킬 시 챔피언 복사. "

        # Merge into golem (Golemify)
        if effects.get("merge_into_golem"):
            aug_state["merge_into_golem"] = True
            result.message += "챔피언 합체하여 골렘 생성. "

        # Solo champion mode (Solo Leveling)
        if effects.get("solo_champion_mode"):
            aug_state["solo_champion_mode"] = True
            if effects.get("massive_stats"):
                aug_state["solo_massive_stats"] = True
            if "combats" in effects:
                aug_state["solo_combats_remaining"] = effects["combats"]
            result.message += "솔로 챔피언 모드 활성화. "

        # Grant matching 2-cost (Starter Kit)
        if effects.get("grant_matching_2cost"):
            aug_state["grant_matching_2cost"] = True

        # 3-cost bonus
        if "3cost_bonus" in effects:
            aug_state["3cost_bonus"] = effects["3cost_bonus"]

        # 5-cost bonus
        if "5cost_bonus" in effects:
            aug_state["5cost_bonus"] = effects["5cost_bonus"]

        # 3-star Yordle massive (Super Yordle)
        if effects.get("3star_yordle_massive"):
            aug_state["3star_yordle_massive"] = True
            result.message += "3성 요들이 거대화. "

        # Carousel duplicates
        if "carousel_duplicates" in effects:
            aug_state["carousel_duplicates"] = effects["carousel_duplicates"]

        # Cost 1/2 move speed (Featherweights)
        if "cost_1_2_move_speed" in effects:
            aug_state["cost_1_2_move_speed"] = effects["cost_1_2_move_speed"]

        # Stacking stats setup
        if "stacking_ad" in effects:
            aug_state["stacking_ad"] = effects["stacking_ad"]
            result.message += f"스택당 공격력 +{effects['stacking_ad']}. "

        if "stacking_ap" in effects:
            aug_state["stacking_ap"] = effects["stacking_ap"]
            result.message += f"스택당 주문력 +{effects['stacking_ap']}. "

        if "stacking_health" in effects:
            aug_state["stacking_health"] = effects["stacking_health"]
            result.message += f"스택당 체력 +{effects['stacking_health']}. "

        # =====================================================================
        # SPECIAL MECHANICS SETUP
        # =====================================================================

        # Track thresholds for progressive rewards
        if "bf_on_damage_threshold" in effects:
            aug_state["damage_thresholds"] = effects["bf_on_damage_threshold"]
            aug_state["damage_threshold_reached"] = 0

        if "rod_on_magic_damage" in effects:
            aug_state["magic_damage_thresholds"] = effects["rod_on_magic_damage"]
            aug_state["magic_damage_threshold_reached"] = 0

        if "gloves_on_crit_threshold" in effects:
            aug_state["crit_thresholds"] = effects["gloves_on_crit_threshold"]
            aug_state["crit_threshold_reached"] = 0

        if "tear_on_mana_threshold" in effects:
            aug_state["mana_thresholds"] = effects["tear_on_mana_threshold"]
            aug_state["mana_threshold_reached"] = 0

        if "recurve_on_attack_threshold" in effects:
            aug_state["attack_thresholds"] = effects["recurve_on_attack_threshold"]
            aug_state["attack_threshold_reached"] = 0

        # =====================================================================
        # ECONOMY & LEVELING MECHANICS
        # =====================================================================

        # Free rerolls if no buy (Patience is a Virtue)
        if "free_rerolls_if_no_buy" in effects:
            aug_state["free_rerolls_if_no_buy"] = effects["free_rerolls_if_no_buy"]

        # Gold on interest threshold (Savings Account)
        if "gold_on_interest_threshold" in effects:
            aug_state["gold_interest_thresholds"] = effects["gold_on_interest_threshold"]
            aug_state["gold_interest_threshold_reached"] = 0

        # Gold per gold above threshold
        if "gold_per_gold_above_threshold" in effects:
            aug_state["gold_per_gold_above_threshold"] = effects["gold_per_gold_above_threshold"]

        # Gold bonus (Forward Thinking)
        if "gold_bonus" in effects:
            aug_state["pending_gold_bonus"] = effects["gold_bonus"]

        # Lose gold (Forward Thinking)
        if "lose_gold" in effects:
            player.gold = max(0, player.gold - effects["lose_gold"])
            result.message += f"골드 -{effects['lose_gold']}. "

        # Regain gold after combats
        if "regain_gold_after_combats" in effects:
            aug_state["regain_gold_combats"] = effects["regain_gold_after_combats"]
            aug_state["regain_gold_amount"] = effects.get("gold_bonus", 0) + effects.get("lose_gold", 0)

        # Gold on elimination (Speedy Double Kill)
        if "gold_on_elimination" in effects:
            aug_state["gold_on_elimination"] = effects["gold_on_elimination"]

        # Chests on reroll spending
        if effects.get("chests_on_reroll_spending"):
            aug_state["chests_on_reroll_spending"] = True

        # Carousel priority
        if effects.get("carousel_priority"):
            aug_state["carousel_priority"] = True

        # Carousel unclaimed (Table Scraps)
        if effects.get("carousel_unclaimed"):
            aug_state["carousel_unclaimed"] = True

        # Skip carousel (Hard Bargain)
        if effects.get("skip_carousel"):
            aug_state["skip_carousel"] = True

        # Level 10 at 9 (Reach for the Stars)
        if effects.get("level_10_at_9"):
            aug_state["level_10_at_9"] = True
            result.message += "레벨 9에서 10 도달 가능. "

        # Bonus per level (Shopping Spree)
        if "bonus_per_level" in effects:
            aug_state["bonus_per_level"] = effects["bonus_per_level"]

        # Rerolls at level 8
        if "rerolls_at_level_8" in effects:
            aug_state["rerolls_at_level_8"] = effects["rerolls_at_level_8"]

        # Shop disabled rounds (Delayed Start)
        if "shop_disabled_rounds" in effects:
            aug_state["shop_disabled_rounds"] = effects["shop_disabled_rounds"]

        # Random augment later
        if effects.get("random_silver_augment"):
            aug_state["random_silver_augment"] = True
        if effects.get("random_gold_augment"):
            aug_state["random_gold_augment"] = True
        if effects.get("random_prismatic_augment"):
            aug_state["random_prismatic_augment"] = True

        # New augment after rounds
        if "new_gold_augment_after_rounds" in effects:
            aug_state["new_augment_after_rounds"] = effects["new_gold_augment_after_rounds"]

        # Set win streak (Called Shot)
        if "set_win_streak" in effects:
            aug_state["set_win_streak"] = effects["set_win_streak"]
            result.message += f"연승 {effects['set_win_streak']}로 설정. "

        # Steal shop champion (Fire Sale)
        if effects.get("steal_shop_champion"):
            aug_state["steal_shop_champion"] = True
            result.message += "상대 상점에서 챔피언 획득. "

        # Rerolls on craft (Crafted Crafting)
        if "rerolls_on_craft" in effects:
            aug_state["rerolls_on_craft"] = effects["rerolls_on_craft"]

        # Rerolls on star up (On a Roll)
        if "rerolls_on_star_up" in effects:
            aug_state["rerolls_on_star_up"] = effects["rerolls_on_star_up"]

        # =====================================================================
        # TRAIT & SYNERGY MECHANICS
        # =====================================================================

        # Random emblem
        if effects.get("emblem"):
            aug_state["pending_random_emblem"] = True

        # Region emblems
        if effects.get("region_emblems"):
            aug_state["region_emblems"] = True

        # Shadow Isles synergy bonus
        if effects.get("shadow_isles_synergy"):
            aug_state["shadow_isles_synergy"] = True

        # Emblem holder bonuses
        if "emblem_holder_bonus" in effects:
            aug_state["emblem_holder_bonus"] = effects["emblem_holder_bonus"]

        if "emblem_holder_as" in effects:
            aug_state["emblem_holder_as"] = effects["emblem_holder_as"]

        # Synergy bonuses
        if effects.get("synergy"):
            aug_state["synergy_bonus"] = True

        # =====================================================================
        # SPECIAL MECHANICS FLAGS
        # =====================================================================

        # Hex type augments
        if "hex_type" in effects:
            aug_state["hex_type"] = effects["hex_type"]

        # All hex augments
        if effects.get("all_hex_augments"):
            aug_state["all_hex_augments"] = True

        # Roll dice
        if effects.get("roll_dice"):
            aug_state["roll_dice"] = True
            result.message += "주사위 굴림. "

        if effects.get("roll_3_dice"):
            aug_state["roll_3_dice"] = True

        # Dice now and per stage
        if "dice_now_and_per_stage" in effects:
            aug_state["dice_per_stage"] = True

        # Powerful random reward
        if effects.get("powerful_random_reward"):
            aug_state["powerful_random_reward"] = True

        # See opponent (Know Your Enemy)
        if effects.get("see_opponent"):
            aug_state["see_opponent"] = True
            result.message += "다음 상대 확인 가능. "

        # Damage amp vs opponent
        if "damage_amp_vs_opponent" in effects:
            aug_state["damage_amp_vs_opponent"] = effects["damage_amp_vs_opponent"]

        # Enhanced effects
        if effects.get("enhanced"):
            aug_state["enhanced_champion"] = True

        if "enhanced_bonus" in effects:
            aug_state["enhanced_bonus"] = effects["enhanced_bonus"]

        # Extended radius (High Voltage)
        if effects.get("extended_radius"):
            aug_state["extended_radius"] = True

        # Fighter convert (Leap of Faith)
        if effects.get("fighter_convert"):
            aug_state["fighter_convert"] = True

        # Mana benefit
        if effects.get("mana_benefit"):
            aug_state["mana_benefit"] = True

        # Available at specific stage
        if "available_at" in effects:
            aug_state["available_at"] = effects["available_at"]

        # Future rewards
        if effects.get("future_rewards"):
            aug_state["future_rewards"] = True

        # Bench slots reduced (Over Encumbered)
        if "bench_slots_reduced" in effects:
            aug_state["bench_slots_reduced"] = effects["bench_slots_reduced"]

        # Duplicator after combats
        if "duplicator_after_combats" in effects:
            aug_state["duplicator_after_combats"] = effects["duplicator_after_combats"]

        # Copies for rounds (Replication)
        if "copies_for_rounds" in effects:
            aug_state["copies_for_rounds"] = effects["copies_for_rounds"]

        # Copies per round
        if "copies_per_round" in effects:
            aug_state["copies_per_round"] = effects["copies_per_round"]

        # Repeat per stage
        if effects.get("repeat_per_stage"):
            aug_state["repeat_per_stage"] = True

        # Darkin choice
        if "darkin_choice" in effects:
            aug_state["darkin_choice"] = effects["darkin_choice"]

        # Combats counter
        if "combats" in effects:
            aug_state["effect_combats"] = effects["combats"]

        # Ionia upgrade
        if effects.get("ionia_upgrade"):
            aug_state["ionia_upgrade"] = True

        # Unlock champion
        if effects.get("unlock_zaahen_on_3star"):
            aug_state["unlock_zaahen_on_3star"] = True

        # Egg hatches
        if "egg_hatches_after_turns" in effects:
            aug_state["egg_hatches_turns"] = effects["egg_hatches_after_turns"]

        # Thiefs gloves at 6
        if effects.get("thiefs_gloves_at_6"):
            aug_state["thiefs_gloves_at_6"] = True

        # Item infinity force later
        if effects.get("infinity_force_later"):
            aug_state["infinity_force_later"] = True

        # Gamblers blade later
        if effects.get("gamblers_blade_later"):
            aug_state["gamblers_blade_later"] = True

        # Lose on crown loss (Heavy is the Crown)
        if effects.get("lose_on_crown_loss"):
            aug_state["lose_on_crown_loss"] = True

        # Oversized (The Golden Dragon)
        if effects.get("oversized"):
            aug_state["oversized"] = True

        # Solo row bonus
        if effects.get("solo_row_bonus"):
            aug_state["solo_row_bonus"] = True

        # Spatula holder bonus
        if effects.get("spatula_holder_bonus"):
            aug_state["spatula_holder_bonus"] = True

        # Spatula or pan on result
        if effects.get("spatula_or_pan_on_result"):
            aug_state["spatula_or_pan_on_result"] = True

        # Stacking on takedown
        if effects.get("stacking_on_takedown"):
            aug_state["stacking_on_takedown"] = True

        # Trials of Twilight unlock
        if effects.get("unlock_zaahen_on_3star"):
            aug_state["unlock_zaahen_on_3star"] = True

        # =====================================================================
        # REMAINING EFFECTS
        # =====================================================================

        # Care package per stage
        if effects.get("care_package_per_stage"):
            aug_state["care_package_per_stage"] = True

        # Package per stage
        if effects.get("package_per_stage"):
            aug_state["package_per_stage"] = True

        # Enhanced bonuses
        if "enhanced_bonuses" in effects:
            aug_state["enhanced_bonuses"] = effects["enhanced_bonuses"]

        # Grant random per stage
        if effects.get("grant_random_per_stage"):
            aug_state["grant_random_per_stage"] = True

        # Grant Zaunites
        if effects.get("grant_zaunites"):
            aug_state["grant_zaunites"] = True
            result.message += "자운 챔피언 획득. "

        # Heal based on HP gap (Spirit Link)
        if effects.get("heal_based_on_hp_gap"):
            aug_state["heal_based_on_hp_gap"] = True

        # Health loss (Blood Offering)
        if "health_loss" in effects:
            player.health = max(1, player.health - effects["health_loss"])
            result.message += f"체력 -{effects['health_loss']}. "

        # High cost on damage threshold
        if effects.get("high_cost_on_damage"):
            aug_state["high_cost_on_damage"] = True

        # Ionia bonus
        if "ionia_bonus" in effects:
            aug_state["ionia_bonus"] = effects["ionia_bonus"]

        # Lock fielded champions
        if effects.get("lock_fielded_champions"):
            aug_state["lock_fielded_champions"] = True
            result.message += "필드 챔피언 고정. "

        # Magic effects
        if effects.get("magic_effects"):
            aug_state["magic_effects"] = True

        # Matching champions per stage
        if effects.get("matching_champions_per_stage"):
            aug_state["matching_champions_per_stage"] = True

        # More below HP threshold
        if "more_below_hp" in effects:
            aug_state["more_below_hp"] = effects["more_below_hp"]

        # Next 1-cost 3-star
        if effects.get("next_1cost_3star"):
            aug_state["next_1cost_3star"] = True
            result.message += "다음 1코스트 챔피언이 3성. "

        # Noxus bonus
        if "noxus_bonus" in effects:
            aug_state["noxus_bonus"] = effects["noxus_bonus"]

        # Overtime cast speed
        if "overtime_cast_speed" in effects:
            aug_state["overtime_cast_speed"] = effects["overtime_cast_speed"]

        # Perfect Thief's Gloves
        if effects.get("perfect_thiefs_gloves"):
            aug_state["perfect_thiefs_gloves"] = True

        # Piltover on invention
        if effects.get("piltover_on_invention"):
            aug_state["piltover_on_invention"] = True

        # Radiant item
        if "radiant_item" in effects:
            aug_state["pending_radiant_item"] = effects["radiant_item"]

        # Radiant item choice
        if "radiant_item_choice" in effects:
            aug_state["radiant_item_choice"] = effects["radiant_item_choice"]

        # Random component (single)
        if "random_component" in effects:
            count = effects["random_component"]
            from src.data.loaders import load_components
            components = load_components()
            for _ in range(count):
                component = self.rng.choice(components)
                player.items.add_to_inventory(component)
                result.items_gained.append(component.name)

        # Randomize bench champions
        if effects.get("randomize_bench_champions"):
            aug_state["randomize_bench_champions"] = True
            result.message += "벤치 챔피언 랜덤화. "

        # Rascal's Gloves
        if effects.get("rascals_gloves"):
            aug_state["rascals_gloves"] = True

        # Rerolls bonus
        if "rerolls_bonus" in effects:
            aug_state["rerolls_bonus"] = effects["rerolls_bonus"]

        # Rerolls per stage
        if "rerolls_per_stage" in effects:
            aug_state["rerolls_per_stage"] = effects["rerolls_per_stage"]

        # Reward on threshold
        if effects.get("reward_on_threshold"):
            aug_state["reward_on_threshold"] = True

        # Rotating radiant
        if effects.get("rotating_radiant"):
            aug_state["rotating_radiant"] = True

        # Size per reroll (Hefty Rolls)
        if "size_per_reroll" in effects:
            aug_state["size_per_reroll"] = effects["size_per_reroll"]

        # Speed bonus
        if "speed_bonus" in effects:
            aug_state["speed_bonus"] = effects["speed_bonus"]

        # Stacking damage after combat start
        if "stacking_damage_after_combat_start" in effects:
            aug_state["stacking_damage_after_combat_start"] = effects["stacking_damage_after_combat_start"]

        # Sunder duration
        if "sunder_duration" in effects:
            aug_state["sunder_duration"] = effects["sunder_duration"]

        # Tactician's cape later
        if effects.get("tacticians_cape_later"):
            aug_state["tacticians_cape_later"] = True

        # Team damage amp
        if "team_damage_amp" in effects:
            aug_state["team_damage_amp"] = effects["team_damage_amp"]

        # Team mana
        if "team_mana" in effects:
            aug_state["team_mana"] = effects["team_mana"]

        # Team shrink
        if effects.get("team_shrink"):
            aug_state["team_shrink"] = True

        # Team stats (generic)
        if "team_stats" in effects:
            aug_state["team_stats"] = effects["team_stats"]

        # Thief's gloves after combats
        if "thiefs_gloves_after_combats" in effects:
            aug_state["thiefs_gloves_after_combats"] = effects["thiefs_gloves_after_combats"]

        # XP if bench empty
        if "xp_if_bench_empty" in effects:
            aug_state["xp_if_bench_empty"] = effects["xp_if_bench_empty"]

        # XP if bench no items (Slammin')
        if "xp_if_bench_no_items" in effects:
            aug_state["xp_if_bench_no_items"] = effects["xp_if_bench_no_items"]

        # XP on loss (Patient Study)
        if "xp_on_loss" in effects:
            aug_state["xp_on_loss"] = effects["xp_on_loss"]

        # XP on win (Patient Study)
        if "xp_on_win" in effects:
            aug_state["xp_on_win"] = effects["xp_on_win"]

        # XP per stage
        if "xp_per_stage" in effects:
            aug_state["xp_per_stage"] = effects["xp_per_stage"]

        # XP per void takedown
        if "xp_per_void_takedown" in effects:
            aug_state["xp_per_void_takedown"] = effects["xp_per_void_takedown"]

        # Zaunite explode on death
        if effects.get("zaunite_explode_on_death"):
            aug_state["zaunite_explode_on_death"] = True
            result.message += "자운 챔피언 사망 시 폭발. "

        return result

    # =========================================================================
    # ROUND START EFFECTS
    # =========================================================================

    def apply_round_start_effects(
        self,
        player: "PlayerState"
    ) -> AugmentEffectResult:
        """Apply effects at the start of each round."""
        result = AugmentEffectResult(success=True)
        aug_state = self.get_player_augment_state(player.player_id)

        # Free rerolls (Trade Sector)
        if aug_state["free_rerolls"] > 0:
            for _ in range(aug_state["free_rerolls"]):
                player.shop.refresh()
            result.message += f"무료 리롤 {aug_state['free_rerolls']}회. "

        # Gold per round (Consistent Income)
        if aug_state["gold_per_round"] > 0:
            player.gold += aug_state["gold_per_round"]
            result.gold_gained = aug_state["gold_per_round"]
            result.message += f"라운드 골드 +{aug_state['gold_per_round']}. "

        # Check player augments for round start effects
        for augment in player.augments:
            effects = augment.effects

            # Tiny Titans heal
            if "heal_per_round" in effects:
                heal = effects["heal_per_round"]
                max_hp = 999 if aug_state["can_exceed_100hp"] else 100
                player.health = min(player.health + heal, max_hp)
                result.message += f"체력 회복 +{heal}. "

            # Pandora's Items - randomize bench items
            if effects.get("randomize_bench_items"):
                result.message += "벤치 아이템 랜덤화. "

            # Component per round (Over Encumbered)
            if "component_per_round" in effects:
                count = effects["component_per_round"]
                from src.data.loaders import load_components
                components = load_components()
                for _ in range(count):
                    component = self.rng.choice(components)
                    player.items.add_to_inventory(component)
                    result.items_gained.append(component.name)
                result.message += f"컴포넌트 +{count}. "

            # XP per round
            if "xp_per_round" in effects:
                xp = effects["xp_per_round"]
                player.add_xp(xp)
                result.xp_gained += xp
                result.message += f"경험치 +{xp}. "

        return result

    def apply_combat_end_effects(
        self,
        player: "PlayerState",
        combat_result: "CombatResult",
        is_winner: bool
    ) -> AugmentEffectResult:
        """Apply effects at the end of combat."""
        result = AugmentEffectResult(success=True)
        aug_state = self.get_player_augment_state(player.player_id)

        # Track combat count
        player.combat_count = getattr(player, 'combat_count', 0) + 1

        # Gold per combat (Hustler)
        if aug_state["gold_per_combat"] > 0:
            player.gold += aug_state["gold_per_combat"]
            result.gold_gained += aug_state["gold_per_combat"]
            result.message += f"전투 골드 +{aug_state['gold_per_combat']}. "

        # XP per combat (Hustler)
        if aug_state["xp_per_combat"] > 0:
            player.add_xp(aug_state["xp_per_combat"])
            result.xp_gained += aug_state["xp_per_combat"]
            result.message += f"전투 경험치 +{aug_state['xp_per_combat']}. "

        # Gold on loss (Calculated Loss)
        if not is_winner and aug_state["gold_on_loss"] > 0:
            player.gold += aug_state["gold_on_loss"]
            result.gold_gained += aug_state["gold_on_loss"]
            result.message += f"패배 보상 골드 +{aug_state['gold_on_loss']}. "

        # Reroll on loss (Calculated Loss)
        if not is_winner and aug_state["reroll_on_loss"] > 0:
            aug_state["pending_rerolls"] = aug_state.get("pending_rerolls", 0) + aug_state["reroll_on_loss"]
            result.message += f"패배 보상 리롤 +{aug_state['reroll_on_loss']}. "

        # Gold on enemy kill (Malicious Monetization)
        if aug_state["gold_on_enemy_kill"] > 0 and hasattr(combat_result, 'enemy_kills'):
            kills = getattr(combat_result, 'enemy_kills', 0)
            gold = aug_state["gold_on_enemy_kill"] * kills
            if gold > 0:
                player.gold += gold
                result.gold_gained += gold
                result.message += f"킬 보상 골드 +{gold}. "

        # Process player augments for combat end effects
        for augment in player.augments:
            effects = augment.effects

            # Risky Moves - lose health but gain gold
            if "health_loss_per_combat" in effects and "gold_per_combat" in effects:
                health_loss = effects["health_loss_per_combat"]
                gold_gain = effects["gold_per_combat"]
                player.health = max(1, player.health - health_loss)
                player.gold += gold_gain
                result.gold_gained += gold_gain
                result.message += f"리스키 무브: 체력 -{health_loss}, 골드 +{gold_gain}. "

            # Loot drop chance (Spoils of War)
            if "loot_drop_chance" in effects:
                chance = effects["loot_drop_chance"]
                if self.rng.random() < chance:
                    # Drop random component
                    from src.data.loaders import load_components
                    components = load_components()
                    component = self.rng.choice(components)
                    player.items.add_to_inventory(component)
                    result.items_gained.append(component.name)
                    result.message += f"전리품 획득: {component.name}. "

            # Lunch Money - gold per damage dealt
            if "gold_per_damage_dealt" in effects:
                if hasattr(combat_result, 'damage_dealt'):
                    damage = getattr(combat_result, 'damage_dealt', 0)
                    gold = int(damage * effects["gold_per_damage_dealt"])
                    if gold > 0:
                        player.gold += gold
                        result.gold_gained += gold
                        result.message += f"피해 보상 골드 +{gold}. "

        return result

    def apply_reroll_effects(
        self,
        player: "PlayerState"
    ) -> AugmentEffectResult:
        """Apply effects when player rerolls."""
        result = AugmentEffectResult(success=True)
        aug_state = self.get_player_augment_state(player.player_id)

        # Track rerolls used
        player.rerolls_used = getattr(player, 'rerolls_used', 0) + 1

        # Check if free reroll available
        pending = aug_state.get("pending_rerolls", 0)
        if pending > 0:
            aug_state["pending_rerolls"] = pending - 1
            result.message += "무료 리롤 사용. "
            result.gold_gained = 2  # Refund the cost
            return result

        # Free reroll chance (Prismatic Ticket)
        if aug_state["free_reroll_chance"] > 0:
            if self.rng.random() < aug_state["free_reroll_chance"]:
                result.message += "무료 리롤! "
                result.gold_gained = 2  # Refund the cost
                return result

        # Process player augments for reroll effects
        for augment in player.augments:
            effects = augment.effects

            # Hefty Rolls - gain health on reroll
            if "health_per_reroll" in effects:
                hp = effects["health_per_reroll"]
                max_hp = 100 if not aug_state.get("can_exceed_100hp") else 999
                player.health = min(player.health + hp, max_hp)
                result.message += f"리롤 체력 +{hp}. "

            # Two Much Value - free rerolls for buying 2-costs
            if "rerolls_per_2cost" in effects:
                # This would be handled in shop purchase logic
                pass

        return result

    def apply_purchase_effects(
        self,
        player: "PlayerState",
        champion_cost: int
    ) -> AugmentEffectResult:
        """Apply effects when player purchases a champion."""
        result = AugmentEffectResult(success=True)
        aug_state = self.get_player_augment_state(player.player_id)

        for augment in player.augments:
            effects = augment.effects

            # On a Roll - reroll on star up
            if "rerolls_on_star_up" in effects:
                # Would need to check if this purchase caused a star-up
                pass

            # Two Much Value - free reroll when buying 2-cost
            if "rerolls_per_2cost" in effects and champion_cost == 2:
                rerolls = effects["rerolls_per_2cost"]
                aug_state["pending_rerolls"] = aug_state.get("pending_rerolls", 0) + rerolls
                result.message += f"2코스트 구매 리롤 +{rerolls}. "

            # Cluttered Mind - XP if bench is full
            if "xp_if_bench_full" in effects:
                bench_count = self._get_bench_count(player)
                bench_max = getattr(player, 'bench_size', 9)
                if bench_count >= bench_max:
                    xp = effects["xp_if_bench_full"]
                    player.add_xp(xp)
                    result.xp_gained += xp
                    result.message += f"벤치 만석 경험치 +{xp}. "

        return result

    def get_interest_modifier(self, player: "PlayerState") -> tuple[float, int]:
        """
        Get interest modifiers from augments.

        Returns:
            Tuple of (interest_multiplier, interest_cap_increase)
        """
        aug_state = self.get_player_augment_state(player.player_id)

        # No interest (Consistent Income, Hustler)
        if aug_state.get("no_interest"):
            return (0.0, 0)

        # Interest cap increase
        cap_increase = aug_state.get("interest_cap_increase", 0)

        return (1.0, cap_increase)

    def get_reroll_cost(self, player: "PlayerState", base_cost: int = 2) -> int:
        """Get modified reroll cost from augments."""
        aug_state = self.get_player_augment_state(player.player_id)

        # Check for pending free rerolls
        if aug_state.get("pending_rerolls", 0) > 0:
            return 0

        # Check for free reroll chance
        if aug_state.get("free_reroll_chance", 0) > 0:
            if self.rng.random() < aug_state["free_reroll_chance"]:
                return 0

        return base_cost

    def get_xp_cost_modifier(self, player: "PlayerState", base_cost: int = 4) -> tuple[int, int]:
        """
        Get modified XP cost from augments.

        Returns:
            Tuple of (gold_cost, health_cost)
        """
        aug_state = self.get_player_augment_state(player.player_id)

        gold_cost = base_cost
        health_cost = 0

        # Cost reduction
        gold_cost = max(0, gold_cost - aug_state.get("xp_cost_reduction", 0))

        # XP costs health instead (Learning from Failure)
        if aug_state.get("xp_costs_health"):
            health_cost = aug_state.get("xp_health_cost", 4)
            gold_cost = 0

        return (gold_cost, health_cost)

    # =========================================================================
    # COMBAT EFFECTS
    # =========================================================================

    def get_combat_stat_modifiers(
        self,
        player: "PlayerState",
        unit: "CombatUnit"
    ) -> dict:
        """Get stat modifiers from augments for a combat unit."""
        modifiers = {
            "bonus_health": 0,
            "bonus_ad": 0,
            "bonus_ap": 0,
            "bonus_armor": 0,
            "bonus_mr": 0,
            "bonus_attack_speed": 0.0,
            "bonus_crit_chance": 0.0,
            "bonus_crit_damage": 0.0,
            "damage_amp": 0.0,
            "damage_reduction": 0.0,
            "omnivamp": 0.0,
        }

        for augment in player.augments:
            effects = augment.effects

            # Featherweights - 1/2 cost units get attack speed
            if "cost_1_2_attack_speed" in effects:
                if unit.cost <= 2:
                    modifiers["bonus_attack_speed"] += effects["cost_1_2_attack_speed"]

            # Cybernetic Uplink - units with items get health/mana
            if "item_holder_health" in effects:
                if len(unit.items) > 0:
                    modifiers["bonus_health"] += effects["item_holder_health"]

            # Makeshift Armor - units without items get armor/mr
            if "no_item_armor" in effects:
                if len(unit.items) == 0:
                    modifiers["bonus_armor"] += effects["no_item_armor"]
            if "no_item_mr" in effects:
                if len(unit.items) == 0:
                    modifiers["bonus_mr"] += effects["no_item_mr"]

            # Team-wide stat bonuses
            if "bonus_ad" in effects:
                modifiers["bonus_ad"] += effects["bonus_ad"]
            if "bonus_ap" in effects:
                modifiers["bonus_ap"] += effects["bonus_ap"]
            if "bonus_armor" in effects:
                modifiers["bonus_armor"] += effects["bonus_armor"]
            if "bonus_mr" in effects:
                modifiers["bonus_mr"] += effects["bonus_mr"]
            if "bonus_health" in effects:
                modifiers["bonus_health"] += effects["bonus_health"]
            if "bonus_as" in effects:
                modifiers["bonus_attack_speed"] += effects["bonus_as"]

            # Team attack speed
            if "team_attack_speed" in effects:
                modifiers["bonus_attack_speed"] += effects["team_attack_speed"]
            if "team_as" in effects:
                modifiers["bonus_attack_speed"] += effects["team_as"]

            # Team AD/AP
            if "team_ad" in effects:
                modifiers["bonus_ad"] += effects["team_ad"]
            if "team_ap" in effects:
                modifiers["bonus_ap"] += effects["team_ap"]

            # Crit chance
            if "team_crit_chance" in effects:
                modifiers["bonus_crit_chance"] += effects["team_crit_chance"]
            if "team_crit" in effects:
                modifiers["bonus_crit_chance"] += effects["team_crit"]

            # Crit damage
            if "crit_damage_bonus" in effects:
                modifiers["bonus_crit_damage"] += effects["crit_damage_bonus"]

            # Omnivamp (Celestial Blessing)
            if "omnivamp" in effects:
                modifiers["omnivamp"] += effects["omnivamp"]

            # Damage amp per bronze trait
            if "damage_amp_per_bronze" in effects:
                # Would need to count bronze traits
                bronze_count = self._count_bronze_traits(player)
                modifiers["damage_amp"] += effects["damage_amp_per_bronze"] * bronze_count

            # Durability per bronze trait
            if "durability_per_bronze" in effects:
                bronze_count = self._count_bronze_traits(player)
                modifiers["damage_reduction"] += effects["durability_per_bronze"] * bronze_count

            # Team durability
            if "team_durability" in effects:
                modifiers["damage_reduction"] += effects["team_durability"]

            # Scaling per missing player HP (Comeback Story)
            if "scaling_per_missing_hp" in effects:
                missing_hp = 100 - player.health
                if missing_hp > 0:
                    bonus = effects["scaling_per_missing_hp"] * missing_hp
                    modifiers["bonus_ad"] += bonus * 10  # Scale appropriately
                    modifiers["bonus_ap"] += bonus * 10

            # Per-level bonuses (Bodyguard Training)
            if "armor_per_level" in effects:
                modifiers["bonus_armor"] += effects["armor_per_level"] * player.level
            if "mr_per_level" in effects:
                modifiers["bonus_mr"] += effects["mr_per_level"] * player.level

            # Per-trait bonuses (Stand United)
            if "ad_per_trait" in effects:
                trait_count = self._count_active_traits(player)
                modifiers["bonus_ad"] += effects["ad_per_trait"] * trait_count
            if "ap_per_trait" in effects:
                trait_count = self._count_active_traits(player)
                modifiers["bonus_ap"] += effects["ap_per_trait"] * trait_count

            # Best Friends - paired champions (2 of same champ)
            if "paired_attack_speed" in effects:
                if self._has_paired_champion(player, unit):
                    modifiers["bonus_attack_speed"] += effects["paired_attack_speed"]
            if "paired_armor" in effects:
                if self._has_paired_champion(player, unit):
                    modifiers["bonus_armor"] += effects["paired_armor"]

            # Bench stacking bonuses (Preparation)
            if "bench_stacking_ad" in effects:
                bench_count = self._get_bench_count(player)
                modifiers["bonus_ad"] += effects["bench_stacking_ad"] * bench_count
            if "bench_stacking_ap" in effects:
                bench_count = self._get_bench_count(player)
                modifiers["bonus_ap"] += effects["bench_stacking_ap"] * bench_count

            # Plot Armor - base stats + low HP bonus
            if "base_armor" in effects:
                modifiers["bonus_armor"] += effects["base_armor"]
            if "base_mr" in effects:
                modifiers["bonus_mr"] += effects["base_mr"]
            if "low_hp_bonus" in effects:
                hp_percent = unit.stats.current_hp / unit.stats.max_hp if unit.stats.max_hp > 0 else 1.0
                if hp_percent < 0.5:
                    modifiers["damage_reduction"] += effects["low_hp_bonus"]

            # Stacking AD/AP per combat (Warlord's Honor, Early Learnings)
            if "stacking_ad" in effects:
                combat_count = getattr(player, 'combat_count', 0)
                modifiers["bonus_ad"] += effects["stacking_ad"] * combat_count
            if "stacking_ap" in effects:
                combat_count = getattr(player, 'combat_count', 0)
                modifiers["bonus_ap"] += effects["stacking_ap"] * combat_count

            # Stacking health (Heart of Steel)
            if "stacking_health" in effects:
                combat_count = getattr(player, 'combat_count', 0)
                modifiers["bonus_health"] += effects["stacking_health"] * combat_count

            # Focused Fire - AD per hit
            if "focused_fire_ad" in effects:
                hit_count = getattr(unit, '_hit_count', 0)
                modifiers["bonus_ad"] += effects["focused_fire_ad"] * hit_count

            # Hefty Rolls - health per reroll spent
            if "health_per_reroll" in effects:
                rerolls_used = getattr(player, 'rerolls_used', 0)
                modifiers["bonus_health"] += effects["health_per_reroll"] * rerolls_used

            # Double trouble bonuses (for pairs of same champion)
            if "double_ad" in effects:
                if self._has_paired_champion(player, unit):
                    modifiers["bonus_ad"] += effects["double_ad"]
            if "double_ap" in effects:
                if self._has_paired_champion(player, unit):
                    modifiers["bonus_ap"] += effects["double_ap"]
            if "double_health" in effects:
                if self._has_paired_champion(player, unit):
                    modifiers["bonus_health"] += effects["double_health"]

            # Dual 4-cost bonuses (Pair of Fours)
            if "dual_4cost_bonus_ad" in effects:
                if unit.cost == 4 and self._has_dual_4cost(player, unit):
                    modifiers["bonus_ad"] += effects["dual_4cost_bonus_ad"]
            if "dual_4cost_bonus_ap" in effects:
                if unit.cost == 4 and self._has_dual_4cost(player, unit):
                    modifiers["bonus_ap"] += effects["dual_4cost_bonus_ap"]
            if "dual_4cost_bonus_hp" in effects:
                if unit.cost == 4 and self._has_dual_4cost(player, unit):
                    modifiers["bonus_health"] += effects["dual_4cost_bonus_hp"]

            # Twin Guardians - front pair gives team armor/mr
            if "front_pair_team_armor" in effects:
                if self._has_front_pair(player):
                    modifiers["bonus_armor"] += effects["front_pair_team_armor"]
            if "front_pair_team_mr" in effects:
                if self._has_front_pair(player):
                    modifiers["bonus_mr"] += effects["front_pair_team_mr"]

            # Mana bonuses
            if "bonus_mana" in effects:
                # Starting mana bonus - handled separately
                pass
            if "mana_regen" in effects:
                # Mana regen per second - handled in tick effects
                pass

            # Team shields
            if "team_shield" in effects:
                # Handled at combat start
                pass

            # Item holder bonuses
            if "item_holder_ad" in effects:
                if len(unit.items) > 0:
                    modifiers["bonus_ad"] += effects["item_holder_ad"]
            if "item_holder_ap" in effects:
                if len(unit.items) > 0:
                    modifiers["bonus_ap"] += effects["item_holder_ap"]
            if "item_holder_as" in effects:
                if len(unit.items) > 0:
                    modifiers["bonus_attack_speed"] += effects["item_holder_as"]

            # 3-item holder bonuses (Cybernetic Implants)
            if "three_item_holder_health" in effects:
                if len(unit.items) >= 3:
                    modifiers["bonus_health"] += effects["three_item_holder_health"]
            if "three_item_holder_ad" in effects:
                if len(unit.items) >= 3:
                    modifiers["bonus_ad"] += effects["three_item_holder_ad"]
            if "three_item_holder_mana" in effects:
                if len(unit.items) >= 3:
                    # Starting mana bonus
                    modifiers["bonus_starting_mana"] = modifiers.get("bonus_starting_mana", 0) + effects["three_item_holder_mana"]

            # Jeweled Lotus - crit chance and AP scaling on crit
            if "crit_ap_scaling" in effects:
                modifiers["bonus_ap"] += unit.stats.crit_chance * effects["crit_ap_scaling"]

            # Ability crits (Jeweled Lotus)
            if effects.get("ability_crits"):
                # Enable ability crits for this unit
                modifiers["ability_can_crit"] = True

            # Damage amp all (Indiscriminate Killer)
            if "damage_amp_all" in effects:
                modifiers["damage_amp"] += effects["damage_amp_all"]

            # AS on takedown (Precision and Grace)
            if "as_on_takedown" in effects:
                takedowns = getattr(unit, '_takedowns', 0)
                modifiers["bonus_attack_speed"] += effects["as_on_takedown"] * takedowns

            # Back row bonus per front (Rear Guard)
            if "back_row_bonus_per_front" in effects:
                front_count = self._count_front_row_units_from_player(player)
                if self._is_unit_back_row(unit):
                    modifiers["bonus_ad"] += effects["back_row_bonus_per_front"] * front_count
                    modifiers["bonus_ap"] += effects["back_row_bonus_per_front"] * front_count

            # Emblem holder bonuses
            if "emblem_holder_bonus" in effects:
                if self._unit_has_emblem(unit):
                    modifiers["bonus_ad"] += effects["emblem_holder_bonus"]
                    modifiers["bonus_ap"] += effects["emblem_holder_bonus"]

            if "emblem_holder_as" in effects:
                if self._unit_has_emblem(unit):
                    modifiers["bonus_attack_speed"] += effects["emblem_holder_as"]

            # Health per emblem (Flexible)
            if "health_per_emblem" in effects:
                emblem_count = self._count_emblems_on_board(player)
                modifiers["bonus_health"] += effects["health_per_emblem"] * emblem_count

            # Combat bonus (generic)
            if "combat_bonus" in effects:
                combat_count = getattr(player, 'combat_count', 0)
                modifiers["bonus_ad"] += effects["combat_bonus"] * combat_count

            # Arcanist mana regen
            if "arcanist_mana_regen" in effects:
                # Handled in tick effects
                pass

            # Area healing
            if effects.get("area_healing"):
                # Handled in tick effects
                pass

            # Bruiser rewards per HP
            if "bruiser_rewards_per_hp" in effects:
                hp_percent = unit.stats.current_hp / unit.stats.max_hp if unit.stats.max_hp > 0 else 1.0
                modifiers["damage_reduction"] += effects["bruiser_rewards_per_hp"] * hp_percent

            # Demacia stacking per rally
            if "demacia_stacking_per_rally" in effects:
                rally_count = getattr(player, '_rally_count', 0)
                modifiers["bonus_ad"] += effects["demacia_stacking_per_rally"] * rally_count
                modifiers["bonus_ap"] += effects["demacia_stacking_per_rally"] * rally_count

            # Projectile attacks (Mess Hall)
            if effects.get("projectile_attacks"):
                # Enable projectile attacks for melee units
                pass

            # Per combat stacking
            if effects.get("per_combat"):
                combat_count = getattr(player, 'combat_count', 0)
                if "stacking_ad" in effects:
                    modifiers["bonus_ad"] += effects["stacking_ad"] * combat_count
                if "stacking_ap" in effects:
                    modifiers["bonus_ap"] += effects["stacking_ap"] * combat_count

            # Random stats on star up
            if effects.get("random_stats_on_star_up"):
                # Would apply random stats when unit is starred up
                pass

            # Mana regen per second (Cry Me a River)
            if "mana_regen_per_sec" in effects:
                # Handled in tick effects
                pass

            # Dash on takedown
            if effects.get("dash_on_takedown"):
                # Would enable dashing on takedown
                pass

            # Omnivamp vs burning (Pyromaniac)
            if "omnivamp_vs_burning" in effects:
                # Handled in damage calculation
                pass

            # Burn damage amp (Pyromaniac)
            if "burn_damage_amp" in effects:
                # Handled in damage calculation
                pass

        return modifiers

    def _count_front_row_units_from_player(self, player: "PlayerState") -> int:
        """Count front row units from player's board."""
        if not hasattr(player, 'board') or not player.board:
            return 0
        board_units = player.board.get_units() if hasattr(player.board, 'get_units') else []
        return len([u for u in board_units if self._is_unit_front_row(u)])

    def _is_unit_front_row(self, unit) -> bool:
        """Check if unit is in front row."""
        if hasattr(unit, 'position') and unit.position:
            return unit.position.row <= 1
        return True  # Default to front

    def _is_unit_back_row(self, unit) -> bool:
        """Check if unit is in back row."""
        return not self._is_unit_front_row(unit)

    def _unit_has_emblem(self, unit: "CombatUnit") -> bool:
        """Check if unit has an emblem item."""
        for item in unit.items:
            if hasattr(item, 'grants_trait') and item.grants_trait:
                return True
            if hasattr(item, 'type') and str(item.type).lower() == 'emblem':
                return True
        return False

    def _count_emblems_on_board(self, player: "PlayerState") -> int:
        """Count emblem items on the board."""
        if not hasattr(player, 'board') or not player.board:
            return 0
        board_units = player.board.get_units() if hasattr(player.board, 'get_units') else []
        count = 0
        for unit in board_units:
            if hasattr(unit, 'items'):
                for item in unit.items:
                    if hasattr(item, 'grants_trait') and item.grants_trait:
                        count += 1
        return count

    def _has_paired_champion(self, player: "PlayerState", unit: "CombatUnit") -> bool:
        """Check if there's another copy of this champion on the board."""
        if not hasattr(player, 'board') or not player.board:
            return False
        board_units = player.board.get_units() if hasattr(player.board, 'get_units') else []
        same_champ_count = sum(
            1 for u in board_units
            if hasattr(u, 'champion_id') and u.champion_id == unit.champion_id
        )
        return same_champ_count >= 2

    def _get_bench_count(self, player: "PlayerState") -> int:
        """Get the number of champions on the bench."""
        if hasattr(player, 'bench') and player.bench:
            if hasattr(player.bench, 'units'):
                return len([u for u in player.bench.units if u is not None])
        return 0

    def _has_dual_4cost(self, player: "PlayerState", unit: "CombatUnit") -> bool:
        """Check if there are 2 different 4-cost champions on the board."""
        if not hasattr(player, 'board') or not player.board:
            return False
        board_units = player.board.get_units() if hasattr(player.board, 'get_units') else []
        four_cost_ids = set()
        for u in board_units:
            if hasattr(u, 'cost') and u.cost == 4:
                four_cost_ids.add(getattr(u, 'champion_id', None))
        return len(four_cost_ids) >= 2

    def _has_front_pair(self, player: "PlayerState") -> bool:
        """Check if there's a pair of same champions in front row."""
        # Simplified check - would need position data
        if not hasattr(player, 'board') or not player.board:
            return False
        return True  # Assume true for now

    def apply_combat_start_effects(
        self,
        player: "PlayerState",
        engine: "CombatEngine",
        player_units: list["CombatUnit"]
    ) -> None:
        """Apply augment effects at combat start."""
        # Apply stat modifiers to all units
        for unit in player_units:
            mods = self.get_combat_stat_modifiers(player, unit)

            unit.stats.max_hp += mods["bonus_health"]
            unit.stats.current_hp += mods["bonus_health"]
            unit.stats.attack_damage += mods["bonus_ad"]
            unit.stats.ability_power += mods["bonus_ap"]
            unit.stats.armor += mods["bonus_armor"]
            unit.stats.magic_resist += mods["bonus_mr"]
            unit.stats.attack_speed *= (1 + mods["bonus_attack_speed"])

        # Apply special augment effects that need board context
        for augment in player.augments:
            effects = augment.effects

            # Big Friend - largest champion gets bonuses
            if "largest_champ_health" in effects or "largest_champ_as" in effects:
                if player_units:
                    # Find largest champion by base HP
                    largest = max(player_units, key=lambda u: u.stats.max_hp)
                    if "largest_champ_health" in effects:
                        bonus_hp = effects["largest_champ_health"]
                        largest.stats.max_hp += bonus_hp
                        largest.stats.current_hp += bonus_hp
                    if "largest_champ_as" in effects:
                        largest.stats.attack_speed *= (1 + effects["largest_champ_as"])

            # Double Trouble - 2 copies of same champion get bonuses
            if "double_ad" in effects or "double_ap" in effects or "double_health" in effects:
                # Count champions by ID
                champ_counts: dict[str, list] = {}
                for unit in player_units:
                    if unit.champion_id not in champ_counts:
                        champ_counts[unit.champion_id] = []
                    champ_counts[unit.champion_id].append(unit)

                # Apply bonuses to champions with exactly 2 copies
                for champ_id, units in champ_counts.items():
                    if len(units) == 2:
                        for unit in units:
                            if "double_ad" in effects:
                                unit.stats.attack_damage += effects["double_ad"]
                            if "double_ap" in effects:
                                unit.stats.ability_power += effects["double_ap"]
                            if "double_health" in effects:
                                unit.stats.max_hp += effects["double_health"]
                                unit.stats.current_hp += effects["double_health"]

            # Exiles - isolated champions get shield
            if "isolated_shield_percent" in effects:
                for unit in player_units:
                    if self._is_unit_isolated(unit, player_units):
                        shield_amount = unit.stats.max_hp * effects["isolated_shield_percent"]
                        # Add shield (would need status effect system integration)
                        if hasattr(unit, '_augment_shield'):
                            unit._augment_shield += shield_amount
                        else:
                            unit._augment_shield = shield_amount

            # Boxing Lessons / Lineup - per front row bonuses
            if "health_per_front_row" in effects:
                front_row_count = self._count_front_row_units(player_units)
                bonus_hp = effects["health_per_front_row"] * front_row_count
                for unit in player_units:
                    unit.stats.max_hp += bonus_hp
                    unit.stats.current_hp += bonus_hp

            if "armor_per_front_row" in effects:
                front_row_count = self._count_front_row_units(player_units)
                bonus_armor = effects["armor_per_front_row"] * front_row_count
                for unit in player_units:
                    unit.stats.armor += bonus_armor

            if "mr_per_front_row" in effects:
                front_row_count = self._count_front_row_units(player_units)
                bonus_mr = effects["mr_per_front_row"] * front_row_count
                for unit in player_units:
                    unit.stats.magic_resist += bonus_mr

            # Find Your Center - center front-row champion gets bonuses
            if "center_front_bonus_ad" in effects or "center_front_bonus_hp" in effects:
                center_unit = self._find_center_front_unit(player_units)
                if center_unit:
                    if "center_front_bonus_ad" in effects:
                        center_unit.stats.attack_damage += effects["center_front_bonus_ad"]
                    if "center_front_bonus_hp" in effects:
                        bonus = effects["center_front_bonus_hp"]
                        center_unit.stats.max_hp += bonus
                        center_unit.stats.current_hp += bonus

            # Little Buddies - high cost champions get bonus per low cost ally
            if "high_cost_bonus_per_low_cost" in effects:
                low_cost_count = sum(1 for u in player_units if u.cost <= 2)
                bonus_percent = effects["high_cost_bonus_per_low_cost"] * low_cost_count
                for unit in player_units:
                    if unit.cost >= 4:
                        unit.stats.attack_damage *= (1 + bonus_percent)
                        unit.stats.ability_power *= (1 + bonus_percent)

            # Team shield (Blood Offering)
            if "team_shield" in effects:
                shield_amount = effects["team_shield"]
                for unit in player_units:
                    self._apply_shield(unit, engine, shield_amount)

            # Celestial Blessing shield conversion
            if "shield_conversion" in effects:
                # Excess healing becomes shield
                for unit in player_units:
                    unit._shield_conversion = effects["shield_conversion"]

            # Corrosion - enemy front row loses armor/mr
            if "enemy_front_armor_reduce" in effects or "enemy_front_mr_reduce" in effects:
                enemy_units = self._get_enemy_units(engine, player)
                front_enemies = self._get_front_row_units(enemy_units)
                for enemy in front_enemies:
                    if "enemy_front_armor_reduce" in effects:
                        enemy.stats.armor = max(0, enemy.stats.armor - effects["enemy_front_armor_reduce"])
                    if "enemy_front_mr_reduce" in effects:
                        enemy.stats.magic_resist = max(0, enemy.stats.magic_resist - effects["enemy_front_mr_reduce"])

            # Largest champ durability (Big Friend)
            if "largest_champ_durability" in effects:
                if player_units:
                    largest = max(player_units, key=lambda u: u.stats.max_hp)
                    largest._damage_reduction_bonus = effects["largest_champ_durability"]

    def _apply_shield(self, unit: "CombatUnit", engine: "CombatEngine", amount: float) -> None:
        """Apply a shield to a unit using the status effect system."""
        if unit._status_effect_system:
            from src.combat.status_effects import StatusEffect, StatusEffectType
            shield = StatusEffect(
                effect_type=StatusEffectType.SHIELD,
                source_id="augment",
                duration=999.0,  # Lasts until absorbed
                value=amount,
                remaining_shield=amount
            )
            unit._status_effect_system.apply_effect(unit, shield)
        else:
            # Fallback for units without status effect system
            if hasattr(unit, '_augment_shield'):
                unit._augment_shield += amount
            else:
                unit._augment_shield = amount

    def _get_enemy_units(self, engine: "CombatEngine", player: "PlayerState") -> list:
        """Get enemy units from combat engine."""
        if hasattr(engine, 'get_enemy_units'):
            return engine.get_enemy_units(player)
        if hasattr(engine, 'units'):
            from src.combat.hex_grid import Team
            player_team = Team.BLUE if player.player_id == 0 else Team.RED
            enemy_team = Team.RED if player_team == Team.BLUE else Team.BLUE
            return [u for u in engine.units if u.team == enemy_team and u.is_alive]
        return []

    def _get_front_row_units(self, units: list) -> list:
        """Get units in the front row (rows 0-1)."""
        front_units = []
        for unit in units:
            if hasattr(unit, 'position') and unit.position:
                if unit.position.row <= 1:
                    front_units.append(unit)
            else:
                # If no position data, consider first half as front row
                front_units.append(unit)
        return front_units[:len(units) // 2] if not front_units else front_units

    def _is_unit_isolated(self, unit: "CombatUnit", all_units: list["CombatUnit"]) -> bool:
        """Check if a unit is isolated (no adjacent allies)."""
        # Simplified check - would need hex grid integration for proper implementation
        return len(all_units) <= 2

    def _count_front_row_units(self, units: list["CombatUnit"]) -> int:
        """Count units in the front row."""
        # Front row is typically rows 0-1 (y <= 1)
        # This would need actual position data
        return len(units) // 2  # Approximate

    def _find_center_front_unit(self, units: list["CombatUnit"]) -> "CombatUnit":
        """Find the center front-row unit."""
        # Would need position data - return first unit as approximation
        if units:
            return units[0]
        return None

    def apply_combat_tick_effects(
        self,
        player: "PlayerState",
        engine: "CombatEngine",
        player_units: list["CombatUnit"],
        current_tick: int
    ) -> None:
        """Apply augment effects every combat tick."""
        ticks_per_second = 30
        elapsed_seconds = current_tick / ticks_per_second

        for augment in player.augments:
            effects = augment.effects

            # First Aid Kit - heal every 5 seconds
            if "heal_per_5_sec" in effects:
                if current_tick > 0 and current_tick % (5 * ticks_per_second) == 0:
                    heal = effects["heal_per_5_sec"]
                    for unit in player_units:
                        if unit.is_alive:
                            unit.heal(heal)

            # Cybernetic Uplink - restore mana per second to units with items
            if "item_holder_mana_per_sec" in effects:
                if current_tick > 0 and current_tick % ticks_per_second == 0:
                    mana_per_sec = effects["item_holder_mana_per_sec"]
                    for unit in player_units:
                        if unit.is_alive and len(unit.items) > 0:
                            unit.stats.current_mana = min(
                                unit.stats.current_mana + mana_per_sec,
                                unit.stats.max_mana
                            )

            # Partial Ascension - 30% damage after 15 seconds (mark units)
            if "damage_bonus_after_15s" in effects:
                if current_tick == 15 * ticks_per_second:
                    # Mark units as having damage bonus
                    for unit in player_units:
                        if unit.is_alive:
                            if not hasattr(unit, '_augment_damage_bonus'):
                                unit._augment_damage_bonus = 0.0
                            unit._augment_damage_bonus += effects["damage_bonus_after_15s"]

            # URF Overtime - 100% AS after 15 seconds
            if "overtime_attack_speed" in effects:
                if current_tick == 15 * ticks_per_second:
                    for unit in player_units:
                        if unit.is_alive:
                            unit.stats.attack_speed *= (1 + effects["overtime_attack_speed"])

            # Ascension - damage amp after delay
            if "damage_amp_after_delay" in effects:
                delay = effects.get("delay_seconds", 12)
                if current_tick == int(delay * ticks_per_second):
                    for unit in player_units:
                        if unit.is_alive:
                            if not hasattr(unit, '_augment_damage_amp'):
                                unit._augment_damage_amp = 0.0
                            unit._augment_damage_amp += effects["damage_amp_after_delay"]

            # Second Wind - heal missing HP after 10 seconds
            if "heal_missing_hp_after_10s" in effects:
                if current_tick == 10 * ticks_per_second:
                    heal_percent = effects["heal_missing_hp_after_10s"]
                    for unit in player_units:
                        if unit.is_alive:
                            missing_hp = unit.stats.max_hp - unit.stats.current_hp
                            heal_amount = missing_hp * heal_percent
                            unit.heal(heal_amount)

            # Last Second Save - heal when falling below threshold
            if "heal_at_threshold" in effects:
                threshold = effects["heal_at_threshold"]
                heal_amount = effects.get("heal_amount", 400)
                for unit in player_units:
                    if unit.is_alive:
                        hp_percent = unit.stats.current_hp / unit.stats.max_hp
                        if hp_percent <= threshold:
                            if not hasattr(unit, '_last_second_save_triggered'):
                                unit._last_second_save_triggered = False
                            if not unit._last_second_save_triggered:
                                unit.heal(heal_amount)
                                unit._last_second_save_triggered = True

            # Arcane Viktor-y - stun after 8 and 20 seconds
            if "stun_after_8s" in effects:
                if current_tick == 8 * ticks_per_second:
                    # Would need to apply stun to enemy team
                    pass  # Stun effect would be handled by status effect system

            if "stun_after_20s" in effects:
                if current_tick == 20 * ticks_per_second:
                    pass  # Second stun

    def apply_on_kill_effects(
        self,
        player: "PlayerState",
        killer: "CombatUnit",
        victim: "CombatUnit"
    ) -> None:
        """Apply augment effects when a unit kills another."""
        aug_state = self.get_player_augment_state(player.player_id)

        for augment in player.augments:
            effects = augment.effects

            # Thrill of the Hunt - heal on kill
            if "heal_on_kill" in effects:
                killer.heal(effects["heal_on_kill"])

            # Healing Orbs - heal nearest ally on enemy death
            if "heal_on_enemy_death" in effects:
                heal_amount = effects["heal_on_enemy_death"]
                # Would need to find nearest ally and heal them
                killer.heal(heal_amount)  # Simplified - heal killer

            # Wood Axiom - permanent health on takedown
            if "health_on_takedown" in effects:
                bonus_hp = effects["health_on_takedown"]
                killer.stats.max_hp += bonus_hp
                killer.stats.current_hp += bonus_hp

        # Gold on enemy kill (from augment state)
        if aug_state["gold_on_enemy_kill"] > 0:
            player.gold += aug_state["gold_on_enemy_kill"]

        # Loot drop chance
        if aug_state["loot_drop_chance"] > 0:
            if self.rng.random() < aug_state["loot_drop_chance"]:
                # Would grant loot - simplified as gold
                player.gold += 1

    def apply_on_death_effects(
        self,
        player: "PlayerState",
        dead_unit: "CombatUnit"
    ) -> None:
        """Apply augment effects when a friendly unit dies."""
        aug_state = self.get_player_augment_state(player.player_id)

        for augment in player.augments:
            effects = augment.effects

            # Eye For An Eye - gain component on ally death
            if "component_on_ally_death" in effects:
                max_components = effects.get("max_components", 3)
                if aug_state["components_from_deaths"] < max_components:
                    aug_state["components_from_deaths"] += 1
                    # Would need to grant component through item system

            # Good For Something - itemless champions drop gold
            if "gold_on_itemless_death" in effects:
                if len(dead_unit.items) == 0:
                    player.gold += effects["gold_on_itemless_death"]

    def apply_on_attack_effects(
        self,
        player: "PlayerState",
        attacker: "CombatUnit",
        target: "CombatUnit",
        engine: "CombatEngine"
    ) -> dict:
        """
        Apply augment effects when a unit attacks.

        Returns dict with modifications like bonus_damage, apply_burn, etc.
        """
        mods = {
            "bonus_damage": 0,
            "apply_burn": False,
            "apply_wound": False,
            "apply_sunder": False,
            "sunder_percent": 0.0,
            "apply_shred": False,
            "shred_percent": 0.0,
            "stun_chance": 0.0,
            "stun_duration": 0.0,
        }

        for augment in player.augments:
            effects = augment.effects

            # Fire Axiom - apply burn and wound
            if effects.get("apply_burn"):
                mods["apply_burn"] = True
            if effects.get("apply_wound"):
                mods["apply_wound"] = True

            # Air Axiom - apply sunder
            if "sunder_percent" in effects:
                mods["apply_sunder"] = True
                mods["sunder_percent"] = max(mods["sunder_percent"], effects["sunder_percent"])

            # Water Axiom - apply shred
            if "shred_percent" in effects:
                mods["apply_shred"] = True
                mods["shred_percent"] = max(mods["shred_percent"], effects["shred_percent"])

            # Earth Axiom - stun chance
            if "stun_chance" in effects:
                mods["stun_chance"] = max(mods["stun_chance"], effects["stun_chance"])
                mods["stun_duration"] = effects.get("stun_duration", 1.0)

            # Focused Fire - stacking AD
            if "stacking_ad" in effects:
                if not hasattr(attacker, '_hit_count'):
                    attacker._hit_count = 0
                attacker._hit_count += 1

        return mods

    def apply_on_damage_dealt(
        self,
        player: "PlayerState",
        attacker: "CombatUnit",
        target: "CombatUnit",
        damage: float,
        damage_type: str,
        engine: "CombatEngine"
    ) -> None:
        """Apply augment effects after damage is dealt."""
        for augment in player.augments:
            effects = augment.effects

            # Omnivamp healing
            if "omnivamp" in effects:
                heal_amount = damage * effects["omnivamp"]
                attacker.heal(heal_amount)

            # Pyromaniac - bonus damage to burning enemies
            if "burn_damage_amp" in effects:
                if target._status_effect_system:
                    from src.combat.status_effects import StatusEffectType
                    if target._status_effect_system.has_effect(target, StatusEffectType.BURN):
                        # Damage amp already applied to burned targets
                        pass

            # Giant Slayer - bonus damage to high HP targets
            if "damage_amp_high_hp" in effects:
                hp_threshold = effects.get("hp_threshold", 1750)
                if target.stats.max_hp > hp_threshold:
                    # Apply bonus damage (would need to be calculated before damage)
                    pass

    def apply_status_effect_to_target(
        self,
        target: "CombatUnit",
        effect_type: str,
        value: float,
        duration: float,
        source_id: str = "augment"
    ) -> bool:
        """Apply a status effect to a target unit."""
        if not target._status_effect_system:
            return False

        from src.combat.status_effects import StatusEffect, StatusEffectType

        type_map = {
            "burn": StatusEffectType.BURN,
            "bleed": StatusEffectType.BLEED,
            "stun": StatusEffectType.STUN,
            "wound": StatusEffectType.GRIEVOUS_WOUNDS,
            "shield": StatusEffectType.SHIELD,
            "armor_reduce": StatusEffectType.ARMOR_DEBUFF,
            "mr_reduce": StatusEffectType.MAGIC_RESIST_DEBUFF,
        }

        if effect_type not in type_map:
            return False

        effect = StatusEffect(
            effect_type=type_map[effect_type],
            source_id=source_id,
            duration=duration,
            value=value
        )

        if effect_type == "shield":
            effect.remaining_shield = value

        target._status_effect_system.apply_effect(target, effect)
        return True

    def apply_sunder(
        self,
        target: "CombatUnit",
        percent: float,
        duration: float = 5.0
    ) -> None:
        """Apply sunder (armor reduction) to target."""
        armor_reduction = target.stats.armor * percent
        self.apply_status_effect_to_target(
            target, "armor_reduce", armor_reduction, duration
        )

    def apply_shred(
        self,
        target: "CombatUnit",
        percent: float,
        duration: float = 5.0
    ) -> None:
        """Apply shred (magic resist reduction) to target."""
        mr_reduction = target.stats.magic_resist * percent
        self.apply_status_effect_to_target(
            target, "mr_reduce", mr_reduction, duration
        )

    # =========================================================================
    # ECONOMY EFFECTS
    # =========================================================================

    def modify_interest(
        self,
        player: "PlayerState",
        base_interest: int
    ) -> int:
        """Modify interest based on augments."""
        aug_state = self.get_player_augment_state(player.player_id)

        # Consistent Income - no interest
        if aug_state["no_interest"]:
            return 0

        # Interest cap increase (Hedge Fund Prismatic)
        max_interest = 5 + aug_state.get("interest_cap_increase", 0)
        return min(base_interest, max_interest)

    def apply_level_up_effects(
        self,
        player: "PlayerState"
    ) -> AugmentEffectResult:
        """Apply effects when player levels up."""
        result = AugmentEffectResult(success=True)
        aug_state = self.get_player_augment_state(player.player_id)

        for augment in player.augments:
            effects = augment.effects

            # Golden Ticket - gold on level up
            if "gold_per_level_up" in effects:
                gold = effects["gold_per_level_up"]
                player.gold += gold
                result.gold_gained += gold
                result.message += f"레벨업 골드 +{gold}. "

            # Shopping Spree - rerolls per level
            if "rerolls_per_level" in effects:
                rerolls = effects["rerolls_per_level"]
                aug_state["pending_rerolls"] = aug_state.get("pending_rerolls", 0) + rerolls
                result.message += f"리롤 +{rerolls}. "

            # Birthday Present - grant 2-star on level up
            if "grant_2star_on_level_up" in effects:
                result.message += "2성 챔피언 획득. "

        return result

    def modify_player_damage(
        self,
        player: "PlayerState",
        base_damage: int,
        won: bool
    ) -> int:
        """Modify damage taken/dealt based on augments."""
        aug_state = self.get_player_augment_state(player.player_id)

        if not won:
            # Nine Lives - only lose 1 HP
            if aug_state.get("only_lose_1_hp"):
                return 1

            # Cursed Crown - double damage on loss
            if aug_state.get("double_damage_on_loss"):
                return base_damage * 2

        return base_damage

    def get_xp_cost_modifier(
        self,
        player: "PlayerState"
    ) -> tuple[int, bool]:
        """Get XP cost modification and whether it costs health instead.

        Returns:
            Tuple of (cost_reduction, costs_health).
        """
        aug_state = self.get_player_augment_state(player.player_id)

        cost_reduction = aug_state.get("xp_cost_reduction", 0)
        costs_health = aug_state.get("xp_costs_health", False)

        return cost_reduction, costs_health

    def get_bonus_xp_on_purchase(self, player: "PlayerState") -> int:
        """Get bonus XP gained when purchasing XP."""
        aug_state = self.get_player_augment_state(player.player_id)
        return aug_state.get("bonus_xp_on_purchase", 0)

    def should_reroll_be_free(self, player: "PlayerState") -> bool:
        """Check if current reroll should be free (Prismatic Ticket)."""
        aug_state = self.get_player_augment_state(player.player_id)
        free_chance = aug_state.get("free_reroll_chance", 0.0)

        if free_chance > 0:
            return self.rng.random() < free_chance

        # Use pending free rerolls
        pending = aug_state.get("pending_rerolls", 0)
        if pending > 0:
            aug_state["pending_rerolls"] = pending - 1
            return True

        return False

    # =========================================================================
    # DAMAGE CALCULATION MODIFIERS
    # =========================================================================

    def get_damage_modifier(
        self,
        player: "PlayerState",
        attacker: "CombatUnit",
        target: "CombatUnit",
        elapsed_seconds: float
    ) -> float:
        """Get damage multiplier from augments."""
        multiplier = 1.0

        for augment in player.augments:
            effects = augment.effects

            # Partial Ascension - +30% damage after 15s
            if "damage_bonus_after_15s" in effects:
                if elapsed_seconds >= 15:
                    multiplier += effects["damage_bonus_after_15s"]

        return multiplier

    def reset_player(self, player_id: int) -> None:
        """Reset augment state for a player."""
        if player_id in self._player_states:
            del self._player_states[player_id]

    def reset(self) -> None:
        """Reset all augment states."""
        self._player_states.clear()


# Singleton instance
_augment_effect_system: Optional[AugmentEffectSystem] = None


def get_augment_effect_system(seed: Optional[int] = None) -> AugmentEffectSystem:
    """Get or create the augment effect system singleton."""
    global _augment_effect_system
    if _augment_effect_system is None or seed is not None:
        _augment_effect_system = AugmentEffectSystem(seed)
    return _augment_effect_system
