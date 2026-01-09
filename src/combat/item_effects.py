"""Item Effects System for TFT Combat.

Handles special effects from equipped items during combat:
- On-hit effects (Giant Slayer, Statikk Shiv, etc.)
- On-cast effects (Blue Buff, Shojin, etc.)
- Passive effects (RFC range, Sunfire burn, etc.)
- On-damage effects (Bloodthirster shield, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING, Set
from enum import Enum, auto

from .status_effects import StatusEffectType

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .attack import DamageEvent, DamageType, AttackSystem
    from .status_effects import StatusEffectSystem
    from .targeting import TargetSelector


class ItemEffectType(Enum):
    """Types of item effects."""
    ON_COMBAT_START = auto()  # Triggered at combat start
    ON_ATTACK = auto()  # Triggered on each attack
    ON_HIT = auto()  # Triggered after damage dealt
    ON_CAST = auto()  # Triggered after ability cast
    ON_DAMAGE_TAKEN = auto()  # Triggered when taking damage
    ON_KILL = auto()  # Triggered on kill
    PASSIVE = auto()  # Always active (stat modifiers)
    PERIODIC = auto()  # Triggered every X seconds


@dataclass
class ItemEffectContext:
    """Context for item effect execution."""
    attack_system: "AttackSystem"
    status_effects: "StatusEffectSystem"
    target_selector: "TargetSelector"
    all_units: Dict[str, "CombatUnit"]
    tick_duration: float = 0.033
    grid: Any = None  # HexGrid reference for position calculations
    current_time: float = 0.0  # Current combat time for expiry tracking


class ItemEffectSystem:
    """
    Manages item effects during combat.

    This system hooks into combat events and triggers appropriate
    item effects based on equipped items.
    """

    # Giant Slayer threshold
    GIANT_SLAYER_HP_THRESHOLD = 1600
    GIANT_SLAYER_BONUS_DAMAGE = 0.25  # 25% bonus damage

    # Statikk Shiv
    STATIKK_SHIV_ATTACK_COUNT = 3
    STATIKK_SHIV_TARGETS = 4
    STATIKK_SHIV_DAMAGE = 70
    STATIKK_SHIV_MR_SHRED = 0.5  # 50% MR shred

    # Sunfire Cape
    SUNFIRE_TICK_INTERVAL = 2.0  # Every 2 seconds
    SUNFIRE_DAMAGE_PERCENT = 0.01  # 1% max HP
    SUNFIRE_RANGE = 1  # Adjacent hexes

    # Bloodthirster
    BLOODTHIRSTER_OMNIVAMP = 0.25  # 25%
    BLOODTHIRSTER_SHIELD_THRESHOLD = 0.40  # Below 40% HP
    BLOODTHIRSTER_SHIELD_PERCENT = 0.25  # 25% max HP shield

    # Rapid Firecannon
    RFC_BONUS_RANGE = 1

    # Last Whisper
    LAST_WHISPER_ARMOR_SHRED = 0.50  # 50% armor shred
    LAST_WHISPER_DURATION = 3.0  # 3 seconds

    # Hand of Justice
    HOJ_AD_AP_BONUS = 0.15  # 15%
    HOJ_OMNIVAMP = 0.15  # 15%

    # Spear of Shojin
    SHOJIN_MANA_PER_ATTACK = 8

    # Guinsoo's Rageblade
    RAGEBLADE_AS_PER_STACK = 0.05  # 5% per stack

    # Hextech Gunblade
    GUNBLADE_OMNIVAMP = 0.25  # 25% omnivamp
    GUNBLADE_EXCESS_SHIELD = True  # Excess healing becomes shield

    # Zeke's Herald
    ZEKES_AS_BONUS = 0.30  # 30% attack speed
    ZEKES_RANGE = 1  # 1 hex range

    # Bramble Vest
    BRAMBLE_DAMAGE_REDUCTION = 0.08  # 8% less damage from attacks
    BRAMBLE_REFLECT_DAMAGE = 75  # Magic damage to attackers

    # Gargoyle Stoneplate
    GARGOYLE_STATS_PER_ENEMY = 10  # 10 armor/MR per targeting enemy

    # Dragon's Claw
    DRAGON_CLAW_ABILITY_REDUCTION = 0.10  # 10% less ability damage
    DRAGON_CLAW_HEAL_PERCENT = 0.012  # 1.2% max HP heal per enemy
    DRAGON_CLAW_INTERVAL = 2.0  # Every 2 seconds

    # Frozen Heart
    FROZEN_HEART_AS_REDUCTION = 0.25  # 25% AS reduction
    FROZEN_HEART_RANGE = 2  # 2 hex range

    # Shroud of Stillness
    SHROUD_MANA_INCREASE = 0.30  # 30% max mana increase
    SHROUD_RANGE = 2  # 2 hex range

    # Chalice of Power
    CHALICE_AP_BONUS = 30  # 30 AP to allies

    # Locket of the Iron Solari
    LOCKET_SHIELD_AMOUNT = 300  # Base shield
    LOCKET_SHIELD_DURATION = 8.0  # 8 seconds

    # Ionic Spark
    IONIC_SPARK_MR_SHRED = 0.50  # 50% MR shred
    IONIC_SPARK_DAMAGE = 160  # Magic damage on cast
    IONIC_SPARK_RANGE = 2  # 2 hex range

    # Morellonomicon
    MORELLO_BURN_DURATION = 10.0  # 10 seconds
    MORELLO_BURN_PERCENT = 0.01  # 1% max HP per second
    MORELLO_HEAL_REDUCTION = 0.33  # 33% healing reduction (Grievous Wounds)

    # Warmog's Armor
    WARMOG_REGEN_PERCENT = 0.03  # 3% max HP per second

    # Sterak's Gage
    STERAK_THRESHOLD = 0.60  # Below 60% HP
    STERAK_SHIELD_PERCENT = 0.25  # 25% max HP shield
    STERAK_AD_BONUS = 0.40  # 40% AD bonus

    # Guardbreaker
    GUARDBREAKER_BONUS_DAMAGE = 0.25  # 25% bonus damage to shielded
    GUARDBREAKER_DURATION = 3.0  # 3 seconds after hitting shield

    def __init__(self):
        """Initialize the item effect system."""
        # Track per-unit state
        self.unit_state: Dict[str, Dict[str, Any]] = {}

        # Periodic effect timers
        self.periodic_timers: Dict[str, Dict[str, float]] = {}

    def initialize_unit(self, unit: "CombatUnit") -> None:
        """
        Initialize item effect tracking for a unit.

        Args:
            unit: The combat unit to initialize.
        """
        self.unit_state[unit.id] = {
            "statikk_counter": 0,
            "rageblade_stacks": 0,
            "deathblade_stacks": 0,
            "titans_stacks": 0,
            "bloodthirster_triggered": False,
            "edge_of_night_triggered": False,
            "sterak_triggered": False,
            "archangels_timer": 0.0,
            "guardbreaker_targets": {},  # unit_id -> expiry_time
            "gunblade_excess_heal": 0.0,  # Tracked for shield conversion
        }
        self.periodic_timers[unit.id] = {
            "sunfire": 0.0,
            "redemption": 0.0,
            "warmog": 0.0,
            "dragon_claw": 0.0,
        }

    def get_equipped_item_ids(self, unit: "CombatUnit") -> List[str]:
        """Get list of equipped item IDs for a unit."""
        if unit.source_instance is None:
            return []

        item_ids = []
        for item_inst in unit.source_instance.items:
            if hasattr(item_inst, 'item'):
                item_ids.append(item_inst.item.id)
            elif hasattr(item_inst, 'id'):
                item_ids.append(item_inst.id)
        return item_ids

    def has_item(self, unit: "CombatUnit", item_id: str) -> bool:
        """Check if unit has a specific item equipped."""
        return item_id in self.get_equipped_item_ids(unit)

    # =========================================================================
    # COMBAT START EFFECTS
    # =========================================================================

    def apply_combat_start_effects(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger at combat start.

        Args:
            unit: The unit to apply effects to.
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(unit)

        # Blue Buff - Start with 20 extra mana
        if "blue_buff" in items:
            unit.stats.current_mana += 20
            events.append({"type": "blue_buff_mana", "unit": unit.id, "mana": 20})

        # Rapid Firecannon - +1 range
        if "rapid_firecannon" in items:
            unit.stats.attack_range += self.RFC_BONUS_RANGE
            events.append({"type": "rfc_range", "unit": unit.id, "bonus": 1})

        # Edge of Night - Initial stealth (simplified as damage reduction)
        # Full implementation would need untargetable state

        # Quicksilver - CC immunity for 14 seconds
        if "quicksilver" in items:
            from .status_effects import create_cc_immunity
            cc_immune = create_cc_immunity(14.0)
            context.status_effects.apply_effect(unit, cc_immune)
            events.append({"type": "quicksilver", "unit": unit.id, "duration": 14})

        # Zeke's Herald - Grant 30% AS to adjacent allies
        if "zekes_herald" in items:
            zekes_events = self._apply_zekes_herald(unit, context)
            events.extend(zekes_events)

        # Chalice of Power - Grant 30 AP to allies in same row
        if "chalice_of_power" in items:
            chalice_events = self._apply_chalice(unit, context)
            events.extend(chalice_events)

        # Locket of the Iron Solari - Shield allies in same row
        if "locket_of_the_iron_solari" in items:
            locket_events = self._apply_locket(unit, context)
            events.extend(locket_events)

        # Shroud of Stillness - Increase enemy mana costs
        if "shroud_of_stillness" in items:
            shroud_events = self._apply_shroud(unit, context)
            events.extend(shroud_events)

        # Frozen Heart - Reduce AS of nearby enemies
        if "frozen_heart" in items:
            frozen_events = self._apply_frozen_heart(unit, context)
            events.extend(frozen_events)

        # Gargoyle Stoneplate - Gain armor/MR per targeting enemy
        if "gargoyle_stoneplate" in items:
            gargoyle_events = self._apply_gargoyle(unit, context)
            events.extend(gargoyle_events)

        return events

    def _apply_zekes_herald(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Zeke's Herald AS buff to adjacent allies."""
        events = []
        buffed = []

        unit_pos = None
        if hasattr(context, 'grid') and context.grid:
            unit_pos = context.grid.get_unit_position(unit.id)

        for uid, other in context.all_units.items():
            if other.team == unit.team and other.id != unit.id and other.is_alive:
                # Check if adjacent (within 1 hex)
                in_range = True
                if unit_pos and hasattr(context, 'grid') and context.grid:
                    other_pos = context.grid.get_unit_position(other.id)
                    if other_pos:
                        in_range = unit_pos.distance_to(other_pos) <= self.ZEKES_RANGE

                if in_range:
                    from .status_effects import create_attack_speed_buff
                    buff = create_attack_speed_buff(
                        f"zekes_{unit.id}",
                        duration=999.0,  # Lasts entire combat
                        bonus=self.ZEKES_AS_BONUS,
                    )
                    context.status_effects.apply_effect(other, buff)
                    buffed.append(other.id)

        if buffed:
            events.append({
                "type": "zekes_herald",
                "source": unit.id,
                "targets": buffed,
                "as_bonus": self.ZEKES_AS_BONUS,
            })
        return events

    def _apply_chalice(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Chalice of Power AP buff to allies in same row."""
        events = []
        buffed = []

        unit_pos = None
        if hasattr(context, 'grid') and context.grid:
            unit_pos = context.grid.get_unit_position(unit.id)

        for uid, other in context.all_units.items():
            if other.team == unit.team and other.is_alive:
                # Check same row
                in_row = True
                if unit_pos and hasattr(context, 'grid') and context.grid:
                    other_pos = context.grid.get_unit_position(other.id)
                    if other_pos:
                        in_row = unit_pos.row == other_pos.row

                if in_row:
                    other.stats.ability_power += self.CHALICE_AP_BONUS
                    buffed.append(other.id)

        if buffed:
            events.append({
                "type": "chalice_of_power",
                "source": unit.id,
                "targets": buffed,
                "ap_bonus": self.CHALICE_AP_BONUS,
            })
        return events

    def _apply_locket(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Locket shield to allies in same row."""
        events = []
        shielded = []

        unit_pos = None
        if hasattr(context, 'grid') and context.grid:
            unit_pos = context.grid.get_unit_position(unit.id)

        for uid, other in context.all_units.items():
            if other.team == unit.team and other.is_alive:
                # Check same row
                in_row = True
                if unit_pos and hasattr(context, 'grid') and context.grid:
                    other_pos = context.grid.get_unit_position(other.id)
                    if other_pos:
                        in_row = unit_pos.row == other_pos.row

                if in_row:
                    from .status_effects import create_shield
                    shield = create_shield(
                        f"locket_{unit.id}",
                        self.LOCKET_SHIELD_DURATION,
                        self.LOCKET_SHIELD_AMOUNT,
                    )
                    context.status_effects.apply_effect(other, shield)
                    shielded.append(other.id)

        if shielded:
            events.append({
                "type": "locket",
                "source": unit.id,
                "targets": shielded,
                "shield": self.LOCKET_SHIELD_AMOUNT,
            })
        return events

    def _apply_shroud(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Shroud of Stillness mana increase to nearby enemies."""
        events = []
        affected = []

        unit_pos = None
        if hasattr(context, 'grid') and context.grid:
            unit_pos = context.grid.get_unit_position(unit.id)

        for uid, other in context.all_units.items():
            if other.team != unit.team and other.is_alive:
                # Check range
                in_range = True
                if unit_pos and hasattr(context, 'grid') and context.grid:
                    other_pos = context.grid.get_unit_position(other.id)
                    if other_pos:
                        in_range = unit_pos.distance_to(other_pos) <= self.SHROUD_RANGE

                if in_range:
                    # Increase max mana by 30%
                    mana_increase = other.stats.max_mana * self.SHROUD_MANA_INCREASE
                    other.stats.max_mana += mana_increase
                    affected.append(other.id)

        if affected:
            events.append({
                "type": "shroud_of_stillness",
                "source": unit.id,
                "targets": affected,
                "mana_increase": self.SHROUD_MANA_INCREASE,
            })
        return events

    def _apply_frozen_heart(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Frozen Heart AS reduction to nearby enemies."""
        events = []
        affected = []

        unit_pos = None
        if hasattr(context, 'grid') and context.grid:
            unit_pos = context.grid.get_unit_position(unit.id)

        for uid, other in context.all_units.items():
            if other.team != unit.team and other.is_alive:
                # Check range
                in_range = True
                if unit_pos and hasattr(context, 'grid') and context.grid:
                    other_pos = context.grid.get_unit_position(other.id)
                    if other_pos:
                        in_range = unit_pos.distance_to(other_pos) <= self.FROZEN_HEART_RANGE

                if in_range:
                    from .status_effects import StatusEffect, StatusEffectType
                    debuff = StatusEffect(
                        effect_type=StatusEffectType.ATTACK_SPEED_DEBUFF,
                        source_id=f"frozen_heart_{unit.id}",
                        duration=999.0,  # Lasts entire combat
                        value=self.FROZEN_HEART_AS_REDUCTION,
                    )
                    context.status_effects.apply_effect(other, debuff)
                    affected.append(other.id)

        if affected:
            events.append({
                "type": "frozen_heart",
                "source": unit.id,
                "targets": affected,
                "as_reduction": self.FROZEN_HEART_AS_REDUCTION,
            })
        return events

    def _apply_gargoyle(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Apply Gargoyle Stoneplate defensive stats."""
        events = []

        # Count enemies targeting this unit (simplified: count all enemies)
        enemy_count = sum(
            1 for uid, other in context.all_units.items()
            if other.team != unit.team and other.is_alive
        )

        bonus = enemy_count * self.GARGOYLE_STATS_PER_ENEMY
        unit.stats.armor += bonus
        unit.stats.magic_resist += bonus

        events.append({
            "type": "gargoyle_stoneplate",
            "unit": unit.id,
            "enemies": enemy_count,
            "armor_bonus": bonus,
            "mr_bonus": bonus,
        })
        return events

    # =========================================================================
    # ON ATTACK EFFECTS
    # =========================================================================

    def apply_on_attack_effects(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger when attacking (before damage).

        Args:
            attacker: The attacking unit.
            target: The target unit.
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(attacker)
        state = self.unit_state.get(attacker.id, {})

        # Statikk Shiv - Count attacks
        if "statikk_shiv" in items:
            state["statikk_counter"] = state.get("statikk_counter", 0) + 1

        # Guinsoo's Rageblade - Stack attack speed
        if "rageblade" in items:
            stacks = state.get("rageblade_stacks", 0)
            state["rageblade_stacks"] = stacks + 1
            bonus_as = self.RAGEBLADE_AS_PER_STACK
            attacker.stats.attack_speed += bonus_as
            events.append({
                "type": "rageblade_stack",
                "unit": attacker.id,
                "stacks": stacks + 1,
                "as_bonus": bonus_as,
            })

        # Spear of Shojin - Gain mana on attack
        if "spear_of_shojin" in items:
            attacker.gain_mana(self.SHOJIN_MANA_PER_ATTACK)
            events.append({
                "type": "shojin_mana",
                "unit": attacker.id,
                "mana": self.SHOJIN_MANA_PER_ATTACK,
            })

        self.unit_state[attacker.id] = state
        return events

    # =========================================================================
    # ON HIT EFFECTS (after damage)
    # =========================================================================

    def apply_on_hit_effects(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
        damage_event: "DamageEvent",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger after dealing damage.

        Args:
            attacker: The attacking unit.
            target: The target unit.
            damage_event: The damage that was dealt.
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(attacker)
        state = self.unit_state.get(attacker.id, {})

        # Giant Slayer - Bonus damage to high HP targets
        if "giant_slayer" in items:
            if target.stats.max_hp >= self.GIANT_SLAYER_HP_THRESHOLD:
                bonus_damage = damage_event.final_damage * self.GIANT_SLAYER_BONUS_DAMAGE
                target.take_damage(bonus_damage, "physical")
                attacker.total_damage_dealt += bonus_damage
                events.append({
                    "type": "giant_slayer",
                    "attacker": attacker.id,
                    "target": target.id,
                    "bonus_damage": bonus_damage,
                })

        # Statikk Shiv - Chain lightning every 3rd attack
        if "statikk_shiv" in items:
            if state.get("statikk_counter", 0) >= self.STATIKK_SHIV_ATTACK_COUNT:
                state["statikk_counter"] = 0
                shiv_events = self._trigger_statikk_shiv(attacker, target, context)
                events.extend(shiv_events)

        # Last Whisper - Armor shred on crit
        if "last_whisper" in items and damage_event.is_critical:
            from .status_effects import create_armor_shred
            shred = create_armor_shred(
                self.LAST_WHISPER_ARMOR_SHRED,
                self.LAST_WHISPER_DURATION
            )
            context.status_effects.apply_effect(target, shred)
            events.append({
                "type": "last_whisper_shred",
                "target": target.id,
                "shred": self.LAST_WHISPER_ARMOR_SHRED,
            })

        # Runaan's Hurricane - Bonus bolts (simplified)
        if "runaans_hurricane" in items:
            bolt_events = self._trigger_runaans(attacker, target, context)
            events.extend(bolt_events)

        # Hextech Gunblade - Heal for 25% of damage dealt
        if "hextech_gunblade" in items:
            heal_amount = damage_event.final_damage * self.GUNBLADE_OMNIVAMP
            excess = attacker.heal(heal_amount)

            # Excess healing becomes shield
            if excess > 0:
                from .status_effects import create_shield
                shield = create_shield(
                    f"gunblade_{attacker.id}",
                    duration=3.0,
                    shield_amount=excess,
                )
                context.status_effects.apply_effect(attacker, shield)
                events.append({
                    "type": "hextech_gunblade_shield",
                    "unit": attacker.id,
                    "heal": heal_amount - excess,
                    "shield": excess,
                })
            else:
                events.append({
                    "type": "hextech_gunblade",
                    "unit": attacker.id,
                    "heal": heal_amount,
                })

        # Guardbreaker - Track shielded targets for bonus damage
        if "guardbreaker" in items:
            if context.status_effects.has_effect(target, StatusEffectType.SHIELD):
                guardbreaker_targets = state.get("guardbreaker_targets", {})
                guardbreaker_targets[target.id] = context.current_time + self.GUARDBREAKER_DURATION
                state["guardbreaker_targets"] = guardbreaker_targets
                events.append({
                    "type": "guardbreaker_marked",
                    "attacker": attacker.id,
                    "target": target.id,
                })

        self.unit_state[attacker.id] = state
        return events

    def _trigger_statikk_shiv(
        self,
        attacker: "CombatUnit",
        primary_target: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Trigger Statikk Shiv chain lightning."""
        events = []
        from .attack import DamageType

        # Find up to 4 targets
        targets_hit = []
        for unit_id, unit in context.all_units.items():
            if unit.team != attacker.team and unit.is_alive:
                targets_hit.append(unit)
                if len(targets_hit) >= self.STATIKK_SHIV_TARGETS:
                    break

        for target in targets_hit:
            # Deal magic damage
            damage = self.STATIKK_SHIV_DAMAGE
            target.take_damage(damage, "magical")
            attacker.total_damage_dealt += damage

            # Apply MR shred
            from .status_effects import create_mr_shred
            shred = create_mr_shred(self.STATIKK_SHIV_MR_SHRED, 3.0)
            context.status_effects.apply_effect(target, shred)

        events.append({
            "type": "statikk_shiv",
            "attacker": attacker.id,
            "targets": [t.id for t in targets_hit],
            "damage": self.STATIKK_SHIV_DAMAGE,
        })

        return events

    def _trigger_runaans(
        self,
        attacker: "CombatUnit",
        primary_target: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Trigger Runaan's Hurricane bonus bolts."""
        events = []

        # Find 2 additional targets
        additional_targets = []
        for unit_id, unit in context.all_units.items():
            if (unit.team != attacker.team and
                unit.is_alive and
                unit.id != primary_target.id):
                additional_targets.append(unit)
                if len(additional_targets) >= 2:
                    break

        for target in additional_targets:
            # 55% AD damage
            damage = attacker.stats.attack_damage * 0.55
            target.take_damage(damage, "physical")
            attacker.total_damage_dealt += damage

        if additional_targets:
            events.append({
                "type": "runaans_hurricane",
                "attacker": attacker.id,
                "targets": [t.id for t in additional_targets],
            })

        return events

    # =========================================================================
    # ON CAST EFFECTS
    # =========================================================================

    def apply_on_cast_effects(
        self,
        caster: "CombatUnit",
        context: ItemEffectContext,
        ability_targets: Optional[List["CombatUnit"]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger after casting an ability.

        Args:
            caster: The unit that cast an ability.
            context: Combat context.
            ability_targets: List of units hit by the ability.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(caster)

        # Blue Buff - Reset mana to 20 after cast
        if "blue_buff" in items:
            caster.stats.current_mana = 20
            events.append({
                "type": "blue_buff_reset",
                "unit": caster.id,
                "mana": 20,
            })

        # Morellonomicon - Apply burn and grievous wounds to ability targets
        if "morellonomicon" in items and ability_targets:
            for target in ability_targets:
                if target.is_alive and target.team != caster.team:
                    # Apply burn (1% max HP per second for 10 seconds)
                    burn_dps = target.stats.max_hp * self.MORELLO_BURN_PERCENT
                    from .status_effects import create_burn, create_grievous_wounds
                    burn = create_burn(
                        f"morello_{caster.id}",
                        self.MORELLO_BURN_DURATION,
                        burn_dps,
                    )
                    context.status_effects.apply_effect(target, burn)

                    # Apply grievous wounds (33% heal reduction)
                    gw = create_grievous_wounds(
                        f"morello_{caster.id}",
                        self.MORELLO_BURN_DURATION,
                    )
                    context.status_effects.apply_effect(target, gw)

            events.append({
                "type": "morellonomicon",
                "caster": caster.id,
                "targets": [t.id for t in ability_targets if t.is_alive and t.team != caster.team],
            })

        # Ionic Spark - Damage nearby enemies when they cast
        # Note: This triggers on ENEMY casts, not self. See apply_enemy_cast_effects

        return events

    def apply_ionic_spark_on_enemy_cast(
        self,
        enemy_caster: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply Ionic Spark damage when an enemy casts.

        Args:
            enemy_caster: The enemy unit that cast an ability.
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []

        # Find all units with Ionic Spark that can hit this caster
        for uid, unit in context.all_units.items():
            if unit.team == enemy_caster.team or not unit.is_alive:
                continue

            items = self.get_equipped_item_ids(unit)
            if "ionic_spark" not in items:
                continue

            # Check if enemy is within range
            in_range = True
            if context.grid:
                unit_pos = context.grid.get_unit_position(unit.id)
                enemy_pos = context.grid.get_unit_position(enemy_caster.id)
                if unit_pos and enemy_pos:
                    in_range = unit_pos.distance_to(enemy_pos) <= self.IONIC_SPARK_RANGE

            if in_range:
                # Deal magic damage
                damage = self.IONIC_SPARK_DAMAGE
                enemy_caster.take_damage(damage, "magical")
                unit.total_damage_dealt += damage

                # Apply MR shred
                from .status_effects import create_mr_shred
                shred = create_mr_shred(self.IONIC_SPARK_MR_SHRED, 3.0)
                context.status_effects.apply_effect(enemy_caster, shred)

                events.append({
                    "type": "ionic_spark",
                    "source": unit.id,
                    "target": enemy_caster.id,
                    "damage": damage,
                })

        return events

    # =========================================================================
    # ON KILL EFFECTS
    # =========================================================================

    def apply_on_kill_effects(
        self,
        killer: "CombatUnit",
        killed: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger on kill.

        Args:
            killer: The unit that got the kill.
            killed: The unit that was killed.
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(killer)
        state = self.unit_state.get(killer.id, {})

        # Deathblade - Gain AD on takedown (max 10 stacks)
        if "deathblade" in items:
            stacks = state.get("deathblade_stacks", 0)
            if stacks < 10:
                state["deathblade_stacks"] = stacks + 1
                killer.stats.attack_damage += 8
                events.append({
                    "type": "deathblade_stack",
                    "unit": killer.id,
                    "stacks": stacks + 1,
                    "ad_bonus": 8,
                })

        self.unit_state[killer.id] = state
        return events

    # =========================================================================
    # ON DAMAGE TAKEN EFFECTS
    # =========================================================================

    def apply_on_damage_taken_effects(
        self,
        unit: "CombatUnit",
        damage: float,
        context: ItemEffectContext,
        attacker: Optional["CombatUnit"] = None,
        is_ability_damage: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Apply effects that trigger when taking damage.

        Args:
            unit: The unit that took damage.
            damage: Amount of damage taken.
            context: Combat context.
            attacker: The unit that dealt the damage (if any).
            is_ability_damage: Whether the damage was from an ability.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(unit)
        state = self.unit_state.get(unit.id, {})

        # Bloodthirster - Shield when below 40% HP (once per combat)
        if "bloodthirster" in items:
            if not state.get("bloodthirster_triggered", False):
                hp_percent = unit.stats.current_hp / unit.stats.max_hp
                if hp_percent < self.BLOODTHIRSTER_SHIELD_THRESHOLD:
                    state["bloodthirster_triggered"] = True
                    shield_amount = unit.stats.max_hp * self.BLOODTHIRSTER_SHIELD_PERCENT
                    from .status_effects import create_shield
                    shield = create_shield(
                        f"bloodthirster_{unit.id}",
                        5.0,
                        shield_amount,
                    )
                    context.status_effects.apply_effect(unit, shield)
                    events.append({
                        "type": "bloodthirster_shield",
                        "unit": unit.id,
                        "shield": shield_amount,
                    })

        # Titan's Resolve - Gain stats on attack/damage
        if "titans_resolve" in items:
            stacks = state.get("titans_stacks", 0)
            if stacks < 25:
                state["titans_stacks"] = stacks + 1
                unit.stats.attack_damage += 2
                unit.stats.ability_power += 2

                # At max stacks, gain armor/MR
                if stacks + 1 == 25:
                    unit.stats.armor += 25
                    unit.stats.magic_resist += 25
                    events.append({
                        "type": "titans_max",
                        "unit": unit.id,
                    })

        # Edge of Night - Stealth and AS when dropping low (once per combat)
        if "edge_of_night" in items:
            if not state.get("edge_of_night_triggered", False):
                hp_percent = unit.stats.current_hp / unit.stats.max_hp
                if hp_percent < 0.5:
                    state["edge_of_night_triggered"] = True
                    unit.stats.attack_speed *= 1.4  # 40% AS boost
                    events.append({
                        "type": "edge_of_night",
                        "unit": unit.id,
                    })

        # Sterak's Gage - Shield and AD when below 60% HP (once per combat)
        if "steraks_gage" in items:
            if not state.get("sterak_triggered", False):
                hp_percent = unit.stats.current_hp / unit.stats.max_hp
                if hp_percent < self.STERAK_THRESHOLD:
                    state["sterak_triggered"] = True

                    # Grant shield
                    shield_amount = unit.stats.max_hp * self.STERAK_SHIELD_PERCENT
                    from .status_effects import create_shield
                    shield = create_shield(
                        f"sterak_{unit.id}",
                        8.0,
                        shield_amount,
                    )
                    context.status_effects.apply_effect(unit, shield)

                    # Grant AD bonus
                    ad_bonus = unit.stats.attack_damage * self.STERAK_AD_BONUS
                    unit.stats.attack_damage += ad_bonus

                    events.append({
                        "type": "steraks_gage",
                        "unit": unit.id,
                        "shield": shield_amount,
                        "ad_bonus": ad_bonus,
                    })

        # Bramble Vest - Reflect damage to attackers (only on attack damage)
        if "bramble_vest" in items and attacker and not is_ability_damage:
            reflect_damage = self.BRAMBLE_REFLECT_DAMAGE
            attacker.take_damage(reflect_damage, "magical")
            unit.total_damage_dealt += reflect_damage
            events.append({
                "type": "bramble_vest",
                "unit": unit.id,
                "attacker": attacker.id,
                "damage": reflect_damage,
            })

        self.unit_state[unit.id] = state
        return events

    def modify_incoming_damage(
        self,
        unit: "CombatUnit",
        damage: float,
        is_ability_damage: bool = False,
    ) -> float:
        """
        Modify incoming damage based on defensive items.

        Args:
            unit: The unit taking damage.
            damage: Base damage before reduction.
            is_ability_damage: Whether the damage is from an ability.

        Returns:
            Modified damage after item reductions.
        """
        items = self.get_equipped_item_ids(unit)
        modified_damage = damage

        # Bramble Vest - 8% less damage from attacks
        if "bramble_vest" in items and not is_ability_damage:
            modified_damage *= (1 - self.BRAMBLE_DAMAGE_REDUCTION)

        # Dragon's Claw - 10% less ability damage
        if "dragons_claw" in items and is_ability_damage:
            modified_damage *= (1 - self.DRAGON_CLAW_ABILITY_REDUCTION)

        return modified_damage

    # =========================================================================
    # PERIODIC EFFECTS
    # =========================================================================

    def update_periodic_effects(
        self,
        unit: "CombatUnit",
        delta_time: float,
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Update periodic item effects.

        Args:
            unit: The unit to update.
            delta_time: Time elapsed (seconds).
            context: Combat context.

        Returns:
            List of effect events.
        """
        events = []
        items = self.get_equipped_item_ids(unit)
        timers = self.periodic_timers.get(unit.id, {})
        state = self.unit_state.get(unit.id, {})

        # Sunfire Cape - Burn adjacent enemies every 2 seconds
        if "red_buff" in items or "sunfire_cape" in items:
            timers["sunfire"] = timers.get("sunfire", 0) + delta_time
            if timers["sunfire"] >= self.SUNFIRE_TICK_INTERVAL:
                timers["sunfire"] = 0
                sunfire_events = self._trigger_sunfire(unit, context)
                events.extend(sunfire_events)

        # Redemption - Heal nearby allies every 5 seconds
        if "redemption" in items:
            timers["redemption"] = timers.get("redemption", 0) + delta_time
            if timers["redemption"] >= 5.0:
                timers["redemption"] = 0
                redemption_events = self._trigger_redemption(unit, context)
                events.extend(redemption_events)

        # Archangel's Staff - Gain 20 AP every 5 seconds
        if "archangels_staff" in items:
            state["archangels_timer"] = state.get("archangels_timer", 0) + delta_time
            if state["archangels_timer"] >= 5.0:
                state["archangels_timer"] = 0
                unit.stats.ability_power += 20
                events.append({
                    "type": "archangels_ap",
                    "unit": unit.id,
                    "ap_gained": 20,
                })

        # Warmog's Armor - Regenerate 3% max HP per second
        if "warmogs_armor" in items:
            heal_amount = unit.stats.max_hp * self.WARMOG_REGEN_PERCENT * delta_time
            unit.heal(heal_amount)
            # No event spam for continuous regen

        # Dragon's Claw - Heal 1.2% max HP per targeting enemy every 2 seconds
        if "dragons_claw" in items:
            timers["dragon_claw"] = timers.get("dragon_claw", 0) + delta_time
            if timers["dragon_claw"] >= self.DRAGON_CLAW_INTERVAL:
                timers["dragon_claw"] = 0

                # Count enemies targeting this unit (simplified: count all enemies)
                enemy_count = sum(
                    1 for uid, other in context.all_units.items()
                    if other.team != unit.team and other.is_alive
                )

                heal_amount = unit.stats.max_hp * self.DRAGON_CLAW_HEAL_PERCENT * enemy_count
                if heal_amount > 0:
                    unit.heal(heal_amount)
                    events.append({
                        "type": "dragons_claw_heal",
                        "unit": unit.id,
                        "heal": heal_amount,
                        "enemy_count": enemy_count,
                    })

        self.periodic_timers[unit.id] = timers
        self.unit_state[unit.id] = state
        return events

    def _trigger_sunfire(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Trigger Sunfire Cape damage."""
        events = []
        targets_hit = []

        # Find adjacent enemies
        for unit_id, other in context.all_units.items():
            if other.team != unit.team and other.is_alive:
                # Simplified: just check if enemy exists
                # Full implementation would check hex distance
                targets_hit.append(other)
                if len(targets_hit) >= 3:
                    break

        for target in targets_hit:
            damage = target.stats.max_hp * self.SUNFIRE_DAMAGE_PERCENT
            target.take_damage(damage, "magical")
            unit.total_damage_dealt += damage

            # Apply burn
            from .status_effects import create_burn
            burn = create_burn(damage * 0.5, 2.0)  # 50% of damage as burn
            context.status_effects.apply_effect(target, burn)

        if targets_hit:
            events.append({
                "type": "sunfire",
                "unit": unit.id,
                "targets": [t.id for t in targets_hit],
            })

        return events

    def _trigger_redemption(
        self,
        unit: "CombatUnit",
        context: ItemEffectContext,
    ) -> List[Dict[str, Any]]:
        """Trigger Redemption healing."""
        events = []
        allies_healed = []

        # Find nearby allies
        for unit_id, other in context.all_units.items():
            if other.team == unit.team and other.is_alive:
                heal_amount = other.stats.max_hp * 0.15
                other.heal(heal_amount)
                allies_healed.append(other)

        if allies_healed:
            events.append({
                "type": "redemption",
                "unit": unit.id,
                "allies": [a.id for a in allies_healed],
            })

        return events

    # =========================================================================
    # DAMAGE MODIFICATION
    # =========================================================================

    def modify_outgoing_damage(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
        base_damage: float,
        is_ability: bool = False,
        context: Optional[ItemEffectContext] = None,
    ) -> float:
        """
        Modify outgoing damage based on item effects.

        Args:
            attacker: The attacking unit.
            target: The target unit.
            base_damage: Base damage before modification.
            is_ability: Whether this is ability damage.
            context: Combat context for time-based effects.

        Returns:
            Modified damage.
        """
        items = self.get_equipped_item_ids(attacker)
        damage = base_damage

        # Giant Slayer - 25% bonus vs high HP
        if "giant_slayer" in items:
            if target.stats.max_hp >= self.GIANT_SLAYER_HP_THRESHOLD:
                damage *= (1 + self.GIANT_SLAYER_BONUS_DAMAGE)

        # Guardbreaker - 25% bonus vs shielded/recently shielded enemies
        if "guardbreaker" in items:
            state = self.unit_state.get(attacker.id, {})
            guardbreaker_targets = state.get("guardbreaker_targets", {})
            current_time = context.current_time if context else 0.0

            # Check if target is currently shielded or was recently shielded
            if context and context.status_effects.has_effect(target, StatusEffectType.SHIELD):
                damage *= (1 + self.GUARDBREAKER_BONUS_DAMAGE)
            elif target.id in guardbreaker_targets:
                if guardbreaker_targets[target.id] > current_time:
                    damage *= (1 + self.GUARDBREAKER_BONUS_DAMAGE)
                else:
                    # Clean up expired entry
                    del guardbreaker_targets[target.id]
                    state["guardbreaker_targets"] = guardbreaker_targets
                    self.unit_state[attacker.id] = state

        return damage

    def can_ability_crit(self, unit: "CombatUnit") -> bool:
        """Check if unit's abilities can crit (IE or JG)."""
        items = self.get_equipped_item_ids(unit)
        return "infinity_edge" in items or "jeweled_gauntlet" in items

    def can_miss(self, attacker: "CombatUnit") -> bool:
        """Check if attacks can miss (RFC prevents miss)."""
        items = self.get_equipped_item_ids(attacker)
        return "rapid_firecannon" not in items

    def clear(self) -> None:
        """Clear all tracked state."""
        self.unit_state.clear()
        self.periodic_timers.clear()
