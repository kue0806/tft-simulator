"""Status Effects System for TFT Combat.

Handles buffs, debuffs, and crowd control effects including:
- Stun, silence, disarm
- Damage over time (burn, bleed)
- Stat modifiers (attack speed buff, armor reduction)
- Shields
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from enum import Enum, auto
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .combat_unit import CombatUnit


class StatusEffectType(Enum):
    """Types of status effects."""

    # Crowd Control
    STUN = auto()  # Cannot act
    SILENCE = auto()  # Cannot cast abilities
    DISARM = auto()  # Cannot attack
    ROOT = auto()  # Cannot move
    KNOCKUP = auto()  # Airborne, cannot act
    TAUNT = auto()  # Forced to attack source
    BLIND = auto()  # Attacks miss
    CHARM = auto()  # Walk toward source

    # Damage Over Time
    BURN = auto()  # Magic damage over time
    BLEED = auto()  # Physical damage over time
    POISON = auto()  # True damage over time

    # Healing/Shields
    SHIELD = auto()  # Absorbs damage
    HEAL_OVER_TIME = auto()  # Regeneration

    # Stat Modifiers
    ATTACK_SPEED_BUFF = auto()
    ATTACK_SPEED_DEBUFF = auto()
    ATTACK_DAMAGE_BUFF = auto()
    ATTACK_DAMAGE_DEBUFF = auto()
    ARMOR_BUFF = auto()
    ARMOR_DEBUFF = auto()
    MAGIC_RESIST_BUFF = auto()
    MAGIC_RESIST_DEBUFF = auto()
    ABILITY_POWER_BUFF = auto()
    DAMAGE_REDUCTION = auto()
    DAMAGE_AMPLIFICATION = auto()

    # Special
    INVULNERABLE = auto()  # Cannot be damaged
    UNTARGETABLE = auto()  # Cannot be targeted
    GRIEVOUS_WOUNDS = auto()  # Reduced healing


@dataclass
class StatusEffect:
    """
    Represents an active status effect.

    Attributes:
        effect_type: Type of the effect.
        source_id: Unit that applied the effect.
        duration: Remaining duration in seconds.
        value: Effect value (damage, stat modifier amount, etc.).
        max_duration: Original duration for percentage calculations.
        stacks: Number of stacks (for stackable effects).
        tick_interval: Time between damage ticks (for DoT effects).
        time_since_tick: Time since last tick.
    """

    effect_type: StatusEffectType
    source_id: str
    duration: float
    value: float = 0.0
    max_duration: float = 0.0
    stacks: int = 1
    max_stacks: int = 1
    tick_interval: float = 1.0
    time_since_tick: float = 0.0

    # For shields
    remaining_shield: float = 0.0

    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if effect has expired."""
        return self.duration <= 0

    @property
    def is_cc(self) -> bool:
        """Check if this is a crowd control effect."""
        return self.effect_type in [
            StatusEffectType.STUN,
            StatusEffectType.SILENCE,
            StatusEffectType.DISARM,
            StatusEffectType.ROOT,
            StatusEffectType.KNOCKUP,
            StatusEffectType.TAUNT,
            StatusEffectType.CHARM,
        ]

    @property
    def prevents_actions(self) -> bool:
        """Check if this effect prevents all actions."""
        return self.effect_type in [
            StatusEffectType.STUN,
            StatusEffectType.KNOCKUP,
            StatusEffectType.CHARM,
        ]

    @property
    def prevents_casting(self) -> bool:
        """Check if this effect prevents ability casting."""
        return self.effect_type in [
            StatusEffectType.STUN,
            StatusEffectType.SILENCE,
            StatusEffectType.KNOCKUP,
            StatusEffectType.CHARM,
        ]

    @property
    def prevents_attacking(self) -> bool:
        """Check if this effect prevents basic attacks."""
        return self.effect_type in [
            StatusEffectType.STUN,
            StatusEffectType.DISARM,
            StatusEffectType.KNOCKUP,
            StatusEffectType.CHARM,
        ]

    @property
    def prevents_movement(self) -> bool:
        """Check if this effect prevents movement."""
        return self.effect_type in [
            StatusEffectType.STUN,
            StatusEffectType.ROOT,
            StatusEffectType.KNOCKUP,
        ]


class StatusEffectSystem:
    """
    Manages status effects on combat units.

    Handles application, stacking, updating, and removal of effects.

    Usage:
        effect_system = StatusEffectSystem()
        effect_system.apply_effect(unit, effect)
        effect_system.update(unit, delta_time)
    """

    def __init__(self):
        """Initialize status effect system."""
        # Effects per unit
        self._unit_effects: Dict[str, List[StatusEffect]] = {}

        # Stat modifier cache (recalculated on change)
        self._stat_modifiers: Dict[str, Dict[str, float]] = {}

    def apply_effect(
        self,
        unit: "CombatUnit",
        effect: StatusEffect,
    ) -> bool:
        """
        Apply a status effect to a unit.

        Handles stacking and effect merging rules.

        Args:
            unit: Target unit.
            effect: Effect to apply.

        Returns:
            True if effect was applied successfully.
        """
        if not unit.is_alive:
            return False

        # Initialize unit effects list
        if unit.id not in self._unit_effects:
            self._unit_effects[unit.id] = []

        effects = self._unit_effects[unit.id]

        # Check for existing effect of same type from same source
        existing = self._find_effect(unit.id, effect.effect_type, effect.source_id)

        if existing:
            # Refresh or stack
            if effect.max_stacks > 1 and existing.stacks < effect.max_stacks:
                existing.stacks += 1
                existing.duration = max(existing.duration, effect.duration)
                existing.value = effect.value  # Use new value
            else:
                # Refresh duration
                existing.duration = max(existing.duration, effect.duration)
            self._invalidate_stat_cache(unit.id)
            return True

        # Apply new effect
        effect.max_duration = effect.duration
        if effect.effect_type == StatusEffectType.SHIELD:
            effect.remaining_shield = effect.value

        effects.append(effect)
        self._invalidate_stat_cache(unit.id)

        # Apply CC flag
        if effect.prevents_actions:
            from .combat_unit import UnitState
            unit.state = UnitState.STUNNED

        return True

    def remove_effect(
        self,
        unit: "CombatUnit",
        effect_type: StatusEffectType,
        source_id: Optional[str] = None,
    ) -> bool:
        """
        Remove a status effect from a unit.

        Args:
            unit: Target unit.
            effect_type: Type of effect to remove.
            source_id: Optional source filter.

        Returns:
            True if effect was removed.
        """
        if unit.id not in self._unit_effects:
            return False

        effects = self._unit_effects[unit.id]
        initial_count = len(effects)

        if source_id:
            self._unit_effects[unit.id] = [
                e for e in effects
                if not (e.effect_type == effect_type and e.source_id == source_id)
            ]
        else:
            self._unit_effects[unit.id] = [
                e for e in effects if e.effect_type != effect_type
            ]

        removed = len(self._unit_effects[unit.id]) < initial_count
        if removed:
            self._invalidate_stat_cache(unit.id)
            self._update_unit_state(unit)

        return removed

    def update(
        self,
        unit: "CombatUnit",
        delta_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Update all effects on a unit.

        Args:
            unit: The unit to update.
            delta_time: Time elapsed (seconds).

        Returns:
            List of effect events (damage, healing, expiration, etc.).
        """
        if unit.id not in self._unit_effects:
            return []

        effects = self._unit_effects[unit.id]
        events = []
        expired = []

        for effect in effects:
            # Update duration
            effect.duration -= delta_time

            # Process damage over time
            if effect.effect_type in [
                StatusEffectType.BURN,
                StatusEffectType.BLEED,
                StatusEffectType.POISON,
            ]:
                effect.time_since_tick += delta_time
                if effect.time_since_tick >= effect.tick_interval:
                    effect.time_since_tick = 0.0
                    damage = effect.value * effect.stacks

                    # Determine damage type
                    if effect.effect_type == StatusEffectType.BURN:
                        damage_type = "magical"
                    elif effect.effect_type == StatusEffectType.BLEED:
                        damage_type = "physical"
                    else:
                        damage_type = "true"

                    actual = unit.take_damage(damage, damage_type)
                    events.append({
                        "type": "dot_damage",
                        "effect": effect.effect_type.name,
                        "damage": actual,
                        "source_id": effect.source_id,
                    })

            # Process heal over time
            elif effect.effect_type == StatusEffectType.HEAL_OVER_TIME:
                effect.time_since_tick += delta_time
                if effect.time_since_tick >= effect.tick_interval:
                    effect.time_since_tick = 0.0
                    healing = effect.value * effect.stacks

                    # Check grievous wounds
                    if self.has_effect(unit, StatusEffectType.GRIEVOUS_WOUNDS):
                        healing *= 0.5  # 50% reduced healing

                    actual = unit.heal(healing)
                    events.append({
                        "type": "hot_healing",
                        "healing": actual,
                        "source_id": effect.source_id,
                    })

            # Check expiration
            if effect.is_expired:
                expired.append(effect)
                events.append({
                    "type": "effect_expired",
                    "effect": effect.effect_type.name,
                })

        # Remove expired effects
        for effect in expired:
            effects.remove(effect)

        if expired:
            self._invalidate_stat_cache(unit.id)
            self._update_unit_state(unit)

        return events

    def absorb_damage(
        self,
        unit: "CombatUnit",
        damage: float,
    ) -> float:
        """
        Attempt to absorb damage with shields.

        Args:
            unit: The unit taking damage.
            damage: Incoming damage amount.

        Returns:
            Remaining damage after shield absorption.
        """
        if unit.id not in self._unit_effects:
            return damage

        remaining = damage
        shields = [e for e in self._unit_effects[unit.id]
                   if e.effect_type == StatusEffectType.SHIELD and e.remaining_shield > 0]

        # Absorb with oldest shields first
        for shield in shields:
            if remaining <= 0:
                break

            absorbed = min(shield.remaining_shield, remaining)
            shield.remaining_shield -= absorbed
            remaining -= absorbed

            # Remove depleted shield
            if shield.remaining_shield <= 0:
                self._unit_effects[unit.id].remove(shield)

        return remaining

    def has_effect(
        self,
        unit: "CombatUnit",
        effect_type: StatusEffectType,
    ) -> bool:
        """Check if unit has a specific effect."""
        if unit.id not in self._unit_effects:
            return False
        return any(e.effect_type == effect_type for e in self._unit_effects[unit.id])

    def get_effects(self, unit: "CombatUnit") -> List[StatusEffect]:
        """Get all effects on a unit."""
        return list(self._unit_effects.get(unit.id, []))

    def get_stat_modifiers(self, unit_id: str) -> Dict[str, float]:
        """
        Get total stat modifiers from effects.

        Returns dict with keys like 'attack_speed', 'armor', etc.
        Values are additive modifiers (can be positive or negative).
        """
        if unit_id in self._stat_modifiers:
            return self._stat_modifiers[unit_id]

        modifiers = {
            "attack_speed": 0.0,
            "attack_damage": 0.0,
            "armor": 0.0,
            "magic_resist": 0.0,
            "ability_power": 0.0,
            "damage_reduction": 0.0,
            "damage_amp": 0.0,
        }

        if unit_id not in self._unit_effects:
            return modifiers

        for effect in self._unit_effects[unit_id]:
            value = effect.value * effect.stacks

            if effect.effect_type == StatusEffectType.ATTACK_SPEED_BUFF:
                modifiers["attack_speed"] += value
            elif effect.effect_type == StatusEffectType.ATTACK_SPEED_DEBUFF:
                modifiers["attack_speed"] -= value
            elif effect.effect_type == StatusEffectType.ATTACK_DAMAGE_BUFF:
                modifiers["attack_damage"] += value
            elif effect.effect_type == StatusEffectType.ATTACK_DAMAGE_DEBUFF:
                modifiers["attack_damage"] -= value
            elif effect.effect_type == StatusEffectType.ARMOR_BUFF:
                modifiers["armor"] += value
            elif effect.effect_type == StatusEffectType.ARMOR_DEBUFF:
                modifiers["armor"] -= value
            elif effect.effect_type == StatusEffectType.MAGIC_RESIST_BUFF:
                modifiers["magic_resist"] += value
            elif effect.effect_type == StatusEffectType.MAGIC_RESIST_DEBUFF:
                modifiers["magic_resist"] -= value
            elif effect.effect_type == StatusEffectType.ABILITY_POWER_BUFF:
                modifiers["ability_power"] += value
            elif effect.effect_type == StatusEffectType.DAMAGE_REDUCTION:
                modifiers["damage_reduction"] += value
            elif effect.effect_type == StatusEffectType.DAMAGE_AMPLIFICATION:
                modifiers["damage_amp"] += value

        self._stat_modifiers[unit_id] = modifiers
        return modifiers

    def is_cc_immune(self, unit: "CombatUnit") -> bool:
        """Check if unit is immune to CC."""
        # Could check for CC immunity effects here
        return False

    def can_act(self, unit: "CombatUnit") -> bool:
        """Check if unit can perform actions (considering CC)."""
        if unit.id not in self._unit_effects:
            return True
        return not any(e.prevents_actions for e in self._unit_effects[unit.id])

    def can_cast(self, unit: "CombatUnit") -> bool:
        """Check if unit can cast abilities."""
        if unit.id not in self._unit_effects:
            return True
        return not any(e.prevents_casting for e in self._unit_effects[unit.id])

    def can_attack(self, unit: "CombatUnit") -> bool:
        """Check if unit can basic attack."""
        if unit.id not in self._unit_effects:
            return True
        return not any(e.prevents_attacking for e in self._unit_effects[unit.id])

    def can_move(self, unit: "CombatUnit") -> bool:
        """Check if unit can move."""
        if unit.id not in self._unit_effects:
            return True
        return not any(e.prevents_movement for e in self._unit_effects[unit.id])

    def clear_unit(self, unit_id: str) -> None:
        """Clear all effects from a unit."""
        if unit_id in self._unit_effects:
            del self._unit_effects[unit_id]
        self._invalidate_stat_cache(unit_id)

    def clear_all(self) -> None:
        """Clear all effects from all units."""
        self._unit_effects.clear()
        self._stat_modifiers.clear()

    def _find_effect(
        self,
        unit_id: str,
        effect_type: StatusEffectType,
        source_id: Optional[str] = None,
    ) -> Optional[StatusEffect]:
        """Find an existing effect on a unit."""
        if unit_id not in self._unit_effects:
            return None

        for effect in self._unit_effects[unit_id]:
            if effect.effect_type == effect_type:
                if source_id is None or effect.source_id == source_id:
                    return effect
        return None

    def _invalidate_stat_cache(self, unit_id: str) -> None:
        """Invalidate stat modifier cache for a unit."""
        if unit_id in self._stat_modifiers:
            del self._stat_modifiers[unit_id]

    def _update_unit_state(self, unit: "CombatUnit") -> None:
        """Update unit state based on current effects."""
        if not self.can_act(unit) and unit.is_alive:
            from .combat_unit import UnitState
            unit.state = UnitState.STUNNED
        elif unit.is_alive:
            from .combat_unit import UnitState
            if unit.state == UnitState.STUNNED:
                unit.state = UnitState.IDLE


# Helper functions
def create_stun(source_id: str, duration: float) -> StatusEffect:
    """Create a stun effect."""
    return StatusEffect(
        effect_type=StatusEffectType.STUN,
        source_id=source_id,
        duration=duration,
    )


def create_burn(source_id: str, duration: float, damage_per_tick: float) -> StatusEffect:
    """Create a burn (magic DoT) effect."""
    return StatusEffect(
        effect_type=StatusEffectType.BURN,
        source_id=source_id,
        duration=duration,
        value=damage_per_tick,
        tick_interval=1.0,
    )


def create_shield(source_id: str, duration: float, shield_amount: float) -> StatusEffect:
    """Create a shield effect."""
    return StatusEffect(
        effect_type=StatusEffectType.SHIELD,
        source_id=source_id,
        duration=duration,
        value=shield_amount,
        remaining_shield=shield_amount,
    )


def create_attack_speed_buff(source_id: str, duration: float, bonus: float) -> StatusEffect:
    """Create an attack speed buff."""
    return StatusEffect(
        effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
        source_id=source_id,
        duration=duration,
        value=bonus,
    )


def create_grievous_wounds(source_id: str, duration: float) -> StatusEffect:
    """Create a grievous wounds effect (50% reduced healing)."""
    return StatusEffect(
        effect_type=StatusEffectType.GRIEVOUS_WOUNDS,
        source_id=source_id,
        duration=duration,
    )


def create_armor_shred(shred_percent: float, duration: float) -> StatusEffect:
    """Create an armor reduction debuff (percentage based)."""
    return StatusEffect(
        effect_type=StatusEffectType.ARMOR_DEBUFF,
        source_id="item",
        duration=duration,
        value=shred_percent,  # Stored as percent, applied multiplicatively
    )


def create_mr_shred(shred_percent: float, duration: float) -> StatusEffect:
    """Create a magic resist reduction debuff (percentage based)."""
    return StatusEffect(
        effect_type=StatusEffectType.MAGIC_RESIST_DEBUFF,
        source_id="item",
        duration=duration,
        value=shred_percent,
    )


def create_cc_immunity(duration: float) -> StatusEffect:
    """Create a CC immunity effect (e.g., Quicksilver)."""
    return StatusEffect(
        effect_type=StatusEffectType.INVULNERABLE,  # Using invulnerable for now
        source_id="item",
        duration=duration,
        custom_data={"cc_immunity": True},
    )
