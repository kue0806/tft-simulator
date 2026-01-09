"""Attack System for TFT Combat.

Handles basic attack execution including:
- Attack timing and cooldowns
- Physical damage calculation
- Critical strikes
- Mana generation on attack
- On-hit effects
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from enum import Enum, auto
import random

if TYPE_CHECKING:
    from .combat_unit import CombatUnit


class DamageType(Enum):
    """Types of damage."""

    PHYSICAL = "physical"
    MAGICAL = "magical"
    TRUE = "true"


@dataclass
class DamageEvent:
    """Represents a damage instance."""

    source_id: str  # Attacking unit ID
    target_id: str  # Target unit ID
    raw_damage: float  # Pre-mitigation damage
    final_damage: float  # Post-mitigation damage
    damage_type: DamageType
    is_critical: bool = False
    is_ability: bool = False
    source_name: str = ""

    # Additional tracking
    overkill: float = 0.0  # Damage beyond killing blow
    healing_done: float = 0.0  # Omnivamp healing


@dataclass
class AttackResult:
    """Result of an attack action."""

    success: bool
    damage_event: Optional[DamageEvent] = None
    target_killed: bool = False
    mana_gained_attacker: float = 0.0
    mana_gained_target: float = 0.0


# Import mana constants from core
from src.core.constants import (
    MANA_PER_ATTACK_BY_ROLE,
    MANA_PER_ATTACK_DEFAULT,
    MANA_ON_DAMAGE_PERCENT,
    MANA_ON_DAMAGE_CAP,
    UnitRole,
)


def get_mana_per_attack(role: str) -> float:
    """Get mana gained per attack based on unit role."""
    return float(MANA_PER_ATTACK_BY_ROLE.get(role, MANA_PER_ATTACK_DEFAULT))


def get_mana_on_damage(pre_mitigation_damage: float) -> float:
    """
    Calculate mana gained from taking damage.

    Based on pre-mitigation damage, capped at 42.
    """
    mana = pre_mitigation_damage * MANA_ON_DAMAGE_PERCENT
    return min(mana, MANA_ON_DAMAGE_CAP)


# Legacy constants (for backward compatibility)
MANA_ON_ATTACK = 10.0
MANA_ON_DAMAGE_TAKEN = 7.0
MANA_ON_DAMAGE_SCALE = 0.01


class AttackSystem:
    """
    Manages basic attacks in combat.

    Handles attack timing, damage calculation, and related effects.

    Usage:
        attack_system = AttackSystem()
        result = attack_system.execute_attack(attacker, target, context)
    """

    def __init__(self, rng: Optional[random.Random] = None):
        """
        Initialize attack system.

        Args:
            rng: Random number generator for deterministic simulation.
        """
        self.rng = rng or random.Random()

        # On-hit effect callbacks
        self._on_hit_effects: List[Callable[["CombatUnit", "CombatUnit", DamageEvent], None]] = []
        self._on_attack_effects: List[Callable[["CombatUnit", "CombatUnit"], None]] = []

    def can_attack(self, attacker: "CombatUnit") -> bool:
        """
        Check if unit can perform an attack.

        Args:
            attacker: The attacking unit.

        Returns:
            True if attack is possible.
        """
        return (
            attacker.is_alive
            and attacker.can_act
            and attacker.attack_cooldown <= 0
            and not attacker.is_casting
        )

    def update_cooldown(self, unit: "CombatUnit", delta_time: float) -> None:
        """
        Update attack cooldown by delta time.

        Args:
            unit: The unit to update.
            delta_time: Time elapsed (seconds).
        """
        if unit.attack_cooldown > 0:
            unit.attack_cooldown = max(0, unit.attack_cooldown - delta_time)

    def execute_attack(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
    ) -> AttackResult:
        """
        Execute a basic attack.

        Args:
            attacker: The attacking unit.
            target: The target unit.

        Returns:
            AttackResult with damage details.
        """
        if not self.can_attack(attacker):
            return AttackResult(success=False)

        if not target.is_targetable:
            return AttackResult(success=False)

        # Set attack cooldown
        attacker.attack_cooldown = attacker.attack_interval

        # Trigger on-attack effects (before damage)
        for effect in self._on_attack_effects:
            effect(attacker, target)

        # Calculate damage
        damage_event = self._calculate_attack_damage(attacker, target)

        # Apply damage
        actual_damage = target.take_damage(
            damage_event.final_damage, damage_event.damage_type.value
        )
        damage_event.final_damage = actual_damage

        # Track damage dealt
        attacker.total_damage_dealt += actual_damage

        # Check for kill
        target_killed = not target.is_alive
        if target_killed:
            attacker.kills += 1
            damage_event.overkill = max(0, actual_damage - target.stats.current_hp)

        # Omnivamp healing
        if attacker.stats.omnivamp > 0 and actual_damage > 0:
            heal_amount = actual_damage * attacker.stats.omnivamp
            healed = attacker.heal(heal_amount)
            damage_event.healing_done = healed

        # Mana generation (role-based for attacker)
        mana_attacker = get_mana_per_attack(attacker.role)

        # Mana from damage taken (based on pre-mitigation damage)
        # Only if damage was actually dealt (not dodged)
        mana_target = 0.0
        if damage_event.raw_damage > 0 and actual_damage > 0:
            mana_target = get_mana_on_damage(damage_event.raw_damage)

        attacker.gain_mana(mana_attacker)
        if target.is_alive and mana_target > 0:
            target.gain_mana(mana_target)

        # Trigger on-hit effects (after damage)
        for effect in self._on_hit_effects:
            effect(attacker, target, damage_event)

        return AttackResult(
            success=True,
            damage_event=damage_event,
            target_killed=target_killed,
            mana_gained_attacker=mana_attacker,
            mana_gained_target=mana_target,
        )

    def _calculate_attack_damage(
        self, attacker: "CombatUnit", target: "CombatUnit"
    ) -> DamageEvent:
        """
        Calculate basic attack damage.

        Args:
            attacker: The attacking unit.
            target: The target unit.

        Returns:
            DamageEvent with calculated values.
        """
        base_damage = attacker.stats.attack_damage

        # Check critical strike
        is_crit = self.rng.random() < attacker.stats.crit_chance

        if is_crit:
            crit_multiplier = attacker.stats.crit_damage
            raw_damage = base_damage * crit_multiplier
        else:
            raw_damage = base_damage

        # Apply damage amplification
        raw_damage *= attacker.stats.damage_amp

        # Physical damage reduction (armor)
        # Formula: reduction = armor / (armor + 100)
        armor = target.stats.armor
        armor_reduction = armor / (armor + 100) if armor >= 0 else 0
        final_damage = raw_damage * (1 - armor_reduction)

        # Note: damage_reduction is applied in CombatUnit.take_damage()
        # to avoid double-application

        # Check dodge
        if self.rng.random() < target.stats.dodge_chance:
            final_damage = 0

        return DamageEvent(
            source_id=attacker.id,
            target_id=target.id,
            raw_damage=raw_damage,
            final_damage=max(0, final_damage),
            damage_type=DamageType.PHYSICAL,
            is_critical=is_crit,
            is_ability=False,
            source_name=attacker.name,
        )

    def calculate_damage(
        self,
        source: "CombatUnit",
        target: "CombatUnit",
        base_damage: float,
        damage_type: DamageType,
        can_crit: bool = False,
        is_ability: bool = False,
    ) -> DamageEvent:
        """
        Calculate damage for abilities or other sources.

        Args:
            source: The damage source unit.
            target: The target unit.
            base_damage: Base damage amount.
            damage_type: Type of damage.
            can_crit: Whether this damage can critically strike.
            is_ability: Whether this is ability damage.

        Returns:
            DamageEvent with calculated values.
        """
        raw_damage = base_damage

        # Check critical strike
        is_crit = False
        if can_crit:
            is_crit = self.rng.random() < source.stats.crit_chance
            if is_crit:
                raw_damage *= source.stats.crit_damage

        # Apply damage amplification
        raw_damage *= source.stats.damage_amp

        # Apply AP scaling for magical damage
        if damage_type == DamageType.MAGICAL and is_ability:
            ap_multiplier = source.stats.ability_power / 100
            raw_damage *= ap_multiplier

        # Calculate mitigation
        if damage_type == DamageType.PHYSICAL:
            armor = target.stats.armor
            reduction = armor / (armor + 100) if armor >= 0 else 0
            final_damage = raw_damage * (1 - reduction)
        elif damage_type == DamageType.MAGICAL:
            mr = target.stats.magic_resist
            reduction = mr / (mr + 100) if mr >= 0 else 0
            final_damage = raw_damage * (1 - reduction)
        else:  # TRUE damage
            final_damage = raw_damage

        # Note: damage_reduction is applied in CombatUnit.take_damage()
        # to avoid double-application

        return DamageEvent(
            source_id=source.id,
            target_id=target.id,
            raw_damage=raw_damage,
            final_damage=max(0, final_damage),
            damage_type=damage_type,
            is_critical=is_crit,
            is_ability=is_ability,
            source_name=source.name,
        )

    def apply_damage(
        self,
        target: "CombatUnit",
        damage_event: DamageEvent,
        grant_mana: bool = True,
    ) -> float:
        """
        Apply a damage event to a target.

        Args:
            target: The target unit.
            damage_event: The damage to apply.
            grant_mana: Whether to grant mana on damage.

        Returns:
            Actual damage dealt.
        """
        actual_damage = target.take_damage(
            damage_event.final_damage, damage_event.damage_type.value
        )

        if grant_mana and actual_damage > 0 and target.is_alive:
            target.gain_mana(MANA_ON_DAMAGE_TAKEN)

        return actual_damage

    def register_on_hit_effect(
        self, callback: Callable[["CombatUnit", "CombatUnit", DamageEvent], None]
    ) -> None:
        """Register a callback for on-hit effects."""
        self._on_hit_effects.append(callback)

    def register_on_attack_effect(
        self, callback: Callable[["CombatUnit", "CombatUnit"], None]
    ) -> None:
        """Register a callback for on-attack effects (before damage)."""
        self._on_attack_effects.append(callback)

    def clear_effects(self) -> None:
        """Clear all registered effects."""
        self._on_hit_effects.clear()
        self._on_attack_effects.clear()


def calculate_effective_hp(
    hp: float, armor: float, mr: float, physical_ratio: float = 0.7
) -> float:
    """
    Calculate effective HP considering resistances.

    Args:
        hp: Current HP.
        armor: Armor value.
        mr: Magic resist value.
        physical_ratio: Ratio of physical damage expected (0.0-1.0).

    Returns:
        Effective HP value.
    """
    physical_ehp = hp * (1 + armor / 100)
    magical_ehp = hp * (1 + mr / 100)
    return physical_ratio * physical_ehp + (1 - physical_ratio) * magical_ehp


def calculate_dps(
    attack_damage: float,
    attack_speed: float,
    crit_chance: float,
    crit_damage: float,
) -> float:
    """
    Calculate expected DPS.

    Args:
        attack_damage: Base attack damage.
        attack_speed: Attacks per second.
        crit_chance: Critical strike chance (0.0-1.0).
        crit_damage: Critical damage multiplier.

    Returns:
        Expected damage per second.
    """
    average_damage = attack_damage * (1 + crit_chance * (crit_damage - 1))
    return average_damage * attack_speed
