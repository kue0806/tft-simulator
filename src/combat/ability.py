"""Ability System for TFT Combat.

Handles ability casting, targeting, and effects.
Abilities are defined per champion and executed when mana is full.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from enum import Enum, auto
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .hex_grid import HexPosition, HexGrid
    from .targeting import TargetSelector
    from .attack import DamageEvent, DamageType
    from .status_effects import StatusEffectSystem, StatusEffect


class AbilityTargetType(Enum):
    """Types of ability targeting."""

    SELF = auto()  # Targets self
    SINGLE_ENEMY = auto()  # Single enemy target
    SINGLE_ALLY = auto()  # Single ally target
    ALL_ENEMIES = auto()  # All enemies
    ALL_ALLIES = auto()  # All allies
    AOE_ENEMY = auto()  # Area around enemy
    AOE_POSITION = auto()  # Area around position
    AOE_SELF = auto()  # Area around self
    LINE = auto()  # Line skillshot
    CONE = auto()  # Cone in front
    NEAREST_ENEMY = auto()  # Auto-target nearest enemy
    FARTHEST_ENEMY = auto()  # Auto-target farthest enemy
    LOWEST_HP_ALLY = auto()  # Auto-target lowest HP ally
    LOWEST_HP_ENEMY = auto()  # Auto-target lowest HP enemy
    RANDOM_ENEMY = auto()  # Random enemy
    MULTI_ENEMY = auto()  # Multiple random enemies (e.g., Bard spirits)
    SELF_BUFF = auto()  # Self-targeting buff ability
    SUMMON = auto()  # Summon minions/units


@dataclass
class AbilityData:
    """
    Static ability data definition.

    Defines how an ability works, loaded from champion data.
    """

    ability_id: str
    name: str
    description: str

    # Targeting
    target_type: AbilityTargetType = AbilityTargetType.SINGLE_ENEMY
    range: int = 99  # Max range (99 = unlimited)
    aoe_radius: int = 0  # For AOE abilities

    # Timing
    cast_time: float = 0.0  # Cast animation time
    channel_time: float = 0.0  # Channel duration (if any)

    # Damage
    base_damage: List[float] = field(default_factory=lambda: [0, 0, 0])  # Per star level
    damage_type: str = "magical"  # physical, magical, true
    damage_scaling: float = 1.0  # AP scaling multiplier
    can_crit: bool = False

    # Effects
    applies_cc: bool = False
    cc_type: str = ""  # stun, silence, knockup, etc.
    cc_duration: List[float] = field(default_factory=lambda: [0, 0, 0])

    # Healing
    base_healing: List[float] = field(default_factory=lambda: [0, 0, 0])
    healing_scaling: float = 0.0

    # Shield
    base_shield: List[float] = field(default_factory=lambda: [0, 0, 0])
    shield_scaling: float = 0.0
    shield_duration: float = 0.0

    # Special flags
    is_auto_cast: bool = True  # Cast automatically when mana full
    interrupts_movement: bool = True
    grants_invulnerability: bool = False

    # Custom data for complex abilities
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AbilityCast:
    """Represents an ability being cast."""

    caster_id: str
    ability: AbilityData
    target_id: Optional[str] = None
    target_position: Optional["HexPosition"] = None
    cast_progress: float = 0.0  # 0.0 to 1.0
    is_complete: bool = False


@dataclass
class AbilityResult:
    """Result of an ability cast."""

    success: bool
    ability_name: str = ""
    caster_id: str = ""
    targets_hit: List[str] = field(default_factory=list)
    total_damage: float = 0.0
    total_healing: float = 0.0
    kills: int = 0
    cc_applied: List[str] = field(default_factory=list)


class AbilitySystem:
    """
    Manages ability casting and execution.

    Handles the full ability lifecycle from cast to effect application.

    Usage:
        ability_system = AbilitySystem(grid, target_selector, attack_system)
        if ability_system.can_cast(unit):
            result = ability_system.start_cast(unit, ability_data)
    """

    def __init__(
        self,
        grid: "HexGrid",
        target_selector: "TargetSelector",
        damage_calculator: Any = None,
        status_effect_system: Optional["StatusEffectSystem"] = None,
    ):
        """
        Initialize ability system.

        Args:
            grid: The hex grid.
            target_selector: Target selection system.
            damage_calculator: Attack system for damage calculation.
            status_effect_system: Status effect system for applying CC.
        """
        self.grid = grid
        self.target_selector = target_selector
        self.damage_calculator = damage_calculator
        self.status_effect_system = status_effect_system

        # Active casts
        self._active_casts: Dict[str, AbilityCast] = {}

        # Ability definitions per champion
        self._ability_registry: Dict[str, AbilityData] = {}

        # Effect callbacks
        self._custom_ability_handlers: Dict[str, Callable] = {}

    def register_ability(self, champion_id: str, ability: AbilityData) -> None:
        """Register an ability for a champion."""
        self._ability_registry[champion_id] = ability

    def register_custom_handler(
        self,
        ability_id: str,
        handler: Callable[["CombatUnit", AbilityData, Dict[str, "CombatUnit"]], AbilityResult],
    ) -> None:
        """Register a custom handler for complex abilities."""
        self._custom_ability_handlers[ability_id] = handler

    def get_ability(self, champion_id: str) -> Optional[AbilityData]:
        """Get ability data for a champion."""
        return self._ability_registry.get(champion_id)

    def can_cast(self, unit: "CombatUnit") -> bool:
        """
        Check if unit can cast their ability.

        Args:
            unit: The unit to check.

        Returns:
            True if ability can be cast.
        """
        return (
            unit.is_alive
            and unit.can_act
            and unit.can_cast
            and not unit.is_casting
            and unit.id not in self._active_casts
        )

    def start_cast(
        self,
        unit: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
        target_id: Optional[str] = None,
    ) -> Optional[AbilityCast]:
        """
        Start casting an ability.

        Args:
            unit: The casting unit.
            ability: Ability data.
            all_units: All units in combat.
            target_id: Optional specific target.

        Returns:
            AbilityCast if started, None if failed.
        """
        if not self.can_cast(unit):
            return None

        # Find target based on ability type
        resolved_target = self._resolve_target(unit, ability, all_units, target_id)

        # Some abilities need a valid target
        if ability.target_type in [
            AbilityTargetType.SINGLE_ENEMY,
            AbilityTargetType.SINGLE_ALLY,
        ]:
            if resolved_target is None:
                return None

        # Create cast
        cast = AbilityCast(
            caster_id=unit.id,
            ability=ability,
            target_id=resolved_target,
            cast_progress=0.0,
        )

        # Spend mana
        unit.spend_mana()
        unit.is_casting = True
        unit.cast_time_remaining = ability.cast_time

        # Track active cast
        self._active_casts[unit.id] = cast

        return cast

    def update_cast(
        self,
        unit: "CombatUnit",
        delta_time: float,
        all_units: Dict[str, "CombatUnit"],
    ) -> Optional[AbilityResult]:
        """
        Update an ongoing cast.

        Args:
            unit: The casting unit.
            delta_time: Time elapsed.
            all_units: All units in combat.

        Returns:
            AbilityResult if cast completed, None if still casting.
        """
        cast = self._active_casts.get(unit.id)
        if cast is None:
            return None

        # Update cast time
        if cast.ability.cast_time > 0:
            unit.cast_time_remaining -= delta_time
            cast.cast_progress = 1.0 - (unit.cast_time_remaining / cast.ability.cast_time)

            if unit.cast_time_remaining > 0:
                return None

        # Cast complete - execute ability
        result = self._execute_ability(unit, cast, all_units)

        # Cleanup
        unit.is_casting = False
        unit.cast_time_remaining = 0.0
        del self._active_casts[unit.id]

        return result

    def interrupt_cast(self, unit: "CombatUnit") -> bool:
        """
        Interrupt an ongoing cast.

        Args:
            unit: The casting unit.

        Returns:
            True if cast was interrupted.
        """
        if unit.id not in self._active_casts:
            return False

        del self._active_casts[unit.id]
        unit.is_casting = False
        unit.cast_time_remaining = 0.0
        return True

    def _resolve_target(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
        explicit_target: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the target for an ability."""
        if explicit_target is not None:
            return explicit_target

        target_type = ability.target_type

        if target_type == AbilityTargetType.SELF:
            return caster.id

        elif target_type == AbilityTargetType.SINGLE_ENEMY:
            # Use current attack target or find new one
            if caster.current_target_id:
                target = all_units.get(caster.current_target_id)
                if target and target.is_targetable:
                    return caster.current_target_id
            return self.target_selector.find_target(caster)

        elif target_type == AbilityTargetType.NEAREST_ENEMY:
            return self.target_selector.find_target(caster)

        elif target_type == AbilityTargetType.FARTHEST_ENEMY:
            from .targeting import TargetingPriority
            return self.target_selector.find_target(
                caster, priority=TargetingPriority.FARTHEST
            )

        elif target_type == AbilityTargetType.LOWEST_HP_ALLY:
            # Find lowest HP ally
            allies = [u for u in all_units.values() if u.team == caster.team and u.is_alive]
            if allies:
                allies.sort(key=lambda u: u.stats.current_hp)
                return allies[0].id
            return None

        elif target_type == AbilityTargetType.LOWEST_HP_ENEMY:
            # Find lowest HP enemy
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                enemies.sort(key=lambda u: u.stats.current_hp)
                return enemies[0].id
            return None

        elif target_type == AbilityTargetType.RANDOM_ENEMY:
            from .targeting import TargetingPriority
            return self.target_selector.find_target(
                caster, priority=TargetingPriority.RANDOM
            )

        elif target_type in [
            AbilityTargetType.ALL_ENEMIES,
            AbilityTargetType.ALL_ALLIES,
            AbilityTargetType.AOE_SELF,
        ]:
            return None  # No specific target needed

        elif target_type in [AbilityTargetType.LINE, AbilityTargetType.CONE]:
            # LINE and CONE need a direction target (use current target or farthest enemy)
            if caster.current_target_id:
                target = all_units.get(caster.current_target_id)
                if target and target.is_targetable:
                    return caster.current_target_id
            # Fallback to farthest enemy for direction
            from .targeting import TargetingPriority
            return self.target_selector.find_target(
                caster, priority=TargetingPriority.FARTHEST
            )

        return None

    def _execute_ability(
        self,
        caster: "CombatUnit",
        cast: AbilityCast,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Execute the ability effect."""
        ability = cast.ability

        # Check for custom handler
        if ability.ability_id in self._custom_ability_handlers:
            handler = self._custom_ability_handlers[ability.ability_id]
            return handler(caster, ability, all_units)

        # Default ability execution
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        # Get targets
        targets = self._get_ability_targets(caster, cast, all_units)
        result.targets_hit = [t.id for t in targets]

        star_idx = min(caster.star_level - 1, 2)

        # Apply damage
        if ability.base_damage[star_idx] > 0:
            for target in targets:
                damage = self._calculate_ability_damage(caster, target, ability, star_idx)
                actual = target.take_damage(damage, ability.damage_type)
                result.total_damage += actual
                caster.total_damage_dealt += actual

                if not target.is_alive:
                    result.kills += 1
                    caster.kills += 1

                # Grant mana on damage
                if actual > 0 and target.is_alive:
                    target.gain_mana(7)  # MANA_ON_DAMAGE_TAKEN

        # Apply healing
        if ability.base_healing[star_idx] > 0:
            for target in targets:
                healing = ability.base_healing[star_idx]
                if ability.healing_scaling > 0:
                    healing *= (1 + caster.stats.ability_power / 100 * ability.healing_scaling)
                healed = target.heal(healing)
                result.total_healing += healed

        # Apply CC via status effect system
        if ability.applies_cc and ability.cc_duration[star_idx] > 0:
            cc_duration = ability.cc_duration[star_idx]
            for target in targets:
                if target.is_alive and self.status_effect_system:
                    cc_effect = self._create_cc_effect(
                        ability.cc_type, caster.id, cc_duration
                    )
                    if cc_effect:
                        self.status_effect_system.apply_effect(target, cc_effect)
                        result.cc_applied.append(target.id)

        return result

    def _create_cc_effect(
        self,
        cc_type: str,
        source_id: str,
        duration: float,
    ) -> Optional["StatusEffect"]:
        """Create a CC status effect based on type string."""
        from .status_effects import StatusEffect, StatusEffectType

        cc_mapping = {
            "stun": StatusEffectType.STUN,
            "silence": StatusEffectType.SILENCE,
            "disarm": StatusEffectType.DISARM,
            "root": StatusEffectType.ROOT,
            "knockup": StatusEffectType.KNOCKUP,
            "taunt": StatusEffectType.TAUNT,
            "blind": StatusEffectType.BLIND,
            "charm": StatusEffectType.CHARM,
        }

        effect_type = cc_mapping.get(cc_type.lower())
        if effect_type is None:
            return None

        return StatusEffect(
            effect_type=effect_type,
            source_id=source_id,
            duration=duration,
        )

    def _get_ability_targets(
        self,
        caster: "CombatUnit",
        cast: AbilityCast,
        all_units: Dict[str, "CombatUnit"],
    ) -> List["CombatUnit"]:
        """Get all targets for an ability."""
        ability = cast.ability
        targets = []

        if ability.target_type == AbilityTargetType.SELF:
            return [caster]

        elif ability.target_type == AbilityTargetType.SINGLE_ENEMY:
            if cast.target_id:
                target = all_units.get(cast.target_id)
                if target and target.is_targetable:
                    return [target]
            return []

        elif ability.target_type == AbilityTargetType.ALL_ENEMIES:
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    targets.append(unit)

        elif ability.target_type == AbilityTargetType.ALL_ALLIES:
            for unit in all_units.values():
                if unit.team == caster.team and unit.is_alive:
                    targets.append(unit)

        elif ability.target_type == AbilityTargetType.AOE_ENEMY:
            if cast.target_id:
                target_pos = self.grid.get_unit_position(cast.target_id)
                if target_pos:
                    for unit in all_units.values():
                        if unit.team != caster.team and unit.is_targetable:
                            unit_pos = self.grid.get_unit_position(unit.id)
                            if unit_pos and target_pos.distance_to(unit_pos) <= ability.aoe_radius:
                                targets.append(unit)

        elif ability.target_type == AbilityTargetType.AOE_SELF:
            caster_pos = self.grid.get_unit_position(caster.id)
            if caster_pos:
                for unit in all_units.values():
                    if unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.grid.get_unit_position(unit.id)
                        if unit_pos and caster_pos.distance_to(unit_pos) <= ability.aoe_radius:
                            targets.append(unit)

        elif ability.target_type in [
            AbilityTargetType.NEAREST_ENEMY,
            AbilityTargetType.FARTHEST_ENEMY,
            AbilityTargetType.LOWEST_HP_ENEMY,
            AbilityTargetType.RANDOM_ENEMY,
        ]:
            if cast.target_id:
                target = all_units.get(cast.target_id)
                if target and target.is_targetable:
                    return [target]

        elif ability.target_type == AbilityTargetType.LINE:
            # Line skillshot - hit enemies between caster and target
            caster_pos = self.grid.get_unit_position(caster.id)
            if cast.target_id:
                target_pos = self.grid.get_unit_position(cast.target_id)
                if caster_pos and target_pos:
                    # Get all enemies in line from caster to target (and beyond)
                    for unit in all_units.values():
                        if unit.team != caster.team and unit.is_targetable:
                            unit_pos = self.grid.get_unit_position(unit.id)
                            if unit_pos and self._is_in_line(caster_pos, target_pos, unit_pos):
                                targets.append(unit)

        elif ability.target_type == AbilityTargetType.CONE:
            # Cone in front of caster toward target
            caster_pos = self.grid.get_unit_position(caster.id)
            if cast.target_id:
                target_pos = self.grid.get_unit_position(cast.target_id)
            else:
                # Use current target or nearest enemy
                target_pos = None
                if caster.current_target_id:
                    target_pos = self.grid.get_unit_position(caster.current_target_id)

            if caster_pos and target_pos:
                cone_range = ability.aoe_radius if ability.aoe_radius > 0 else 3
                for unit in all_units.values():
                    if unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.grid.get_unit_position(unit.id)
                        if unit_pos:
                            dist = caster_pos.distance_to(unit_pos)
                            if dist <= cone_range and self._is_in_cone(caster_pos, target_pos, unit_pos):
                                targets.append(unit)

        return targets

    def _is_in_line(
        self,
        start: "HexPosition",
        end: "HexPosition",
        point: "HexPosition",
        tolerance: float = 1.5,
    ) -> bool:
        """Check if a point is roughly on the line from start through end."""
        # Calculate direction vector
        dx = end.col - start.col
        dy = end.row - start.row

        # Point relative to start
        px = point.col - start.col
        py = point.row - start.row

        # Project point onto line direction
        line_len_sq = dx * dx + dy * dy
        if line_len_sq == 0:
            return start.distance_to(point) <= tolerance

        # Calculate perpendicular distance to line
        cross = abs(dx * py - dy * px)
        perp_dist = cross / (line_len_sq ** 0.5)

        return perp_dist <= tolerance

    def _is_in_cone(
        self,
        origin: "HexPosition",
        direction: "HexPosition",
        point: "HexPosition",
        cone_angle: float = 60.0,  # degrees from center
    ) -> bool:
        """Check if a point is within a cone from origin toward direction."""
        import math

        # Direction vector
        dx = direction.col - origin.col
        dy = direction.row - origin.row

        # Point vector
        px = point.col - origin.col
        py = point.row - origin.row

        # Angle between direction and point
        dir_angle = math.atan2(dy, dx)
        point_angle = math.atan2(py, px)

        angle_diff = abs(dir_angle - point_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        return math.degrees(angle_diff) <= cone_angle

    def _calculate_ability_damage(
        self,
        caster: "CombatUnit",
        target: "CombatUnit",
        ability: AbilityData,
        star_idx: int,
    ) -> float:
        """Calculate ability damage."""
        base = ability.base_damage[star_idx]

        # AP scaling
        ap_mult = 1.0 + (caster.stats.ability_power / 100) * ability.damage_scaling

        raw_damage = base * ap_mult * caster.stats.damage_amp

        # Mitigation
        if ability.damage_type == "physical":
            reduction = target.stats.armor / (target.stats.armor + 100)
        elif ability.damage_type == "magical":
            reduction = target.stats.magic_resist / (target.stats.magic_resist + 100)
        else:  # true
            reduction = 0

        final_damage = raw_damage * (1 - reduction) * (1 - target.stats.damage_reduction)

        return max(0, final_damage)

    def is_casting(self, unit_id: str) -> bool:
        """Check if unit is currently casting."""
        return unit_id in self._active_casts

    def get_cast_progress(self, unit_id: str) -> float:
        """Get casting progress (0.0 to 1.0)."""
        cast = self._active_casts.get(unit_id)
        return cast.cast_progress if cast else 0.0

    def clear(self) -> None:
        """Clear all active casts."""
        self._active_casts.clear()


def create_simple_damage_ability(
    ability_id: str,
    name: str,
    base_damage: List[float],
    damage_type: str = "magical",
    target_type: AbilityTargetType = AbilityTargetType.SINGLE_ENEMY,
    aoe_radius: int = 0,
    cast_time: float = 0.5,
) -> AbilityData:
    """Helper to create a simple damage ability."""
    return AbilityData(
        ability_id=ability_id,
        name=name,
        description=f"{name} deals damage",
        target_type=target_type,
        aoe_radius=aoe_radius,
        cast_time=cast_time,
        base_damage=base_damage,
        damage_type=damage_type,
        damage_scaling=1.0,
    )
