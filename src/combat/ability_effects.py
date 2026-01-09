"""Custom Ability Effect Handlers for TFT Combat.

Implements special ability effects that go beyond basic damage/CC:
- Self-buffs (Briar Blood Frenzy, etc.)
- Execute/Missing HP bonus damage
- DoT effects
- Multi-hit abilities
- Summons
- Special mechanics
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable
from dataclasses import dataclass

from .ability import AbilityData, AbilityResult, AbilityTargetType
from .status_effects import StatusEffect, StatusEffectType

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .ability import AbilitySystem


@dataclass
class OnHitEffect:
    """Temporary on-hit effect from abilities."""
    source_ability: str
    duration: float
    heal_percent: float = 0.0  # % of damage dealt
    bonus_damage: float = 0.0
    bonus_damage_type: str = "physical"


class AbilityEffectHandlers:
    """
    Registry of custom ability effect handlers.

    Each handler takes (caster, ability, all_units, ability_system) and returns AbilityResult.
    """

    def __init__(self, ability_system: "AbilitySystem"):
        self.ability_system = ability_system

        # Track active on-hit effects per unit
        self._on_hit_effects: Dict[str, List[OnHitEffect]] = {}

        # Track summons per owner
        self._summons: Dict[str, List[str]] = {}

        # Track DoT effects applied by abilities
        self._ability_dots: Dict[str, Dict[str, Any]] = {}

    def register_all_handlers(self) -> None:
        """Register all custom ability handlers."""
        handlers = {
            # 1-cost self-buffs
            "briar_blood_frenzy": self.handle_briar_blood_frenzy,

            # 1-cost special
            "jhin_curtain_call": self.handle_jhin_curtain_call,
            "viego_blade_of_the_ruined_king": self.handle_viego_lifesteal,

            # 2-cost
            "reksai_furious_bite": self.handle_reksai_execute,
            "teemo_noxious_trap": self.handle_teemo_dot,
            "shen_spirit_blade": self.handle_shen_damage_reduction,

            # 3-cost (add these ability IDs when needed)
            "draven_spinning_axe": self.handle_draven_axes,
            "vayne_silver_bolts": self.handle_vayne_silver_bolts,
            "dr_mundo_sadism": self.handle_mundo_regen,

            # 4-cost
            "darius_noxian_guillotine": self.handle_darius_execute,
            "jinx_super_mega_death_rocket": self.handle_jinx_execute,
            "malzahar_malefic_visions": self.handle_malzahar_dot,
            "xerath_rite_of_the_arcane": self.handle_xerath_barrage,

            # 5-cost
            "garen_demacian_justice": self.handle_garen_execute,
            "veigar_primordial_burst": self.handle_veigar_execute,

            # DoT abilities
            "rumble_flamespitter": self.handle_rumble_flamespitter,

            # Summon abilities
            "annie_summon_tibbers": self.handle_annie_tibbers,
            "yorick_dark_procession": self.handle_yorick_ghouls,
            "azir_emperors_divide": self.handle_azir_soldiers,

            # ===== Set 16 Self-Buff Handlers =====
            "swain_demonic_ascension": self.handle_swain_demonic_ascension,
            "nasus_fury_of_sands": self.handle_nasus_fury_of_sands,
            "renekton_dominus": self.handle_renekton_dominus,
            "shyvana_dragons_descent": self.handle_shyvana_dragons_descent,
            "belveth_endless_banquet": self.handle_belveth_endless_banquet,
            "singed_insanity_potion": self.handle_singed_insanity_potion,
            "xin_zhao_crescent_guard": self.handle_xin_zhao_crescent_guard,

            # ===== Set 16 HP% Damage Handlers =====
            "sett_showstopper": self.handle_sett_showstopper,
            "tahm_kench_devour": self.handle_tahm_kench_devour,
            "nidalee_pounce": self.handle_nidalee_pounce,

            # ===== Set 16 Multi-Hit Handlers =====
            "miss_fortune_bullet_time": self.handle_miss_fortune_bullet_time,
            "kaisa_icathian_rain": self.handle_kaisa_icathian_rain,
            "kindred_lambs_respite": self.handle_kindred_lambs_respite,
            "ambessa_dashing_slash": self.handle_ambessa_dashing_slash,
            "kogmaw_void_artillery": self.handle_kogmaw_void_artillery,

            # ===== Set 16 Ally Buff Handlers =====
            "bard_tempered_fate": self.handle_bard_tempered_fate,
            "taric_cosmic_radiance": self.handle_taric_cosmic_radiance,

            # ===== Set 16 Special Mechanic Handlers =====
            "zilean_time_bomb": self.handle_zilean_time_bomb,
            "warwick_primal_howl": self.handle_warwick_primal_howl,
            "malzahar_void_gate": self.handle_malzahar_void_gate,
            "kalista_rend": self.handle_kalista_rend,
            "gwen_snip_snip": self.handle_gwen_snip_snip,
            "sylas_hijack": self.handle_sylas_hijack,
            "fiddlesticks_crowstorm": self.handle_fiddlesticks_crowstorm,
            "seraphine_note_progression": self.handle_seraphine_note_progression,
            "lucian_senna_culling": self.handle_lucian_senna_culling,

            # ===== 5-Cost & 7-Cost Legendary Handlers =====
            "aatrox_darkin_blade": self.handle_aatrox_darkin_blade,
            "aurelion_sol_skies_descend": self.handle_aurelion_sol_skies_descend,
            "baron_nashor_void_eruption": self.handle_baron_nashor_void_eruption,
            "ryze_realm_warp": self.handle_ryze_realm_warp,
            "zaahen_divine_challenge": self.handle_zaahen_divine_challenge,
            "brock_seismic_slam": self.handle_brock_seismic_slam,

            # ===== Missing Handlers =====
            "dr_mundo_goes_where_he_pleases": self.handle_dr_mundo_maximum_dosage,
            "mel_golden_barrier": self.handle_mel_councils_blessing,
            "milio_cozy_campfire": self.handle_milio_breath_of_life,
        }

        for ability_id, handler in handlers.items():
            self.ability_system.register_custom_handler(ability_id, handler)

    # =========================================================================
    # SELF-BUFF HANDLERS
    # =========================================================================

    def handle_briar_blood_frenzy(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Briar: Gain attack speed and heal on hit."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Attack speed buff
        as_bonus = custom.get("attack_speed_bonus", [0.4, 0.5, 0.7])[star_idx]
        duration = custom.get("duration", 4.0)

        effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=duration,
            value=as_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # Add on-hit heal effect
        heal_percent = custom.get("heal_per_attack_percent", [0.15, 0.2, 0.3])[star_idx]
        on_hit = OnHitEffect(
            source_ability=ability.ability_id,
            duration=duration,
            heal_percent=heal_percent,
        )

        if caster.id not in self._on_hit_effects:
            self._on_hit_effects[caster.id] = []
        self._on_hit_effects[caster.id].append(on_hit)

        return result

    def handle_mundo_regen(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Dr. Mundo: Regenerate health over time and gain AD."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Get healing values
        heal_total = custom.get("heal_percent_max_hp", [0.3, 0.4, 0.6])[star_idx]
        duration = custom.get("duration", 5.0)

        # Apply heal over time
        heal_per_second = (caster.stats.max_hp * heal_total) / duration

        effect = StatusEffect(
            effect_type=StatusEffectType.HEAL_OVER_TIME,
            source_id=caster.id,
            duration=duration,
            value=heal_per_second,
            tick_interval=0.5,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # AD buff
        ad_bonus = custom.get("ad_bonus", [30, 45, 70])[star_idx]
        ad_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_DAMAGE_BUFF,
            source_id=caster.id,
            duration=duration,
            value=ad_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, ad_effect)

        return result

    def handle_vayne_silver_bolts(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Vayne: Gain attack speed and deal bonus true damage on 3rd hit."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Attack speed buff
        as_bonus = custom.get("attack_speed_bonus", [0.5, 0.6, 0.8])[star_idx]
        duration = custom.get("duration", 6.0)

        effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=duration,
            value=as_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        return result

    def handle_shen_damage_reduction(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Shen: Create zone that reduces attack damage, then deal damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Damage reduction for self
        reduction = custom.get("damage_reduction", [0.25, 0.3, 0.4])[star_idx]
        duration = custom.get("duration", 3.0)

        effect = StatusEffect(
            effect_type=StatusEffectType.DAMAGE_REDUCTION,
            source_id=caster.id,
            duration=duration,
            value=reduction,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # Deal AOE damage
        base_damage = ability.base_damage[star_idx]
        if base_damage > 0:
            caster_pos = self.ability_system.grid.get_unit_position(caster.id)
            if caster_pos:
                for unit in all_units.values():
                    if unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                        if unit_pos and caster_pos.distance_to(unit_pos) <= ability.aoe_radius:
                            damage = self._calculate_damage(caster, unit, base_damage, ability)
                            actual = unit.take_damage(damage, ability.damage_type)
                            result.total_damage += actual
                            result.targets_hit.append(unit.id)
                            caster.total_damage_dealt += actual

                            if not unit.is_alive:
                                result.kills += 1
                                caster.kills += 1

        return result

    # =========================================================================
    # EXECUTE / MISSING HP HANDLERS
    # =========================================================================

    def handle_reksai_execute(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Rek'Sai: Deal bonus damage based on target's missing HP."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        # Get target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            # Find lowest HP enemy
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = min(enemies, key=lambda u: u.stats.current_hp)

        if not target:
            result.success = False
            return result

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        # Calculate missing HP bonus
        missing_hp_bonus = ability.custom_data.get("missing_hp_bonus", 0.5)
        missing_hp_percent = 1 - (target.stats.current_hp / target.stats.max_hp)

        # Bonus damage: +X% per 1% missing HP (capped at 50%)
        bonus_mult = 1 + min(missing_hp_percent * missing_hp_bonus * 100, 50) / 100

        damage = base_damage * bonus_mult
        actual = target.take_damage(damage, ability.damage_type)

        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        return result

    def handle_darius_execute(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Darius: Execute low HP target, reset on kill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        # Find lowest HP enemy
        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        target = min(enemies, key=lambda u: u.stats.current_hp / u.stats.max_hp)

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        # Execute threshold
        execute_threshold = ability.custom_data.get("execute_threshold", [0.2, 0.25, 0.4])[star_idx]
        hp_percent = target.stats.current_hp / target.stats.max_hp

        damage = base_damage
        if hp_percent <= execute_threshold:
            # Execute: deal massive bonus damage
            damage *= 3

        actual = target.take_damage(damage, ability.damage_type)
        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1
            # Reset: refund mana
            caster.stats.current_mana = caster.stats.max_mana * 0.8

        return result

    def handle_jinx_execute(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Jinx: Fire rocket dealing damage, executes low HP targets."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        # Find farthest enemy
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        # Farthest enemy
        farthest = None
        max_dist = -1
        for enemy in enemies:
            enemy_pos = self.ability_system.grid.get_unit_position(enemy.id)
            if enemy_pos:
                dist = caster_pos.distance_to(enemy_pos)
                if dist > max_dist:
                    max_dist = dist
                    farthest = enemy

        if not farthest:
            result.success = False
            return result

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        # Missing HP bonus
        missing_hp_mult = ability.custom_data.get("missing_hp_mult", [0.15, 0.2, 0.3])[star_idx]
        missing_hp_percent = 1 - (farthest.stats.current_hp / farthest.stats.max_hp)

        damage = base_damage * (1 + missing_hp_percent * missing_hp_mult * 10)
        actual = farthest.take_damage(damage, ability.damage_type)

        result.total_damage = actual
        result.targets_hit = [farthest.id]
        caster.total_damage_dealt += actual

        if not farthest.is_alive:
            result.kills += 1
            caster.kills += 1
            # Get Excited: gain attack speed
            excited_as = ability.custom_data.get("excited_as", 0.5)
            effect = StatusEffect(
                effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
                source_id=caster.id,
                duration=999,  # Permanent
                value=excited_as,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(caster, effect)

        return result

    def handle_garen_execute(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Garen: Deal damage based on target's missing HP."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = min(enemies, key=lambda u: u.stats.current_hp)

        if not target:
            result.success = False
            return result

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        # Missing HP bonus
        missing_hp_mult = ability.custom_data.get("missing_hp_mult", [0.01, 0.015, 0.025])[star_idx]
        missing_hp = target.stats.max_hp - target.stats.current_hp

        damage = base_damage + missing_hp * missing_hp_mult * 100
        actual = target.take_damage(damage, "true")  # True damage

        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        return result

    def handle_veigar_execute(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Veigar: Deal damage, executes champions below HP threshold."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        # Find lowest HP enemy
        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        target = min(enemies, key=lambda u: u.stats.current_hp)

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        # Execute threshold based on AP
        execute_threshold = ability.custom_data.get("execute_base", [0.1, 0.15, 0.25])[star_idx]
        ap_scaling = ability.custom_data.get("execute_ap_scaling", 0.001)
        execute_threshold += caster.stats.ability_power * ap_scaling

        hp_percent = target.stats.current_hp / target.stats.max_hp

        if hp_percent <= execute_threshold:
            # Execute: instant kill
            damage = target.stats.current_hp + 9999
        else:
            damage = base_damage * (1 + caster.stats.ability_power / 100)

        actual = target.take_damage(damage, ability.damage_type)
        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        return result

    # =========================================================================
    # DOT HANDLERS
    # =========================================================================

    def handle_teemo_dot(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Teemo: Place poison traps dealing DoT."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        total_damage = ability.base_damage[star_idx]
        dot_duration = ability.custom_data.get("dot_duration", 3.0)

        # Find target position for AOE
        target_id = caster.current_target_id
        if not target_id:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target_id = enemies[0].id

        if not target_id:
            result.success = False
            return result

        target_pos = self.ability_system.grid.get_unit_position(target_id)
        if not target_pos:
            result.success = False
            return result

        # Apply DoT to enemies in AOE
        damage_per_second = total_damage / dot_duration

        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and target_pos.distance_to(unit_pos) <= ability.aoe_radius:
                    effect = StatusEffect(
                        effect_type=StatusEffectType.POISON,
                        source_id=caster.id,
                        duration=dot_duration,
                        value=damage_per_second,
                        tick_interval=1.0,
                    )
                    if self.ability_system.status_effect_system:
                        self.ability_system.status_effect_system.apply_effect(unit, effect)
                    result.targets_hit.append(unit.id)

        return result

    def handle_malzahar_dot(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Malzahar: Apply Malefic Visions DoT that spreads on kill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)

        # Find multiple targets
        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        num_targets = ability.custom_data.get("num_targets", [2, 3, 4])[star_idx]
        targets = enemies[:min(num_targets, len(enemies))]

        total_damage = ability.base_damage[star_idx]
        duration = ability.custom_data.get("duration", 4.0)
        damage_per_second = total_damage / duration

        for target in targets:
            effect = StatusEffect(
                effect_type=StatusEffectType.BURN,
                source_id=caster.id,
                duration=duration,
                value=damage_per_second,
                tick_interval=0.5,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(target, effect)
            result.targets_hit.append(target.id)

        return result

    def handle_rumble_flamespitter(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Rumble: Torch enemies in a cone with DoT."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        total_damage = ability.base_damage[star_idx]

        duration = ability.custom_data.get("duration", 3.0)
        ticks = ability.custom_data.get("ticks", 6)

        # Get targets in cone
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        target_pos = None

        if caster.current_target_id:
            target_pos = self.ability_system.grid.get_unit_position(caster.current_target_id)

        if not caster_pos or not target_pos:
            # Fallback: find nearest enemy
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies and caster_pos:
                nearest = min(enemies, key=lambda u: caster_pos.distance_to(
                    self.ability_system.grid.get_unit_position(u.id) or caster_pos
                ))
                target_pos = self.ability_system.grid.get_unit_position(nearest.id)

        if not caster_pos or not target_pos:
            result.success = False
            return result

        # Apply burn DoT to cone targets
        damage_per_tick = total_damage / ticks
        tick_interval = duration / ticks

        cone_range = ability.aoe_radius if ability.aoe_radius > 0 else 3

        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    dist = caster_pos.distance_to(unit_pos)
                    if dist <= cone_range and self.ability_system._is_in_cone(caster_pos, target_pos, unit_pos):
                        effect = StatusEffect(
                            effect_type=StatusEffectType.BURN,
                            source_id=caster.id,
                            duration=duration,
                            value=damage_per_tick / tick_interval,
                            tick_interval=tick_interval,
                        )
                        if self.ability_system.status_effect_system:
                            self.ability_system.status_effect_system.apply_effect(unit, effect)
                        result.targets_hit.append(unit.id)

        return result

    # =========================================================================
    # MULTI-HIT HANDLERS
    # =========================================================================

    def handle_jhin_curtain_call(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Jhin: Fire 4 shots at enemies."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        damage_per_shot = ability.base_damage[star_idx]
        num_shots = ability.custom_data.get("num_shots", 4)

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        # Get enemies sorted by distance (farthest first)
        enemies = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    enemies.append((unit, caster_pos.distance_to(unit_pos)))

        enemies.sort(key=lambda x: x[1], reverse=True)

        if not enemies:
            result.success = False
            return result

        # Fire shots at different targets if possible
        for i in range(num_shots):
            target_idx = i % len(enemies)
            target = enemies[target_idx][0]

            if not target.is_alive:
                continue

            # Last shot always crits
            is_crit = (i == num_shots - 1) or (ability.can_crit and self._check_crit(caster))
            damage = damage_per_shot
            if is_crit:
                damage *= caster.stats.crit_damage

            actual = target.take_damage(damage, ability.damage_type)
            result.total_damage += actual
            caster.total_damage_dealt += actual

            if target.id not in result.targets_hit:
                result.targets_hit.append(target.id)

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1

        return result

    def handle_xerath_barrage(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Xerath: Fire multiple energy barrages at random enemies."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        damage_per_hit = ability.base_damage[star_idx]
        num_hits = ability.custom_data.get("num_hits", [3, 4, 6])[star_idx]

        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        import random

        for _ in range(num_hits):
            # Filter to alive enemies
            alive_enemies = [e for e in enemies if e.is_alive]
            if not alive_enemies:
                break

            target = random.choice(alive_enemies)
            damage = self._calculate_damage(caster, target, damage_per_hit, ability)
            actual = target.take_damage(damage, ability.damage_type)

            result.total_damage += actual
            caster.total_damage_dealt += actual

            if target.id not in result.targets_hit:
                result.targets_hit.append(target.id)

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1

        return result

    def handle_draven_axes(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Draven: Empower next attacks with spinning axes."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Bonus AD
        ad_bonus = custom.get("ad_bonus", [50, 75, 120])[star_idx]
        duration = custom.get("duration", 6.0)

        effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_DAMAGE_BUFF,
            source_id=caster.id,
            duration=duration,
            value=ad_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # Attack speed buff
        as_bonus = custom.get("as_bonus", [0.3, 0.4, 0.6])[star_idx]
        as_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=duration,
            value=as_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, as_effect)

        return result

    # =========================================================================
    # SPECIAL MECHANICS
    # =========================================================================

    def handle_viego_lifesteal(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Viego: Stab target and heal for % of damage dealt."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            result.success = False
            return result

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]

        damage = self._calculate_damage(caster, target, base_damage, ability)
        actual = target.take_damage(damage, ability.damage_type)

        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        # Heal for % of damage
        heal_percent = ability.custom_data.get("heal_percent", 0.25)
        heal_amount = actual * heal_percent
        healed = caster.heal(heal_amount)
        result.total_healing = healed

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_damage(
        self,
        caster: "CombatUnit",
        target: "CombatUnit",
        base_damage: float,
        ability: AbilityData,
    ) -> float:
        """Calculate ability damage with AP scaling."""
        ap_mult = 1.0 + (caster.stats.ability_power / 100) * ability.damage_scaling
        return base_damage * ap_mult * caster.stats.damage_amp

    def _check_crit(self, caster: "CombatUnit") -> bool:
        """Check if attack crits."""
        import random
        return random.random() < caster.stats.crit_chance

    def process_on_hit(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
        damage_dealt: float,
    ) -> float:
        """Process on-hit effects and return bonus effects applied."""
        if attacker.id not in self._on_hit_effects:
            return 0.0

        total_healing = 0.0
        expired = []

        for i, effect in enumerate(self._on_hit_effects[attacker.id]):
            # Heal on hit
            if effect.heal_percent > 0:
                heal = damage_dealt * effect.heal_percent
                healed = attacker.heal(heal)
                total_healing += healed

            # Bonus damage on hit
            if effect.bonus_damage > 0:
                target.take_damage(effect.bonus_damage, effect.bonus_damage_type)

            # Check expiration (tracked separately by status effects)

        return total_healing

    def update_effects(self, delta_time: float) -> None:
        """Update time-based on-hit effects."""
        for unit_id in list(self._on_hit_effects.keys()):
            effects = self._on_hit_effects[unit_id]
            remaining = []

            for effect in effects:
                effect.duration -= delta_time
                if effect.duration > 0:
                    remaining.append(effect)

            if remaining:
                self._on_hit_effects[unit_id] = remaining
            else:
                del self._on_hit_effects[unit_id]

    def clear(self) -> None:
        """Clear all tracked effects."""
        self._on_hit_effects.clear()
        self._summons.clear()
        self._ability_dots.clear()

    # =========================================================================
    # SUMMON HANDLERS
    # =========================================================================

    def handle_annie_tibbers(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Annie: Summon Tibbers to fight alongside her."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Tibbers stats based on star level
        tibbers_hp = custom.get("tibbers_hp", [1500, 2700, 4860])[star_idx]
        tibbers_ad = custom.get("tibbers_ad", [90, 162, 292])[star_idx]
        tibbers_armor = custom.get("tibbers_armor", 80)
        tibbers_mr = custom.get("tibbers_mr", 80)
        tibbers_as = custom.get("tibbers_as", 0.75)

        # Initial impact damage
        impact_damage = ability.base_damage[star_idx]

        # Find target position for summon
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        # Deal AOE damage on summon
        aoe_radius = ability.aoe_radius if ability.aoe_radius > 0 else 2

        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and caster_pos.distance_to(unit_pos) <= aoe_radius:
                    damage = self._calculate_damage(caster, unit, impact_damage, ability)
                    actual = unit.take_damage(damage, ability.damage_type)
                    result.total_damage += actual
                    result.targets_hit.append(unit.id)
                    caster.total_damage_dealt += actual

                    if not unit.is_alive:
                        result.kills += 1
                        caster.kills += 1

        # Create Tibbers as a combat unit
        # For now, we apply Tibbers' effects as buffs to nearby allies
        # Full summon implementation would require adding a new CombatUnit to the grid

        # Apply Tibbers' aura effect - AS buff to Annie
        tibbers_aura_as = custom.get("aura_as_bonus", [0.2, 0.3, 0.5])[star_idx]
        effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=999,  # Permanent while Tibbers is alive
            value=tibbers_aura_as,
            custom_data={"is_tibbers_buff": True, "tibbers_hp": tibbers_hp},
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # Track summon
        if caster.id not in self._summons:
            self._summons[caster.id] = []
        self._summons[caster.id].append("tibbers")

        return result

    def handle_yorick_ghouls(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Yorick: Throw dark mist, dealing damage and applying Chill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        base_damage = ability.base_damage[star_idx]
        custom = ability.custom_data

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        # Deal damage
        damage = self._calculate_damage(caster, target, base_damage, ability)
        actual = target.take_damage(damage, ability.damage_type)
        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        # Heal self
        heal_amount = custom.get("heal", [100, 150, 250])[star_idx]
        healed = caster.heal(heal_amount)
        result.total_healing = healed

        # Apply Chill (attack speed slow) to target
        chill_duration = custom.get("chill_duration", 3.0)
        chill_slow = custom.get("chill_slow", 0.3)

        if target.is_alive:
            effect = StatusEffect(
                effect_type=StatusEffectType.ATTACK_SPEED_DEBUFF,
                source_id=caster.id,
                duration=chill_duration,
                value=chill_slow,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(target, effect)

        return result

    def handle_azir_soldiers(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Azir: Summon soldiers that attack with him."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Soldier count and damage
        num_soldiers = custom.get("num_soldiers", [2, 3, 4])[star_idx]
        soldier_damage = ability.base_damage[star_idx]

        # Soldiers provide bonus on-hit damage
        on_hit = OnHitEffect(
            source_ability=ability.ability_id,
            duration=custom.get("duration", 6.0),
            bonus_damage=soldier_damage * num_soldiers / 4,  # Averaged per hit
            bonus_damage_type=ability.damage_type,
        )

        if caster.id not in self._on_hit_effects:
            self._on_hit_effects[caster.id] = []
        self._on_hit_effects[caster.id].append(on_hit)

        # Attack speed buff while soldiers are active
        as_bonus = custom.get("as_bonus", [0.3, 0.4, 0.6])[star_idx]
        effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=custom.get("duration", 6.0),
            value=as_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, effect)

        # Track summon
        if caster.id not in self._summons:
            self._summons[caster.id] = []
        self._summons[caster.id].extend([f"soldier_{i}" for i in range(num_soldiers)])

        return result

    # =========================================================================
    # SET 16 SELF-BUFF HANDLERS
    # =========================================================================

    def handle_swain_demonic_ascension(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Swain: Passive heal/damage aura + Active stun on largest group."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Passive: Heal over time (22/24/100 + 1.5% HP per second)
        base_heal = custom.get("passive_heal", [22, 24, 100])[star_idx]
        hp_percent_heal = custom.get("passive_heal_percent", 0.015)
        heal_per_sec = base_heal + caster.stats.max_hp * hp_percent_heal
        duration = custom.get("duration", 6.0)

        heal_effect = StatusEffect(
            effect_type=StatusEffectType.HEAL_OVER_TIME,
            source_id=caster.id,
            duration=duration,
            value=heal_per_sec,
            tick_interval=1.0,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, heal_effect)

        # Active: Deal damage and stun largest group
        active_damage = ability.base_damage[star_idx]
        stun_duration = ability.cc_duration[star_idx] if ability.cc_duration else 1.75

        # Find largest group of enemies (AOE centered on densest area)
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if caster_pos:
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and caster_pos.distance_to(unit_pos) <= ability.aoe_radius:
                        damage = self._calculate_damage(caster, unit, active_damage, ability)
                        actual = unit.take_damage(damage, ability.damage_type)
                        result.total_damage += actual
                        result.targets_hit.append(unit.id)
                        caster.total_damage_dealt += actual

                        # Apply stun
                        if unit.is_alive and self.ability_system.status_effect_system:
                            stun = StatusEffect(
                                effect_type=StatusEffectType.STUN,
                                source_id=caster.id,
                                duration=stun_duration,
                            )
                            self.ability_system.status_effect_system.apply_effect(unit, stun)
                            result.cc_applied.append(unit.id)

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        return result

    def handle_nasus_fury_of_sands(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Nasus: Steal HP from enemies, gain defenses, deal AOE damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Health steal from 4/4/8 enemies
        num_targets = custom.get("num_targets", [4, 4, 8])[star_idx]
        total_hp_steal = custom.get("hp_steal", [485, 620, 3000])[star_idx]
        hp_per_target = total_hp_steal / num_targets

        # Find nearest enemies
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    enemies.append((unit, caster_pos.distance_to(unit_pos)))

        enemies.sort(key=lambda x: x[1])
        targets = [e[0] for e in enemies[:num_targets]]

        # Steal HP from each target
        total_stolen = 0
        for target in targets:
            steal = min(hp_per_target, target.stats.current_hp - 1)
            if steal > 0:
                target.take_damage(steal, "true")
                total_stolen += steal
                result.targets_hit.append(target.id)

        # Heal Nasus
        caster.heal(total_stolen)
        result.total_healing = total_stolen

        # Gain armor/MR buff
        duration = custom.get("duration", 8.0)
        defense_bonus = custom.get("defense_bonus", 30)

        armor_effect = StatusEffect(
            effect_type=StatusEffectType.ARMOR_BUFF,
            source_id=caster.id,
            duration=duration,
            value=defense_bonus,
        )
        mr_effect = StatusEffect(
            effect_type=StatusEffectType.MAGIC_RESIST_BUFF,
            source_id=caster.id,
            duration=duration,
            value=defense_bonus,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, armor_effect)
            self.ability_system.status_effect_system.apply_effect(caster, mr_effect)

        return result

    def handle_renekton_dominus(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Renekton: Gain max HP, attacks deal % max HP bonus damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Bonus max HP (200/250/300% AP scaling)
        hp_percent = custom.get("hp_percent", [2.0, 2.5, 3.0])[star_idx]
        bonus_hp = caster.stats.max_hp * hp_percent * (1 + caster.stats.ability_power / 100)

        caster.stats.max_hp += bonus_hp
        caster.stats.current_hp += bonus_hp

        # Duration
        duration = custom.get("duration", 8.0)

        # On-hit: 6% max HP bonus magic damage
        on_hit_percent = custom.get("on_hit_percent", 0.06)
        on_hit = OnHitEffect(
            source_ability=ability.ability_id,
            duration=duration,
            bonus_damage=caster.stats.max_hp * on_hit_percent,
            bonus_damage_type="magical",
        )

        if caster.id not in self._on_hit_effects:
            self._on_hit_effects[caster.id] = []
        self._on_hit_effects[caster.id].append(on_hit)

        return result

    def handle_shyvana_dragons_descent(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Shyvana: Transform, gain HP/durability, dive bomb damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Gain max HP
        bonus_hp = custom.get("bonus_hp", [600, 1000, 9999])[star_idx]
        caster.stats.max_hp += bonus_hp
        caster.stats.current_hp += bonus_hp

        # Durability buff
        durability = custom.get("durability", [0.2, 0.2, 0.9])[star_idx]
        duration = custom.get("duration", 3.0)  # Brief durability

        dur_effect = StatusEffect(
            effect_type=StatusEffectType.DAMAGE_REDUCTION,
            source_id=caster.id,
            duration=duration,
            value=durability,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, dur_effect)

        # Dive bomb damage (100/150/3000% AD in 3 hex circle)
        dive_damage_percent = custom.get("dive_damage_percent", [1.0, 1.5, 30.0])[star_idx]
        dive_damage = caster.stats.attack_damage * dive_damage_percent

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if caster_pos:
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and caster_pos.distance_to(unit_pos) <= 3:
                        actual = unit.take_damage(dive_damage, "physical")
                        result.total_damage += actual
                        result.targets_hit.append(unit.id)
                        caster.total_damage_dealt += actual

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        return result

    def handle_belveth_endless_banquet(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Bel'Veth: Transform, gain AS/HP, true damage on hit."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Initial AOE damage
        base_damage = ability.base_damage[star_idx]
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)

        if caster_pos and base_damage > 0:
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and caster_pos.distance_to(unit_pos) <= 2:
                        actual = unit.take_damage(base_damage, "physical")
                        result.total_damage += actual
                        result.targets_hit.append(unit.id)
                        caster.total_damage_dealt += actual

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        # Transform buffs (permanent)
        as_bonus = custom.get("transform_as", [0.33, 0.33, 1.33])[star_idx]
        hp_bonus_percent = custom.get("transform_hp_percent", 0.33)

        as_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=999,
            value=as_bonus,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, as_effect)

        # Bonus HP
        bonus_hp = caster.stats.max_hp * hp_bonus_percent
        caster.stats.max_hp += bonus_hp
        caster.stats.current_hp += bonus_hp

        # True damage on hit
        true_damage = custom.get("true_damage", [15, 25, 100])[star_idx]
        on_hit = OnHitEffect(
            source_ability=ability.ability_id,
            duration=999,
            bonus_damage=true_damage,
            bonus_damage_type="true",
        )

        if caster.id not in self._on_hit_effects:
            self._on_hit_effects[caster.id] = []
        self._on_hit_effects[caster.id].append(on_hit)

        return result

    def handle_singed_insanity_potion(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Singed: Heal per second, gain armor/MR."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        duration = custom.get("duration", 4.0)
        heal_per_sec = custom.get("heal_per_sec", [80, 100, 500])[star_idx]
        defense_bonus = custom.get("defense_bonus", 50)

        # Heal over time
        heal_effect = StatusEffect(
            effect_type=StatusEffectType.HEAL_OVER_TIME,
            source_id=caster.id,
            duration=duration,
            value=heal_per_sec,
            tick_interval=1.0,
        )

        # Armor/MR buff
        armor_effect = StatusEffect(
            effect_type=StatusEffectType.ARMOR_BUFF,
            source_id=caster.id,
            duration=duration,
            value=defense_bonus,
        )
        mr_effect = StatusEffect(
            effect_type=StatusEffectType.MAGIC_RESIST_BUFF,
            source_id=caster.id,
            duration=duration,
            value=defense_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, heal_effect)
            self.ability_system.status_effect_system.apply_effect(caster, armor_effect)
            self.ability_system.status_effect_system.apply_effect(caster, mr_effect)

        return result

    def handle_xin_zhao_crescent_guard(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Xin Zhao: AOE damage, Challenge enemies (200% dmg to them, 85% reduction from others)."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # AOE damage
        damage_percent = custom.get("damage_percent", [2.0, 2.5, 3.5])[star_idx]
        base_damage = caster.stats.attack_damage * damage_percent * (1 + caster.stats.ability_power / 100)
        challenge_duration = custom.get("challenge_duration", 6.0)

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if caster_pos:
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and caster_pos.distance_to(unit_pos) <= ability.aoe_radius:
                        actual = unit.take_damage(base_damage, "physical")
                        result.total_damage += actual
                        result.targets_hit.append(unit.id)
                        caster.total_damage_dealt += actual

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        # Damage reduction from other sources (self buff)
        reduction = custom.get("damage_reduction", 0.85)
        dr_effect = StatusEffect(
            effect_type=StatusEffectType.DAMAGE_REDUCTION,
            source_id=caster.id,
            duration=challenge_duration,
            value=reduction,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, dr_effect)

        return result

    # =========================================================================
    # SET 16 HP% DAMAGE HANDLERS
    # =========================================================================

    def handle_sett_showstopper(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Sett: Grab target, deal % max HP damage, AOE to others."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = max(enemies, key=lambda u: u.stats.current_hp)

        if not target:
            result.success = False
            return result

        # Primary target: % max HP damage
        hp_percent = custom.get("hp_percent", [0.4, 0.6, 8.0])[star_idx]
        primary_damage = target.stats.max_hp * hp_percent * (1 + caster.stats.ability_power / 100)

        actual = target.take_damage(primary_damage, "magical")
        result.total_damage += actual
        result.targets_hit.append(target.id)
        caster.total_damage_dealt += actual

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        # AOE damage to others (0.5% of primary damage)
        aoe_mult = custom.get("aoe_mult", 0.005)
        aoe_damage = primary_damage * aoe_mult

        target_pos = self.ability_system.grid.get_unit_position(target.id)
        if target_pos:
            for unit in all_units.values():
                if unit.id != target.id and unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and target_pos.distance_to(unit_pos) <= 3:
                        aoe_actual = unit.take_damage(aoe_damage, "magical")
                        result.total_damage += aoe_actual
                        caster.total_damage_dealt += aoe_actual

                        if unit.id not in result.targets_hit:
                            result.targets_hit.append(unit.id)

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        return result

    def handle_tahm_kench_devour(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Tahm Kench: Devour target for massive damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        # Check CC immunity
        is_cc_immune = False
        if self.ability_system.status_effect_system:
            # Check for QSS or similar effects
            pass  # Simplified: assume not CC immune

        # Damage based on CC immunity
        if is_cc_immune:
            damage = custom.get("cc_immune_damage", [600, 900, 20000])[star_idx]
        else:
            damage = custom.get("normal_damage", [850, 1275, 30000])[star_idx]

        damage *= (1 + caster.stats.ability_power / 100)
        actual = target.take_damage(damage, "magical")
        result.total_damage = actual
        result.targets_hit = [target.id]
        caster.total_damage_dealt += actual

        # Gain durability while devouring
        durability = custom.get("durability", 0.3)
        duration = custom.get("devour_duration", 2.5)

        dur_effect = StatusEffect(
            effect_type=StatusEffectType.DAMAGE_REDUCTION,
            source_id=caster.id,
            duration=duration,
            value=durability,
        )
        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, dur_effect)

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        return result

    def handle_nidalee_pounce(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Nidalee: Pounce on lowest HP enemy, recast on kill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data
        base_damage = ability.base_damage[star_idx]

        # Track remaining casts
        remaining_casts = custom.get("max_casts", 3)
        damage_reduction = 1.0

        while remaining_casts > 0:
            # Find lowest HP adjacent enemy
            caster_pos = self.ability_system.grid.get_unit_position(caster.id)
            if not caster_pos:
                break

            enemies = []
            for unit in all_units.values():
                if unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and caster_pos.distance_to(unit_pos) <= 2:
                        enemies.append(unit)

            if not enemies:
                break

            target = min(enemies, key=lambda u: u.stats.current_hp)

            # Bonus damage to tanks/fighters
            role_mult = 1.0
            if target.role == "tank":
                role_mult = 1.5
            elif target.role == "fighter":
                role_mult = 1.25

            damage = base_damage * damage_reduction * role_mult
            actual = target.take_damage(damage, ability.damage_type)
            result.total_damage += actual
            caster.total_damage_dealt += actual

            if target.id not in result.targets_hit:
                result.targets_hit.append(target.id)

            remaining_casts -= 1

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1
                # Recast at 70% damage
                damage_reduction = 0.7
            else:
                break  # Stop if target survives

        return result

    # =========================================================================
    # SET 16 MULTI-HIT HANDLERS
    # =========================================================================

    def handle_miss_fortune_bullet_time(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Miss Fortune: Fire waves of bullets at nearest enemies."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Number of targets
        num_targets = custom.get("num_targets", [2, 2, 6])[star_idx]

        # Wave damage (AD + AP scaling)
        ad_percent = custom.get("first_wave_ad", [1.3, 2.0, 30.0])[star_idx]
        ap_percent = custom.get("first_wave_ap", [0.15, 0.25, 0.7])[star_idx]
        first_wave_damage = (caster.stats.attack_damage * ad_percent +
                            caster.stats.ability_power * ap_percent)

        # Rest waves deal 60% of first wave
        rest_wave_mult = custom.get("rest_wave_mult", [0.6, 0.6, 1.0])[star_idx]
        num_waves = custom.get("num_waves", 3)

        # Find nearest enemies
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    enemies.append((unit, caster_pos.distance_to(unit_pos)))

        enemies.sort(key=lambda x: x[1])
        targets = [e[0] for e in enemies[:num_targets]]

        # Apply waves
        for wave_idx in range(num_waves):
            wave_damage = first_wave_damage if wave_idx == 0 else first_wave_damage * rest_wave_mult

            for target in targets:
                if not target.is_alive:
                    continue

                actual = target.take_damage(wave_damage, "physical")
                result.total_damage += actual
                caster.total_damage_dealt += actual

                if target.id not in result.targets_hit:
                    result.targets_hit.append(target.id)

                if not target.is_alive:
                    result.kills += 1
                    caster.kills += 1

        return result

    def handle_kaisa_icathian_rain(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Kai'Sa: Dash away, fire missiles at 4 nearest enemies."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        num_missiles = custom.get("num_missiles", [15, 15, 25])[star_idx]
        damage_per_missile = custom.get("missile_damage", [44, 66, 155])[star_idx]
        num_targets = custom.get("num_targets", 4)

        # Find nearest enemies
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    enemies.append((unit, caster_pos.distance_to(unit_pos)))

        enemies.sort(key=lambda x: x[1])
        targets = [e[0] for e in enemies[:num_targets]]

        if not targets:
            result.success = False
            return result

        # Distribute missiles among targets
        missiles_per_target = num_missiles // len(targets)
        extra_missiles = num_missiles % len(targets)

        for i, target in enumerate(targets):
            missiles = missiles_per_target + (1 if i < extra_missiles else 0)
            total_damage = damage_per_missile * missiles

            actual = target.take_damage(total_damage, "physical")
            result.total_damage += actual
            result.targets_hit.append(target.id)
            caster.total_damage_dealt += actual

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1

        return result

    def handle_kindred_lambs_respite(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Kindred: Create zone preventing ally deaths, double AS, bonus arrows."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        duration = custom.get("zone_duration", [2.5, 2.5, 99])[star_idx]
        arrow_damage_percent = custom.get("arrow_damage", [0.25, 0.38, 9.99])[star_idx]

        # Double attack speed
        as_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=duration,
            value=1.0,  # +100% AS
        )

        # Invulnerability for allies (simplified: just caster)
        invuln_effect = StatusEffect(
            effect_type=StatusEffectType.INVULNERABLE,
            source_id=caster.id,
            duration=duration,
            custom_data={"prevents_death": True},
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, as_effect)
            self.ability_system.status_effect_system.apply_effect(caster, invuln_effect)

        # Bonus arrow on-hit
        arrow_damage = caster.stats.attack_damage * arrow_damage_percent
        on_hit = OnHitEffect(
            source_ability=ability.ability_id,
            duration=duration,
            bonus_damage=arrow_damage,
            bonus_damage_type="physical",
        )

        if caster.id not in self._on_hit_effects:
            self._on_hit_effects[caster.id] = []
        self._on_hit_effects[caster.id].append(on_hit)

        return result

    def handle_ambessa_dashing_slash(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Ambessa: Dash and slash, chain slam, recast on kill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        # Swipe damage
        ad_percent = custom.get("swipe_ad", [0.45, 0.65, 3.2])[star_idx]
        ap_percent = custom.get("swipe_ap", [0.05, 0.1, 0.3])[star_idx]
        swipe_damage = (caster.stats.attack_damage * ad_percent +
                       caster.stats.ability_power * ap_percent)

        # Chain slam (275% of swipe)
        chain_mult = custom.get("chain_mult", 2.75)
        chain_damage = swipe_damage * chain_mult

        # Apply swipe damage to target
        actual = target.take_damage(swipe_damage, "physical")
        result.total_damage += actual
        result.targets_hit.append(target.id)
        caster.total_damage_dealt += actual

        killed = not target.is_alive
        if killed:
            result.kills += 1
            caster.kills += 1

        # Chain slam in line
        target_pos = self.ability_system.grid.get_unit_position(target.id)
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)

        if target_pos and caster_pos:
            for unit in all_units.values():
                if unit.id != target.id and unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and self.ability_system._is_in_line(caster_pos, target_pos, unit_pos):
                        chain_actual = unit.take_damage(chain_damage, "physical")
                        result.total_damage += chain_actual
                        caster.total_damage_dealt += chain_actual

                        if unit.id not in result.targets_hit:
                            result.targets_hit.append(unit.id)

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1
                            killed = True

        # Recast on kill (simplified: just bonus mana)
        if killed:
            caster.stats.current_mana = min(
                caster.stats.current_mana + caster.stats.max_mana * 0.5,
                caster.stats.max_mana
            )

        return result

    def handle_kogmaw_void_artillery(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Kog'Maw: Deal damage and shred armor/MR."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data
        base_damage = ability.base_damage[star_idx]

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = max(enemies, key=lambda u: u.stats.current_hp)

        if not target:
            result.success = False
            return result

        # Primary damage
        damage = self._calculate_damage(caster, target, base_damage, ability)
        actual = target.take_damage(damage, ability.damage_type)
        result.total_damage += actual
        result.targets_hit.append(target.id)
        caster.total_damage_dealt += actual

        # Shred armor/MR
        shred_amount = custom.get("shred", [8, 10, 15])[star_idx]
        target.stats.armor = max(0, target.stats.armor - shred_amount)
        target.stats.magic_resist = max(0, target.stats.magic_resist - shred_amount)

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        # Adjacent enemies take 50% damage
        target_pos = self.ability_system.grid.get_unit_position(target.id)
        if target_pos:
            for unit in all_units.values():
                if unit.id != target.id and unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and target_pos.distance_to(unit_pos) <= 1:
                        adj_damage = damage * 0.5
                        adj_actual = unit.take_damage(adj_damage, ability.damage_type)
                        result.total_damage += adj_actual
                        caster.total_damage_dealt += adj_actual

                        if unit.id not in result.targets_hit:
                            result.targets_hit.append(unit.id)

        return result

    # =========================================================================
    # SET 16 ALLY BUFF HANDLERS
    # =========================================================================

    def handle_bard_tempered_fate(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Bard: Stun enemies and increase damage taken."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        stun_duration = custom.get("stun_duration", [1.25, 2.0, 15.0])[star_idx]
        damage_amp = custom.get("damage_amp", [0.15, 0.2, 99.99])[star_idx]

        # Find largest group of enemies
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        # Target all enemies in range (simplified)
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and caster_pos.distance_to(unit_pos) <= ability.aoe_radius:
                    # Stun
                    stun_effect = StatusEffect(
                        effect_type=StatusEffectType.STUN,
                        source_id=caster.id,
                        duration=stun_duration,
                    )
                    # Damage amplification
                    amp_effect = StatusEffect(
                        effect_type=StatusEffectType.DAMAGE_AMPLIFICATION,
                        source_id=caster.id,
                        duration=stun_duration,
                        value=damage_amp,
                    )

                    if self.ability_system.status_effect_system:
                        self.ability_system.status_effect_system.apply_effect(unit, stun_effect)
                        self.ability_system.status_effect_system.apply_effect(unit, amp_effect)

                    result.targets_hit.append(unit.id)
                    result.cc_applied.append(unit.id)

        return result

    def handle_taric_cosmic_radiance(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Taric: Grant invulnerability to nearby allies."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        invuln_duration = custom.get("invuln_duration", [2.5, 2.5, 8.0])[star_idx]
        radius = custom.get("radius", [2, 3, 4])[star_idx]

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        # Apply invulnerability to allies in range
        for unit in all_units.values():
            if unit.team == caster.team and unit.is_alive:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and caster_pos.distance_to(unit_pos) <= radius:
                    invuln_effect = StatusEffect(
                        effect_type=StatusEffectType.INVULNERABLE,
                        source_id=caster.id,
                        duration=invuln_duration,
                    )

                    if self.ability_system.status_effect_system:
                        self.ability_system.status_effect_system.apply_effect(unit, invuln_effect)

                    result.targets_hit.append(unit.id)

        return result

    # =========================================================================
    # SET 16 SPECIAL MECHANIC HANDLERS
    # =========================================================================

    def handle_zilean_time_bomb(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Zilean: Place time bomb that deals DoT and executes after duration."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        dot_damage = custom.get("dot_damage", [70, 105, 1000])[star_idx]
        explode_damage = custom.get("explode_damage", [150, 225, 4000])[star_idx]
        execute_time = custom.get("execute_time", [18, 16, 1])[star_idx]

        # Find nearest enemy without bomb
        enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
        if not enemies:
            result.success = False
            return result

        target = enemies[0]  # Simplified: nearest

        # Apply DoT (burn effect)
        dot_effect = StatusEffect(
            effect_type=StatusEffectType.BURN,
            source_id=caster.id,
            duration=execute_time,
            value=dot_damage,
            tick_interval=1.0,
            custom_data={"is_time_bomb": True, "explode_damage": explode_damage},
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(target, dot_effect)

        result.targets_hit.append(target.id)

        return result

    def handle_warwick_primal_howl(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Warwick: Gain massive AS, omnivamp, bonus damage for rest of combat."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Attack speed buff (permanent)
        as_bonus = custom.get("as_bonus", [1.0, 1.0, 4.0])[star_idx]
        as_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
            source_id=caster.id,
            duration=999,
            value=as_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, as_effect)

        # Omnivamp
        omnivamp = custom.get("omnivamp", 0.15)
        caster.stats.omnivamp += omnivamp

        return result

    def handle_malzahar_void_gate(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Malzahar: Line damage + infection DoT that spreads on kill."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        base_damage = ability.base_damage[star_idx]
        infection_damage = custom.get("infection_damage", [16, 24, 400])[star_idx]
        shred_percent = custom.get("shred_percent", 0.2)
        shred_duration = custom.get("shred_duration", 4.0)

        # Find target for line
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        target_pos = self.ability_system.grid.get_unit_position(target.id)

        if not caster_pos or not target_pos:
            result.success = False
            return result

        # Hit enemies in 5-hex line
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and self.ability_system._is_in_line(caster_pos, target_pos, unit_pos, tolerance=2.5):
                    # Initial damage
                    damage = self._calculate_damage(caster, unit, base_damage, ability)
                    actual = unit.take_damage(damage, ability.damage_type)
                    result.total_damage += actual
                    result.targets_hit.append(unit.id)
                    caster.total_damage_dealt += actual

                    # MR shred
                    shred_amount = unit.stats.magic_resist * shred_percent
                    unit.stats.magic_resist -= shred_amount

                    # Infection DoT (infinite duration, stacks)
                    infection_effect = StatusEffect(
                        effect_type=StatusEffectType.POISON,
                        source_id=caster.id,
                        duration=999,
                        value=infection_damage * (1 + caster.stats.ability_power / 100),
                        tick_interval=1.0,
                        max_stacks=99,
                        custom_data={"spreads_on_kill": True},
                    )

                    if self.ability_system.status_effect_system:
                        self.ability_system.status_effect_system.apply_effect(unit, infection_effect)

                    if not unit.is_alive:
                        result.kills += 1
                        caster.kills += 1

        return result

    def handle_kalista_rend(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Kalista: Throw spears at 3 nearest enemies based on souls."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        base_damage = custom.get("base_damage", [28, 42, 450])[star_idx]
        ad_scaling = custom.get("ad_scaling", [2, 3, 15])[star_idx]
        armor_shred = custom.get("armor_shred", 1)

        # Find 3 nearest enemies
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    enemies.append((unit, caster_pos.distance_to(unit_pos)))

        enemies.sort(key=lambda x: x[1])
        targets = [e[0] for e in enemies[:3]]

        for target in targets:
            damage = base_damage + caster.stats.attack_damage * ad_scaling
            actual = target.take_damage(damage, "physical")
            result.total_damage += actual
            result.targets_hit.append(target.id)
            caster.total_damage_dealt += actual

            # Armor shred
            target.stats.armor = max(0, target.stats.armor - armor_shred)

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1

        return result

    def handle_gwen_snip_snip(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Gwen: Dash around target and snip 5 times."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        primary_damage = custom.get("primary_damage", [45, 68, 105])[star_idx]
        cone_damage = custom.get("cone_damage", [20, 30, 50])[star_idx]
        num_snips = custom.get("num_snips", 5)

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        # Snip target multiple times
        for _ in range(num_snips):
            if not target.is_alive:
                break

            damage = self._calculate_damage(caster, target, primary_damage, ability)
            actual = target.take_damage(damage, ability.damage_type)
            result.total_damage += actual
            caster.total_damage_dealt += actual

        if target.id not in result.targets_hit:
            result.targets_hit.append(target.id)

        if not target.is_alive:
            result.kills += 1
            caster.kills += 1

        # Cone damage to others
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        target_pos = self.ability_system.grid.get_unit_position(target.id)

        if caster_pos and target_pos:
            for unit in all_units.values():
                if unit.id != target.id and unit.team != caster.team and unit.is_targetable:
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and self.ability_system._is_in_cone(caster_pos, target_pos, unit_pos):
                        cone_total = cone_damage * num_snips
                        cone_dmg = self._calculate_damage(caster, unit, cone_total, ability)
                        cone_actual = unit.take_damage(cone_dmg, ability.damage_type)
                        result.total_damage += cone_actual
                        caster.total_damage_dealt += cone_actual

                        if unit.id not in result.targets_hit:
                            result.targets_hit.append(unit.id)

                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

        return result

    def handle_sylas_hijack(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Sylas: Cycle through 3 abilities (Cataclysm, Demacian Justice, Final Spark)."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # Track which ability to use (stored in custom_data of unit or random)
        import random
        ability_choice = random.choice(["cataclysm", "demacian_justice", "final_spark"])

        if ability_choice == "cataclysm":
            # AOE damage + stun
            damage = custom.get("cataclysm_damage", [120, 180, 1000])[star_idx]
            stun_duration = custom.get("cataclysm_stun", [1.5, 1.75, 30])[star_idx]

            caster_pos = self.ability_system.grid.get_unit_position(caster.id)
            if caster_pos:
                for unit in all_units.values():
                    if unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                        if unit_pos and caster_pos.distance_to(unit_pos) <= 2:
                            dmg = self._calculate_damage(caster, unit, damage, ability)
                            actual = unit.take_damage(dmg, ability.damage_type)
                            result.total_damage += actual
                            result.targets_hit.append(unit.id)
                            caster.total_damage_dealt += actual

                            if unit.is_alive and self.ability_system.status_effect_system:
                                stun = StatusEffect(
                                    effect_type=StatusEffectType.STUN,
                                    source_id=caster.id,
                                    duration=stun_duration,
                                )
                                self.ability_system.status_effect_system.apply_effect(unit, stun)
                                result.cc_applied.append(unit.id)

                            if not unit.is_alive:
                                result.kills += 1
                                caster.kills += 1

        elif ability_choice == "demacian_justice":
            # Single target damage + execute
            damage = custom.get("justice_damage", [700, 1050, 99999])[star_idx]
            execute_threshold = custom.get("execute_threshold", 0.15)

            target = None
            if caster.current_target_id:
                target = all_units.get(caster.current_target_id)

            if not target:
                enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
                if enemies:
                    target = min(enemies, key=lambda u: u.stats.current_hp)

            if target:
                hp_percent = target.stats.current_hp / target.stats.max_hp
                if hp_percent <= execute_threshold:
                    # Execute
                    actual = target.take_damage(target.stats.current_hp + 9999, ability.damage_type)
                else:
                    dmg = self._calculate_damage(caster, target, damage, ability)
                    actual = target.take_damage(dmg, ability.damage_type)

                result.total_damage += actual
                result.targets_hit.append(target.id)
                caster.total_damage_dealt += actual

                if not target.is_alive:
                    result.kills += 1
                    caster.kills += 1

        else:  # final_spark
            # Shield + line damage
            shield = custom.get("spark_shield", [400, 450, 800])[star_idx]
            damage = custom.get("spark_damage", [360, 540, 1000])[star_idx]

            # Apply shield
            shield_effect = StatusEffect(
                effect_type=StatusEffectType.SHIELD,
                source_id=caster.id,
                duration=2.0,
                value=shield,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(caster, shield_effect)

            # Line damage
            target = None
            if caster.current_target_id:
                target = all_units.get(caster.current_target_id)

            if not target:
                enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
                if enemies:
                    target = enemies[0]

            if target:
                caster_pos = self.ability_system.grid.get_unit_position(caster.id)
                target_pos = self.ability_system.grid.get_unit_position(target.id)

                if caster_pos and target_pos:
                    for unit in all_units.values():
                        if unit.team != caster.team and unit.is_targetable:
                            unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                            if unit_pos and self.ability_system._is_in_line(caster_pos, target_pos, unit_pos):
                                dmg = self._calculate_damage(caster, unit, damage, ability)
                                actual = unit.take_damage(dmg, ability.damage_type)
                                result.total_damage += actual
                                result.targets_hit.append(unit.id)
                                caster.total_damage_dealt += actual

                                if not unit.is_alive:
                                    result.kills += 1
                                    caster.kills += 1

        return result

    def handle_fiddlesticks_crowstorm(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Fiddlesticks: Teleport to largest group, stun, deal DoT."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        stun_duration = custom.get("stun_duration", 1.75)
        damage_per_sec = custom.get("damage_per_sec", [100, 150, 6666])[star_idx]
        duration = custom.get("channel_duration", 5.0)
        close_bonus = custom.get("close_bonus", 0.33)

        # Find enemies in 2 hex radius
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if not caster_pos:
            result.success = False
            return result

        enemies_in_range = []
        for unit in all_units.values():
            if unit.team != caster.team and unit.is_targetable:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and caster_pos.distance_to(unit_pos) <= 2:
                    enemies_in_range.append((unit, caster_pos.distance_to(unit_pos)))

        enemies_in_range.sort(key=lambda x: x[1])

        # Stun all in range
        for unit, dist in enemies_in_range:
            stun = StatusEffect(
                effect_type=StatusEffectType.STUN,
                source_id=caster.id,
                duration=stun_duration,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(unit, stun)
            result.cc_applied.append(unit.id)

        # Apply burn DoT (closest 2 get 33% bonus)
        for i, (unit, dist) in enumerate(enemies_in_range):
            bonus = close_bonus if i < 2 else 0
            burn_damage = damage_per_sec * (1 + bonus) * (1 + caster.stats.ability_power / 100)

            burn = StatusEffect(
                effect_type=StatusEffectType.BURN,
                source_id=caster.id,
                duration=duration,
                value=burn_damage,
                tick_interval=1.0,
            )
            if self.ability_system.status_effect_system:
                self.ability_system.status_effect_system.apply_effect(unit, burn)

            result.targets_hit.append(unit.id)

        return result

    def handle_seraphine_note_progression(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Seraphine: Deal damage with notes, at 12 notes heal allies and big damage."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        note_damage = custom.get("note_damage", [25, 40, 200])[star_idx]
        num_notes = custom.get("num_notes", 3)

        # Track notes (simplified: check if this is empowered cast)
        # For simplicity, alternate between normal and empowered
        is_empowered = getattr(caster, '_seraphine_notes', 0) >= 9  # Every 4th cast

        if is_empowered:
            # Empowered: heal allies, big damage to enemies
            heal_amount = custom.get("empowered_heal", [60, 90, 400])[star_idx]
            wave_damage = custom.get("empowered_damage", [270, 405, 2200])[star_idx]

            # Heal allies
            for unit in all_units.values():
                if unit.team == caster.team and unit.is_alive:
                    healed = unit.heal(heal_amount * (1 + caster.stats.ability_power / 100))
                    result.total_healing += healed
                    if unit.id not in result.targets_hit:
                        result.targets_hit.append(unit.id)

            # Damage enemies (reduced by 30% per enemy passed)
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            for i, enemy in enumerate(enemies):
                reduction = 0.7 ** i  # 30% reduction per enemy
                damage = wave_damage * reduction * (1 + caster.stats.ability_power / 100)
                actual = enemy.take_damage(damage, ability.damage_type)
                result.total_damage += actual
                caster.total_damage_dealt += actual

                if enemy.id not in result.targets_hit:
                    result.targets_hit.append(enemy.id)

                if not enemy.is_alive:
                    result.kills += 1
                    caster.kills += 1

            caster._seraphine_notes = 0
        else:
            # Normal: deal note damage to nearby enemies
            caster_pos = self.ability_system.grid.get_unit_position(caster.id)
            if caster_pos:
                enemies = []
                for unit in all_units.values():
                    if unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                        if unit_pos:
                            enemies.append((unit, caster_pos.distance_to(unit_pos)))

                enemies.sort(key=lambda x: x[1])

                for i in range(min(num_notes, len(enemies))):
                    enemy = enemies[i][0]
                    damage = note_damage * (1 + caster.stats.ability_power / 100)
                    actual = enemy.take_damage(damage, ability.damage_type)
                    result.total_damage += actual
                    result.targets_hit.append(enemy.id)
                    caster.total_damage_dealt += actual

                    if not enemy.is_alive:
                        result.kills += 1
                        caster.kills += 1

            caster._seraphine_notes = getattr(caster, '_seraphine_notes', 0) + num_notes

        return result

    def handle_lucian_senna_culling(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Lucian & Senna: Fire multiple shots that explode on first enemy."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        num_shots = custom.get("num_shots", [12, 12, 42])[star_idx]
        damage_per_shot = custom.get("damage_per_shot", [83, 132, 900])[star_idx]

        # Find target
        target = None
        if caster.current_target_id:
            target = all_units.get(caster.current_target_id)

        if not target or not target.is_targetable:
            enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
            if enemies:
                target = enemies[0]

        if not target:
            result.success = False
            return result

        target_pos = self.ability_system.grid.get_unit_position(target.id)

        # Fire shots
        for _ in range(num_shots):
            if not target.is_alive:
                # Retarget
                enemies = [u for u in all_units.values() if u.team != caster.team and u.is_targetable]
                if enemies:
                    target = enemies[0]
                    target_pos = self.ability_system.grid.get_unit_position(target.id)
                else:
                    break

            # Direct hit
            actual = target.take_damage(damage_per_shot, "physical")
            result.total_damage += actual
            caster.total_damage_dealt += actual

            if target.id not in result.targets_hit:
                result.targets_hit.append(target.id)

            # Explosion hits nearby
            if target_pos:
                for unit in all_units.values():
                    if unit.id != target.id and unit.team != caster.team and unit.is_targetable:
                        unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                        if unit_pos and target_pos.distance_to(unit_pos) <= 1:
                            splash = damage_per_shot * 0.5
                            splash_actual = unit.take_damage(splash, "physical")
                            result.total_damage += splash_actual
                            caster.total_damage_dealt += splash_actual

            if not target.is_alive:
                result.kills += 1
                caster.kills += 1

        return result

    # =========================================================================
    # 5-COST & 7-COST LEGENDARY HANDLERS
    # =========================================================================

    def handle_aatrox_darkin_blade(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Aatrox: The Darkin Blade - 3가지 스킬 로테이션.
        - Slash: 라인 피해 + 20% Sunder (4초)
        - Swipe: 콘 피해 + 1초 넉업
        - Slam: 원형 AOE 피해
        - 15% 이하 적 처형
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # 로테이션 추적
        if not hasattr(caster, '_aatrox_rotation'):
            caster._aatrox_rotation = 0

        rotation = caster._aatrox_rotation % 3
        caster._aatrox_rotation += 1

        # 스킬별 피해량
        slash_damage = custom.get("slash_damage", [100, 195, 2500])[star_idx]
        swipe_damage = custom.get("swipe_damage", [120, 234, 3000])[star_idx]
        slam_damage = custom.get("slam_damage", [160, 312, 4000])[star_idx]
        execute_threshold = custom.get("execute_threshold", 0.15)
        sunder_percent = custom.get("sunder_percent", 0.20)

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        from src.combat.combat_unit import UnitState

        if rotation == 0:
            # Slash - 라인 피해
            if caster_pos and caster.current_target_id:
                target = all_units.get(caster.current_target_id)
                if target:
                    target_pos = self.ability_system.grid.get_unit_position(target.id)
                    if target_pos:
                        line_targets = self.ability_system.grid.get_units_in_line(
                            caster_pos, target_pos, all_units
                        )
                        for unit_id in line_targets:
                            unit = all_units.get(unit_id)
                            if unit and unit.team != caster.team and unit.is_targetable:
                                # 처형 체크
                                if unit.stats.current_hp / unit.stats.max_hp <= execute_threshold:
                                    unit.stats.current_hp = 0
                                    unit.state = UnitState.DEAD
                                    result.kills += 1
                                    caster.kills += 1
                                else:
                                    actual = unit.take_damage(slash_damage, "physical")
                                    result.total_damage += actual
                                    caster.total_damage_dealt += actual
                                    # Sunder - 방어력 감소
                                    unit.stats.armor *= (1 - sunder_percent)
                                result.targets_hit.append(unit.id)

        elif rotation == 1:
            # Swipe - 콘 피해 + 넉업
            if caster_pos:
                cone_targets = self.ability_system.grid.get_units_in_cone(
                    caster_pos, 2, all_units
                )
                for unit_id in cone_targets:
                    unit = all_units.get(unit_id)
                    if unit and unit.team != caster.team and unit.is_targetable:
                        if unit.stats.current_hp / unit.stats.max_hp <= execute_threshold:
                            unit.stats.current_hp = 0
                            unit.state = UnitState.DEAD
                            result.kills += 1
                            caster.kills += 1
                        else:
                            actual = unit.take_damage(swipe_damage, "physical")
                            result.total_damage += actual
                            caster.total_damage_dealt += actual
                            # 1초 스턴
                            if self.ability_system.status_effect_system:
                                stun = StatusEffect(
                                    effect_type=StatusEffectType.STUN,
                                    source_id=caster.id,
                                    duration=1.0,
                                    value=0,
                                )
                                self.ability_system.status_effect_system.apply_effect(unit, stun)
                        result.targets_hit.append(unit.id)

        else:
            # Slam - 타겟 주변 원형 AOE
            target = all_units.get(caster.current_target_id) if caster.current_target_id else None
            if not target:
                target = min(enemies, key=lambda u: caster_pos.distance_to(
                    self.ability_system.grid.get_unit_position(u.id) or caster_pos
                ) if caster_pos else 0) if enemies else None

            if target:
                target_pos = self.ability_system.grid.get_unit_position(target.id)
                if target_pos:
                    for unit in enemies:
                        unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                        if unit_pos and target_pos.distance_to(unit_pos) <= 1:
                            if unit.stats.current_hp / unit.stats.max_hp <= execute_threshold:
                                unit.stats.current_hp = 0
                                unit.state = UnitState.DEAD
                                result.kills += 1
                                caster.kills += 1
                            else:
                                actual = unit.take_damage(slam_damage, "physical")
                                result.total_damage += actual
                                caster.total_damage_dealt += actual
                            result.targets_hit.append(unit.id)

        return result

    def handle_aurelion_sol_skies_descend(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Aurelion Sol: The Skies Descend - Stardust 스택 시스템.
        - 스킬 피해 시 Stardust 획득
        - 스택에 따라 강화: 추가피해, 범위증가, 넉업, 트루뎀, 메테오
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        base_damage = ability.base_damage[star_idx]
        shockwave_damage = custom.get("shockwave_damage", [50, 75, 1000])[star_idx]
        meteor_damage = custom.get("meteor_damage", [1000, 1500, 9999])[star_idx]

        # Stardust 추적
        if not hasattr(caster, '_stardust'):
            caster._stardust = 0

        stardust = caster._stardust

        # 스택에 따른 보너스 계산
        base_radius = 2
        bonus_damage_mult = 1.0
        has_knockup = False
        has_true_damage = False
        has_meteor = False
        shockwave_radius = 0

        if stardust >= 15:
            shockwave_radius = 3
        if stardust >= 60:
            bonus_damage_mult += 0.15
        if stardust >= 100:
            shockwave_radius = 4
        if stardust >= 175:
            has_knockup = True
        if stardust >= 250:
            shockwave_radius = 5  # 간소화: 10 대신 5
        if stardust >= 400:
            has_true_damage = True
        if stardust >= 700:
            has_meteor = True

        # 타겟 찾기
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        if not enemies:
            return result

        # 가장 많은 적이 모인 곳 찾기
        target = max(enemies, key=lambda u: sum(
            1 for e in enemies
            if self.ability_system.grid.get_unit_position(e.id) and
            self.ability_system.grid.get_unit_position(u.id) and
            self.ability_system.grid.get_unit_position(e.id).distance_to(
                self.ability_system.grid.get_unit_position(u.id)
            ) <= base_radius
        ))

        target_pos = self.ability_system.grid.get_unit_position(target.id)
        if not target_pos:
            return result

        # 메인 피해
        final_damage = base_damage * bonus_damage_mult
        damage_type = "true" if has_true_damage else "magical"

        for unit in enemies:
            unit_pos = self.ability_system.grid.get_unit_position(unit.id)
            if unit_pos and target_pos.distance_to(unit_pos) <= base_radius:
                actual = unit.take_damage(final_damage, damage_type)
                result.total_damage += actual
                caster.total_damage_dealt += actual
                result.targets_hit.append(unit.id)

                # Stardust 획득
                caster._stardust += 5

                # 넉업
                if has_knockup and self.ability_system.status_effect_system:
                    stun = StatusEffect(
                        effect_type=StatusEffectType.STUN,
                        source_id=caster.id,
                        duration=2.0,
                        value=0,
                    )
                    self.ability_system.status_effect_system.apply_effect(unit, stun)

                if not unit.is_alive:
                    result.kills += 1
                    caster.kills += 1

        # 쇼크웨이브
        if shockwave_radius > 0:
            for unit in enemies:
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos:
                    dist = target_pos.distance_to(unit_pos)
                    if base_radius < dist <= shockwave_radius:
                        actual = unit.take_damage(shockwave_damage, damage_type)
                        result.total_damage += actual
                        caster.total_damage_dealt += actual
                        if unit.id not in result.targets_hit:
                            result.targets_hit.append(unit.id)
                        caster._stardust += 3

        # 메테오 스톰
        if has_meteor:
            living_enemies = [u for u in enemies if u.is_alive]
            if living_enemies:
                damage_per_enemy = meteor_damage / len(living_enemies)
                for unit in living_enemies:
                    actual = unit.take_damage(damage_per_enemy, "magical")
                    result.total_damage += actual
                    caster.total_damage_dealt += actual
                    if not unit.is_alive:
                        result.kills += 1
                        caster.kills += 1

        return result

    def handle_baron_nashor_void_eruption(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Baron Nashor: Void Eruption.
        - CC 면역 (패시브)
        - 슬램 + 스파이크 (2헥스 원형)
        - 3초간 10발 산성 발사
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        slam_damage = ability.base_damage[star_idx]
        acid_damage = custom.get("acid_damage", [250, 375, 20000])[star_idx]
        num_acid_globs = custom.get("num_acid_globs", 10)

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        if not enemies or not caster_pos:
            return result

        # 슬램 - 2헥스 원형
        for unit in enemies:
            unit_pos = self.ability_system.grid.get_unit_position(unit.id)
            if unit_pos and caster_pos.distance_to(unit_pos) <= 2:
                actual = unit.take_damage(slam_damage, "physical")
                result.total_damage += actual
                caster.total_damage_dealt += actual
                result.targets_hit.append(unit.id)

                # 넉업
                if self.ability_system.status_effect_system:
                    stun = StatusEffect(
                        effect_type=StatusEffectType.STUN,
                        source_id=caster.id,
                        duration=1.5,
                        value=0,
                    )
                    self.ability_system.status_effect_system.apply_effect(unit, stun)

                if not unit.is_alive:
                    result.kills += 1
                    caster.kills += 1

        # 산성 발사 (10발을 즉시 발사로 간소화)
        living_enemies = [u for u in enemies if u.is_alive]
        for i in range(num_acid_globs):
            if not living_enemies:
                break
            target = living_enemies[i % len(living_enemies)]
            actual = target.take_damage(acid_damage, "physical")
            result.total_damage += actual
            caster.total_damage_dealt += actual
            if target.id not in result.targets_hit:
                result.targets_hit.append(target.id)
            if not target.is_alive:
                result.kills += 1
                caster.kills += 1
                living_enemies = [u for u in living_enemies if u.is_alive]

        return result

    def handle_ryze_realm_warp(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Ryze: Realm Warp - 지역 특성에 따른 보너스.
        - 기본: 마법 피해 + 분열 (33% 피해로 2명에게)
        - 지역 보너스: Freljord(트루뎀), Noxus(관통), Demacia(처형) 등
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        base_damage = ability.base_damage[star_idx]
        split_ratio = custom.get("split_ratio", 0.33)
        split_count = custom.get("split_count", 2)

        # 지역 보너스 체크 (source_instance에서 특성 확인)
        active_regions = set()
        if caster.source_instance:
            # 간소화: 챔피언 특성에서 지역 확인
            for trait in getattr(caster.source_instance.champion, 'traits', []):
                trait_lower = trait.lower()
                if trait_lower in ['freljord', 'noxus', 'demacia', 'ionia', 'bilgewater',
                                   'shurima', 'targon', 'piltover', 'shadow_isles', 'ixtal']:
                    active_regions.add(trait_lower)

        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        if not enemies:
            return result

        # 메인 타겟
        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        if caster_pos:
            target = min(enemies, key=lambda u: caster_pos.distance_to(
                self.ability_system.grid.get_unit_position(u.id) or caster_pos
            ))
        else:
            target = enemies[0]

        # 지역 보너스 적용
        damage_type = "magical"
        bonus_mult = 1.0
        execute_threshold = 0.0
        extra_splits = 0

        if 'freljord' in active_regions:
            damage_type = "true"
        if 'demacia' in active_regions:
            execute_threshold = 0.15
        if 'ionia' in active_regions:
            extra_splits = 2
        if 'piltover' in active_regions:
            if not hasattr(caster, '_ryze_cast_count'):
                caster._ryze_cast_count = 0
            caster._ryze_cast_count += 1
            if caster._ryze_cast_count % 3 == 0:
                bonus_mult = 1.5
        if 'targon' in active_regions:
            # 힐은 피해 후 처리
            pass

        final_damage = base_damage * bonus_mult

        from src.combat.combat_unit import UnitState

        # 메인 피해
        if execute_threshold > 0 and target.stats.current_hp / target.stats.max_hp <= execute_threshold:
            target.stats.current_hp = 0
            target.state = UnitState.DEAD
            result.kills += 1
            caster.kills += 1
        else:
            actual = target.take_damage(final_damage, damage_type)
            result.total_damage += actual
            caster.total_damage_dealt += actual
        result.targets_hit.append(target.id)

        # 분열 피해
        other_enemies = [u for u in enemies if u.id != target.id and u.is_alive]
        split_damage = final_damage * split_ratio
        total_splits = split_count + extra_splits

        for i, unit in enumerate(other_enemies[:total_splits]):
            if execute_threshold > 0 and unit.stats.current_hp / unit.stats.max_hp <= execute_threshold:
                unit.stats.current_hp = 0
                unit.state = UnitState.DEAD
                result.kills += 1
                caster.kills += 1
            else:
                actual = unit.take_damage(split_damage, damage_type)
                result.total_damage += actual
                caster.total_damage_dealt += actual
            result.targets_hit.append(unit.id)

            if not unit.is_alive:
                result.kills += 1
                caster.kills += 1

        # Targon 힐
        if 'targon' in active_regions:
            allies = [u for u in all_units.values()
                      if u.team == caster.team and u.is_alive and u.id != caster.id]
            if allies:
                lowest_ally = min(allies, key=lambda u: u.stats.current_hp / u.stats.max_hp)
                heal_amount = result.total_damage * 0.25
                lowest_ally.heal(heal_amount)

        return result

    def handle_zaahen_divine_challenge(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Zaahen: Divine Challenge.
        - 대시 후 원형 슬래시
        - 타겟이 안 죽으면 즉시 재시전
        - 22회 연속 시전 시 즉사 + 3헥스 AOE
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        main_damage = ability.base_damage[star_idx]
        splash_damage = custom.get("splash_damage", [44, 66, 680])[star_idx]
        execute_damage = custom.get("execute_damage", [220, 330, 3400])[star_idx]
        max_casts = custom.get("max_casts", 22)

        # 시전 횟수 추적
        if not hasattr(caster, '_zaahen_cast_count'):
            caster._zaahen_cast_count = 0

        caster._zaahen_cast_count += 1
        cast_count = caster._zaahen_cast_count

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        if not enemies:
            return result

        # 타겟 선택
        target = all_units.get(caster.current_target_id) if caster.current_target_id else None
        if not target or not target.is_targetable:
            target = enemies[0]

        target_pos = self.ability_system.grid.get_unit_position(target.id)

        from src.combat.combat_unit import UnitState

        # 22회 도달 시 즉사
        if cast_count >= max_casts:
            caster._zaahen_cast_count = 0  # 리셋

            # 타겟 즉사
            target.stats.current_hp = 0
            target.state = UnitState.DEAD
            result.kills += 1
            caster.kills += 1
            result.targets_hit.append(target.id)

            # 3헥스 AOE
            if target_pos:
                for unit in enemies:
                    if unit.id == target.id:
                        continue
                    unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                    if unit_pos and target_pos.distance_to(unit_pos) <= 3:
                        actual = unit.take_damage(execute_damage, "physical")
                        result.total_damage += actual
                        caster.total_damage_dealt += actual
                        result.targets_hit.append(unit.id)
                        if not unit.is_alive:
                            result.kills += 1
                            caster.kills += 1

            return result

        # 일반 슬래시
        actual = target.take_damage(main_damage, "physical")
        result.total_damage += actual
        caster.total_damage_dealt += actual
        result.targets_hit.append(target.id)

        # 주변 적 스플래시
        if target_pos:
            for unit in enemies:
                if unit.id == target.id:
                    continue
                unit_pos = self.ability_system.grid.get_unit_position(unit.id)
                if unit_pos and target_pos.distance_to(unit_pos) <= 1:
                    splash_actual = unit.take_damage(splash_damage, "physical")
                    result.total_damage += splash_actual
                    caster.total_damage_dealt += splash_actual
                    result.targets_hit.append(unit.id)

        # 타겟이 살아있으면 마나 즉시 회복 (재시전 유도)
        if target.is_alive:
            caster.stats.current_mana = caster.stats.max_mana
        else:
            result.kills += 1
            caster.kills += 1
            caster._zaahen_cast_count = 0  # 킬 시 리셋

        return result

    def handle_brock_seismic_slam(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """
        Brock: Seismic Slam.
        - CC 면역 (패시브)
        - 전체 적에게 피해 (거리에 따라 감소, 최소 40%)
        - 첫 피격 시 넉업
        - 12개 돌 낙하
        """
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        base_slam_damage = ability.base_damage[star_idx]
        hp_percent_damage = custom.get("hp_percent_damage", 0.02)
        rock_damage = custom.get("rock_damage", [250, 375, 600])[star_idx]
        num_rocks = custom.get("num_rocks", 12)
        min_damage_percent = custom.get("min_damage_percent", 0.40)
        reduction_per_hex = custom.get("reduction_per_hex", 0.15)

        caster_pos = self.ability_system.grid.get_unit_position(caster.id)
        enemies = [u for u in all_units.values()
                   if u.team != caster.team and u.is_targetable]

        if not enemies:
            return result

        # 첫 슬램 추적
        if not hasattr(caster, '_brock_slam_targets'):
            caster._brock_slam_targets = set()

        # 전체 적에게 슬램
        for unit in enemies:
            unit_pos = self.ability_system.grid.get_unit_position(unit.id)

            # 거리에 따른 피해 감소
            if unit_pos and caster_pos:
                distance = caster_pos.distance_to(unit_pos)
                damage_mult = max(min_damage_percent, 1.0 - (distance * reduction_per_hex))
            else:
                damage_mult = 1.0

            # HP% 피해 + 기본 피해
            total_damage = (caster.stats.max_hp * hp_percent_damage + base_slam_damage) * damage_mult
            actual = unit.take_damage(total_damage, "physical")
            result.total_damage += actual
            caster.total_damage_dealt += actual
            result.targets_hit.append(unit.id)

            # 첫 피격 시 넉업
            if unit.id not in caster._brock_slam_targets:
                caster._brock_slam_targets.add(unit.id)
                if self.ability_system.status_effect_system:
                    stun = StatusEffect(
                        effect_type=StatusEffectType.STUN,
                        source_id=caster.id,
                        duration=1.75,
                        value=0,
                    )
                    self.ability_system.status_effect_system.apply_effect(unit, stun)

            if not unit.is_alive:
                result.kills += 1
                caster.kills += 1

        # 12개 돌 낙하 (랜덤 타겟)
        living_enemies = [u for u in enemies if u.is_alive]
        import random
        for i in range(num_rocks):
            if not living_enemies:
                break
            # 랜덤 타겟 선택
            target = random.choice(living_enemies)
            actual = target.take_damage(rock_damage, "physical")
            result.total_damage += actual
            caster.total_damage_dealt += actual
            if not target.is_alive:
                result.kills += 1
                caster.kills += 1
                living_enemies = [u for u in living_enemies if u.is_alive]

        return result

    # =========================================================================
    # SET 16 MISSING HANDLERS (Dr. Mundo, Mel, Milio)
    # =========================================================================

    def handle_dr_mundo_maximum_dosage(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Dr. Mundo: Inject himself with chemicals, healing and gaining AD over time."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
            targets_hit=[caster.id],
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # HP% 힐량
        hp_percent_heal = custom.get("hp_percent_heal", [0.20, 0.25, 0.80])[star_idx]
        duration = custom.get("duration", 4.0)
        ad_bonus_percent = custom.get("ad_bonus_percent", [0.15, 0.20, 0.50])[star_idx]

        # 총 힐량 계산 (max HP 기준)
        total_heal = caster.stats.max_hp * hp_percent_heal
        heal_per_sec = total_heal / duration

        # Heal over time 효과 적용
        hot_effect = StatusEffect(
            effect_type=StatusEffectType.HEAL_OVER_TIME,
            source_id=caster.id,
            duration=duration,
            value=heal_per_sec,
            tick_interval=1.0,
        )

        # AD 버프 적용
        ad_bonus = caster.stats.attack_damage * ad_bonus_percent
        ad_effect = StatusEffect(
            effect_type=StatusEffectType.ATTACK_DAMAGE_BUFF,
            source_id=caster.id,
            duration=duration,
            value=ad_bonus,
        )

        if self.ability_system.status_effect_system:
            self.ability_system.status_effect_system.apply_effect(caster, hot_effect)
            self.ability_system.status_effect_system.apply_effect(caster, ad_effect)

        result.total_healing = total_heal

        return result

    def handle_mel_councils_blessing(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Mel: Shield all allies and grant AP buff."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # 쉴드량 (Mel의 AP 스케일링)
        base_shield = custom.get("base_shield", [200, 300, 1000])[star_idx]
        ap_scaling = custom.get("ap_scaling", 0.4)
        shield_amount = base_shield + (caster.stats.ability_power * ap_scaling)

        # AP 버프량
        ap_buff = custom.get("ap_buff", [10, 15, 40])[star_idx]
        duration = custom.get("duration", 4.0)

        # 모든 아군에게 적용
        for unit in all_units.values():
            if unit.team == caster.team and unit.is_alive:
                # 쉴드 적용
                shield_effect = StatusEffect(
                    effect_type=StatusEffectType.SHIELD,
                    source_id=caster.id,
                    duration=duration,
                    value=shield_amount,
                )

                # AP 버프 적용
                ap_effect = StatusEffect(
                    effect_type=StatusEffectType.ABILITY_POWER_BUFF,
                    source_id=caster.id,
                    duration=duration,
                    value=ap_buff,
                )

                if self.ability_system.status_effect_system:
                    self.ability_system.status_effect_system.apply_effect(unit, shield_effect)
                    self.ability_system.status_effect_system.apply_effect(unit, ap_effect)

                result.targets_hit.append(unit.id)

        return result

    def handle_milio_breath_of_life(
        self,
        caster: "CombatUnit",
        ability: AbilityData,
        all_units: Dict[str, "CombatUnit"],
    ) -> AbilityResult:
        """Milio: Heal all allies and cleanse CC effects."""
        result = AbilityResult(
            success=True,
            ability_name=ability.name,
            caster_id=caster.id,
        )

        star_idx = min(caster.star_level - 1, 2)
        custom = ability.custom_data

        # 힐량 (Milio의 AP 스케일링)
        base_heal = custom.get("base_heal", [150, 225, 800])[star_idx]
        ap_scaling = custom.get("ap_scaling", 0.35)
        heal_amount = base_heal + (caster.stats.ability_power * ap_scaling)

        # 모든 아군에게 적용
        for unit in all_units.values():
            if unit.team == caster.team and unit.is_alive:
                # CC 해제
                if self.ability_system.status_effect_system:
                    # 스턴, 침묵, 루트 등 CC 효과 제거
                    cc_types = [
                        StatusEffectType.STUN,
                        StatusEffectType.SILENCE,
                        StatusEffectType.ROOT,
                        StatusEffectType.DISARM,
                        StatusEffectType.KNOCKUP,
                    ]
                    for cc_type in cc_types:
                        self.ability_system.status_effect_system.remove_effect(unit, cc_type)

                # 힐 적용
                actual_heal = unit.heal(heal_amount)
                result.total_healing += actual_heal
                result.targets_hit.append(unit.id)

        return result
