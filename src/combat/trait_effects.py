"""Trait Effects System for TFT Combat.

Handles special effects from traits during combat:
- Combat start effects (Warden shields, Juggernaut setup, etc.)
- On-kill effects (Bilgewater serpents, Shadow Isles souls, etc.)
- On-damage effects (Demacia rally, Noxus Atakhan, etc.)
- Periodic effects (trait-based regeneration, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .combat_unit import CombatUnit
    from .status_effects import StatusEffectSystem
    from src.core.unique_traits import UniqueTraitHandler
    from src.core.synergy_calculator import ActiveTrait


@dataclass
class TraitEffectContext:
    """Context for trait effect execution."""
    status_effects: "StatusEffectSystem"
    all_units: Dict[str, "CombatUnit"]
    tick_duration: float = 0.033
    grid: Any = None  # HexGrid reference for distance calculations


class TraitEffectSystem:
    """
    Manages trait effects during combat.

    This system tracks active traits and their handlers,
    applying effects at appropriate combat events.
    """

    def __init__(self):
        """Initialize the trait effect system."""
        # Track trait handlers per team {team: {trait_id: handler}}
        self.team_handlers: Dict[str, Dict[str, "UniqueTraitHandler"]] = {
            "blue": {},
            "red": {},
        }

        # Track active traits per team {team: {trait_id: ActiveTrait}}
        self.team_active_traits: Dict[str, Dict[str, "ActiveTrait"]] = {
            "blue": {},
            "red": {},
        }

        # Track team health for rally effects
        self.team_health: Dict[str, Dict[str, float]] = {
            "blue": {"current": 0, "max": 0},
            "red": {"current": 0, "max": 0},
        }

        # Track combat stats
        self.combat_stats: Dict[str, Dict[str, Any]] = {}

    def setup_team_traits(
        self,
        team: str,
        active_traits: Dict[str, "ActiveTrait"],
        units: List["CombatUnit"],
    ) -> None:
        """
        Set up trait handlers for a team at combat start.

        Args:
            team: Team identifier ("blue" or "red")
            active_traits: Dictionary of active traits for the team
            units: List of CombatUnits on this team
        """
        from src.core.unique_traits import get_handler

        self.team_active_traits[team] = active_traits.copy()
        self.team_handlers[team] = {}

        # Initialize handlers for each active trait
        for trait_id, active_trait in active_traits.items():
            if not active_trait.is_active:
                continue

            handler = get_handler(trait_id)
            if handler:
                self.team_handlers[team][trait_id] = handler

                # Get champions with this trait
                trait_champions = self._get_trait_champions(units, trait_id)

                # Apply initial setup
                handler.apply(None, trait_champions)

        # Calculate initial team health
        total_hp = sum(u.stats.max_hp for u in units)
        self.team_health[team] = {"current": total_hp, "max": total_hp}

    def _get_trait_champions(
        self,
        units: List["CombatUnit"],
        trait_id: str,
    ) -> List[Any]:
        """Get champion instances that have a specific trait."""
        result = []
        for unit in units:
            if unit.source_instance:
                champion = unit.source_instance.champion
                # Check if champion has this trait
                if hasattr(champion, "traits"):
                    if trait_id in champion.traits:
                        result.append(unit.source_instance)
        return result

    def apply_combat_start_effects(
        self,
        team: str,
        units: List["CombatUnit"],
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects at combat start.

        Args:
            team: Team identifier
            units: Team's combat units
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []

        handlers = self.team_handlers.get(team, {})

        # Apply Warden shields
        if "warden" in handlers:
            warden_handler = handlers["warden"]
            if hasattr(warden_handler, "shield_amount") and warden_handler.shield_amount > 0:
                warden_units = self._get_units_with_trait(units, "warden")
                for unit in warden_units:
                    from .status_effects import StatusEffectType
                    context.status_effects.add_effect(
                        unit,
                        StatusEffectType.SHIELD,
                        warden_handler.shield_amount,
                        duration=float("inf"),
                    )
                    events.append({
                        "type": "trait_shield",
                        "trait": "warden",
                        "unit": unit.id,
                        "amount": warden_handler.shield_amount,
                    })

        # Apply Ionia path if not set
        if "ionia" in handlers:
            ionia_handler = handlers["ionia"]
            if ionia_handler.current_path is None:
                ionia_handler.roll_path()
                events.append({
                    "type": "trait_ionia_path",
                    "trait": "ionia",
                    "path": ionia_handler.current_path,
                })

        # Apply Quickstriker attack speed buff at combat start
        if "quickstriker" in handlers:
            quickstriker_handler = handlers["quickstriker"]
            bonus = quickstriker_handler.get_bonus(None, [])
            as_bonus = bonus.get("attack_speed", 0) / 100  # Convert to decimal
            if as_bonus > 0:
                quickstriker_units = self._get_units_with_trait(units, "quickstriker")
                for unit in quickstriker_units:
                    # Apply as combat start AS buff (first 10 seconds)
                    from .status_effects import StatusEffect, StatusEffectType
                    as_effect = StatusEffect(
                        effect_type=StatusEffectType.ATTACK_SPEED_BUFF,
                        source_id="quickstriker_trait",
                        duration=10.0,
                        value=as_bonus,
                    )
                    context.status_effects.apply_effect(unit, as_effect)
                    events.append({
                        "type": "trait_quickstriker_as",
                        "trait": "quickstriker",
                        "unit": unit.id,
                        "attack_speed": as_bonus,
                    })

        # Apply Bruiser HP bonus
        if "bruiser" in handlers:
            bruiser_handler = handlers["bruiser"]
            bonus = bruiser_handler.get_bonus(None, [])
            hp_percent = bonus.get("hp_percent", 0) / 100  # Convert to decimal
            if hp_percent > 0:
                for unit in units:
                    bonus_hp = unit.stats.max_hp * hp_percent
                    unit.stats.max_hp += bonus_hp
                    unit.stats.current_hp += bonus_hp
                    events.append({
                        "type": "trait_bruiser_hp",
                        "trait": "bruiser",
                        "unit": unit.id,
                        "bonus_hp": bonus_hp,
                    })

        # Apply Defender armor/MR bonus
        if "defender" in handlers:
            defender_handler = handlers["defender"]
            bonus = defender_handler.get_bonus(None, [])
            armor = bonus.get("armor", 0)
            mr = bonus.get("magic_resist", 0)
            if armor > 0 or mr > 0:
                for unit in units:
                    unit.stats.armor += armor
                    unit.stats.magic_resist += mr
                    events.append({
                        "type": "trait_defender_defense",
                        "trait": "defender",
                        "unit": unit.id,
                        "armor": armor,
                        "magic_resist": mr,
                    })

        # Apply Arcanist AP bonus
        if "arcanist" in handlers:
            arcanist_handler = handlers["arcanist"]
            bonus = arcanist_handler.get_bonus(None, [])
            ap = bonus.get("ability_power", 0)
            if ap > 0:
                for unit in units:
                    unit.stats.ability_power += ap
                    events.append({
                        "type": "trait_arcanist_ap",
                        "trait": "arcanist",
                        "unit": unit.id,
                        "ability_power": ap,
                    })

        return events

    def _get_units_with_trait(
        self,
        units: List["CombatUnit"],
        trait_id: str,
    ) -> List["CombatUnit"]:
        """Get combat units that have a specific trait."""
        result = []
        for unit in units:
            if unit.source_instance:
                champion = unit.source_instance.champion
                if hasattr(champion, "traits") and trait_id in champion.traits:
                    result.append(unit)
        return result

    def apply_on_damage_effects(
        self,
        team: str,
        units: List["CombatUnit"],
        damage_amount: float,
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects when the team takes damage.

        Args:
            team: Team that took damage
            units: Team's combat units
            damage_amount: Amount of damage taken
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []

        # Update team health
        team_hp = self.team_health.get(team)
        if team_hp:
            team_hp["current"] = max(0, team_hp["current"] - damage_amount)
            current_percent = (team_hp["current"] / team_hp["max"]) * 100 if team_hp["max"] > 0 else 0

            # Check Demacia rally
            handlers = self.team_handlers.get(team, {})
            if "demacia" in handlers:
                demacia_handler = handlers["demacia"]
                if demacia_handler.on_health_lost(current_percent):
                    events.append({
                        "type": "trait_rally",
                        "trait": "demacia",
                        "rally_count": demacia_handler.rally_count,
                    })

        return events

    def apply_on_kill_effects(
        self,
        killer: "CombatUnit",
        victim: "CombatUnit",
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects when a unit kills another.

        Args:
            killer: Unit that got the kill
            victim: Unit that was killed
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []
        team = "blue" if killer.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        # Bilgewater - earn Silver Serpents
        if "bilgewater" in handlers:
            bilgewater_handler = handlers["bilgewater"]
            bilgewater_handler.on_kill()
            events.append({
                "type": "trait_bilgewater_serpent",
                "trait": "bilgewater",
                "killer": killer.id,
                "serpents": bilgewater_handler.silver_serpents,
            })

        # Void - Voracious mutation AS bonus
        if "void" in handlers:
            void_handler = handlers["void"]
            mutation = void_handler.get_champion_mutation(killer.source_instance) if killer.source_instance else None
            if mutation == "voracious":
                # Grant attack speed on kill
                bonus_as = void_handler.MUTATIONS["voracious"]["attack_speed_on_kill"] / 100
                killer.stats.attack_speed *= (1 + bonus_as)
                events.append({
                    "type": "trait_void_voracious",
                    "trait": "void",
                    "unit": killer.id,
                    "attack_speed_bonus": bonus_as,
                })

        # Shadow Isles - collect souls
        both_teams = ["blue", "red"]
        for check_team in both_teams:
            team_handlers = self.team_handlers.get(check_team, {})
            if "shadow_isles" in team_handlers:
                shadow_handler = team_handlers["shadow_isles"]
                is_ally = (check_team == "blue" and victim.team.value == "blue") or \
                          (check_team == "red" and victim.team.value == "red")
                shadow_handler.on_death(is_ally)
                events.append({
                    "type": "trait_shadow_isles_soul",
                    "trait": "shadow_isles",
                    "team": check_team,
                    "souls": shadow_handler.souls,
                })

        return events

    def apply_on_death_effects(
        self,
        dead_unit: "CombatUnit",
        all_units: List["CombatUnit"],
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects when a unit dies.

        Args:
            dead_unit: Unit that died
            all_units: All combat units
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []
        team = "blue" if dead_unit.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        # Juggernaut - heal remaining Juggernauts on ally death
        if "juggernaut" in handlers:
            juggernaut_handler = handlers["juggernaut"]
            if juggernaut_handler.heal_amount > 0:
                # Heal remaining Juggernaut allies
                alive_units = [u for u in all_units if u.is_alive and u.team == dead_unit.team]
                juggernaut_units = self._get_units_with_trait(alive_units, "juggernaut")

                for unit in juggernaut_units:
                    healed = unit.heal(juggernaut_handler.heal_amount)
                    if healed > 0:
                        events.append({
                            "type": "trait_juggernaut_heal",
                            "trait": "juggernaut",
                            "unit": unit.id,
                            "heal": healed,
                        })

        return events

    def apply_tick_effects(
        self,
        units: List["CombatUnit"],
        tick: int,
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects every combat tick.

        Args:
            units: All combat units
            tick: Current tick number
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []

        # Process each team
        for team in ["blue", "red"]:
            team_units = [u for u in units if u.team.value == team and u.is_alive]
            handlers = self.team_handlers.get(team, {})

            # Invoker mana regen
            if "invoker" in handlers:
                invoker_handler = handlers["invoker"]
                bonus = invoker_handler.get_bonus(None, [])
                mana_regen = bonus.get("mana_regen", 0)
                if mana_regen > 0:
                    # Apply mana regen every second (30 ticks)
                    if tick % 30 == 0:
                        for unit in team_units:
                            unit.gain_mana(mana_regen)

        return events

    def apply_on_ability_cast(
        self,
        caster: "CombatUnit",
        targets: List["CombatUnit"],
        context: TraitEffectContext,
    ) -> List[Dict[str, Any]]:
        """
        Apply trait effects when a unit casts an ability.

        Args:
            caster: Unit that cast the ability
            targets: Units hit by the ability
            context: Trait effect context

        Returns:
            List of effect events
        """
        events = []
        team = "blue" if caster.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        # Disruptor - apply Dazzle to targets
        if "disruptor" in handlers and caster.source_instance:
            champion = caster.source_instance.champion
            if hasattr(champion, "traits") and "disruptor" in champion.traits:
                from .status_effects import StatusEffect, StatusEffectType
                dazzle_duration = 1.5  # Dazzle lasts 1.5 seconds

                for target in targets:
                    if target.team != caster.team and target.is_alive:
                        # Dazzle = can't attack (DISARM)
                        dazzle_effect = StatusEffect(
                            effect_type=StatusEffectType.DISARM,
                            source_id=f"disruptor_{caster.id}",
                            duration=dazzle_duration,
                        )
                        context.status_effects.apply_effect(target, dazzle_effect)
                        events.append({
                            "type": "trait_disruptor_dazzle",
                            "trait": "disruptor",
                            "caster": caster.id,
                            "target": target.id,
                            "duration": dazzle_duration,
                        })

        # Vanquisher - ability can crit
        if "vanquisher" in handlers and caster.source_instance:
            champion = caster.source_instance.champion
            if hasattr(champion, "traits") and "vanquisher" in champion.traits:
                vanquisher_handler = handlers["vanquisher"]
                bonus = vanquisher_handler.get_bonus(None, [])
                crit_chance = bonus.get("crit_chance", 0) / 100  # Convert to decimal
                # Note: actual crit application happens in ability damage calculation
                events.append({
                    "type": "trait_vanquisher_crit_chance",
                    "trait": "vanquisher",
                    "caster": caster.id,
                    "crit_chance": crit_chance,
                })

        return events

    def get_huntress_leap_target(
        self,
        unit: "CombatUnit",
        enemy_units: List["CombatUnit"],
        context: TraitEffectContext,
    ) -> Optional["CombatUnit"]:
        """
        Get the leap target for Huntress trait.
        Huntress units leap to lowest health enemy at combat start.

        Args:
            unit: Huntress unit
            enemy_units: List of enemy units
            context: Trait effect context

        Returns:
            Target unit to leap to, or None if no valid target
        """
        team = "blue" if unit.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        if "huntress" not in handlers:
            return None

        if not unit.source_instance:
            return None

        champion = unit.source_instance.champion
        if not hasattr(champion, "traits") or "huntress" not in champion.traits:
            return None

        # Find lowest HP enemy
        alive_enemies = [e for e in enemy_units if e.is_alive and e.is_targetable]
        if not alive_enemies:
            return None

        return min(alive_enemies, key=lambda e: e.stats.current_hp)

    def should_huntress_leap(
        self,
        unit: "CombatUnit",
    ) -> bool:
        """Check if a unit should perform Huntress leap."""
        team = "blue" if unit.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        if "huntress" not in handlers:
            return False

        if not unit.source_instance:
            return False

        champion = unit.source_instance.champion
        return hasattr(champion, "traits") and "huntress" in champion.traits

    def get_damage_modifier(
        self,
        attacker: "CombatUnit",
        target: "CombatUnit",
        context: Optional[TraitEffectContext] = None,
    ) -> float:
        """
        Get damage modifier from traits.

        Args:
            attacker: Attacking unit
            target: Target unit
            context: Trait effect context with grid reference

        Returns:
            Damage multiplier (1.0 = no modification)
        """
        multiplier = 1.0
        team = "blue" if attacker.team.value == "blue" else "red"
        handlers = self.team_handlers.get(team, {})

        # Slayer - bonus damage based on missing health
        if "slayer" in handlers and attacker.source_instance:
            champion = attacker.source_instance.champion
            if hasattr(champion, "traits") and "slayer" in champion.traits:
                # Bonus damage scales with attacker's missing health
                missing_hp_percent = 1 - (attacker.stats.current_hp / attacker.stats.max_hp)
                # Up to 50% bonus damage at low health
                multiplier += missing_hp_percent * 0.5

        # Longshot - bonus damage based on distance
        if "longshot" in handlers and attacker.source_instance:
            champion = attacker.source_instance.champion
            if hasattr(champion, "traits") and "longshot" in champion.traits:
                longshot_handler = handlers["longshot"]
                bonus = longshot_handler.get_bonus(None, [])
                base_damage_amp = bonus.get("damage_amp", 0) / 100  # Convert to decimal

                # Calculate distance bonus if grid is available
                if context and context.grid:
                    attacker_pos = context.grid.get_unit_position(attacker.id)
                    target_pos = context.grid.get_unit_position(target.id)
                    if attacker_pos and target_pos:
                        distance = attacker_pos.distance_to(target_pos)
                        # Longshot: +8% damage per hex distance (max 5 hexes = +40%)
                        distance_bonus = min(distance, 5) * 0.08
                        multiplier += base_damage_amp + distance_bonus

        # Disruptor - bonus damage to dazzled targets
        if "disruptor" in handlers:
            disruptor_handler = handlers["disruptor"]
            bonus = disruptor_handler.get_bonus(None, [])
            dazzle_damage = bonus.get("dazzle_damage", 0)
            if dazzle_damage > 0 and context and context.status_effects:
                # Check if target is dazzled
                from .status_effects import StatusEffectType
                if context.status_effects.has_effect(target, StatusEffectType.DISARM):
                    # Dazzle = DISARM in our status effects (prevents attacking)
                    multiplier += dazzle_damage / 100

        # Huntress - bonus damage to low health targets
        if "huntress" in handlers and attacker.source_instance:
            champion = attacker.source_instance.champion
            if hasattr(champion, "traits") and "huntress" in champion.traits:
                huntress_handler = handlers["huntress"]
                bonus = huntress_handler.get_bonus(None, [])
                damage_amp = bonus.get("damage_amp", 0) / 100  # Convert to decimal
                multiplier += damage_amp

        return multiplier

    def get_trait_bonuses(
        self,
        unit: "CombatUnit",
        team: str,
    ) -> Dict[str, float]:
        """
        Get stat bonuses from traits for a specific unit.

        Args:
            unit: The combat unit
            team: Team identifier

        Returns:
            Dictionary of stat bonuses
        """
        bonuses: Dict[str, float] = {}
        handlers = self.team_handlers.get(team, {})
        active_traits = self.team_active_traits.get(team, {})

        if not unit.source_instance:
            return bonuses

        champion = unit.source_instance.champion
        champion_traits = getattr(champion, "traits", [])

        for trait_id, handler in handlers.items():
            if trait_id not in active_traits:
                continue

            active_trait = active_traits[trait_id]
            if not active_trait.is_active:
                continue

            # Get champions with this trait for the handler
            all_units_team = [u for u in self.combat_stats.get("units", []) if u.team.value == team]
            trait_champions = self._get_trait_champions(all_units_team, trait_id)

            # Get bonuses from handler
            handler_bonuses = handler.get_bonus(None, trait_champions)

            # Apply bonuses based on whether this champion has the trait
            for stat, value in handler_bonuses.items():
                if trait_id in champion_traits:
                    # Champion has this trait - gets full bonus
                    bonuses[stat] = bonuses.get(stat, 0) + value
                else:
                    # Some traits give reduced bonuses to non-trait champions
                    # (e.g., Arcanist gives AP to all, but more to Arcanists)
                    pass

        return bonuses

    def check_noxus_atakhan(
        self,
        team: str,
        enemy_damage_percent: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if Noxus should summon Atakhan.

        Args:
            team: Team with Noxus trait
            enemy_damage_percent: Percent of enemy health lost

        Returns:
            Atakhan stats if should be summoned, None otherwise
        """
        handlers = self.team_handlers.get(team, {})

        if "noxus" in handlers:
            noxus_handler = handlers["noxus"]
            if noxus_handler.on_enemy_damage(enemy_damage_percent):
                # Get Noxus champions for power calculation
                # Note: Would need actual champion instances here
                return noxus_handler.calculate_atakhan_power([])

        return None

    def reset_handlers(self) -> None:
        """Reset all trait handlers for new combat."""
        for team in ["blue", "red"]:
            for trait_id, handler in self.team_handlers.get(team, {}).items():
                if hasattr(handler, "reset"):
                    handler.reset()

        self.team_health = {
            "blue": {"current": 0, "max": 0},
            "red": {"current": 0, "max": 0},
        }

    def clear(self) -> None:
        """Clear all trait effect state."""
        self.team_handlers = {"blue": {}, "red": {}}
        self.team_active_traits = {"blue": {}, "red": {}}
        self.team_health = {
            "blue": {"current": 0, "max": 0},
            "red": {"current": 0, "max": 0},
        }
        self.combat_stats = {}
