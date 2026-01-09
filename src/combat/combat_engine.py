"""Combat Engine for TFT.

The main combat simulation loop that orchestrates all combat systems:
- Unit placement and initialization
- Combat loop (targeting, movement, attacking, abilities)
- Win condition checking
- Result calculation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum, auto
import random

from .hex_grid import HexGrid, HexPosition, Team
from .combat_unit import CombatUnit, CombatStats, CombatResult, UnitState
from .targeting import TargetSelector, TargetingContext, TargetingPriority
from .pathfinding import PathFinder, get_blocked_positions
from .movement import MovementSystem, MOVE_TIME_PER_HEX
from .attack import AttackSystem, DamageEvent, AttackResult
from .ability import AbilitySystem, AbilityData
from .status_effects import StatusEffectSystem
from .item_effects import ItemEffectSystem, ItemEffectContext
from .trait_effects import TraitEffectSystem, TraitEffectContext
from .champion_abilities import register_all_abilities
from src.core.constants import BASE_STAGE_DAMAGE, UNIT_DAMAGE_BY_STAR_AND_COST
from src.core.augment_effects import AugmentEffectSystem, get_augment_effect_system

if TYPE_CHECKING:
    from ..core.player_units import ChampionInstance
    from ..core.game_state import PlayerState


# Combat constants
TICK_RATE = 30  # Updates per second
TICK_DURATION = 1.0 / TICK_RATE  # ~0.033 seconds
MAX_COMBAT_TICKS = 30 * TICK_RATE  # 30 second timeout (TFT standard)


class CombatPhase(Enum):
    """Combat phases."""

    SETUP = auto()  # Pre-combat setup
    RUNNING = auto()  # Combat in progress
    FINISHED = auto()  # Combat complete


@dataclass
class CombatState:
    """Current state of combat."""

    phase: CombatPhase = CombatPhase.SETUP
    current_tick: int = 0
    elapsed_time: float = 0.0

    # Unit tracking
    blue_units_alive: int = 0
    red_units_alive: int = 0

    # Combat events log
    events: List[Dict[str, Any]] = field(default_factory=list)


class CombatEngine:
    """
    Main combat simulation engine.

    Orchestrates all combat systems to simulate TFT combat.

    Usage:
        engine = CombatEngine()
        engine.setup_combat(blue_units, red_units, blue_positions, red_positions)
        result = engine.run_combat()

    Or step-by-step:
        engine.setup_combat(...)
        while not engine.is_finished():
            engine.tick()
        result = engine.get_result()
    """

    def __init__(self, seed: Optional[int] = None, stage: int = 2):
        """
        Initialize combat engine.

        Args:
            seed: Random seed for deterministic simulation.
            stage: Current game stage (for player damage calculation).
        """
        self.rng = random.Random(seed)
        self.stage = stage  # For player damage calculation

        # Combat grid
        self.grid = HexGrid()

        # Combat units
        self.units: Dict[str, CombatUnit] = {}

        # Cached team-separated unit lists (performance optimization)
        self._blue_units: List[CombatUnit] = []
        self._red_units: List[CombatUnit] = []
        self._units_cache_dirty: bool = True

        # Player references for augment effects
        self.blue_player: Optional["PlayerState"] = None
        self.red_player: Optional["PlayerState"] = None

        # Augment effect system
        self.augment_effects = get_augment_effect_system()

        # Trait effect system
        self.trait_effects = TraitEffectSystem()

        # Subsystems
        self._init_subsystems()

        # State
        self.state = CombatState()

    def _init_subsystems(self) -> None:
        """Initialize all combat subsystems."""
        # Targeting
        self.targeting_context = TargetingContext(
            grid=self.grid,
            all_units=self.units,
        )
        self.target_selector = TargetSelector(self.targeting_context, self.rng)

        # Movement
        self.movement = MovementSystem(self.grid)

        # Attack
        self.attack_system = AttackSystem(self.rng)

        # Status effects (initialized before AbilitySystem needs it)
        self.status_effects = StatusEffectSystem()

        # Item effects
        self.item_effects = ItemEffectSystem()
        self.item_effect_context = ItemEffectContext(
            attack_system=self.attack_system,
            status_effects=self.status_effects,
            target_selector=self.target_selector,
            all_units=self.units,
            tick_duration=TICK_DURATION,
        )

        # Trait effects context
        self.trait_effect_context = TraitEffectContext(
            status_effects=self.status_effects,
            all_units=self.units,
            tick_duration=TICK_DURATION,
        )

        # Abilities
        self.ability_system = AbilitySystem(
            self.grid,
            self.target_selector,
            self.attack_system,
            self.status_effects,
        )

        # Register all champion abilities
        register_all_abilities(self.ability_system)

        # Register custom ability effect handlers
        from .ability_effects import AbilityEffectHandlers
        self.ability_effect_handlers = AbilityEffectHandlers(self.ability_system)
        self.ability_effect_handlers.register_all_handlers()

    def setup_combat(
        self,
        blue_instances: List["ChampionInstance"],
        red_instances: List["ChampionInstance"],
        blue_positions: List[HexPosition],
        red_positions: List[HexPosition],
        stat_calculator: Any = None,
    ) -> None:
        """
        Setup combat with units from both teams.

        Args:
            blue_instances: Blue team champion instances.
            red_instances: Red team champion instances.
            blue_positions: Positions for blue team (rows 0-3).
            red_positions: Positions for red team (will be mirrored to rows 4-7).
            stat_calculator: Optional stat calculator for computing final stats.
        """
        self._clear()

        # Place blue team units
        for i, instance in enumerate(blue_instances):
            if i < len(blue_positions):
                pos = blue_positions[i]
                self._add_unit(instance, Team.BLUE, pos, stat_calculator)

        # Place red team units (mirror positions)
        for i, instance in enumerate(red_instances):
            if i < len(red_positions):
                # Mirror the position for red team
                original_pos = red_positions[i]
                mirrored_pos = HexGrid.mirror_position(original_pos)
                self._add_unit(instance, Team.RED, mirrored_pos, stat_calculator)

        # Rebuild unit cache after setup
        self._units_cache_dirty = True
        self._rebuild_unit_cache()

        # Update counts using cached lists
        self.state.blue_units_alive = len(self._blue_units)
        self.state.red_units_alive = len(self._red_units)

        # Apply combat start item effects (RFC range, Blue Buff mana, Quicksilver, etc.)
        for unit in self.units.values():
            events = self.item_effects.apply_combat_start_effects(unit, self.item_effect_context)
            self.state.events.extend([{"tick": 0, **e} for e in events])

        # Apply combat start augment effects
        self._apply_augment_combat_start()

        # Apply combat start trait effects
        self._apply_trait_combat_start()

        self.state.phase = CombatPhase.RUNNING

    def setup_combat_from_boards(
        self,
        blue_board: Dict[HexPosition, "ChampionInstance"],
        red_board: Dict[HexPosition, "ChampionInstance"],
        stat_calculator: Any = None,
        blue_player: Optional["PlayerState"] = None,
        red_player: Optional["PlayerState"] = None,
    ) -> None:
        """
        Setup combat from board dictionaries.

        Args:
            blue_board: Blue team {position: instance} mapping.
            red_board: Red team {position: instance} mapping (will be mirrored).
            stat_calculator: Optional stat calculator.
            blue_player: Optional PlayerState for blue team augments.
            red_player: Optional PlayerState for red team augments.
        """
        # Store player references for augment effects
        self.blue_player = blue_player
        self.red_player = red_player

        blue_instances = list(blue_board.values())
        blue_positions = list(blue_board.keys())
        red_instances = list(red_board.values())
        red_positions = list(red_board.keys())

        self.setup_combat(
            blue_instances,
            red_instances,
            blue_positions,
            red_positions,
            stat_calculator,
        )

    def _add_unit(
        self,
        instance: "ChampionInstance",
        team: Team,
        position: HexPosition,
        stat_calculator: Any = None,
    ) -> Optional[CombatUnit]:
        """Add a unit to combat."""
        # Calculate final stats
        if stat_calculator:
            calculated_stats = stat_calculator.calculate(instance, {})
        else:
            calculated_stats = self._get_default_stats(instance)

        # Create combat unit
        combat_unit = CombatUnit.from_champion_instance(
            instance, team, calculated_stats
        )

        # Link status effect system for shield/invulnerability checks
        combat_unit._status_effect_system = self.status_effects

        # Place on grid
        if self.grid.place_unit(combat_unit.id, position):
            self.units[combat_unit.id] = combat_unit

            # Initialize item effects for this unit
            self.item_effects.initialize_unit(combat_unit)

            return combat_unit

        return None

    def _get_default_stats(self, instance: "ChampionInstance") -> Dict[str, float]:
        """Get default stats when no stat calculator provided.

        This uses the ChampionInstance.get_stats() method which includes
        base stats at star level plus all item bonuses.
        """
        # Get stats from ChampionInstance (includes items)
        stats = instance.get_stats()

        return {
            "max_health": stats["health"],
            "attack_damage": stats["attack_damage"],
            "ability_power": stats.get("ability_power", 100.0),
            "armor": stats["armor"],
            "magic_resist": stats["magic_resist"],
            "attack_speed": stats["attack_speed"],
            "crit_chance": stats["crit_chance"],
            "crit_damage": stats["crit_damage"],
            "mana_start": stats["mana_start"],
            "omnivamp": stats.get("omnivamp", 0.0),
        }

    def tick(self) -> bool:
        """
        Execute one combat tick.

        Returns:
            True if combat is still running, False if finished.
        """
        if self.state.phase != CombatPhase.RUNNING:
            return False

        self.state.current_tick += 1
        self.state.elapsed_time += TICK_DURATION

        # Check timeout
        if self.state.current_tick >= MAX_COMBAT_TICKS:
            self._end_combat_timeout()
            return False

        # Update all living units
        living_units = [u for u in self.units.values() if u.is_alive]

        for unit in living_units:
            self._update_unit(unit)

        # Apply augment tick effects (First Aid Kit heal, URF Overtime, etc.)
        self._apply_augment_tick_effects()

        # Apply trait tick effects (Invoker mana regen, etc.)
        self._apply_trait_tick_effects()

        # Update ability effect handler timers (on-hit buffs, etc.)
        self.ability_effect_handlers.update_effects(TICK_DURATION)

        # Check win condition
        winner = self._check_win_condition()
        if winner is not None:
            self._end_combat(winner)
            return False

        return True

    def _update_unit(self, unit: CombatUnit) -> None:
        """Update a single unit for one tick."""
        if not unit.is_alive:
            return

        # Update status effects
        effect_events = self.status_effects.update(unit, TICK_DURATION)
        self.state.events.extend(effect_events)

        # Update periodic item effects (Sunfire, Redemption, Archangel's)
        item_events = self.item_effects.update_periodic_effects(
            unit, TICK_DURATION, self.item_effect_context
        )
        self.state.events.extend([{"tick": self.state.current_tick, **e} for e in item_events])

        # Skip if CC'd
        if not self.status_effects.can_act(unit):
            return

        # Update attack cooldown
        self.attack_system.update_cooldown(unit, TICK_DURATION)

        # Handle casting
        if unit.is_casting:
            result = self.ability_system.update_cast(unit, TICK_DURATION, self.units)
            if result:
                self._log_event("ability_cast", {
                    "caster": unit.id,
                    "ability": result.ability_name,
                    "damage": result.total_damage,
                    "targets": result.targets_hit,
                })
                # Apply on-cast item effects (Blue Buff mana reset)
                on_cast_events = self.item_effects.apply_on_cast_effects(
                    unit, self.item_effect_context
                )
                self.state.events.extend([{"tick": self.state.current_tick, **e} for e in on_cast_events])
            return

        # Acquire target
        target_id = self.target_selector.acquire_target(unit)
        if target_id is None:
            return

        target = self.units.get(target_id)
        if target is None or not target.is_targetable:
            unit.current_target_id = None
            return

        target_pos = self.grid.get_unit_position(target_id)
        if target_pos is None:
            return

        # Check if in attack range
        in_range = self.target_selector.is_in_range(
            unit, target_id, unit.stats.attack_range
        )

        if in_range:
            # Stop movement
            self.movement.stop_movement(unit.id)

            # Try to cast ability if mana full
            if unit.can_cast:
                ability = self.ability_system.get_ability(unit.champion_id)
                if ability:
                    cast = self.ability_system.start_cast(unit, ability, self.units)
                    if cast:
                        self._log_event("ability_start", {
                            "caster": unit.id,
                            "ability": ability.name,
                        })
                        return

            # Try to attack
            if self.attack_system.can_attack(unit):
                # Apply on-attack item effects (Rageblade, Shojin, Statikk counter)
                on_attack_events = self.item_effects.apply_on_attack_effects(
                    unit, target, self.item_effect_context
                )
                self.state.events.extend([{"tick": self.state.current_tick, **e} for e in on_attack_events])

                result = self.attack_system.execute_attack(unit, target)
                if result.success and result.damage_event:
                    # Apply on-hit item effects (Giant Slayer, Statikk proc, Runaan's, Last Whisper)
                    on_hit_events = self.item_effects.apply_on_hit_effects(
                        unit, target, result.damage_event, self.item_effect_context
                    )
                    self.state.events.extend([{"tick": self.state.current_tick, **e} for e in on_hit_events])

                    # Apply ability on-hit effects (Briar heal, etc.)
                    ability_on_hit_heal = self.ability_effect_handlers.process_on_hit(
                        unit, target, result.damage_event.final_damage
                    )
                    if ability_on_hit_heal > 0:
                        self._log_event("ability_on_hit_heal", {
                            "unit": unit.id,
                            "heal": ability_on_hit_heal,
                        })

                    self._log_event("attack", {
                        "attacker": unit.id,
                        "target": target_id,
                        "damage": result.damage_event.final_damage,
                        "crit": result.damage_event.is_critical,
                        "killed": result.target_killed,
                    })

                    # Update alive counts on kill
                    if result.target_killed:
                        # Apply on-kill item effects (Deathblade stacks)
                        on_kill_events = self.item_effects.apply_on_kill_effects(
                            unit, target, self.item_effect_context
                        )
                        self.state.events.extend([{"tick": self.state.current_tick, **e} for e in on_kill_events])

                        # Apply on-kill augment effects (Thrill of the Hunt)
                        self._apply_augment_on_kill(unit, target)

                        # Apply on-kill trait effects (Bilgewater serpents, etc.)
                        self._apply_trait_on_kill(unit, target)

                        # Apply on-death trait effects (Juggernaut heal, etc.)
                        self._apply_trait_on_death(target)

                        if target.team == Team.BLUE:
                            self.state.blue_units_alive -= 1
                        else:
                            self.state.red_units_alive -= 1
        else:
            # Need to move toward target
            if not self.movement.is_moving(unit.id):
                self.movement.start_move_to_target(
                    unit, target_pos, unit.stats.attack_range
                )
            else:
                # Continue movement - just update, don't recalculate every tick
                self.movement.update(unit, TICK_DURATION)

    def _check_win_condition(self) -> Optional[Team]:
        """Check if combat has a winner."""
        blue_alive = any(
            u.is_alive for u in self.units.values() if u.team == Team.BLUE
        )
        red_alive = any(
            u.is_alive for u in self.units.values() if u.team == Team.RED
        )

        if not blue_alive and not red_alive:
            # Draw - shouldn't happen but handle it
            return None
        elif not blue_alive:
            return Team.RED
        elif not red_alive:
            return Team.BLUE

        return None

    def _end_combat(self, winner: Team) -> None:
        """End combat with a winner."""
        self.state.phase = CombatPhase.FINISHED
        self._log_event("combat_end", {
            "winner": winner.value,
            "ticks": self.state.current_tick,
            "time": self.state.elapsed_time,
        })

    def _end_combat_timeout(self) -> None:
        """End combat due to timeout."""
        self.state.phase = CombatPhase.FINISHED
        self._log_event("combat_timeout", {
            "ticks": self.state.current_tick,
        })

    def run_combat(self) -> CombatResult:
        """
        Run combat to completion.

        Returns:
            CombatResult with winner and statistics.
        """
        while self.tick():
            pass

        return self.get_result()

    def get_result(self) -> CombatResult:
        """Get the combat result."""
        # Determine winner
        blue_alive = [u for u in self.units.values() if u.team == Team.BLUE and u.is_alive]
        red_alive = [u for u in self.units.values() if u.team == Team.RED and u.is_alive]

        if blue_alive and not red_alive:
            winner = Team.BLUE
            winning_units = blue_alive
            losing_units = []
        elif red_alive and not blue_alive:
            winner = Team.RED
            winning_units = red_alive
            losing_units = []
        else:
            # Timeout or draw - winner is team with more health
            blue_hp = sum(u.stats.current_hp for u in blue_alive)
            red_hp = sum(u.stats.current_hp for u in red_alive)
            if blue_hp >= red_hp:
                winner = Team.BLUE
                winning_units = blue_alive
                losing_units = red_alive
            else:
                winner = Team.RED
                winning_units = red_alive
                losing_units = blue_alive

        # Calculate player damage
        damage_to_loser = self._calculate_player_damage(winning_units)

        # Compile unit stats
        unit_stats = {}
        for unit in self.units.values():
            unit_stats[unit.id] = {
                "name": unit.name,
                "team": unit.team.value,
                "damage_dealt": unit.total_damage_dealt,
                "damage_taken": unit.total_damage_taken,
                "kills": unit.kills,
                "healing_done": unit.total_healing_done,
                "alive": unit.is_alive,
            }

        return CombatResult(
            winner=winner,
            winning_units_remaining=len(winning_units),
            losing_units_remaining=len(losing_units),
            rounds_taken=self.state.current_tick,
            total_damage_to_loser=damage_to_loser,
            unit_stats=unit_stats,
        )

    def _calculate_player_damage(self, winning_units: List[CombatUnit]) -> float:
        """
        Calculate damage dealt to losing player.

        Formula: Base Stage Damage + Sum of Unit Damages
        Unit damage is based on cost tier and star level.
        """
        if not winning_units:
            return 0

        # Base stage damage
        base_damage = BASE_STAGE_DAMAGE.get(self.stage, BASE_STAGE_DAMAGE.get(7, 17))

        # Sum unit damage based on cost and star level
        unit_damage = 0
        for unit in winning_units:
            star_damage_table = UNIT_DAMAGE_BY_STAR_AND_COST.get(
                unit.star_level, UNIT_DAMAGE_BY_STAR_AND_COST[1]
            )
            unit_damage += star_damage_table.get(unit.cost, unit.cost)

        return float(base_damage + unit_damage)

    def is_finished(self) -> bool:
        """Check if combat is finished."""
        return self.state.phase == CombatPhase.FINISHED

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a combat event."""
        self.state.events.append({
            "tick": self.state.current_tick,
            "type": event_type,
            **data,
        })

    def _clear(self) -> None:
        """Clear all combat state."""
        self.grid.clear()
        self.units.clear()
        self.movement.clear()
        self.ability_system.clear()
        self.status_effects.clear_all()
        self.item_effects.clear()
        self.trait_effects.clear()
        self.ability_effect_handlers.clear()
        self.state = CombatState()

        # Clear cached unit lists
        self._blue_units.clear()
        self._red_units.clear()
        self._units_cache_dirty = True

        # Reinitialize targeting context reference
        self.targeting_context.all_units = self.units
        self.item_effect_context.all_units = self.units
        self.trait_effect_context.all_units = self.units

    def _rebuild_unit_cache(self) -> None:
        """Rebuild cached team-separated unit lists."""
        self._blue_units = [u for u in self.units.values() if u.team == Team.BLUE]
        self._red_units = [u for u in self.units.values() if u.team == Team.RED]
        self._units_cache_dirty = False

    def _get_blue_units(self, alive_only: bool = False) -> List[CombatUnit]:
        """Get blue team units (cached)."""
        if self._units_cache_dirty:
            self._rebuild_unit_cache()
        if alive_only:
            return [u for u in self._blue_units if u.is_alive]
        return self._blue_units

    def _get_red_units(self, alive_only: bool = False) -> List[CombatUnit]:
        """Get red team units (cached)."""
        if self._units_cache_dirty:
            self._rebuild_unit_cache()
        if alive_only:
            return [u for u in self._red_units if u.is_alive]
        return self._red_units

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all combat events."""
        return list(self.state.events)

    def get_unit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current stats for all units."""
        return {
            uid: {
                "name": u.name,
                "hp": u.stats.current_hp,
                "max_hp": u.stats.max_hp,
                "mana": u.stats.current_mana,
                "max_mana": u.stats.max_mana,
                "position": self.grid.get_unit_position(uid),
                "alive": u.is_alive,
            }
            for uid, u in self.units.items()
        }

    # =========================================================================
    # AUGMENT EFFECTS
    # =========================================================================

    def _apply_augment_combat_start(self) -> None:
        """Apply augment effects at combat start."""
        # Apply to blue team
        if self.blue_player:
            self.augment_effects.apply_combat_start_effects(
                self.blue_player, self, self._get_blue_units()
            )

        # Apply to red team
        if self.red_player:
            self.augment_effects.apply_combat_start_effects(
                self.red_player, self, self._get_red_units()
            )

    def _apply_augment_tick_effects(self) -> None:
        """Apply augment effects every tick."""
        # Apply to blue team
        if self.blue_player:
            self.augment_effects.apply_combat_tick_effects(
                self.blue_player, self, self._get_blue_units(alive_only=True), self.state.current_tick
            )

        # Apply to red team
        if self.red_player:
            self.augment_effects.apply_combat_tick_effects(
                self.red_player, self, self._get_red_units(alive_only=True), self.state.current_tick
            )

    def _apply_augment_on_kill(self, killer: CombatUnit, victim: CombatUnit) -> None:
        """Apply augment effects when a unit kills another."""
        player = self.blue_player if killer.team == Team.BLUE else self.red_player
        if player:
            self.augment_effects.apply_on_kill_effects(player, killer, victim)

    def get_augment_damage_modifier(
        self, attacker: CombatUnit, target: CombatUnit
    ) -> float:
        """Get damage multiplier from augments."""
        player = self.blue_player if attacker.team == Team.BLUE else self.red_player
        if not player:
            return 1.0

        return self.augment_effects.get_damage_modifier(
            player, attacker, target, self.state.elapsed_time
        )

    # =========================================================================
    # TRAIT EFFECTS
    # =========================================================================

    def _apply_trait_combat_start(self) -> None:
        """Apply trait effects at combat start."""
        # Set up blue team traits
        if self.blue_player:
            blue_units = self._get_blue_units()
            active_traits = getattr(self.blue_player, "active_traits", {})
            self.trait_effects.setup_team_traits("blue", active_traits, blue_units)

            events = self.trait_effects.apply_combat_start_effects(
                "blue", blue_units, self.trait_effect_context
            )
            self.state.events.extend([{"tick": 0, **e} for e in events])

        # Set up red team traits
        if self.red_player:
            red_units = self._get_red_units()
            active_traits = getattr(self.red_player, "active_traits", {})
            self.trait_effects.setup_team_traits("red", active_traits, red_units)

            events = self.trait_effects.apply_combat_start_effects(
                "red", red_units, self.trait_effect_context
            )
            self.state.events.extend([{"tick": 0, **e} for e in events])

    def _apply_trait_on_kill(self, killer: CombatUnit, victim: CombatUnit) -> None:
        """Apply trait effects when a unit kills another."""
        events = self.trait_effects.apply_on_kill_effects(
            killer, victim, self.trait_effect_context
        )
        self.state.events.extend([{"tick": self.state.current_tick, **e} for e in events])

    def _apply_trait_on_death(self, dead_unit: CombatUnit) -> None:
        """Apply trait effects when a unit dies."""
        all_units = list(self.units.values())
        events = self.trait_effects.apply_on_death_effects(
            dead_unit, all_units, self.trait_effect_context
        )
        self.state.events.extend([{"tick": self.state.current_tick, **e} for e in events])

    def _apply_trait_tick_effects(self) -> None:
        """Apply trait effects every tick."""
        units = list(self.units.values())
        events = self.trait_effects.apply_tick_effects(
            units, self.state.current_tick, self.trait_effect_context
        )
        self.state.events.extend([{"tick": self.state.current_tick, **e} for e in events])

    def get_trait_damage_modifier(
        self, attacker: CombatUnit, target: CombatUnit
    ) -> float:
        """Get damage multiplier from traits."""
        return self.trait_effects.get_damage_modifier(attacker, target)
