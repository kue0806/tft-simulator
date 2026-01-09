"""Combat Unit for TFT.

Manages unit state during combat, including HP, mana, and status effects.
Connects with ChampionInstance for base stats and items.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum, auto
import uuid

from src.combat.hex_grid import Team, HexPosition

if TYPE_CHECKING:
    from src.core.player_units import ChampionInstance
    from src.combat.status_effects import StatusEffectSystem


class UnitState(Enum):
    """Unit combat state."""

    IDLE = auto()  # Waiting
    MOVING = auto()  # Moving towards target
    ATTACKING = auto()  # Performing attack
    CASTING = auto()  # Casting ability
    STUNNED = auto()  # Crowd controlled
    DEAD = auto()  # Dead


@dataclass
class CombatStats:
    """
    Real-time combat stats.

    These are calculated from base stats + items + trait bonuses.
    """

    # Health
    max_hp: float
    current_hp: float

    # Offensive stats
    attack_damage: float
    ability_power: float
    attack_speed: float  # Attacks per second
    crit_chance: float  # 0.0 to 1.0
    crit_damage: float  # Multiplier (default 1.4 = 140%)

    # Defensive stats
    armor: float
    magic_resist: float

    # Mana
    max_mana: float
    current_mana: float
    starting_mana: float

    # Range
    attack_range: int  # In hex units (1 = melee, 4 = ranged, etc.)

    # Other
    dodge_chance: float  # 0.0 to 1.0
    omnivamp: float  # Heal for % of all damage dealt
    damage_amp: float  # Damage multiplier (1.0 = 100%)
    damage_reduction: float  # Damage taken reduction (0.0 = no reduction)


@dataclass
class CombatUnit:
    """
    A unit participating in combat.

    This is the combat version of ChampionInstance, tracking real-time state.
    """

    # Identification
    id: str
    name: str
    champion_id: str
    star_level: int
    team: Team

    # Combat stats (calculated at combat start)
    stats: CombatStats

    # Role for mana generation and targeting priority
    role: str = "fighter"  # tank, fighter, marksman, caster, assassin, specialist

    # Cost tier (for player damage calculation)
    cost: int = 1

    # State
    state: UnitState = UnitState.IDLE

    # Targeting
    current_target_id: Optional[str] = None

    # Attack timing (seconds)
    attack_cooldown: float = 0.0

    # Ability casting
    is_casting: bool = False
    cast_time_remaining: float = 0.0

    # Movement
    move_progress: float = 0.0  # 0.0 to 1.0, progress to next hex
    move_target: Optional[HexPosition] = None

    # Combat statistics
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    total_healing_done: float = 0.0
    kills: int = 0

    # Status effects (implemented in Stage 5.5)
    status_effects: List[Any] = field(default_factory=list)

    # Reference to status effect system (set by CombatEngine)
    _status_effect_system: Optional["StatusEffectSystem"] = field(default=None, repr=False)

    # Reference to original instance (for items, traits)
    source_instance: Optional["ChampionInstance"] = None

    @property
    def items(self) -> list:
        """Get items from source instance."""
        if self.source_instance:
            return self.source_instance.items or []
        return []

    @classmethod
    def from_champion_instance(
        cls,
        instance: "ChampionInstance",
        team: Team,
        calculated_stats: Dict[str, float],
    ) -> "CombatUnit":
        """
        Create CombatUnit from ChampionInstance.

        Args:
            instance: Original champion instance.
            team: Team assignment.
            calculated_stats: Stats from StatCalculator.

        Returns:
            New CombatUnit ready for combat.
        """
        champion = instance.champion
        base_stats = champion.stats
        star_idx = instance.star_level - 1

        # Build CombatStats from calculated stats with fallbacks
        stats = CombatStats(
            max_hp=calculated_stats.get("max_health", base_stats.health[star_idx]),
            current_hp=calculated_stats.get("max_health", base_stats.health[star_idx]),
            attack_damage=calculated_stats.get(
                "attack_damage", base_stats.attack_damage[star_idx]
            ),
            ability_power=calculated_stats.get("ability_power", 100.0),
            armor=calculated_stats.get("armor", base_stats.armor),
            magic_resist=calculated_stats.get("magic_resist", base_stats.magic_resist),
            attack_speed=calculated_stats.get("attack_speed", base_stats.attack_speed),
            crit_chance=calculated_stats.get("crit_chance", base_stats.crit_chance),
            crit_damage=calculated_stats.get("crit_damage", base_stats.crit_damage),
            max_mana=base_stats.mana[1],  # Max mana from base stats
            current_mana=calculated_stats.get("mana_start", base_stats.mana[0]),
            starting_mana=calculated_stats.get("mana_start", base_stats.mana[0]),
            attack_range=base_stats.attack_range,
            dodge_chance=calculated_stats.get("dodge_chance", 0.0),
            omnivamp=calculated_stats.get("omnivamp", 0.0),
            damage_amp=calculated_stats.get("damage_amp", 1.0),
            damage_reduction=calculated_stats.get("damage_reduction", 0.0),
        )

        # Determine role from champion data (default to fighter)
        role = getattr(champion, 'role', 'fighter')
        if role is None:
            role = 'fighter'

        return cls(
            id=str(uuid.uuid4()),
            name=champion.name,
            champion_id=champion.id,
            star_level=instance.star_level,
            team=team,
            stats=stats,
            role=role,
            cost=champion.cost,
            source_instance=instance,
        )

    @property
    def is_alive(self) -> bool:
        """Check if unit is alive."""
        return self.stats.current_hp > 0 and self.state != UnitState.DEAD

    @property
    def is_targetable(self) -> bool:
        """Check if unit can be targeted."""
        return self.is_alive  # Add invulnerability checks later

    @property
    def can_act(self) -> bool:
        """Check if unit can perform actions."""
        return self.is_alive and self.state not in [UnitState.STUNNED, UnitState.DEAD]

    @property
    def can_attack(self) -> bool:
        """Check if unit can attack."""
        return self.can_act and self.attack_cooldown <= 0 and not self.is_casting

    @property
    def can_cast(self) -> bool:
        """Check if unit can cast ability."""
        return (
            self.can_act
            and self.stats.current_mana >= self.stats.max_mana
            and not self.is_casting
        )

    @property
    def attack_interval(self) -> float:
        """Get time between attacks in seconds."""
        if self.stats.attack_speed <= 0:
            return float("inf")
        return 1.0 / self.stats.attack_speed

    def take_damage(
        self,
        amount: float,
        damage_type: str = "physical",
        ignore_shields: bool = False,
    ) -> float:
        """
        Receive damage.

        Args:
            amount: Pre-mitigated damage amount (armor/MR already applied by AttackSystem).
            damage_type: "physical", "magical", or "true" (used for logging/effects only).
            ignore_shields: If True, bypass shields (for some special effects).

        Returns:
            Actual damage taken after reductions.

        Note:
            Armor/MR reduction is handled in AttackSystem._calculate_attack_damage()
            and AttackSystem.calculate_damage() to avoid double-application.
            This method only applies damage_reduction stat and shields.
        """
        if not self.is_alive:
            return 0.0

        # Check invulnerability
        if self._status_effect_system:
            from src.combat.status_effects import StatusEffectType
            if self._status_effect_system.has_effect(self, StatusEffectType.INVULNERABLE):
                return 0.0

        # Apply damage reduction stat (from items/traits like Warmog's, etc.)
        # Note: Armor/MR is already applied in AttackSystem, so we only apply damage_reduction here
        effective_damage = amount * (1 - self.stats.damage_reduction)

        effective_damage = max(0, effective_damage)

        # Absorb damage with shields first
        if not ignore_shields and self._status_effect_system:
            effective_damage = self._status_effect_system.absorb_damage(
                self, effective_damage
            )

        self.stats.current_hp -= effective_damage
        self.total_damage_taken += effective_damage

        if self.stats.current_hp <= 0:
            self.stats.current_hp = 0
            self.state = UnitState.DEAD

        return effective_damage

    def heal(self, amount: float) -> float:
        """
        Heal the unit.

        Args:
            amount: Heal amount.

        Returns:
            Actual amount healed.
        """
        if not self.is_alive:
            return 0.0

        old_hp = self.stats.current_hp
        self.stats.current_hp = min(self.stats.current_hp + amount, self.stats.max_hp)
        healed = self.stats.current_hp - old_hp
        self.total_healing_done += healed
        return healed

    def gain_mana(self, amount: float) -> None:
        """
        Gain mana.

        Args:
            amount: Mana to gain.
        """
        if not self.is_alive:
            return
        self.stats.current_mana = min(
            self.stats.current_mana + amount, self.stats.max_mana
        )

    def spend_mana(self) -> None:
        """Spend mana after casting ability (resets to starting mana)."""
        self.stats.current_mana = self.stats.starting_mana

    def reset_for_combat(self) -> None:
        """Reset unit state for a new combat."""
        self.stats.current_hp = self.stats.max_hp
        self.stats.current_mana = self.stats.starting_mana
        self.state = UnitState.IDLE
        self.current_target_id = None
        self.attack_cooldown = 0.0
        self.is_casting = False
        self.cast_time_remaining = 0.0
        self.move_progress = 0.0
        self.move_target = None
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.total_healing_done = 0.0
        self.kills = 0
        self.status_effects.clear()

    def __repr__(self) -> str:
        stars = "*" * self.star_level
        hp_pct = (
            int(self.stats.current_hp / self.stats.max_hp * 100)
            if self.stats.max_hp > 0
            else 0
        )
        return f"{self.name}{stars} ({hp_pct}% HP)"


@dataclass
class CombatResult:
    """Result of a combat round."""

    winner: Team
    winning_units_remaining: int
    losing_units_remaining: int  # Usually 0
    rounds_taken: int  # Combat ticks/rounds elapsed
    total_damage_to_loser: float  # Player damage dealt to loser

    # Per-unit statistics
    unit_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {unit_id: {"damage_dealt": x, "damage_taken": y, "kills": z}}
