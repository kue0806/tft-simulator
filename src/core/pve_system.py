"""PvE System for TFT Set 16.

Manages PvE rounds including monster encounters and loot drops.
Uses data from data/loot/set16_loot_tables.json for accurate drop rates.
Simulates actual combat using CombatEngine.
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any, List, Dict, TYPE_CHECKING

from src.core.constants import PVE_ROUNDS, PVE_MONSTERS

if TYPE_CHECKING:
    from src.core.player_units import PlayerUnits


class OrbType(Enum):
    """Types of orbs that can drop."""
    GRAY = "gray"
    BLUE = "blue"
    GOLD = "gold"
    PRISMATIC = "prismatic"


class RewardType(Enum):
    """Types of rewards from orbs."""
    GOLD = "gold"
    CHAMPION = "champion"
    COMPONENT = "component"
    COMPLETED_ITEM = "completed_item"
    SPECIAL_ITEM = "special_item"
    CONSUMABLE = "consumable"


@dataclass
class OrbReward:
    """A reward from opening an orb."""
    reward_type: RewardType
    value: Any  # gold amount, item id, champion cost, etc.
    quantity: int = 1

    def __repr__(self) -> str:
        if self.reward_type == RewardType.GOLD:
            return f"{self.value} gold"
        elif self.reward_type == RewardType.CHAMPION:
            return f"{self.quantity}x {self.value}-cost champion"
        elif self.reward_type == RewardType.COMPONENT:
            return f"{self.value}"
        else:
            return f"{self.value}"


@dataclass
class Orb:
    """An orb dropped from PvE combat."""
    orb_type: OrbType
    rewards: list[OrbReward] = field(default_factory=list)
    gold_value: int = 0  # Approximate gold value

    def __repr__(self) -> str:
        rewards_str = ", ".join(str(r) for r in self.rewards)
        return f"{self.orb_type.value} orb: [{rewards_str}]"


@dataclass
class PvEResult:
    """Result of a PvE encounter."""
    won: bool
    round_stage: str
    monster_type: str
    damage_taken: int = 0
    orbs: list[Orb] = field(default_factory=list)
    items_gained: list[str] = field(default_factory=list)
    gold_gained: int = 0
    champions_gained: list[tuple[int, int]] = field(default_factory=list)  # (cost, quantity)
    special_items_gained: list[str] = field(default_factory=list)

    def total_value(self) -> int:
        """Calculate total gold value of loot."""
        return self.gold_gained + sum(orb.gold_value for orb in self.orbs)


class LootTableLoader:
    """Loads and provides access to loot table data."""

    _instance: Optional["LootTableLoader"] = None
    _data: Optional[dict] = None

    @classmethod
    def get_instance(cls) -> "LootTableLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_data()

    def _load_data(self) -> None:
        """Load loot table data from JSON file."""
        loot_file = Path(__file__).parent.parent.parent / "data" / "loot" / "set16_loot_tables.json"
        if loot_file.exists():
            with open(loot_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            # Fallback to default data
            self._data = self._get_default_data()

    def _get_default_data(self) -> dict:
        """Return default loot data if file not found."""
        return {
            "pve_rounds": {
                "1-2": {"monster": "minions", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gray", "blue"], "component_chance": 0.33},
                "1-3": {"monster": "minions", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gray", "blue"], "component_chance": 0.33},
                "1-4": {"monster": "minions", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gray", "blue"], "component_chance": 0.34},
                "2-7": {"monster": "krugs", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gray", "blue", "gold"], "component_chance": 0.5},
                "3-7": {"monster": "wolves", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gray", "blue", "gold"], "component_chance": 0.5},
                "4-7": {"monster": "raptors", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["blue", "gold"], "component_chance": 0.6},
                "5-7": {"monster": "rift_herald", "guaranteed_orbs": 1, "max_orbs": 2, "orb_pool": ["gold", "prismatic"], "guaranteed_item": True},
                "6-7": {"monster": "elder_dragon", "guaranteed_orbs": 2, "max_orbs": 3, "orb_pool": ["gold", "prismatic"], "guaranteed_item": True},
                "7-7": {"monster": "baron_nashor", "guaranteed_orbs": 2, "max_orbs": 4, "orb_pool": ["gold", "prismatic"], "guaranteed_item": True},
            },
            "components": ["bf_sword", "recurve_bow", "needlessly_large_rod", "tear_of_the_goddess",
                          "chain_vest", "negatron_cloak", "giants_belt", "sparring_gloves", "spatula"],
            "stage1_outcomes": {
                "outcomes": [
                    {"id": "items_heavy", "probability": 0.33, "total_items": 3, "total_gold": 0},
                    {"id": "balanced", "probability": 0.34, "total_items": 1, "total_gold": 3},
                    {"id": "gold_heavy", "probability": 0.33, "total_items": 0, "total_gold": 9},
                ]
            },
            "orbs": {}
        }

    def get_pve_round_config(self, stage: str) -> Optional[dict]:
        """Get configuration for a PvE round."""
        return self._data.get("pve_rounds", {}).get(stage)

    def get_orb_contents(self, orb_type: str, stage_num: int) -> list[dict]:
        """Get possible contents for an orb type at a given stage."""
        orbs = self._data.get("orbs", {})
        orb_data = orbs.get(orb_type, {})

        # Select stage-appropriate contents
        if orb_type == "gray":
            if stage_num <= 2:
                return orb_data.get("stages_1_2", {}).get("contents", [])
            return orb_data.get("stages_3_plus", {}).get("contents", [])
        elif orb_type == "blue":
            if stage_num <= 3:
                return orb_data.get("stages_1_3", {}).get("contents", [])
            return orb_data.get("stages_4_plus", {}).get("contents", [])
        elif orb_type == "gold":
            if stage_num <= 3:
                return orb_data.get("stages_2_3", {}).get("contents", [])
            return orb_data.get("stages_4_plus", {}).get("contents", [])
        elif orb_type == "prismatic":
            return orb_data.get("stages_3_plus", {}).get("contents", [])

        return []

    def get_components(self) -> list[str]:
        """Get list of component items."""
        return self._data.get("components", [])

    def get_stage1_outcomes(self) -> list[dict]:
        """Get Stage 1 RNG protection outcomes."""
        return self._data.get("stage1_outcomes", {}).get("outcomes", [])

    @property
    def data(self) -> dict:
        return self._data


@dataclass
class MonsterStats:
    """Stats for a PvE monster."""
    health: float
    attack_damage: float
    attack_speed: float
    armor: float
    magic_resist: float
    attack_range: int
    mana: tuple[int, int] = (0, 100)  # (starting, max)
    crit_chance: float = 0.0
    crit_damage: float = 1.4


@dataclass
class MonsterSpawn:
    """A monster to spawn in PvE round."""
    monster_type: str
    position: tuple[int, int]  # (row, col) on enemy side


class PvESystem:
    """Manages PvE encounters and loot generation."""

    # Detailed monster stats - balanced for TFT gameplay
    # Stage 1 minions are weak, later rounds scale up
    MONSTER_STATS = {
        # Stage 1 minions - very weak, 1-cost champion easily beats
        "melee_minion": MonsterStats(
            health=180, attack_damage=20, attack_speed=0.6,
            armor=10, magic_resist=10, attack_range=1
        ),
        "caster_minion": MonsterStats(
            health=120, attack_damage=25, attack_speed=0.5,
            armor=5, magic_resist=15, attack_range=3
        ),
        # Krugs - Stage 2-7, moderately tanky
        "ancient_krug": MonsterStats(
            health=800, attack_damage=35, attack_speed=0.5,
            armor=40, magic_resist=20, attack_range=1
        ),
        "krug": MonsterStats(
            health=350, attack_damage=25, attack_speed=0.6,
            armor=30, magic_resist=15, attack_range=1
        ),
        # Wolves - Stage 3-7, fast assassin-like behavior
        "greater_murk_wolf": MonsterStats(
            health=700, attack_damage=50, attack_speed=0.8,
            armor=25, magic_resist=25, attack_range=1
        ),
        "murk_wolf": MonsterStats(
            health=350, attack_damage=35, attack_speed=0.85,
            armor=20, magic_resist=20, attack_range=1
        ),
        # Raptors - Stage 4-7, become stronger when allies die
        "crimson_raptor": MonsterStats(
            health=600, attack_damage=55, attack_speed=0.7,
            armor=20, magic_resist=30, attack_range=1
        ),
        "raptor": MonsterStats(
            health=300, attack_damage=40, attack_speed=0.75,
            armor=15, magic_resist=25, attack_range=1
        ),
        # Rift Herald - Stage 5-7, boss monster
        "rift_herald": MonsterStats(
            health=2500, attack_damage=80, attack_speed=0.5,
            armor=50, magic_resist=50, attack_range=1,
            mana=(50, 100)
        ),
        # Elder Dragon - Stage 6-7, powerful boss
        "elder_dragon": MonsterStats(
            health=3500, attack_damage=100, attack_speed=0.55,
            armor=60, magic_resist=80, attack_range=2,
            mana=(0, 80)
        ),
        # Baron Nashor - Stage 7-7, final boss
        "baron_nashor": MonsterStats(
            health=5000, attack_damage=120, attack_speed=0.45,
            armor=80, magic_resist=80, attack_range=2,
            mana=(0, 100)
        ),
    }

    # Spawn configurations per stage - (monster_type, row, col)
    # Rows 0-3 are player side, rows 4-7 are enemy side
    STAGE_SPAWNS = {
        "1-2": [
            ("melee_minion", 5, 2),
            ("melee_minion", 5, 4),
        ],
        "1-3": [
            ("melee_minion", 5, 1),
            ("melee_minion", 5, 3),
            ("caster_minion", 6, 2),
        ],
        "1-4": [
            ("melee_minion", 5, 1),
            ("melee_minion", 5, 3),
            ("melee_minion", 5, 5),
            ("caster_minion", 6, 2),
            ("caster_minion", 6, 4),
        ],
        "2-7": [
            ("ancient_krug", 5, 1),
            ("krug", 5, 3),
            ("krug", 5, 5),
            ("krug", 6, 2),
            ("krug", 6, 4),
        ],
        "3-7": [
            ("greater_murk_wolf", 5, 3),
            ("murk_wolf", 5, 1),
            ("murk_wolf", 5, 5),
            ("murk_wolf", 6, 2),
            ("murk_wolf", 6, 4),
        ],
        "4-7": [
            ("crimson_raptor", 5, 3),
            ("raptor", 5, 1),
            ("raptor", 5, 5),
            ("raptor", 6, 2),
            ("raptor", 6, 4),
        ],
        "5-7": [
            ("rift_herald", 5, 3),
        ],
        "6-7": [
            ("elder_dragon", 5, 3),
        ],
        "7-7": [
            ("baron_nashor", 5, 3),
        ],
    }

    # Legacy monster configs for damage calculation
    MONSTER_CONFIGS = {
        "minions": {"name": "Minions", "health": 100, "damage": 5, "count": 3},
        "krugs": {"name": "Krugs", "health": 300, "damage": 10, "count": 6},
        "wolves": {"name": "Wolves", "health": 400, "damage": 15, "count": 5},
        "raptors": {"name": "Raptors", "health": 500, "damage": 20, "count": 4},
        "rift_herald": {"name": "Rift Herald", "health": 1000, "damage": 30, "count": 1},
        "elder_dragon": {"name": "Elder Dragon", "health": 1500, "damage": 40, "count": 1},
        "baron_nashor": {"name": "Baron Nashor", "health": 2000, "damage": 50, "count": 1},
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize PvE system.

        Args:
            seed: Random seed for reproducible results.
        """
        self.rng = random.Random(seed)
        self.loot_tables = LootTableLoader.get_instance()

        # Track items per player for RNG protection
        self._player_item_counts: dict[int, int] = {}
        # Track Stage 1 outcome per player
        self._player_stage1_outcome: dict[int, str] = {}

    def is_pve_round(self, stage: str) -> bool:
        """Check if a stage is a PvE round."""
        return stage in PVE_ROUNDS

    def get_monster_type(self, stage: str) -> Optional[str]:
        """Get the monster type for a given stage."""
        return PVE_MONSTERS.get(stage)

    def get_monster_config(self, stage: str) -> Optional[dict]:
        """Get monster configuration for a stage."""
        monster_type = self.get_monster_type(stage)
        if monster_type:
            return self.MONSTER_CONFIGS.get(monster_type)
        return None

    def _determine_stage1_outcome(self, player_id: int) -> dict:
        """Determine the Stage 1 loot outcome for a player."""
        if player_id not in self._player_stage1_outcome:
            outcomes = self.loot_tables.get_stage1_outcomes()
            if outcomes:
                probs = [o["probability"] for o in outcomes]
                chosen = self.rng.choices(outcomes, weights=probs, k=1)[0]
                self._player_stage1_outcome[player_id] = chosen["id"]
                return chosen
            # Default outcome
            return {"id": "balanced", "total_items": 1, "total_gold": 3}

        # Return the already-determined outcome
        outcome_id = self._player_stage1_outcome[player_id]
        outcomes = self.loot_tables.get_stage1_outcomes()
        for o in outcomes:
            if o["id"] == outcome_id:
                return o
        return {"id": "balanced", "total_items": 1, "total_gold": 3}

    def _select_orb_type(self, orb_pool: list[str], stage_num: int) -> OrbType:
        """Select an orb type from the pool with weighted probabilities."""
        # Weight higher tier orbs less frequently
        weights = {
            "gray": 40,
            "blue": 35,
            "gold": 20,
            "prismatic": 5,
        }

        pool_weights = [weights.get(orb, 10) for orb in orb_pool]
        chosen = self.rng.choices(orb_pool, weights=pool_weights, k=1)[0]
        return OrbType(chosen)

    def _parse_reward(self, reward_str: str) -> list[OrbReward]:
        """Parse a reward string into OrbReward objects."""
        rewards = []

        # Handle common patterns
        if "gold" in reward_str.lower():
            # Extract gold amount
            import re
            gold_match = re.search(r"(\d+)\s*gold", reward_str.lower())
            if gold_match:
                gold = int(gold_match.group(1))
                rewards.append(OrbReward(RewardType.GOLD, gold))

        if "cost champion" in reward_str.lower():
            import re
            # Match patterns like "2x 3-cost champion" or "1x 4-cost champion"
            champ_match = re.search(r"(\d+)x?\s*(\d)-cost", reward_str.lower())
            if champ_match:
                qty = int(champ_match.group(1))
                cost = int(champ_match.group(2))
                rewards.append(OrbReward(RewardType.CHAMPION, cost, qty))

        # Special items
        special_items = [
            "reforger", "magnetic_remover", "lesser_champion_duplicator",
            "champion_duplicator", "component_anvil", "completed_item_anvil",
            "artifact_item_anvil", "radiant_armory", "masterwork_upgrade",
            "thiefs_gloves", "spatula", "frying_pan"
        ]
        for item in special_items:
            if item.replace("_", " ") in reward_str.lower() or item in reward_str.lower():
                rewards.append(OrbReward(RewardType.SPECIAL_ITEM, item))

        return rewards

    def _generate_orb(self, orb_type: OrbType, stage_num: int, can_drop_components: bool) -> Orb:
        """Generate an orb with rewards based on loot tables."""
        orb = Orb(orb_type=orb_type)
        contents = self.loot_tables.get_orb_contents(orb_type.value, stage_num)

        if contents:
            # Select reward based on probabilities
            probs = [c.get("chance", 0.1) for c in contents]
            if sum(probs) > 0:
                chosen = self.rng.choices(contents, weights=probs, k=1)[0]
                reward_str = chosen.get("reward", "")
                orb.rewards = self._parse_reward(reward_str)

        # Fallback: generate default rewards if parsing failed
        if not orb.rewards:
            orb.rewards = self._generate_default_rewards(orb_type, stage_num, can_drop_components)

        # Calculate approximate gold value
        orb.gold_value = self._calculate_orb_value(orb_type, stage_num)

        return orb

    def _generate_default_rewards(self, orb_type: OrbType, stage_num: int, can_drop_components: bool) -> list[OrbReward]:
        """Generate default rewards when loot table parsing fails."""
        rewards = []
        components = self.loot_tables.get_components()

        if orb_type == OrbType.GRAY:
            # Gray orb: champions or small gold
            if self.rng.random() < 0.5:
                cost = 1 if stage_num <= 2 else 2
                rewards.append(OrbReward(RewardType.CHAMPION, cost, 2))
            else:
                rewards.append(OrbReward(RewardType.GOLD, self.rng.randint(1, 3)))

        elif orb_type == OrbType.BLUE:
            # Blue orb: better champions or gold
            if self.rng.random() < 0.5:
                cost = 2 if stage_num <= 3 else 3
                rewards.append(OrbReward(RewardType.CHAMPION, cost, 2))
            else:
                rewards.append(OrbReward(RewardType.GOLD, self.rng.randint(3, 6)))

        elif orb_type == OrbType.GOLD:
            # Gold orb: components, completed items, or good gold
            roll = self.rng.random()
            if can_drop_components and roll < 0.3 and components:
                comp = self.rng.choice(components)
                rewards.append(OrbReward(RewardType.COMPONENT, comp))
            elif roll < 0.5:
                rewards.append(OrbReward(RewardType.SPECIAL_ITEM, "completed_item_anvil"))
            else:
                rewards.append(OrbReward(RewardType.GOLD, self.rng.randint(8, 15)))

        elif orb_type == OrbType.PRISMATIC:
            # Prismatic: high value items
            roll = self.rng.random()
            if roll < 0.3:
                rewards.append(OrbReward(RewardType.SPECIAL_ITEM, "artifact_item_anvil"))
            elif roll < 0.5:
                rewards.append(OrbReward(RewardType.SPECIAL_ITEM, "radiant_armory"))
            else:
                rewards.append(OrbReward(RewardType.GOLD, self.rng.randint(15, 25)))

        return rewards

    def _calculate_orb_value(self, orb_type: OrbType, stage_num: int) -> int:
        """Calculate approximate gold value of an orb."""
        values = {
            OrbType.GRAY: 2 if stage_num <= 2 else 3,
            OrbType.BLUE: 6 if stage_num <= 3 else 8,
            OrbType.GOLD: 15 if stage_num <= 3 else 18,
            OrbType.PRISMATIC: 30,
        }
        return values.get(orb_type, 5)

    def _create_monster_units(self, stage: str) -> List[Dict]:
        """
        Create monster unit data for combat simulation.

        Args:
            stage: The stage string (e.g., "1-2").

        Returns:
            List of monster unit dictionaries with stats and positions.
        """
        spawns = self.STAGE_SPAWNS.get(stage, [])
        monsters = []

        for monster_type, row, col in spawns:
            stats = self.MONSTER_STATS.get(monster_type)
            if stats:
                monsters.append({
                    "id": f"monster_{monster_type}_{uuid.uuid4().hex[:8]}",
                    "name": monster_type.replace("_", " ").title(),
                    "monster_type": monster_type,
                    "position": (row, col),
                    "stats": stats,
                })

        return monsters

    def simulate_pve_combat_with_engine(
        self,
        stage: str,
        player_units: "PlayerUnits",
        player_id: int,
    ) -> PvEResult:
        """
        Simulate PvE combat using the actual CombatEngine.

        Args:
            stage: The current stage (e.g., "1-2", "2-7").
            player_units: The player's units (with board positions).
            player_id: ID of the player.

        Returns:
            PvEResult with combat outcome and loot.
        """
        from src.combat.combat_engine import CombatEngine
        from src.combat.combat_unit import CombatUnit, CombatStats
        from src.combat.hex_grid import HexPosition, Team

        monster_type = self.get_monster_type(stage)
        if not monster_type:
            raise ValueError(f"Stage {stage} is not a PvE round")

        round_config = self.loot_tables.get_pve_round_config(stage) or {}
        stage_num = int(stage.split("-")[0])

        # Get player's board units
        board_units = list(player_units.board.items())

        # No units on board = auto loss
        # Damage = Base Stage Damage + Number of Monsters
        if not board_units:
            from src.core.constants import BASE_STAGE_DAMAGE

            spawns = self.STAGE_SPAWNS.get(stage, [])
            base_damage = BASE_STAGE_DAMAGE.get(stage_num, 0)
            unit_damage = len(spawns)  # All monsters survive
            damage = base_damage + unit_damage

            return PvEResult(
                won=False,
                round_stage=stage,
                monster_type=monster_type,
                damage_taken=damage,
            )

        # Create combat engine
        engine = CombatEngine(seed=self.rng.randint(0, 1000000), stage=stage_num)

        # Add player units (Blue team)
        for pos, instance in board_units:
            calculated_stats = {
                "max_health": instance.champion.stats.health[instance.star_level - 1],
                "attack_damage": instance.champion.stats.attack_damage[instance.star_level - 1],
                "ability_power": 100.0,
                "armor": instance.champion.stats.armor,
                "magic_resist": instance.champion.stats.magic_resist,
                "attack_speed": instance.champion.stats.attack_speed,
                "crit_chance": instance.champion.stats.crit_chance,
                "crit_damage": instance.champion.stats.crit_damage,
                "mana_start": instance.champion.stats.mana[0],
            }
            combat_unit = CombatUnit.from_champion_instance(
                instance, Team.BLUE, calculated_stats
            )
            hex_pos = HexPosition(pos[0], pos[1])
            if engine.grid.place_unit(combat_unit.id, hex_pos):
                engine.units[combat_unit.id] = combat_unit

        # Create and add monster units (Red team)
        monsters = self._create_monster_units(stage)
        for monster in monsters:
            stats = monster["stats"]
            combat_stats = CombatStats(
                max_hp=stats.health,
                current_hp=stats.health,
                attack_damage=stats.attack_damage,
                ability_power=100.0,
                attack_speed=stats.attack_speed,
                crit_chance=stats.crit_chance,
                crit_damage=stats.crit_damage,
                armor=stats.armor,
                magic_resist=stats.magic_resist,
                max_mana=stats.mana[1],
                current_mana=stats.mana[0],
                starting_mana=stats.mana[0],
                attack_range=stats.attack_range,
                dodge_chance=0.0,
                omnivamp=0.0,
                damage_amp=1.0,
                damage_reduction=0.0,
            )
            monster_unit = CombatUnit(
                id=monster["id"],
                name=monster["name"],
                champion_id=monster["monster_type"],
                star_level=1,
                team=Team.RED,
                stats=combat_stats,
                role="fighter",
                cost=0,  # Monsters have 0 cost (no player damage from surviving)
            )
            hex_pos = HexPosition(monster["position"][0], monster["position"][1])
            if engine.grid.place_unit(monster_unit.id, hex_pos):
                engine.units[monster_unit.id] = monster_unit

        # Update unit counts
        engine.state.blue_units_alive = len([u for u in engine.units.values() if u.team == Team.BLUE])
        engine.state.red_units_alive = len([u for u in engine.units.values() if u.team == Team.RED])

        # Run combat
        from src.combat.combat_engine import CombatPhase
        engine.state.phase = CombatPhase.RUNNING

        combat_result = engine.run_combat()

        # Determine winner
        won = combat_result.winner == Team.BLUE

        result = PvEResult(
            won=won,
            round_stage=stage,
            monster_type=monster_type,
        )

        if not won:
            # Lost PvE - calculate damage using TFT formula:
            # Total Damage = Base Stage Damage + Number of Surviving Units
            from src.core.constants import BASE_STAGE_DAMAGE

            surviving_monsters = [
                u for u in engine.units.values()
                if u.team == Team.RED and u.is_alive
            ]

            base_damage = BASE_STAGE_DAMAGE.get(stage_num, 0)
            unit_damage = len(surviving_monsters)
            result.damage_taken = base_damage + unit_damage
            return result

        # Won - generate loot
        self._generate_pve_loot(result, stage, stage_num, player_id, round_config)

        return result

    def _generate_pve_loot(
        self,
        result: PvEResult,
        stage: str,
        stage_num: int,
        player_id: int,
        round_config: dict,
    ) -> None:
        """Generate loot for a won PvE round."""
        orb_pool = round_config.get("orb_pool", ["gray", "blue"])
        guaranteed_orbs = round_config.get("guaranteed_orbs", 1)
        max_orbs = round_config.get("max_orbs", 2)
        can_drop_components = stage in ["1-2", "1-3", "1-4", "2-7", "3-7", "4-7"]

        # Special handling for Stage 1
        if stage in ["1-2", "1-3", "1-4"]:
            outcome = self._determine_stage1_outcome(player_id)

            # Distribute Stage 1 rewards across rounds
            round_index = {"1-2": 0, "1-3": 1, "1-4": 2}[stage]
            total_items = outcome.get("total_items", 1)
            total_gold = outcome.get("total_gold", 3)

            # Simple distribution: items in earlier rounds, gold spread across
            if round_index < total_items:
                # Drop a component
                components = self.loot_tables.get_components()
                if components:
                    comp = self.rng.choice([c for c in components if c != "spatula"])
                    result.items_gained.append(comp)
                    orb = Orb(orb_type=OrbType.BLUE)
                    orb.rewards.append(OrbReward(RewardType.COMPONENT, comp))
                    result.orbs.append(orb)

            # Distribute gold across rounds
            gold_per_round = total_gold // 3 + (1 if round_index < total_gold % 3 else 0)
            if gold_per_round > 0:
                result.gold_gained += gold_per_round
        else:
            # Normal PvE rounds (Krugs, Wolves, etc.)
            # RNG protection: players with fewer items get better odds
            player_items = self._player_item_counts.get(player_id, 0)
            if player_items < 5:
                guaranteed_orbs = max(guaranteed_orbs, 1)

            num_orbs = self.rng.randint(guaranteed_orbs, max_orbs)

            for _ in range(num_orbs):
                orb_type = self._select_orb_type(orb_pool, stage_num)
                orb = self._generate_orb(orb_type, stage_num, can_drop_components)
                result.orbs.append(orb)

                # Extract rewards
                for reward in orb.rewards:
                    if reward.reward_type == RewardType.GOLD:
                        result.gold_gained += reward.value
                    elif reward.reward_type == RewardType.COMPONENT:
                        result.items_gained.append(reward.value)
                        self._player_item_counts[player_id] = player_items + 1
                    elif reward.reward_type == RewardType.CHAMPION:
                        result.champions_gained.append((reward.value, reward.quantity))
                    elif reward.reward_type == RewardType.SPECIAL_ITEM:
                        result.special_items_gained.append(reward.value)

            # Late game PvE always drops items
            if round_config.get("guaranteed_item") and not result.items_gained and not result.special_items_gained:
                result.special_items_gained.append("completed_item_anvil")

    def simulate_pve_combat(
        self,
        stage: str,
        player_id: int,
        player_power: int = 100,
        board_unit_count: int = 0,
    ) -> PvEResult:
        """
        Legacy method for probability-based PvE combat.
        Use simulate_pve_combat_with_engine for actual combat simulation.

        Args:
            stage: The current stage (e.g., "1-2", "2-7").
            player_id: ID of the player.
            player_power: Simplified measure of player's combat power.
            board_unit_count: Number of units on the player's board.

        Returns:
            PvEResult with combat outcome and loot.
        """
        monster_type = self.get_monster_type(stage)
        if not monster_type:
            raise ValueError(f"Stage {stage} is not a PvE round")

        monster_config = self.MONSTER_CONFIGS.get(monster_type, {})
        round_config = self.loot_tables.get_pve_round_config(stage) or {}
        stage_num = int(stage.split("-")[0])

        # No units on board = auto loss
        if board_unit_count == 0:
            damage = monster_config.get("damage", 10) * monster_config.get("count", 1)
            return PvEResult(
                won=False,
                round_stage=stage,
                monster_type=monster_type,
                damage_taken=damage,
            )

        # Determine win/loss based on board strength
        base_chance = min(0.5, board_unit_count * 0.1)
        power_bonus = min(0.48, player_power / 500)
        win_chance = min(0.98, base_chance + power_bonus)
        won = self.rng.random() < win_chance

        result = PvEResult(
            won=won,
            round_stage=stage,
            monster_type=monster_type,
        )

        if not won:
            result.damage_taken = monster_config.get("damage", 10) * max(1, monster_config.get("count", 1) // 2)
            return result

        # Generate loot
        self._generate_pve_loot(result, stage, stage_num, player_id, round_config)

        return result

    def reset_game(self) -> None:
        """Reset tracking for a new game."""
        self._player_item_counts.clear()
        self._player_stage1_outcome.clear()


# Singleton instance for easy access
_pve_system: Optional[PvESystem] = None


def get_pve_system(seed: Optional[int] = None) -> PvESystem:
    """Get or create the PvE system singleton."""
    global _pve_system
    if _pve_system is None or seed is not None:
        _pve_system = PvESystem(seed)
    return _pve_system
