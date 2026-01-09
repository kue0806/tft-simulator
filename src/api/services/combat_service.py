"""
Combat simulation service.
"""

from typing import List, Dict, Any, Optional
import random

from ..schemas.combat import (
    TeamSetup,
    SimulationResultSchema,
    CombatResultSchema,
    UnitCombatStats,
)
from .game_service import GameService


class CombatService:
    """Combat simulation service."""

    def __init__(self, game_service: GameService):
        self.game_service = game_service
        self._last_combat_result: Optional[CombatResultSchema] = None

    def simulate(
        self,
        team_blue: TeamSetup,
        team_red: TeamSetup,
        iterations: int,
    ) -> SimulationResultSchema:
        """
        Simulate combat between two teams.

        Args:
            team_blue: Blue team setup.
            team_red: Red team setup.
            iterations: Number of iterations.

        Returns:
            Simulation results.
        """
        blue_wins = 0
        red_wins = 0
        total_rounds = 0
        total_damage = 0

        blue_stats: Dict[str, Dict[str, float]] = {}
        red_stats: Dict[str, Dict[str, float]] = {}

        for _ in range(iterations):
            result = self._simulate_single_combat(team_blue, team_red)

            if result.winner == "blue":
                blue_wins += 1
            else:
                red_wins += 1

            total_rounds += result.rounds_taken
            total_damage += result.damage_to_loser

            # Accumulate stats
            for stat in result.unit_stats:
                stats_dict = blue_stats if "blue" in stat.unit_id else red_stats
                if stat.unit_id not in stats_dict:
                    stats_dict[stat.unit_id] = {
                        "damage_dealt": 0,
                        "damage_taken": 0,
                        "kills": 0,
                        "survived": 0,
                    }
                stats_dict[stat.unit_id]["damage_dealt"] += stat.damage_dealt
                stats_dict[stat.unit_id]["damage_taken"] += stat.damage_taken
                stats_dict[stat.unit_id]["kills"] += stat.kills
                if stat.survived:
                    stats_dict[stat.unit_id]["survived"] += 1

        # Average stats
        for stats_dict in [blue_stats, red_stats]:
            for unit_id in stats_dict:
                for key in stats_dict[unit_id]:
                    stats_dict[unit_id][key] /= iterations

        return SimulationResultSchema(
            iterations=iterations,
            blue_wins=blue_wins,
            red_wins=red_wins,
            blue_win_rate=blue_wins / iterations,
            average_rounds=total_rounds / iterations,
            average_damage=total_damage / iterations,
            blue_unit_stats=list(blue_stats.values()),
            red_unit_stats=list(red_stats.values()),
        )

    def _simulate_single_combat(
        self, team_blue: TeamSetup, team_red: TeamSetup
    ) -> CombatResultSchema:
        """Simulate a single combat."""
        # Simplified combat simulation
        blue_power = self._calculate_team_power(team_blue)
        red_power = self._calculate_team_power(team_red)

        # Add some randomness
        blue_roll = blue_power * random.uniform(0.8, 1.2)
        red_roll = red_power * random.uniform(0.8, 1.2)

        winner = "blue" if blue_roll > red_roll else "red"
        loser_power = red_power if winner == "blue" else blue_power
        winner_power = blue_power if winner == "blue" else red_power

        # Calculate remaining units and damage
        power_diff = abs(blue_roll - red_roll)
        units_remaining = max(1, int(len(team_blue.units if winner == "blue" else team_red.units) * (power_diff / winner_power + 0.5)))
        damage = max(1, int(power_diff / 100))
        rounds = random.randint(10, 30)

        # Generate unit stats
        unit_stats = []
        for i, unit in enumerate(team_blue.units):
            unit_stats.append(
                UnitCombatStats(
                    unit_id=f"blue_{i}",
                    champion_name=unit.get("champion_id", "Unknown"),
                    damage_dealt=random.uniform(500, 2000),
                    damage_taken=random.uniform(300, 1500),
                    healing_done=random.uniform(0, 500),
                    kills=random.randint(0, 3),
                    survived=winner == "blue" and random.random() > 0.3,
                )
            )
        for i, unit in enumerate(team_red.units):
            unit_stats.append(
                UnitCombatStats(
                    unit_id=f"red_{i}",
                    champion_name=unit.get("champion_id", "Unknown"),
                    damage_dealt=random.uniform(500, 2000),
                    damage_taken=random.uniform(300, 1500),
                    healing_done=random.uniform(0, 500),
                    kills=random.randint(0, 3),
                    survived=winner == "red" and random.random() > 0.3,
                )
            )

        result = CombatResultSchema(
            winner=winner,
            units_remaining=units_remaining,
            damage_to_loser=damage,
            rounds_taken=rounds,
            unit_stats=unit_stats,
        )

        self._last_combat_result = result
        return result

    def _calculate_team_power(self, team: TeamSetup) -> float:
        """Calculate approximate team power."""
        power = 0.0
        for unit in team.units:
            cost = unit.get("cost", 1)
            star = unit.get("star_level", 1)
            items = len(unit.get("items", []))

            # Base power from cost and star level
            unit_power = cost * 100 * (1.8 ** (star - 1))
            # Item bonus
            unit_power *= 1 + (items * 0.15)

            power += unit_power

        return power

    def simulate_players(
        self,
        game_id: str,
        player_id: int,
        opponent_id: int,
        iterations: int,
    ) -> SimulationResultSchema:
        """Simulate combat between two game players."""
        player = self.game_service.get_player_raw(game_id, player_id)
        opponent = self.game_service.get_player_raw(game_id, opponent_id)

        if not player or not opponent:
            raise ValueError("Player not found")

        # Convert player boards to TeamSetup
        blue_units = []
        for unit in player.units.board.values():
            blue_units.append(
                {
                    "champion_id": unit.champion.id,
                    "star_level": unit.star_level,
                    "items": unit.items,
                    "position": unit.position,
                    "cost": unit.champion.cost,
                }
            )

        red_units = []
        for unit in opponent.units.board.values():
            red_units.append(
                {
                    "champion_id": unit.champion.id,
                    "star_level": unit.star_level,
                    "items": unit.items,
                    "position": unit.position,
                    "cost": unit.champion.cost,
                }
            )

        return self.simulate(
            TeamSetup(units=blue_units),
            TeamSetup(units=red_units),
            iterations,
        )

    def get_last_combat_result(self) -> Optional[CombatResultSchema]:
        """Get the last combat result."""
        return self._last_combat_result

    def get_player_combat_stats(self, game_id: str, player_id: int) -> Dict[str, Any]:
        """Get player combat statistics."""
        # Placeholder - would track historical stats
        return {
            "total_combats": 0,
            "wins": 0,
            "losses": 0,
            "average_damage_dealt": 0,
            "average_damage_taken": 0,
        }
