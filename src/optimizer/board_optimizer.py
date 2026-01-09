"""Board Positioning Optimizer.

Optimizes unit positioning on the hex grid:
- Role-based positioning
- Counter-positioning against enemies
- Simulation-based optimization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import random

from src.combat.hex_grid import HexGrid, HexPosition, Team

if TYPE_CHECKING:
    from src.core.game_state import PlayerState
    from src.core.player_units import ChampionInstance
    from src.combat.simulation import CombatSimulator


@dataclass
class PositionScore:
    """Score for a single position."""

    position: HexPosition
    unit_id: str
    score: float
    reasons: List[str]


@dataclass
class BoardLayout:
    """A complete board layout."""

    positions: Dict[str, HexPosition]  # unit_id -> position
    total_score: float
    win_rate: float
    description: str


class BoardOptimizer:
    """
    Board positioning optimizer.

    Optimizes unit placement using role-based heuristics
    and optional simulation-based evaluation.

    Usage:
        optimizer = BoardOptimizer()
        layout = optimizer.optimize(player)
    """

    # Row classifications (player's perspective, rows 0-3)
    FRONTLINE_ROWS = [3]  # Row closest to enemy
    MIDLINE_ROWS = [2]
    BACKLINE_ROWS = [0, 1]  # Rows furthest from enemy

    # Grid dimensions
    ROWS = 4
    COLS = 7

    def __init__(self, simulator: Optional["CombatSimulator"] = None):
        """
        Initialize board optimizer.

        Args:
            simulator: Optional combat simulator for win rate calculations.
        """
        self.simulator = simulator
        self._rng = random.Random()

    def optimize(
        self,
        player: "PlayerState",
        enemy_layouts: Optional[List[BoardLayout]] = None,
        iterations: int = 100,
    ) -> BoardLayout:
        """
        Optimize board positioning.

        Args:
            player: Player state with units.
            enemy_layouts: Optional enemy layouts to counter.
            iterations: Optimization iterations.

        Returns:
            Optimized BoardLayout.
        """
        units = list(player.units.board.keys())
        instances = player.units.board

        if not units:
            return BoardLayout({}, 0, 0, "No units")

        # Classify unit roles
        roles = self._classify_roles(instances)

        # Generate initial layout
        best_layout = self._generate_initial_layout(units, roles)
        best_score = self._evaluate_layout(best_layout, player, enemy_layouts)

        # Iterative optimization
        for _ in range(iterations):
            # Mutate layout
            new_layout = self._mutate_layout(best_layout, units)
            new_score = self._evaluate_layout(new_layout, player, enemy_layouts)

            if new_score > best_score:
                best_layout = new_layout
                best_score = new_score

        # Calculate win rate if simulator available
        win_rate = self._calculate_win_rate(best_layout, player, enemy_layouts)

        return BoardLayout(
            positions=best_layout,
            total_score=best_score,
            win_rate=win_rate,
            description=self._describe_layout(best_layout, roles),
        )

    def suggest_position(
        self, player: "PlayerState", unit_id: str
    ) -> List[PositionScore]:
        """
        Suggest positions for a specific unit.

        Args:
            player: Player state.
            unit_id: Unit to position.

        Returns:
            List of position suggestions sorted by score.
        """
        instance = player.units.board.get(unit_id)
        if not instance:
            return []

        role = self._get_unit_role(instance)
        suggestions = []

        # Get target rows for role
        if role == "tank":
            target_rows = self.FRONTLINE_ROWS
        elif role == "assassin":
            target_rows = self.BACKLINE_ROWS  # Start at back to jump
        elif role == "carry":
            target_rows = self.BACKLINE_ROWS
        else:
            target_rows = self.MIDLINE_ROWS

        # Score each position in target rows
        for row in target_rows:
            for col in range(self.COLS):
                pos = HexPosition(row, col)
                score, reasons = self._score_position(pos, instance, player, role)
                suggestions.append(
                    PositionScore(
                        position=pos, unit_id=unit_id, score=score, reasons=reasons
                    )
                )

        suggestions.sort(key=lambda s: s.score, reverse=True)
        return suggestions[:5]

    def counter_position(
        self, player: "PlayerState", enemy_layout: BoardLayout
    ) -> BoardLayout:
        """
        Create counter-positioning against enemy.

        Args:
            player: Player state.
            enemy_layout: Enemy's board layout.

        Returns:
            Counter-positioned layout.
        """
        return self.optimize(player, [enemy_layout], iterations=50)

    def get_recommended_swap(
        self, player: "PlayerState"
    ) -> Optional[Tuple[str, str]]:
        """
        Find best swap to improve positioning.

        Args:
            player: Player state.

        Returns:
            Tuple of (unit1_id, unit2_id) to swap, or None.
        """
        units = list(player.units.board.keys())
        if len(units) < 2:
            return None

        current_layout = {}
        for uid, inst in player.units.board.items():
            if inst.position:
                pos = inst.position
                if isinstance(pos, tuple):
                    current_layout[uid] = HexPosition(pos[0], pos[1])
                else:
                    current_layout[uid] = pos
        current_score = self._evaluate_layout(current_layout, player, None)

        best_swap = None
        best_improvement = 0

        # Try all swaps
        for i, u1 in enumerate(units):
            for u2 in units[i + 1 :]:
                # Create swapped layout
                new_layout = dict(current_layout)
                if u1 in new_layout and u2 in new_layout:
                    new_layout[u1], new_layout[u2] = new_layout[u2], new_layout[u1]

                    new_score = self._evaluate_layout(new_layout, player, None)
                    improvement = new_score - current_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (u1, u2)

        return best_swap

    def _classify_roles(
        self, instances: Dict[str, "ChampionInstance"]
    ) -> Dict[str, str]:
        """Classify unit roles."""
        roles = {}
        for unit_id, inst in instances.items():
            roles[unit_id] = self._get_unit_role(inst)
        return roles

    def _get_unit_role(self, instance: "ChampionInstance") -> str:
        """
        Determine unit's tactical role.

        Args:
            instance: Champion instance.

        Returns:
            Role string: "tank", "assassin", "carry", "support", "flex".
        """
        champion = instance.champion
        traits = [t.lower() for t in champion.traits]

        # Check traits for role hints
        tank_traits = {"tank", "guardian", "warden", "bastion", "juggernaut"}
        if any(t in tank_traits for t in traits):
            return "tank"

        if "assassin" in traits:
            return "assassin"

        support_traits = {"support", "enchanter", "invoker"}
        if any(t in support_traits for t in traits):
            return "support"

        # Check stats
        stats = champion.stats

        # Ranged units are usually carries
        if stats.attack_range >= 3:
            return "carry"

        # High defensive stats = tank
        if stats.armor > 50 or stats.health[0] > 800:
            return "tank"

        # High attack damage = carry
        if stats.attack_damage[0] > 55:
            return "carry"

        return "flex"

    def _generate_initial_layout(
        self, units: List[str], roles: Dict[str, str]
    ) -> Dict[str, HexPosition]:
        """Generate initial layout based on roles."""
        layout = {}

        # Categorize units
        tanks = [u for u in units if roles.get(u) == "tank"]
        carries = [u for u in units if roles.get(u) == "carry"]
        assassins = [u for u in units if roles.get(u) == "assassin"]
        others = [
            u for u in units if roles.get(u) not in ["tank", "carry", "assassin"]
        ]

        used_positions = set()

        # Place tanks in frontline (center)
        for i, unit in enumerate(tanks):
            col = 3 + (i % 2) * (1 if i % 4 < 2 else -1) * ((i // 2) + 1)
            col = max(0, min(6, col))
            pos = HexPosition(3, col)
            while pos in used_positions:
                col = (col + 1) % 7
                pos = HexPosition(3, col)
            layout[unit] = pos
            used_positions.add(pos)

        # Place carries in backline (center)
        for i, unit in enumerate(carries):
            col = 3 + (i % 2) * (1 if i % 4 < 2 else -1) * ((i // 2) + 1)
            col = max(0, min(6, col))
            row = 0 if i % 2 == 0 else 1
            pos = HexPosition(row, col)
            while pos in used_positions:
                col = (col + 1) % 7
                pos = HexPosition(row, col)
            layout[unit] = pos
            used_positions.add(pos)

        # Place assassins in corners
        corners = [
            HexPosition(0, 0),
            HexPosition(0, 6),
            HexPosition(1, 0),
            HexPosition(1, 6),
        ]
        for i, unit in enumerate(assassins):
            if i < len(corners):
                pos = corners[i]
                if pos not in used_positions:
                    layout[unit] = pos
                    used_positions.add(pos)

        # Place remaining units in empty spots
        remaining = others + [u for u in units if u not in layout]
        empty = []
        for row in range(4):
            for col in range(7):
                pos = HexPosition(row, col)
                if pos not in used_positions:
                    empty.append(pos)

        for unit in remaining:
            if empty:
                layout[unit] = empty.pop(0)

        return layout

    def _mutate_layout(
        self, layout: Dict[str, HexPosition], units: List[str]
    ) -> Dict[str, HexPosition]:
        """Create mutated layout by swapping two units."""
        new_layout = layout.copy()

        if len(units) < 2:
            return new_layout

        u1, u2 = self._rng.sample(units, 2)
        if u1 in new_layout and u2 in new_layout:
            new_layout[u1], new_layout[u2] = new_layout[u2], new_layout[u1]

        return new_layout

    def _evaluate_layout(
        self,
        layout: Dict[str, HexPosition],
        player: "PlayerState",
        enemy_layouts: Optional[List[BoardLayout]],
    ) -> float:
        """Evaluate layout quality."""
        score = 0.0
        roles = self._classify_roles(player.units.board)

        for unit_id, pos in layout.items():
            role = roles.get(unit_id, "flex")
            instance = player.units.board.get(unit_id)
            if instance:
                pos_score, _ = self._score_position(pos, instance, player, role)
                score += pos_score

        return score

    def _score_position(
        self,
        position: HexPosition,
        instance: "ChampionInstance",
        player: "PlayerState",
        role: str,
    ) -> Tuple[float, List[str]]:
        """Score a position for a unit."""
        score = 0.0
        reasons = []

        # Role-based scoring
        if role == "tank" and position.row == 3:
            score += 20
            reasons.append("Frontline tank")
        elif role == "tank" and position.row == 2:
            score += 10
            reasons.append("Midline tank")

        if role == "carry" and position.row in [0, 1]:
            score += 20
            reasons.append("Backline carry")

        if role == "assassin" and position.col in [0, 6]:
            score += 15
            reasons.append("Corner assassin")

        # Center protection bonus for carries
        if role == "carry" and 2 <= position.col <= 4:
            score += 10
            reasons.append("Center protection")

        # Spread penalty for clumping
        # (In a full implementation, check nearby ally positions)

        return score, reasons

    def _calculate_win_rate(
        self,
        layout: Dict[str, HexPosition],
        player: "PlayerState",
        enemy_layouts: Optional[List[BoardLayout]],
    ) -> float:
        """Calculate win rate using simulator."""
        if self.simulator is None or enemy_layouts is None or not enemy_layouts:
            return 0.5

        # Would run actual simulations here
        # For now, return estimate based on score
        return 0.5

    def _describe_layout(
        self, layout: Dict[str, HexPosition], roles: Dict[str, str]
    ) -> str:
        """Generate layout description."""
        front = sum(1 for u, p in layout.items() if p.row == 3)
        mid = sum(1 for u, p in layout.items() if p.row == 2)
        back = sum(1 for u, p in layout.items() if p.row in [0, 1])

        return f"Front: {front}, Mid: {mid}, Back: {back}"

    def apply_layout(
        self, player: "PlayerState", layout: BoardLayout
    ) -> bool:
        """
        Apply a layout to player's board.

        Args:
            player: Player state.
            layout: Layout to apply.

        Returns:
            True if successful.
        """
        # In a full implementation, this would actually move units
        # For now, just validate
        for unit_id, pos in layout.positions.items():
            if unit_id not in player.units.board:
                return False
            if not (0 <= pos.row < self.ROWS and 0 <= pos.col < self.COLS):
                return False

        return True

    def get_formation_templates(self) -> Dict[str, Dict[str, HexPosition]]:
        """
        Get common formation templates.

        Returns:
            Dict of formation name to position template.
        """
        return {
            "box": {
                "tank1": HexPosition(3, 2),
                "tank2": HexPosition(3, 4),
                "carry1": HexPosition(0, 2),
                "carry2": HexPosition(0, 4),
                "support": HexPosition(1, 3),
            },
            "line": {
                "tank1": HexPosition(3, 1),
                "tank2": HexPosition(3, 3),
                "tank3": HexPosition(3, 5),
                "carry1": HexPosition(0, 3),
            },
            "corner_left": {
                "tank1": HexPosition(3, 0),
                "tank2": HexPosition(3, 1),
                "carry1": HexPosition(0, 0),
                "carry2": HexPosition(1, 0),
            },
            "corner_right": {
                "tank1": HexPosition(3, 5),
                "tank2": HexPosition(3, 6),
                "carry1": HexPosition(0, 6),
                "carry2": HexPosition(1, 6),
            },
            "spread": {
                "unit1": HexPosition(3, 0),
                "unit2": HexPosition(3, 3),
                "unit3": HexPosition(3, 6),
                "unit4": HexPosition(0, 1),
                "unit5": HexPosition(0, 5),
            },
        }
