"""Monte Carlo Combat Simulation for TFT.

Runs multiple combat simulations to estimate win rates,
expected damage, and other statistical outcomes.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import statistics

from .combat_engine import CombatEngine
from .combat_unit import CombatResult
from .hex_grid import HexPosition, Team

if TYPE_CHECKING:
    from ..core.player_units import ChampionInstance


@dataclass
class SimulationResult:
    """
    Result of Monte Carlo combat simulation.

    Contains statistical analysis of multiple combat runs.
    """

    # Win statistics
    blue_win_rate: float  # 0.0 to 1.0
    red_win_rate: float
    draw_rate: float

    # Damage statistics
    avg_damage_to_blue: float  # Average damage when blue loses
    avg_damage_to_red: float  # Average damage when red loses

    # Time statistics
    avg_combat_duration: float  # In ticks
    min_combat_duration: int
    max_combat_duration: int

    # Survival statistics
    avg_blue_survivors: float
    avg_red_survivors: float

    # Sample size
    iterations: int

    # Confidence interval (95%)
    win_rate_confidence: Tuple[float, float] = (0.0, 1.0)

    # Raw results for detailed analysis
    individual_results: List[CombatResult] = field(default_factory=list)


@dataclass
class PositioningAnalysis:
    """Analysis of different positioning options."""

    position_win_rates: Dict[str, float]  # position_key -> win_rate
    best_position: str
    best_win_rate: float
    position_damage_dealt: Dict[str, float]
    position_damage_taken: Dict[str, float]


class CombatSimulator:
    """
    Monte Carlo combat simulator.

    Runs multiple combat simulations to estimate outcomes.

    Usage:
        simulator = CombatSimulator()
        result = simulator.simulate(
            blue_units, red_units,
            blue_positions, red_positions,
            iterations=1000
        )
        print(f"Blue win rate: {result.blue_win_rate:.1%}")
    """

    def __init__(self, base_seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            base_seed: Base seed for reproducibility (seeds will be derived).
        """
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)

    def simulate(
        self,
        blue_instances: List["ChampionInstance"],
        red_instances: List["ChampionInstance"],
        blue_positions: List[HexPosition],
        red_positions: List[HexPosition],
        iterations: int = 100,
        stat_calculator: Any = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            blue_instances: Blue team champion instances.
            red_instances: Red team champion instances.
            blue_positions: Blue team positions.
            red_positions: Red team positions.
            iterations: Number of simulation runs.
            stat_calculator: Optional stat calculator.
            parallel: Whether to run simulations in parallel.
            max_workers: Max parallel workers.

        Returns:
            SimulationResult with statistical analysis.
        """
        results: List[CombatResult] = []

        if parallel and iterations > 10:
            results = self._run_parallel(
                blue_instances, red_instances,
                blue_positions, red_positions,
                iterations, stat_calculator, max_workers
            )
        else:
            results = self._run_sequential(
                blue_instances, red_instances,
                blue_positions, red_positions,
                iterations, stat_calculator
            )

        return self._analyze_results(results, iterations)

    def _run_sequential(
        self,
        blue_instances: List["ChampionInstance"],
        red_instances: List["ChampionInstance"],
        blue_positions: List[HexPosition],
        red_positions: List[HexPosition],
        iterations: int,
        stat_calculator: Any,
    ) -> List[CombatResult]:
        """Run simulations sequentially."""
        results = []

        for i in range(iterations):
            seed = self._get_iteration_seed(i)
            engine = CombatEngine(seed=seed)
            engine.setup_combat(
                blue_instances, red_instances,
                blue_positions, red_positions,
                stat_calculator
            )
            result = engine.run_combat()
            results.append(result)

        return results

    def _run_parallel(
        self,
        blue_instances: List["ChampionInstance"],
        red_instances: List["ChampionInstance"],
        blue_positions: List[HexPosition],
        red_positions: List[HexPosition],
        iterations: int,
        stat_calculator: Any,
        max_workers: int,
    ) -> List[CombatResult]:
        """Run simulations in parallel."""
        results = []

        def run_single(iteration: int) -> CombatResult:
            seed = self._get_iteration_seed(iteration)
            engine = CombatEngine(seed=seed)
            engine.setup_combat(
                blue_instances, red_instances,
                blue_positions, red_positions,
                stat_calculator
            )
            return engine.run_combat()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single, i): i
                for i in range(iterations)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def _get_iteration_seed(self, iteration: int) -> int:
        """Get deterministic seed for an iteration."""
        if self.base_seed is not None:
            return self.base_seed + iteration
        return self.rng.randint(0, 2**31)

    def _analyze_results(
        self, results: List[CombatResult], iterations: int
    ) -> SimulationResult:
        """Analyze simulation results."""
        blue_wins = sum(1 for r in results if r.winner == Team.BLUE)
        red_wins = sum(1 for r in results if r.winner == Team.RED)
        draws = iterations - blue_wins - red_wins

        blue_win_rate = blue_wins / iterations if iterations > 0 else 0
        red_win_rate = red_wins / iterations if iterations > 0 else 0
        draw_rate = draws / iterations if iterations > 0 else 0

        # Damage statistics
        blue_loss_damages = [
            r.total_damage_to_loser for r in results if r.winner == Team.RED
        ]
        red_loss_damages = [
            r.total_damage_to_loser for r in results if r.winner == Team.BLUE
        ]

        avg_damage_to_blue = (
            statistics.mean(blue_loss_damages) if blue_loss_damages else 0
        )
        avg_damage_to_red = (
            statistics.mean(red_loss_damages) if red_loss_damages else 0
        )

        # Duration statistics
        durations = [r.rounds_taken for r in results]
        avg_duration = statistics.mean(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # Survival statistics
        blue_survivors = [r.winning_units_remaining for r in results if r.winner == Team.BLUE]
        red_survivors = [r.winning_units_remaining for r in results if r.winner == Team.RED]

        avg_blue_survivors = statistics.mean(blue_survivors) if blue_survivors else 0
        avg_red_survivors = statistics.mean(red_survivors) if red_survivors else 0

        # Confidence interval (Wilson score interval approximation)
        confidence = self._calculate_confidence_interval(blue_wins, iterations)

        return SimulationResult(
            blue_win_rate=blue_win_rate,
            red_win_rate=red_win_rate,
            draw_rate=draw_rate,
            avg_damage_to_blue=avg_damage_to_blue,
            avg_damage_to_red=avg_damage_to_red,
            avg_combat_duration=avg_duration,
            min_combat_duration=min_duration,
            max_combat_duration=max_duration,
            avg_blue_survivors=avg_blue_survivors,
            avg_red_survivors=avg_red_survivors,
            iterations=iterations,
            win_rate_confidence=confidence,
            individual_results=results,
        )

    def _calculate_confidence_interval(
        self, successes: int, n: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if n == 0:
            return (0.0, 1.0)

        z = 1.96  # 95% confidence
        p = successes / n

        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator

        spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)

        return (lower, upper)

    def simulate_positioning_options(
        self,
        blue_instances: List["ChampionInstance"],
        red_instances: List["ChampionInstance"],
        blue_position_options: List[List[HexPosition]],
        red_positions: List[HexPosition],
        iterations_per_position: int = 50,
        stat_calculator: Any = None,
    ) -> PositioningAnalysis:
        """
        Compare different positioning options.

        Args:
            blue_instances: Blue team units.
            red_instances: Red team units.
            blue_position_options: List of position configurations to test.
            red_positions: Fixed red team positions.
            iterations_per_position: Simulations per position option.
            stat_calculator: Optional stat calculator.

        Returns:
            PositioningAnalysis with comparison results.
        """
        position_results: Dict[str, SimulationResult] = {}

        for i, positions in enumerate(blue_position_options):
            position_key = f"option_{i}"
            result = self.simulate(
                blue_instances, red_instances,
                positions, red_positions,
                iterations=iterations_per_position,
                stat_calculator=stat_calculator,
            )
            position_results[position_key] = result

        # Analyze
        win_rates = {k: r.blue_win_rate for k, r in position_results.items()}
        damage_dealt = {k: r.avg_damage_to_red for k, r in position_results.items()}
        damage_taken = {k: r.avg_damage_to_blue for k, r in position_results.items()}

        best_key = max(win_rates.keys(), key=lambda k: win_rates[k])

        return PositioningAnalysis(
            position_win_rates=win_rates,
            best_position=best_key,
            best_win_rate=win_rates[best_key],
            position_damage_dealt=damage_dealt,
            position_damage_taken=damage_taken,
        )

    def calculate_expected_placement_change(
        self,
        current_hp: int,
        win_rate: float,
        avg_damage_on_loss: float,
        avg_damage_on_win: float = 0,
    ) -> float:
        """
        Calculate expected HP change.

        Args:
            current_hp: Current player HP.
            win_rate: Win probability.
            avg_damage_on_loss: Average damage taken on loss.
            avg_damage_on_win: Average damage dealt on win (usually 0).

        Returns:
            Expected HP change (negative means losing HP).
        """
        loss_rate = 1 - win_rate
        expected_change = (
            -loss_rate * avg_damage_on_loss
            + win_rate * avg_damage_on_win
        )
        return expected_change


def quick_simulate(
    blue_units: List["ChampionInstance"],
    red_units: List["ChampionInstance"],
    blue_positions: List[HexPosition],
    red_positions: List[HexPosition],
    iterations: int = 100,
) -> float:
    """
    Quick simulation helper returning blue win rate.

    Args:
        blue_units: Blue team units.
        red_units: Red team units.
        blue_positions: Blue team positions.
        red_positions: Red team positions.
        iterations: Number of simulations.

    Returns:
        Blue team win rate (0.0 to 1.0).
    """
    simulator = CombatSimulator()
    result = simulator.simulate(
        blue_units, red_units,
        blue_positions, red_positions,
        iterations=iterations,
    )
    return result.blue_win_rate


def estimate_board_strength(
    units: List["ChampionInstance"],
    positions: List[HexPosition],
    iterations: int = 50,
) -> Dict[str, float]:
    """
    Estimate board strength by simulating against itself.

    This gives a baseline measure of how strong a board is.

    Args:
        units: Champion instances.
        positions: Unit positions.
        iterations: Simulations to run.

    Returns:
        Dict with strength metrics.
    """
    simulator = CombatSimulator()

    # Mirror match
    result = simulator.simulate(
        units, units,
        positions, positions,
        iterations=iterations,
    )

    # A stronger board should win 50% in mirror matches
    # The interesting metrics are damage dealt and survivors
    return {
        "avg_survivors": (result.avg_blue_survivors + result.avg_red_survivors) / 2,
        "avg_combat_duration": result.avg_combat_duration,
        "avg_damage_dealt": result.avg_damage_to_red,  # Should equal avg_damage_to_blue in mirror
    }
