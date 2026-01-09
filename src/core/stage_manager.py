"""Stage Manager for TFT Set 16.

Manages game stages and rounds, including round types and timing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.constants import (
    PVE_ROUNDS,
    CAROUSEL_ROUNDS,
    AUGMENT_ROUNDS,
    PVE_MONSTERS,
    PASSIVE_XP_PER_ROUND,
    get_round_passive_gold,
)


class RoundType(Enum):
    """Types of rounds in TFT."""

    PVE = "pve"
    PVP = "pvp"
    CAROUSEL = "carousel"


class RoundPhase(Enum):
    """Phases within a round."""

    AUGMENT_SELECTION = "augment_selection"  # Augment choice phase
    CAROUSEL = "carousel"                     # Shared draft
    PLANNING = "planning"                     # Shop/positioning phase
    COMBAT = "combat"                         # Battle phase
    LOOT = "loot"                            # PvE loot collection


@dataclass
class RoundInfo:
    """Information about a specific round."""

    stage: str  # e.g., "2-1"
    stage_num: int  # e.g., 2
    round_num: int  # e.g., 1
    round_type: RoundType
    passive_gold: int  # Gold gained this round
    passive_xp: int  # XP gained this round
    is_carousel: bool = False
    is_augment: bool = False
    is_pve: bool = False
    monster_type: Optional[str] = None  # For PvE rounds
    phases: list[RoundPhase] = field(default_factory=list)

    def __post_init__(self):
        """Build the phases for this round."""
        if not self.phases:
            self.phases = self._build_phases()

    def _build_phases(self) -> list[RoundPhase]:
        """Determine the sequence of phases for this round."""
        phases = []

        # Augment selection happens at start of specific rounds
        if self.is_augment:
            phases.append(RoundPhase.AUGMENT_SELECTION)

        # Carousel rounds
        if self.is_carousel:
            phases.append(RoundPhase.CAROUSEL)
            # After 1-1 carousel, there's no combat - go to next round
            if self.stage != "1-1":
                phases.append(RoundPhase.PLANNING)
                phases.append(RoundPhase.COMBAT)
        else:
            # Normal round: planning then combat
            phases.append(RoundPhase.PLANNING)
            phases.append(RoundPhase.COMBAT)

        # PvE rounds have loot phase after combat
        if self.is_pve:
            phases.append(RoundPhase.LOOT)

        return phases


class StageManager:
    """Manage game stages and rounds."""

    # Round types by stage-round
    ROUND_TYPES = {
        # Stage 1: Opening carousel + PvE
        "1-1": RoundType.CAROUSEL,  # Opening carousel
        "1-2": RoundType.PVE,
        "1-3": RoundType.PVE,
        "1-4": RoundType.PVE,
        # Stage 2+: PvP with special rounds
        "2-1": RoundType.PVP,  # First augment
        "2-2": RoundType.PVP,
        "2-3": RoundType.PVP,
        "2-4": RoundType.CAROUSEL,
        "2-5": RoundType.PVP,
        "2-6": RoundType.PVP,
        "2-7": RoundType.PVE,  # Krugs
        # Stage 3
        "3-1": RoundType.PVP,
        "3-2": RoundType.PVP,  # Second augment
        "3-3": RoundType.PVP,
        "3-4": RoundType.CAROUSEL,
        "3-5": RoundType.PVP,
        "3-6": RoundType.PVP,
        "3-7": RoundType.PVE,  # Wolves
        # Stage 4
        "4-1": RoundType.PVP,
        "4-2": RoundType.PVP,  # Third augment
        "4-3": RoundType.PVP,
        "4-4": RoundType.CAROUSEL,
        "4-5": RoundType.PVP,
        "4-6": RoundType.PVP,
        "4-7": RoundType.PVE,  # Raptors (last component drops)
        # Stage 5
        "5-1": RoundType.PVP,
        "5-2": RoundType.PVP,
        "5-3": RoundType.PVP,
        "5-4": RoundType.CAROUSEL,  # Combined items
        "5-5": RoundType.PVP,
        "5-6": RoundType.PVP,
        "5-7": RoundType.PVE,  # Dragon/Rift Herald
        # Stage 6
        "6-1": RoundType.PVP,
        "6-2": RoundType.PVP,
        "6-3": RoundType.PVP,
        "6-4": RoundType.CAROUSEL,  # Combined + components, 5-costs
        "6-5": RoundType.PVP,
        "6-6": RoundType.PVP,
        "6-7": RoundType.PVE,  # Elder Dragon
        # Stage 7
        "7-1": RoundType.PVP,
        "7-2": RoundType.PVP,
        "7-3": RoundType.PVP,
        "7-4": RoundType.CAROUSEL,
        "7-5": RoundType.PVP,
        "7-6": RoundType.PVP,
        "7-7": RoundType.PVE,  # Baron Nashor
    }

    def __init__(self):
        """Initialize stage manager at stage 1-1."""
        self.current_stage = 1
        self.current_round = 1
        self.total_rounds = 0
        self.current_phase_index = 0  # Track current phase within round

    def get_stage_string(self) -> str:
        """
        Get current stage as string (e.g., '2-3').

        Returns:
            Stage string in format "X-Y".
        """
        return f"{self.current_stage}-{self.current_round}"

    def advance_round(self) -> RoundInfo:
        """
        Advance to next round and return info.

        Returns:
            RoundInfo for the new current round.
        """
        self.current_round += 1
        self.total_rounds += 1
        self.current_phase_index = 0  # Reset phase

        # Check for stage advancement
        max_rounds = self._get_max_rounds_in_stage()
        if self.current_round > max_rounds:
            self.current_stage += 1
            self.current_round = 1

        return self.get_current_round_info()

    def _get_max_rounds_in_stage(self) -> int:
        """
        Get number of rounds in current stage.

        Returns:
            Maximum round number in current stage.
        """
        if self.current_stage == 1:
            return 4  # Stage 1 has 4 rounds (carousel + 3 PvE)
        return 7  # Other stages have 7 rounds

    def get_current_round_info(self) -> RoundInfo:
        """
        Get info about current round.

        Returns:
            RoundInfo with all round details.
        """
        stage_str = self.get_stage_string()

        # Determine round type
        round_type = self.ROUND_TYPES.get(stage_str, RoundType.PVP)

        # Passive gold based on round
        passive_gold = get_round_passive_gold(stage_str)

        # Passive XP (2 per round starting from 1-2)
        passive_xp = PASSIVE_XP_PER_ROUND if stage_str != "1-1" else 0

        # Check special round types
        is_carousel = stage_str in CAROUSEL_ROUNDS
        is_augment = stage_str in AUGMENT_ROUNDS
        is_pve = stage_str in PVE_ROUNDS or round_type == RoundType.PVE
        monster_type = PVE_MONSTERS.get(stage_str)

        return RoundInfo(
            stage=stage_str,
            stage_num=self.current_stage,
            round_num=self.current_round,
            round_type=round_type,
            passive_gold=passive_gold,
            passive_xp=passive_xp,
            is_carousel=is_carousel,
            is_augment=is_augment,
            is_pve=is_pve,
            monster_type=monster_type,
        )

    def get_current_phase(self) -> Optional[RoundPhase]:
        """Get the current phase of the round."""
        round_info = self.get_current_round_info()
        if self.current_phase_index < len(round_info.phases):
            return round_info.phases[self.current_phase_index]
        return None

    def advance_phase(self) -> Optional[RoundPhase]:
        """
        Advance to the next phase within the current round.

        Returns:
            The new current phase, or None if round is complete.
        """
        round_info = self.get_current_round_info()
        self.current_phase_index += 1

        if self.current_phase_index < len(round_info.phases):
            return round_info.phases[self.current_phase_index]
        return None

    def is_round_complete(self) -> bool:
        """Check if all phases in the current round are complete."""
        round_info = self.get_current_round_info()
        return self.current_phase_index >= len(round_info.phases)

    def get_rounds_until(self, target_stage: str) -> int:
        """
        Calculate rounds until target stage.

        Args:
            target_stage: Target stage string (e.g., "4-2").

        Returns:
            Number of rounds until target.
        """
        target_parts = target_stage.split("-")
        target_stage_num = int(target_parts[0])
        target_round_num = int(target_parts[1])

        # If already past target
        if self.current_stage > target_stage_num:
            return 0
        if self.current_stage == target_stage_num and self.current_round >= target_round_num:
            return 0

        rounds = 0

        # Rounds remaining in current stage
        if self.current_stage == target_stage_num:
            return target_round_num - self.current_round

        # Complete current stage
        rounds += self._get_max_rounds_in_stage() - self.current_round

        # Add full stages in between
        for stage in range(self.current_stage + 1, target_stage_num):
            if stage == 1:
                rounds += 4
            else:
                rounds += 7

        # Add rounds in target stage
        rounds += target_round_num

        return rounds

    def is_rolldown_timing(self) -> bool:
        """
        Check if current round is a common rolldown timing.

        Returns:
            True if this is a standard rolldown round.
        """
        stage_str = self.get_stage_string()
        rolldown_timings = ["3-2", "4-1", "4-2", "4-5", "5-1"]
        return stage_str in rolldown_timings

    def is_level_timing(self) -> bool:
        """
        Check if current round is a standard leveling timing.

        Returns:
            True if this is a standard leveling round.
        """
        timings = {
            "2-1": 4,  # Level 4
            "2-5": 5,  # Level 5
            "3-2": 6,  # Level 6
            "4-1": 7,  # Level 7
            "4-2": 8,  # Level 8
            "5-1": 9,  # Level 9
        }
        return self.get_stage_string() in timings

    def get_recommended_level(self) -> int:
        """
        Get recommended level for current stage.

        Returns:
            Recommended player level.
        """
        stage_str = self.get_stage_string()
        level_timings = {
            "2-1": 4,
            "2-5": 5,
            "3-2": 6,
            "4-1": 7,
            "4-2": 8,
            "5-1": 9,
        }
        # Find the most recent timing
        for timing, level in reversed(list(level_timings.items())):
            timing_parts = timing.split("-")
            timing_stage = int(timing_parts[0])
            timing_round = int(timing_parts[1])
            if (
                self.current_stage > timing_stage
                or (self.current_stage == timing_stage and self.current_round >= timing_round)
            ):
                return level
        return 3  # Default starting level recommendation

    def is_pve_round(self) -> bool:
        """
        Check if current round is PvE.

        Returns:
            True if current round is PvE.
        """
        return self.get_current_round_info().is_pve

    def is_carousel_round(self) -> bool:
        """
        Check if current round is a carousel.

        Returns:
            True if current round is a carousel.
        """
        return self.get_current_round_info().is_carousel

    def is_augment_round(self) -> bool:
        """
        Check if current round has augment selection.

        Returns:
            True if current round has augment selection.
        """
        return self.get_current_round_info().is_augment

    def set_stage(self, stage_str: str) -> None:
        """
        Set the current stage directly.

        Args:
            stage_str: Stage string in format "X-Y".
        """
        parts = stage_str.split("-")
        self.current_stage = int(parts[0])
        self.current_round = int(parts[1])
