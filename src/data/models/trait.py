"""Trait data model for TFT Set 16."""

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class TraitType(StrEnum):
    """Trait classification."""
    ORIGIN = "origin"
    CLASS = "class"
    UNIQUE = "unique"  # Single champion traits (e.g., Heroic for Galio)


class TraitBreakpoint(BaseModel):
    """A single breakpoint for a trait."""
    count: int = Field(..., ge=1, description="Number of units required")
    effect: str = Field(..., description="Description of the effect at this breakpoint")
    stats: dict[str, float] = Field(default_factory=dict, description="Stat bonuses granted")


class Trait(BaseModel):
    """TFT Trait model (Origin or Class)."""
    id: str = Field(..., description="Unique identifier (lowercase, no spaces)")
    name: str = Field(..., description="Display name")
    type: TraitType
    description: str = Field(..., description="General trait description")
    breakpoints: list[TraitBreakpoint] = Field(..., min_length=1)
    champions: list[str] = Field(default_factory=list, description="List of champion IDs with this trait")
    mechanic: Optional[str] = Field(default=None, description="Special mechanic name (e.g., 'Silver Serpents')")

    model_config = {"use_enum_values": True}

    @property
    def min_units(self) -> int:
        """Minimum units needed to activate this trait."""
        return min(bp.count for bp in self.breakpoints)

    @property
    def max_units(self) -> int:
        """Maximum breakpoint for this trait."""
        return max(bp.count for bp in self.breakpoints)

    def get_active_breakpoint(self, count: int) -> Optional[TraitBreakpoint]:
        """Get the active breakpoint for a given unit count."""
        active = None
        for bp in sorted(self.breakpoints, key=lambda x: x.count):
            if count >= bp.count:
                active = bp
            else:
                break
        return active
