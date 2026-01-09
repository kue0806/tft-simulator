"""
Dependency injection for API services.
"""

from functools import lru_cache

from .services.game_service import GameService
from .services.combat_service import CombatService
from .services.optimizer_service import OptimizerService


@lru_cache()
def get_game_service() -> GameService:
    """Get GameService singleton."""
    return GameService()


@lru_cache()
def get_combat_service() -> CombatService:
    """Get CombatService singleton."""
    return CombatService(get_game_service())


@lru_cache()
def get_optimizer_service() -> OptimizerService:
    """Get OptimizerService singleton."""
    return OptimizerService(get_game_service())
