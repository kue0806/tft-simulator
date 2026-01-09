"""
API configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """API settings."""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Simulation
    DEFAULT_SIMULATION_COUNT: int = 100
    MAX_SIMULATION_COUNT: int = 1000

    # Session
    SESSION_TIMEOUT: int = 3600  # 1 hour
    MAX_SESSIONS: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
