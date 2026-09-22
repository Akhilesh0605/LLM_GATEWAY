# app/config.py
"""
Centralized configuration loaded from .env file.
All env vars are validated and typed via Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Infrastructure
    REDIS_SERVER_LINK: str
    POSTGRESQL_LINK: str
    ENCRYPTION_KEY: str

    # Semantic Cache Thresholds (cosine similarity)
    SIMILARITY_THRESHOLD_SIMPLE: float = 0.90
    SIMILARITY_THRESHOLD_MEDIUM: float = 0.92
    SIMILARITY_THRESHOLD_COMPLEX: float = 0.95

    # Cache TTL (seconds)
    CACHE_TTL_SIMPLE: int = 7200
    CACHE_TTL_MEDIUM: int = 3600
    CACHE_TTL_COMPLEX: int = 1800

    # Default daily budget for new users (USD)
    DAILY_BUDGET_USD: float = 10.0

    # Shadow evaluation sampling rate (10% of queries)
    EVALUATION_LOOP_RATE: float = 0.10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()