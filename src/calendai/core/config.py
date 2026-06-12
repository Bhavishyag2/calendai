"""Application settings, loaded from environment / .env.

Model tiers are env-swappable by design: the agent model does the reasoning,
the utility model handles memory extraction and eval judging. Keeping them
separate is the latency/cost vs intelligence lever discussed in docs/tradeoffs.md.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    calendai_agent_model: str = "claude-sonnet-4-6"
    calendai_utility_model: str = "claude-haiku-4-5"

    calendai_db_path: str = "data/calendai.db"

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/callback"

    calendai_fernet_key: str = ""
    calendai_base_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
