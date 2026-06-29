"""Application configuration.

Fail-loud principle: required settings have no defaults, so a missing value raises
at load time with a clear message. Provider settings for *generation* (LLM) are
optional at startup and enforced only when a request actually uses ``generate``
(see :func:`require_llm`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Inbound auth (Bearer) — required.
    ragstore_api_key: str

    # SQLite (relational source of truth + job queue) — required.
    sqlite_path: str

    # Weaviate (vectors + hybrid search) — required host/port, secure off by default.
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_http_secure: bool = False
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_grpc_secure: bool = False
    weaviate_api_key: str | None = None

    # Embeddings — component-owned, external provider. All required.
    embedding_base_url: str
    embedding_model: str
    embedding_api_key: str
    embedding_dim: int

    # Generation (LLM) — optional at startup, required only when used.
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None


def load_settings(**overrides: object) -> Settings:
    """Load settings, raising a clear RuntimeError if anything required is missing."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise RuntimeError(f"Invalid ragstore configuration:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def require_llm(settings: Settings) -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for generation, failing loud if unconfigured."""
    if not settings.llm_base_url or not settings.llm_model or not settings.llm_api_key:
        raise RuntimeError(
            "Generation requested but LLM is not configured: set "
            "LLM_BASE_URL, LLM_MODEL and LLM_API_KEY."
        )
    return settings.llm_base_url, settings.llm_model, settings.llm_api_key
