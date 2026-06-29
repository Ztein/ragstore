"""Config is fail-loud: missing required env raises; provided env loads cleanly."""

import pytest

from ragstore.config import load_settings, require_llm

REQUIRED = {
    "RAGSTORE_API_KEY": "secret",
    "SQLITE_PATH": "/tmp/ragstore.db",
    "EMBEDDING_BASE_URL": "http://embed.local/v1",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_API_KEY": "ek",
    "EMBEDDING_DIM": "1024",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Wipe anything that could leak in from the real environment or a .env file.
    for key in (*REQUIRED, "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _seed(monkeypatch):
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)


def test_missing_required_raises_runtime_error(monkeypatch):
    with pytest.raises(RuntimeError) as exc:
        load_settings(_env_file=None)
    msg = str(exc.value)
    assert "configuration" in msg.lower()
    # The clear message names the missing fields.
    assert "ragstore_api_key" in msg.lower()


def test_loads_when_required_present(monkeypatch):
    _seed(monkeypatch)
    settings = load_settings(_env_file=None)
    assert settings.ragstore_api_key == "secret"
    assert settings.embedding_dim == 1024
    # LLM optional at startup.
    assert settings.llm_base_url is None


def test_require_llm_fails_loud_when_unset(monkeypatch):
    _seed(monkeypatch)
    settings = load_settings(_env_file=None)
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        require_llm(settings)


def test_require_llm_returns_triple_when_set(monkeypatch):
    _seed(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-x")
    monkeypatch.setenv("LLM_API_KEY", "lk")
    settings = load_settings(_env_file=None)
    assert require_llm(settings) == ("http://llm.local/v1", "gpt-x", "lk")
