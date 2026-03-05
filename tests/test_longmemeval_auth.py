from __future__ import annotations

import json

from benchmarks.longmemeval.auth import resolve_openai_bearer_token, resolve_provider_token


def _clear_openai_auth_env(monkeypatch) -> None:
    for key in (
        "OPENAI_OAUTH_ACCESS_TOKEN",
        "OPENAI_OAUTH_TOKEN",
        "OPENAI_ACCESS_TOKEN",
        "OPENAI_OAUTH_TOKEN_FILE",
        "OPENAI_API_KEY",
        "CODEX_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_oauth_env_preferred_over_api_key(monkeypatch):
    _clear_openai_auth_env(monkeypatch)
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key")

    token = resolve_openai_bearer_token("auto")

    assert token == "oauth-token"


def test_oauth_file_fallback(monkeypatch, tmp_path):
    _clear_openai_auth_env(monkeypatch)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"access_token": "file-oauth-token"}))
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_FILE", str(auth_file))

    token = resolve_openai_bearer_token("oauth")

    assert token == "file-oauth-token"


def test_api_key_mode_uses_only_api_key(monkeypatch):
    _clear_openai_auth_env(monkeypatch)
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key")

    token = resolve_openai_bearer_token("api-key")

    assert token == "api-key"


def test_provider_resolver_openai_uses_oauth_mode(monkeypatch):
    _clear_openai_auth_env(monkeypatch)
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key")

    token = resolve_provider_token("openai", auth_mode="oauth")

    assert token == "oauth-token"
