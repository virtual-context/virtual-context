"""Auth token resolution helpers for LongMemEval benchmark providers."""

from __future__ import annotations

import json
import os
from pathlib import Path

OPENAI_OAUTH_TOKEN_ENVS: tuple[str, ...] = (
    "OPENAI_OAUTH_ACCESS_TOKEN",
    "OPENAI_OAUTH_TOKEN",
    "OPENAI_ACCESS_TOKEN",
)

PROVIDER_API_KEY_ENVS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _load_openai_oauth_token_from_file(path: Path) -> str:
    """Best-effort OAuth token loader for Codex/OpenAI auth JSON files."""
    if not path.exists():
        return ""

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    data = raw

    candidates = [
        data.get("access_token"),
        data.get("token"),
        data.get("openai_access_token"),
        (data.get("tokens") or {}).get("access_token") if isinstance(data.get("tokens"), dict) else "",
        (data.get("openai") or {}).get("access_token") if isinstance(data.get("openai"), dict) else "",
        (data.get("oauth") or {}).get("access_token") if isinstance(data.get("oauth"), dict) else "",
    ]
    for token in candidates:
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


def resolve_openai_bearer_token(auth_mode: str = "auto") -> str:
    """Resolve OpenAI bearer token via OAuth and/or API key based on mode."""
    mode = (auth_mode or "auto").strip().lower()

    if mode in {"auto", "oauth"}:
        for env_name in OPENAI_OAUTH_TOKEN_ENVS:
            tok = os.environ.get(env_name, "").strip()
            if tok:
                return tok

        explicit_file = os.environ.get("OPENAI_OAUTH_TOKEN_FILE", "").strip()
        search_paths: list[Path] = []
        if explicit_file:
            search_paths.append(Path(explicit_file).expanduser())
        codex_home = os.environ.get("CODEX_HOME", "").strip()
        if codex_home:
            search_paths.append(Path(codex_home).expanduser() / "auth.json")
        search_paths.append(Path.home() / ".codex" / "auth.json")

        for path in search_paths:
            tok = _load_openai_oauth_token_from_file(path)
            if tok:
                return tok

    if mode in {"auto", "api-key"}:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            return key

    return ""


def resolve_provider_token(
    provider: str,
    *,
    explicit_token: str | None = None,
    auth_mode: str = "auto",
) -> str:
    """Resolve bearer/API token for a provider."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    if provider in {"openai", "openai-codex", "openai_codex"}:
        return resolve_openai_bearer_token(auth_mode)

    env_name = PROVIDER_API_KEY_ENVS.get(provider, "")
    if not env_name:
        return ""
    return os.environ.get(env_name, "").strip()
