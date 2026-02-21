"""AnthropicProvider: calls Messages API via httpx (no SDK dependency)."""

from __future__ import annotations

import os

from ..types import LLMProviderError
from .base import BaseProvider

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """LLM provider using Anthropic Messages API directly via httpx."""

    _timeout = 60.0

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        model: str = "claude-haiku-4-5",
        temperature: float = 0.3,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.model = model
        self.temperature = temperature
        if not self.api_key:
            raise LLMProviderError(
                f"No API key found. Set {api_key_env} env var or pass api_key.",
                provider="anthropic",
            )

    def _provider_name(self) -> str:
        return "anthropic"

    def _get_url(self) -> str:
        return API_URL

    def _get_headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def _extract_text(self, data: dict) -> str:
        content = data.get("content", [])
        text_parts = [
            block["text"]
            for block in content
            if block.get("type") == "text"
        ]
        return "\n".join(text_parts)
