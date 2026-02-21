"""AnthropicProvider: calls Messages API via httpx (no SDK dependency)."""

from __future__ import annotations

import os
import time

import httpx

from ..types import LLMProviderError

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]


class AnthropicProvider:
    """LLM provider using Anthropic Messages API directly via httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        model: str = "claude-haiku-4-5",
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.model = model
        self.temperature = temperature
        self.last_usage: dict = {}  # populated after each complete() call
        if not self.api_key:
            raise LLMProviderError(
                f"No API key found. Set {api_key_env} env var or pass api_key.",
                provider="anthropic",
            )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Send a completion request to Anthropic Messages API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(API_URL, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    self.last_usage = data.get("usage", {})
                    content = data.get("content", [])
                    text_parts = [
                        block["text"]
                        for block in content
                        if block.get("type") == "text"
                    ]
                    return "\n".join(text_parts)

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMProviderError(
                        f"HTTP {response.status_code}: {response.text}",
                        provider="anthropic",
                        status_code=response.status_code,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                    continue

                raise LLMProviderError(
                    f"HTTP {response.status_code}: {response.text}",
                    provider="anthropic",
                    status_code=response.status_code,
                )

            except httpx.HTTPError as e:
                last_error = LLMProviderError(
                    f"HTTP error: {e}",
                    provider="anthropic",
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                continue

        raise last_error or LLMProviderError(
            "Max retries exceeded", provider="anthropic"
        )
