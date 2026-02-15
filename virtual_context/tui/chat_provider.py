"""Anthropic streaming chat provider using httpx SSE."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

from ..types import LLMProviderError

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class ChatProvider:
    """Streams chat responses from the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise LLMProviderError(
                "No API key. Set ANTHROPIC_API_KEY or pass --api-key.",
                provider="anthropic",
            )

    def stream_message(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Yield text delta chunks from Anthropic streaming API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise LLMProviderError(
                        f"HTTP {resp.status_code}: {resp.text}",
                        provider="anthropic",
                        status_code=resp.status_code,
                    )

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                    elif event.get("type") == "error":
                        raise LLMProviderError(
                            event.get("error", {}).get("message", "Unknown error"),
                            provider="anthropic",
                        )
