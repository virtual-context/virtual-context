"""GenericOpenAIProvider: OpenAI-compatible endpoint via httpx.

Works with Ollama, vLLM, LM Studio, or any server exposing /v1/chat/completions.
"""

from __future__ import annotations

import time

import httpx

from ..types import LLMProviderError

MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]


class GenericOpenAIProvider:
    """LLM provider using any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen3:4b-instruct-2507-fp16",
        temperature: float = 0.3,
        api_key: str = "not-needed",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.last_usage: dict = {}  # populated after each complete() call

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Send a chat completion request."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    self.last_usage = data.get("usage", {})
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMProviderError(
                        f"HTTP {response.status_code}: {response.text}",
                        provider="generic_openai",
                        status_code=response.status_code,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                    continue

                raise LLMProviderError(
                    f"HTTP {response.status_code}: {response.text}",
                    provider="generic_openai",
                    status_code=response.status_code,
                )

            except httpx.HTTPError as e:
                last_error = LLMProviderError(
                    f"HTTP error: {e}",
                    provider="generic_openai",
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                continue

        raise last_error or LLMProviderError(
            "Max retries exceeded", provider="generic_openai"
        )
