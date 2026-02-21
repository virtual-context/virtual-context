"""GenericOpenAIProvider: OpenAI-compatible endpoint via httpx.

Works with Ollama, vLLM, LM Studio, or any server exposing /v1/chat/completions.
"""

from __future__ import annotations

from .base import BaseProvider


class GenericOpenAIProvider(BaseProvider):
    """LLM provider using any OpenAI-compatible chat completions API."""

    _timeout = 120.0

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen3:4b-instruct-2507-fp16",
        temperature: float = 0.3,
        api_key: str = "not-needed",
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def _provider_name(self) -> str:
        return "generic_openai"

    def _get_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

    def _extract_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
