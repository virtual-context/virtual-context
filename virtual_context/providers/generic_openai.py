"""GenericOpenAIProvider: OpenAI-compatible endpoint via httpx.

Works with Ollama, vLLM, LM Studio, or any server exposing /v1/chat/completions.
"""

from __future__ import annotations

from .base import BaseProvider


class GenericOpenAIProvider(BaseProvider):
    """LLM provider using any OpenAI-compatible chat completions API."""

    _timeout = 20.0

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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Some OpenAI-hosted models (e.g., gpt-5-mini) only support default
        # temperature=1 and reject explicit non-default values.
        if "api.openai.com" not in self.base_url or self.temperature == 1:
            payload["temperature"] = self.temperature
        # OpenAI-hosted chat models (notably GPT-5 family) expect
        # max_completion_tokens instead of max_tokens.
        if "api.openai.com" in self.base_url:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        return payload

    def _extract_text(self, data: dict) -> str:
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
