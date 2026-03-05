"""OllamaNativeProvider: Ollama's native /api/chat endpoint.

Required for qwen3 models where the OpenAI-compatible /v1/chat/completions
endpoint puts thinking output in 'reasoning' field with empty 'content'.
Uses think: false + format: json to get clean JSON output.
"""

from __future__ import annotations

from .base import BaseProvider


class OllamaNativeProvider(BaseProvider):
    """LLM provider using Ollama's native /api/chat endpoint."""

    _timeout = 60.0

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:30b-a3b",
        temperature: float = 0.3,
        num_predict: int = 500,
        force_json: bool = True,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.force_json = force_json

    def _provider_name(self) -> str:
        return "ollama_native"

    def _get_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _get_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens or self.num_predict,
                "temperature": self.temperature,
            },
        }
        if self.force_json:
            payload["format"] = "json"
        return payload

    def _extract_text(self, data: dict) -> str:
        return data.get("message", {}).get("content", "")
