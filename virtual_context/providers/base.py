"""LLM Provider base class with shared retry logic."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import httpx

from ..types import LLMProviderError

MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]


class BaseProvider(ABC):
    """Abstract base for LLM providers. Subclasses override hook methods;
    the retry loop in ``complete()`` is shared."""

    _timeout: float = 60.0

    def __init__(self) -> None:
        self.last_usage: dict = {}

    # -- hook methods subclasses must implement --

    @abstractmethod
    def _provider_name(self) -> str: ...

    @abstractmethod
    def _get_url(self) -> str: ...

    @abstractmethod
    def _get_headers(self) -> dict: ...

    @abstractmethod
    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict: ...

    @abstractmethod
    def _extract_text(self, data: dict) -> str: ...

    # -- shared retry logic --

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Send a completion request with automatic retry on transient errors."""
        url = self._get_url()
        headers = self._get_headers()
        payload = self._build_payload(system, user, max_tokens)

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    self.last_usage = data.get("usage", {})
                    return self._extract_text(data)

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMProviderError(
                        f"HTTP {response.status_code}: {response.text}",
                        provider=self._provider_name(),
                        status_code=response.status_code,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                    continue

                raise LLMProviderError(
                    f"HTTP {response.status_code}: {response.text}",
                    provider=self._provider_name(),
                    status_code=response.status_code,
                )

            except httpx.HTTPError as e:
                last_error = LLMProviderError(
                    f"HTTP error: {e}",
                    provider=self._provider_name(),
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                continue

        raise last_error or LLMProviderError(
            "Max retries exceeded", provider=self._provider_name()
        )


# Re-exports kept for backward compatibility
from ..types import LLMProvider  # noqa: E402, F401

__all__ = ["BaseProvider", "LLMProvider", "LLMProviderError"]
