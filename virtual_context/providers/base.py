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
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        """Send a completion request with automatic retry on transient errors.

        Returns:
            A ``(text, usage)`` tuple where *usage* is the token usage dict
            from the API response (keys vary by provider but typically include
            ``input_tokens``/``prompt_tokens`` and ``output_tokens``/
            ``completion_tokens``).

        .. deprecated::
            The ``last_usage`` instance attribute is still written for
            backward compatibility but should not be relied upon in
            concurrent contexts.  Use the returned *usage* dict instead.
        """
        url = self._get_url()
        headers = self._get_headers()
        payload = self._build_payload(system, user, max_tokens)
        client = self._get_client()

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = client.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    self.last_usage = usage  # deprecated -- kept for backward compat
                    return self._extract_text(data), usage

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
