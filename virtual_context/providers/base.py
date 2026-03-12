"""LLM Provider base class with shared retry logic."""

from __future__ import annotations

import json as _json
import logging
import time
from abc import ABC, abstractmethod

import httpx

from ..types import LLMProviderError

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF = [1.0, 2.0, 4.0, 4.0, 4.0]

# Shared httpx client with HTTP/2 — avoids urllib3 chunked-response hangs
# that occur with requests/urllib3 on HTTP/1.1 during sequential high-volume calls.
_http_client: httpx.Client | None = None


def _get_client(timeout: float) -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        # pool_timeout covers waiting for a connection from the pool.
        # All other timeouts set to `timeout` to enforce a hard per-request
        # deadline — prevents OpenRouter/Cloudflare drip-feeding bytes
        # that kept connections alive indefinitely with default settings.
        _http_client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(timeout, pool=10.0),
        )
    return _http_client


class BaseProvider(ABC):
    """Abstract base for LLM providers. Subclasses override hook methods;
    the retry loop in ``complete()`` is shared."""

    _timeout: float = 30.0

    def __init__(self) -> None:
        self.last_usage: dict = {}

    def close(self) -> None:
        global _http_client
        if _http_client is not None and not _http_client.is_closed:
            _http_client.close()
            _http_client = None

    def __del__(self) -> None:
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

        Uses httpx with HTTP/2 to avoid urllib3 chunked-response hangs
        observed with requests/urllib3 on HTTP/1.1 during sequential
        high-volume calls to OpenRouter/Cloudflare.

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

        model = payload.get("model", "unknown")
        prompt_preview = str(payload.get("messages", [{}])[-1].get("content", ""))[:80]
        logger.debug("LLM request: provider=%s model=%s url=%s prompt=%.80s...",
                      self._provider_name(), model, url, prompt_preview)

        client = _get_client(self._timeout)
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            t0 = time.time()
            logger.debug("LLM attempt %d/%d: provider=%s model=%s",
                          attempt + 1, MAX_RETRIES, self._provider_name(), model)
            try:
                resp = client.post(url, headers=headers, json=payload)
                elapsed_ms = (time.time() - t0) * 1000
                data = resp.json()

                if resp.status_code >= 400 or "error" in data:
                    err_msg = str(data.get("error", f"HTTP {resp.status_code}"))[:500]
                    status = resp.status_code
                    logger.warning("LLM API error: provider=%s model=%s %.0fms status=%d error=%.200s",
                                    self._provider_name(), model, elapsed_ms, status, err_msg)
                    if status in (429, 500, 502, 503):
                        last_error = LLMProviderError(
                            f"API error: {err_msg}",
                            provider=self._provider_name(),
                            status_code=status,
                        )
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    raise LLMProviderError(
                        f"API error: {err_msg}",
                        provider=self._provider_name(),
                        status_code=status,
                    )

                usage = data.get("usage", {})
                self.last_usage = usage
                text = self._extract_text(data)
                logger.debug("LLM success: provider=%s model=%s %.0fms status=%d response=%.100s",
                              self._provider_name(), model, elapsed_ms, resp.status_code, text)
                return text, usage

            except (_json.JSONDecodeError, ValueError) as e:
                elapsed_ms = (time.time() - t0) * 1000
                logger.warning("LLM JSON error: provider=%s model=%s %.0fms error=%s",
                                self._provider_name(), model, elapsed_ms, e)
                last_error = LLMProviderError(
                    f"JSON decode error: {e}",
                    provider=self._provider_name(),
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                continue

            except (httpx.TimeoutException, httpx.HTTPError, OSError) as e:
                elapsed_ms = (time.time() - t0) * 1000
                logger.warning("LLM HTTP error: provider=%s model=%s %.0fms error=%s",
                                self._provider_name(), model, elapsed_ms, e)
                last_error = LLMProviderError(
                    f"HTTP error: {e}",
                    provider=self._provider_name(),
                )
                if attempt < MAX_RETRIES - 1:
                    logger.debug("LLM retry: sleeping %.1fs before attempt %d",
                                  RETRY_BACKOFF[attempt], attempt + 2)
                    time.sleep(RETRY_BACKOFF[attempt])
                continue

        raise last_error or LLMProviderError(
            "Max retries exceeded", provider=self._provider_name()
        )


# Re-exports kept for backward compatibility
from ..types import LLMProvider  # noqa: E402, F401

__all__ = ["BaseProvider", "LLMProvider", "LLMProviderError"]
