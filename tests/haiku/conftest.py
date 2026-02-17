"""Shared fixtures and auto-skip logic for Haiku integration tests."""

from __future__ import annotations

import os

import httpx
import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.providers.anthropic import AnthropicProvider
from virtual_context.types import CompactorConfig, TagGeneratorConfig

HAIKU_MODEL = "claude-haiku-4-5"


def _haiku_available() -> bool:
    """Check if ANTHROPIC_API_KEY is set and Haiku responds."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return False
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30.0,
        )
        return resp.status_code == 200
    except (httpx.HTTPError, Exception):
        return False


_haiku_ok = _haiku_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-add the 'haiku' marker to every test in this directory."""
    for item in items:
        if "/haiku/" in str(item.fspath):
            item.add_marker(pytest.mark.haiku)
            if not _haiku_ok:
                item.add_marker(
                    pytest.mark.skip(reason="ANTHROPIC_API_KEY not set or Haiku unavailable")
                )


@pytest.fixture(scope="session")
def haiku_provider() -> AnthropicProvider:
    """Session-scoped AnthropicProvider pointed at Haiku."""
    if not _haiku_ok:
        pytest.skip("Haiku not available")
    return AnthropicProvider(model=HAIKU_MODEL, temperature=0.0)


@pytest.fixture(scope="session")
def haiku_tag_generator(haiku_provider) -> LLMTagGenerator:
    """Session-scoped LLMTagGenerator backed by real Haiku."""
    config = TagGeneratorConfig(type="llm", max_tags=10, min_tags=5, max_tokens=8192)
    return LLMTagGenerator(llm_provider=haiku_provider, config=config)


@pytest.fixture(scope="session")
def haiku_compactor(haiku_provider) -> DomainCompactor:
    """Session-scoped DomainCompactor backed by real Haiku."""
    config = CompactorConfig(
        summary_ratio=0.15,
        min_summary_tokens=50,
        max_summary_tokens=500,
        llm_token_overhead=8000,
    )
    return DomainCompactor(
        llm_provider=haiku_provider,
        config=config,
        model_name=HAIKU_MODEL,
    )
