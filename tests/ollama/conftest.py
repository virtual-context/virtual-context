"""Shared fixtures and auto-skip logic for Ollama integration tests."""

from __future__ import annotations

import httpx
import pytest

from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.providers.generic_openai import GenericOpenAIProvider
from virtual_context.types import CompactorConfig, TagGeneratorConfig

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-fp16"


def _ollama_available() -> bool:
    """Check if Ollama is running and has the required model."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        return any(OLLAMA_MODEL in m.get("name", "") for m in models)
    except (httpx.HTTPError, Exception):
        return False


_ollama_ok = _ollama_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-add the 'ollama' marker to every test in this directory."""
    for item in items:
        if "/ollama/" in str(item.fspath):
            item.add_marker(pytest.mark.ollama)
            if not _ollama_ok:
                item.add_marker(
                    pytest.mark.skip(reason="Ollama not running or qwen3:4b-instruct-2507-fp16 not available")
                )


@pytest.fixture(scope="session")
def _warmup_ollama():
    """Warmup: send a tiny completion so the model is loaded into memory."""
    if not _ollama_ok:
        pytest.skip("Ollama not available")

    # Low max_tokens — just enough to load the model into VRAM
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, f"Warmup failed: {resp.text}"


@pytest.fixture(scope="session")
def ollama_provider(_warmup_ollama) -> GenericOpenAIProvider:
    """Session-scoped GenericOpenAIProvider pointed at local Ollama."""
    return GenericOpenAIProvider(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        model=OLLAMA_MODEL,
        temperature=0.3,
    )


@pytest.fixture
def ollama_tag_generator(ollama_provider) -> LLMTagGenerator:
    """Function-scoped LLMTagGenerator backed by real Ollama."""
    config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1, max_tokens=8192)
    return LLMTagGenerator(llm_provider=ollama_provider, config=config)


@pytest.fixture
def ollama_compactor(ollama_provider) -> DomainCompactor:
    """Function-scoped DomainCompactor backed by real Ollama."""
    config = CompactorConfig(
        summary_ratio=0.15,
        min_summary_tokens=50,
        max_summary_tokens=500,
        llm_token_overhead=8000,
    )
    return DomainCompactor(
        llm_provider=ollama_provider,
        config=config,
        model_name=OLLAMA_MODEL,
    )
