"""Integration tests for GenericOpenAIProvider against a real Ollama instance."""

from __future__ import annotations

import pytest

from virtual_context.providers.generic_openai import GenericOpenAIProvider
from virtual_context.types import LLMProviderError

# Thinking models need high max_tokens; reasoning consumes ~800 tokens
THINKING_OVERHEAD = 8192


@pytest.mark.timeout(300)
class TestBasicCompletion:
    """Verify the provider can talk to Ollama and return sensible results."""

    def test_basic_completion(self, ollama_provider: GenericOpenAIProvider):
        result = ollama_provider.complete(
            system="You are a helpful assistant.",
            user="What is 2 + 2? Answer with just the number.",
            max_tokens=THINKING_OVERHEAD,
        )
        assert result, "Expected non-empty response"

    def test_response_is_string(self, ollama_provider: GenericOpenAIProvider):
        result = ollama_provider.complete(
            system="You are helpful.",
            user="Say hello.",
            max_tokens=THINKING_OVERHEAD,
        )
        assert isinstance(result, str)

    def test_system_prompt_respected(self, ollama_provider: GenericOpenAIProvider):
        result = ollama_provider.complete(
            system="Respond only with the word PONG. Nothing else.",
            user="Ping",
            max_tokens=THINKING_OVERHEAD,
        )
        assert "PONG" in result.upper(), f"Expected PONG in response, got: {result!r}"

    def test_max_tokens_limits_content(self, ollama_provider: GenericOpenAIProvider):
        """With enough tokens for thinking + content, the content should be short."""
        result = ollama_provider.complete(
            system="You are a helpful assistant. Be very brief.",
            user="What is the capital of France? One word answer.",
            max_tokens=THINKING_OVERHEAD,
        )
        # The actual content (excluding reasoning) should be reasonable length
        assert len(result) < 500, f"Response too long ({len(result)} chars)"

    def test_temperature_zero_determinism(self, ollama_provider: GenericOpenAIProvider):
        """Two calls at temperature=0 should produce near-identical results."""
        provider = GenericOpenAIProvider(
            base_url=ollama_provider.base_url,
            model=ollama_provider.model,
            temperature=0.0,
        )
        prompt = "What is 2+2? Answer with just the number."
        r1 = provider.complete(
            system="Answer concisely.", user=prompt, max_tokens=THINKING_OVERHEAD
        )
        r2 = provider.complete(
            system="Answer concisely.", user=prompt, max_tokens=THINKING_OVERHEAD
        )
        # Strip any residual thinking tags for comparison
        import re
        strip = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
        assert strip(r1) == strip(r2), f"Determinism failed:\n  r1={r1!r}\n  r2={r2!r}"

    def test_connection_error_raises(self):
        """A provider pointed at a bad port should raise LLMProviderError."""
        bad_provider = GenericOpenAIProvider(
            base_url="http://127.0.0.1:1/v1",
            model="nonexistent",
        )
        with pytest.raises(LLMProviderError):
            bad_provider.complete(
                system="test", user="test", max_tokens=5,
            )
