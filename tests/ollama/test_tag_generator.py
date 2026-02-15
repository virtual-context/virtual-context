"""Integration tests for LLMTagGenerator with real Ollama."""

from __future__ import annotations

import re

import pytest

from virtual_context.core.tag_generator import LLMTagGenerator

TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$")

LEGAL_TEXT = (
    "User: What's the deadline for the court filing in case 24-cv-1234?\n"
    "Assistant: The filing deadline is January 30th. The motion must be submitted "
    "to the court by 5pm. The attorney has already drafted the brief."
)

MEDICAL_TEXT = (
    "User: My blood glucose was 180 this morning. Should I adjust my insulin?\n"
    "Assistant: A reading of 180 is above target. Consider adjusting your insulin "
    "dosage by 1 unit. Schedule an appointment with your endocrinologist."
)

CODE_TEXT = (
    "User: How do I add a new REST endpoint to our FastAPI application?\n"
    "Assistant: Create a new router in app/routers/, define the path operation with "
    "@router.get or @router.post, add Pydantic models for request/response, "
    "then include the router in main.py."
)


@pytest.mark.timeout(600)
class TestTagGeneratorStructure:
    """Assert tag structure, not exact content."""

    def test_legal_text_returns_tags(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(LEGAL_TEXT)
        assert result.tags, "Expected non-empty tags"
        assert result.primary, "Expected non-empty primary tag"
        assert result.source == "llm"

    def test_medical_text_returns_tags(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(MEDICAL_TEXT)
        assert result.tags, "Expected non-empty tags"
        assert result.primary, "Expected non-empty primary tag"
        assert result.source == "llm"

    def test_code_text_returns_tags(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(CODE_TEXT)
        assert result.tags, "Expected non-empty tags"
        assert result.primary, "Expected non-empty primary tag"
        assert result.source == "llm"

    def test_tags_are_normalized(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(LEGAL_TEXT)
        for tag in result.tags:
            assert TAG_PATTERN.match(tag), (
                f"Tag {tag!r} doesn't match normalized pattern [a-z0-9][a-z0-9-]*[a-z0-9]"
            )

    def test_tag_count_within_bounds(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(MEDICAL_TEXT)
        assert 1 <= len(result.tags) <= 5, f"Expected 1-5 tags, got {len(result.tags)}"

    def test_primary_in_tags(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(CODE_TEXT)
        assert result.primary in result.tags, (
            f"Primary {result.primary!r} not in tags {result.tags}"
        )

    def test_think_tag_stripping(self, ollama_tag_generator: LLMTagGenerator):
        result = ollama_tag_generator.generate_tags(LEGAL_TEXT)
        for tag in result.tags:
            assert "<think>" not in tag, f"<think> found in tag: {tag!r}"
            assert "</think>" not in tag, f"</think> found in tag: {tag!r}"
        assert "<think>" not in result.primary

    def test_fallback_on_gibberish(self, ollama_tag_generator: LLMTagGenerator):
        """Random characters should not cause an exception."""
        gibberish = "asdkjfh 2389!@# $*&^ xzcvbn qwerty 12345 zzzzz"
        result = ollama_tag_generator.generate_tags(gibberish)
        # Should return *something* — either real tags or fallback
        assert result.tags, "Expected at least fallback tags"
        assert result.primary, "Expected a primary tag"

    def test_vocabulary_accumulates(self, ollama_tag_generator: LLMTagGenerator):
        """After multiple calls, the internal vocabulary should have entries."""
        ollama_tag_generator.generate_tags(LEGAL_TEXT)
        ollama_tag_generator.generate_tags(MEDICAL_TEXT)
        ollama_tag_generator.generate_tags(CODE_TEXT)
        assert len(ollama_tag_generator._tag_vocabulary) > 0, (
            "Expected vocabulary to have entries after 3 calls"
        )

    @pytest.mark.xfail(reason="LLM may or may not use existing tags -- nondeterministic")
    def test_existing_tags_influence(self, ollama_tag_generator: LLMTagGenerator):
        """Passing existing_tags may influence the output tags."""
        existing = ["legal-filing", "court-deadline"]
        result = ollama_tag_generator.generate_tags(LEGAL_TEXT, existing_tags=existing)
        # Check if any existing tag was reused
        overlap = set(result.tags) & set(existing)
        assert overlap, f"Expected overlap with existing tags, got {result.tags}"
