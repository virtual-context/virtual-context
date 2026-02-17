"""Tests for tag generator: LLM-based and keyword-based."""

import pytest

from virtual_context.core.tag_generator import (
    KeywordTagGenerator,
    LLMTagGenerator,
    build_tag_generator,
    detect_broad_heuristic,
    _compile_broad_patterns,
)
from virtual_context.core.tag_canonicalizer import TagCanonicalizer
from virtual_context.types import DEFAULT_BROAD_PATTERNS, KeywordTagConfig, TagGeneratorConfig

from conftest import MockLLMProvider


class TestKeywordTagGenerator:
    @pytest.fixture
    def keyword_config(self):
        return KeywordTagConfig(
            tag_keywords={
                "legal": ["court", "filing", "motion", "attorney"],
                "medical": ["insulin", "medication", "doctor", "blood"],
                "code": ["function", "bug", "deploy", "API"],
            },
            tag_patterns={
                "legal": [r"\b\d{2}-cv-\d+"],
                "code": [r"\bdef \w+\("],
            },
        )

    @pytest.fixture
    def generator(self, keyword_config):
        return KeywordTagGenerator(config=keyword_config)

    def test_keyword_match(self, generator):
        result = generator.generate_tags("The attorney filed a motion in court")
        assert "legal" in result.tags
        assert result.primary == "legal"
        assert result.source == "keyword"

    def test_pattern_match(self, generator):
        result = generator.generate_tags("Check case 24-cv-1234 details")
        assert "legal" in result.tags

    def test_no_match_returns_general(self, generator):
        result = generator.generate_tags("What's the weather like today?")
        assert result.tags == ["_general"]
        assert result.primary == "_general"
        assert result.source == "fallback"

    def test_multiple_tag_match(self, generator):
        result = generator.generate_tags(
            "The doctor said the bug in the insulin pump needs fixing"
        )
        assert len(result.tags) >= 1



class TestLLMTagGenerator:
    @pytest.fixture
    def llm_response(self):
        return '{"tags": ["database-schema", "migration"], "primary": "database-schema"}'

    @pytest.fixture
    def generator(self, llm_response):
        provider = MockLLMProvider(response=llm_response)
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        return LLMTagGenerator(llm_provider=provider, config=config)

    def test_llm_tag_generation(self, generator):
        result = generator.generate_tags("We need to create the users table")
        assert "database-schema" in result.tags
        assert result.primary == "database-schema"
        assert result.source == "llm"

    def test_normalize_tags(self, generator):
        tags = generator._normalize_tags(["Database Schema", "API Design", "auth-flow"])
        assert tags == ["database-schema", "api-design", "auth-flow"]

    def test_normalize_single_tag(self, generator):
        assert generator._normalize_tag("Database Schema") == "database-schema"
        assert generator._normalize_tag("  auth-flow  ") == "auth-flow"
        assert generator._normalize_tag("API_Design") == "api-design"

    def test_vocabulary_tracking(self, generator):
        generator.generate_tags("First call")
        assert "database-schema" in generator._tag_vocabulary
        assert generator._tag_vocabulary["database-schema"] == 1

        generator.generate_tags("Second call")
        assert generator._tag_vocabulary["database-schema"] == 2

    def test_alias_resolution(self):
        canonicalizer = TagCanonicalizer()
        canonicalizer.register_alias("db-schema", "database-schema")
        provider = MockLLMProvider(response='{"tags": ["database-schema"], "primary": "database-schema"}')
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config, canonicalizer=canonicalizer)
        assert generator._normalize_tag("db-schema") == "database-schema"

    def test_fallback_on_error(self):
        """Test fallback when LLM fails."""
        provider = MockLLMProvider(response="invalid json {{{")
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)
        result = generator.generate_tags("test text")
        assert result.tags == ["_general"]
        assert result.source == "fallback"

    def test_strips_thinking_tags(self):
        """Test that <think>...</think> tags are stripped."""
        provider = MockLLMProvider(
            response='<think>Let me analyze this...</think>{"tags": ["api-design"], "primary": "api-design"}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)
        result = generator.generate_tags("Design an API")
        assert "api-design" in result.tags

    def test_load_vocabulary(self, generator):
        generator.load_vocabulary({"auth": 10, "database": 5})
        assert generator._tag_vocabulary["auth"] == 10
        assert generator._tag_vocabulary["database"] == 5


class TestBroadHeuristic:
    """Test deterministic broad-query detection."""

    @pytest.fixture
    def patterns(self):
        return _compile_broad_patterns(DEFAULT_BROAD_PATTERNS)

    @pytest.mark.parametrize("text", [
        "What did you say about the image storage earlier?",
        "What did we discuss about authentication?",
        "What did you mention about the database?",
        "Remind me what we decided about the API",
        "Remind me about the deployment plan",
        "Looking back at everything we've discussed",
        "Looking back at our conversation",
        "Summarize what we've covered so far",
        "Recap everything we talked about",
        "Can you summarize the project?",
        "Can you recap our discussion?",
        "What have we covered today?",
        "What have we discussed so far?",
        "You mentioned earlier that we should use Redis",
        "We discussed before that JWT was the way to go",
        "We talked about previously using S3",
        "Go back over what we said",
        "Go over everything again",
        "If I had to explain the most important thing I learned from each of these threads",
        "What's the takeaway from each of these topics?",
        "Across everything we've discussed, what matters most?",
        "From all we've covered, what should I remember?",
    ])
    def test_broad_detected(self, patterns, text):
        assert detect_broad_heuristic(text, patterns) is True, f"Should detect as broad: {text!r}"

    @pytest.mark.parametrize("text", [
        "How do I implement pagination?",
        "Fix the auth bug",
        "What is the best database for this?",
        "Can you write a function that sorts a list?",
        "Deploy the app to production",
        "I want to add a new feature",
        "The tests are failing",
        "What's the weather like?",
    ])
    def test_not_broad(self, patterns, text):
        assert detect_broad_heuristic(text, patterns) is False, f"Should NOT detect as broad: {text!r}"

    @pytest.mark.regression("BUG-007")
    def test_llm_broad_miss_overridden(self):
        """LLM returns broad=false, heuristic overrides to true (the T45 bug)."""
        provider = MockLLMProvider(
            response='{"tags": ["image-storage"], "primary": "image-storage", "broad": false}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("What did you say about the image storage earlier?")
        assert result.broad is True, "Heuristic should override LLM's missed broad"

    def test_llm_broad_true_preserved(self):
        """LLM returns broad=true, heuristic doesn't interfere."""
        provider = MockLLMProvider(
            response='{"tags": ["storage"], "primary": "storage", "broad": true}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("What did you say about the image storage earlier?")
        assert result.broad is True

    def test_non_broad_not_overridden(self):
        """Specific query stays broad=false even with heuristic active."""
        provider = MockLLMProvider(
            response='{"tags": ["pagination"], "primary": "pagination", "broad": false}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("How do I implement pagination?")
        assert result.broad is False

    def test_custom_patterns(self):
        """User-configured broad patterns work."""
        config = TagGeneratorConfig(
            type="llm", max_tags=5, min_tags=1,
            broad_patterns=[r"\brecuerda\b"],  # Spanish: "remember"
        )
        provider = MockLLMProvider(
            response='{"tags": ["general"], "primary": "general", "broad": false}'
        )
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("Recuerda lo que dijimos sobre la base de datos")
        assert result.broad is True

    def test_empty_patterns_disables_heuristic(self):
        """Empty broad_patterns list disables the heuristic."""
        config = TagGeneratorConfig(
            type="llm", max_tags=5, min_tags=1,
            broad_patterns=[],
        )
        provider = MockLLMProvider(
            response='{"tags": ["image-storage"], "primary": "image-storage", "broad": false}'
        )
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("What did you say about the image storage earlier?")
        assert result.broad is False, "Empty patterns should disable heuristic"

    def test_invalid_pattern_skipped(self):
        """Invalid regex is skipped without crashing."""
        patterns = _compile_broad_patterns(["[invalid", r"\bvalid\b"])
        assert len(patterns) == 1
        assert detect_broad_heuristic("this is valid", patterns) is True


class TestRelatedTagsParsing:
    """Test that related_tags are parsed and normalized from LLM response."""

    def test_related_tags_parsed(self):
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database", "broad": false, "related_tags": ["sql", "postgres"]}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("database design")
        assert result.related_tags == ["sql", "postgres"]

    def test_related_tags_normalized(self):
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database", "broad": false, "related_tags": ["SQL Server", "Post Gres"]}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("database design")
        assert "sql-server" in result.related_tags
        assert "post-gres" in result.related_tags

    def test_related_tags_deduped_against_primary(self):
        provider = MockLLMProvider(
            response='{"tags": ["database", "schema"], "primary": "database", "broad": false, "related_tags": ["database", "sql", "schema"]}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("database schema")
        # "database" and "schema" should be removed since they're already in primary tags
        assert "database" not in result.related_tags
        assert "schema" not in result.related_tags
        assert "sql" in result.related_tags

    def test_no_related_tags_returns_empty(self):
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database", "broad": false}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("database design")
        assert result.related_tags == []

    def test_related_tags_invalid_type_returns_empty(self):
        """If related_tags is not a list, return empty."""
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database", "related_tags": "not-a-list"}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        result = generator.generate_tags("database design")
        assert result.related_tags == []


class TestContextLookback:
    """Test that context_turns are injected into the tagger prompt."""

    def test_context_turns_in_prompt(self):
        """When context_turns is provided, the prompt should contain a context section."""
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database"}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        context = [
            "How do I optimize the users table query?",
            "Add an index on the email column and use EXPLAIN ANALYZE.",
        ]
        generator.generate_tags("of course", context_turns=context)

        # Check the prompt sent to the LLM
        assert len(provider.calls) == 1
        prompt = provider.calls[0]["user"]
        assert "Recent conversation context:" in prompt
        assert "optimize the users table" in prompt

    def test_no_context_turns_no_section(self):
        """When context_turns is not provided, no context section in prompt."""
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database"}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        generator.generate_tags("How do I optimize queries?")

        prompt = provider.calls[0]["user"]
        assert "Recent conversation context:" not in prompt

    def test_context_turns_none_backward_compat(self):
        """context_turns=None is identical to omitting it."""
        provider = MockLLMProvider(
            response='{"tags": ["database"], "primary": "database"}'
        )
        config = TagGeneratorConfig(type="llm", max_tags=5, min_tags=1)
        generator = LLMTagGenerator(llm_provider=provider, config=config)

        generator.generate_tags("test", context_turns=None)

        prompt = provider.calls[0]["user"]
        assert "Recent conversation context:" not in prompt


class TestBuildTagGenerator:
    def test_build_keyword_generator(self):
        config = TagGeneratorConfig(
            type="keyword",
            keyword_fallback=KeywordTagConfig(
                tag_keywords={"test": ["hello"]},
            ),
        )
        gen = build_tag_generator(config)
        assert isinstance(gen, KeywordTagGenerator)

    def test_build_llm_generator(self):
        config = TagGeneratorConfig(type="llm")
        provider = MockLLMProvider()
        gen = build_tag_generator(config, llm_provider=provider)
        assert isinstance(gen, LLMTagGenerator)

    def test_fallback_to_keyword_when_no_provider(self):
        config = TagGeneratorConfig(
            type="llm",
            keyword_fallback=KeywordTagConfig(tag_keywords={"x": ["y"]}),
        )
        gen = build_tag_generator(config, llm_provider=None)
        assert isinstance(gen, KeywordTagGenerator)
