"""Haiku tagging accuracy tests — curated payloads with structural assertions."""

from __future__ import annotations

import pytest


class TestTopicContamination:
    """Verify the tagger doesn't leak vocabulary topics into unrelated messages."""

    def test_electronics_message_ignores_planes_vocab(self, haiku_tag_generator):
        """An electronics message should not pick up 'planes' from existing vocab."""
        result = haiku_tag_generator.generate_tags(
            "I'm building an Arduino circuit with a 10k resistor and an LED. "
            "The breadboard layout has the microcontroller connected via SPI.",
            existing_tags=["planes", "aviation", "flight-training", "electronics", "arduino"],
        )
        assert "planes" not in result.tags
        assert "aviation" not in result.tags
        assert any(t in result.tags for t in ("electronics", "arduino", "circuits", "hardware"))

    def test_planes_message_ignores_electronics_vocab(self, haiku_tag_generator):
        """A planes message should not pick up 'electronics' from existing vocab."""
        result = haiku_tag_generator.generate_tags(
            "I flew a Cessna 172 today and practiced crosswind landings. "
            "The ATIS reported winds at 15 knots gusting to 25.",
            existing_tags=["electronics", "arduino", "circuits", "planes", "aviation"],
        )
        assert "electronics" not in result.tags
        assert "arduino" not in result.tags
        assert any(t in result.tags for t in ("planes", "aviation", "flying", "flight", "pilot"))

    def test_topic_switching(self, haiku_tag_generator):
        """Switching from electronics to jiu-jitsu should produce martial arts tags."""
        result = haiku_tag_generator.generate_tags(
            "I just started training jiu-jitsu and learned the closed guard position. "
            "My coach showed me an armbar from mount.",
            existing_tags=["electronics", "arduino", "circuits", "jiu-jitsu", "martial-arts"],
        )
        assert "electronics" not in result.tags
        assert any(t in result.tags for t in ("jiu-jitsu", "martial-arts", "bjj", "grappling"))


class TestTemporalDetection:
    """Verify temporal query detection in LLM tagger."""

    def test_first_thing_discussed_is_temporal(self, haiku_tag_generator):
        """'Very first thing we discussed' should be detected as temporal."""
        result = haiku_tag_generator.generate_tags(
            "What was the very first thing we discussed at the start?",
            existing_tags=["database", "auth", "frontend"],
        )
        assert result.temporal is True

    def test_specific_question_is_not_temporal(self, haiku_tag_generator):
        """A normal question should not be temporal."""
        result = haiku_tag_generator.generate_tags(
            "How do I add an index to the users table?",
            existing_tags=["database", "schema", "postgres"],
        )
        assert result.temporal is False


class TestEnvelopeIsolation:
    """Verify the tagger ignores messaging infrastructure in vocab."""

    def test_weather_query_ignores_telegram_vocab(self, haiku_tag_generator):
        """A weather query should not pick up 'telegram' or 'messaging' from vocab."""
        result = haiku_tag_generator.generate_tags(
            "What's the weather forecast for New York this weekend?",
            existing_tags=["telegram", "messaging", "weather", "climate", "protocol"],
        )
        assert "telegram" not in result.tags
        assert "messaging" not in result.tags
        assert "protocol" not in result.tags
        assert any(t in result.tags for t in ("weather", "climate", "forecast"))


class TestTagStructure:
    """Verify structural properties of tag results."""

    def test_tags_are_lowercase_kebab(self, haiku_tag_generator):
        """All tags should be lowercase kebab-case."""
        result = haiku_tag_generator.generate_tags(
            "We're building a REST API with PostgreSQL and Redis caching.",
            existing_tags=[],
        )
        for tag in result.tags:
            assert tag == tag.lower(), f"Tag should be lowercase: {tag!r}"
            assert " " not in tag, f"Tag should not contain spaces: {tag!r}"
            assert "_" not in tag, f"Tag should use hyphens not underscores: {tag!r}"

    def test_primary_in_tags(self, haiku_tag_generator):
        """The primary tag should always be in the tags list."""
        result = haiku_tag_generator.generate_tags(
            "Let's discuss the database migration strategy for the users table.",
            existing_tags=["database", "migration", "schema"],
        )
        assert result.primary in result.tags

    def test_respects_existing_vocabulary(self, haiku_tag_generator):
        """When existing tags match the topic, tagger should reuse them."""
        result = haiku_tag_generator.generate_tags(
            "I need to optimize the database query that joins users and orders.",
            existing_tags=["database", "schema", "query-optimization", "postgres"],
        )
        # At least one existing tag should be reused
        overlap = set(result.tags) & {"database", "schema", "query-optimization", "postgres"}
        assert len(overlap) >= 1, f"Expected vocab reuse, got tags: {result.tags}"
