"""Tests for virtual_context.ingest.supersession — fact contradiction detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from virtual_context.ingest.supersession import (
    FactSupersessionChecker,
    _extract_object_keyword,
)
from virtual_context.types import Fact, SupersessionConfig


# ── helpers ──────────────────────────────────────────────────────────────

def _make_checker(
    llm_response: str = "[]",
    config: SupersessionConfig | None = None,
) -> tuple[FactSupersessionChecker, MagicMock, MagicMock]:
    """Create a FactSupersessionChecker with mocked LLM and store.

    Returns (checker, mock_store, mock_llm).
    """
    llm = MagicMock()
    llm.complete.return_value = (llm_response, {})
    llm.last_usage = {}

    store = MagicMock()
    store.query_facts.return_value = []

    cfg = config or SupersessionConfig(enabled=True, batch_size=20)
    checker = FactSupersessionChecker(
        llm_provider=llm,
        model="test-model",
        store=store,
        config=cfg,
    )
    return checker, store, llm


def _make_fact(
    id: str = "fact-new",
    subject: str = "user",
    verb: str = "lives-in",
    object: str = "New York",
    status: str = "active",
    tags: list[str] | None = None,
    when_date: str = "",
    what: str = "",
) -> Fact:
    return Fact(
        id=id,
        subject=subject,
        verb=verb,
        object=object,
        status=status,
        tags=tags or [],
        when_date=when_date,
        what=what,
    )


# ── _extract_object_keyword tests ───────────────────────────────────────

class TestExtractObjectKeyword:
    """Tests for keyword extraction from fact object strings."""

    def test_proper_noun_preferred(self):
        """Proper nouns (capitalized) are preferred over common words."""
        kw = _extract_object_keyword("lives in Chicago with family")
        assert kw == "Chicago"

    def test_longest_proper_noun(self):
        kw = _extract_object_keyword("visited Philadelphia and Boston")
        assert kw == "Philadelphia"  # longer proper noun

    def test_fallback_to_longest_common(self):
        """Without proper nouns, falls back to longest word >= 5 chars."""
        kw = _extract_object_keyword("plays guitar every night")
        assert kw == "guitar"  # "guitar" (6 chars) is the longest non-stopword

    def test_stopwords_filtered(self):
        """Stopwords are filtered out even if they're long enough."""
        kw = _extract_object_keyword("started another hobby recently")
        assert kw == "hobby"  # "started", "another", "recently" are stopwords

    def test_short_words_excluded(self):
        """Words shorter than 5 characters are excluded."""
        kw = _extract_object_keyword("has a red car")
        assert kw is None  # all words < 5 chars

    def test_empty_string(self):
        kw = _extract_object_keyword("")
        assert kw is None

    def test_only_stopwords(self):
        kw = _extract_object_keyword("from with that about into")
        assert kw is None


# ── _parse_response tests ───────────────────────────────────────────────

class TestParseResponse:
    """Tests for the supersession check response parser."""

    def test_simple_json_array(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0"), _make_fact(id="c1"), _make_fact(id="c2")]
        result = checker._parse_response("[0, 2]", candidates)
        assert result == ["c0", "c2"]

    def test_empty_array(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response("[]", candidates)
        assert result == []

    def test_json_object_with_updated_key(self):
        """Qwen3-style response: {"updated": [0]}."""
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0"), _make_fact(id="c1")]
        result = checker._parse_response('{"updated": [0]}', candidates)
        assert result == ["c0"]

    def test_json_object_with_superseded_key(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0"), _make_fact(id="c1")]
        result = checker._parse_response('{"superseded": [1]}', candidates)
        assert result == ["c1"]

    def test_json_object_with_indices_key(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response('{"indices": [0]}', candidates)
        assert result == ["c0"]

    def test_thinking_tags_stripped(self):
        """<think>...</think> tags from Qwen3 are stripped before parsing."""
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response(
            "<think>let me think about this</think>[0]",
            candidates,
        )
        assert result == ["c0"]

    def test_out_of_bounds_index_skipped(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response("[0, 5, -1]", candidates)
        assert result == ["c0"]  # 5 and -1 are out of bounds

    def test_none_response(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response(None, candidates)
        assert result == []

    def test_non_json_with_embedded_array(self):
        """Fallback regex extraction of [0, 1] from prose."""
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0"), _make_fact(id="c1")]
        result = checker._parse_response(
            "Based on my analysis, the superseded facts are: [0, 1].",
            candidates,
        )
        assert result == ["c0", "c1"]

    def test_completely_unparseable(self):
        checker, store, _ = _make_checker()
        candidates = [_make_fact(id="c0")]
        result = checker._parse_response("I have no idea what JSON is", candidates)
        assert result == []


# ── _parse_merge_response tests ──────────────────────────────────────────

class TestParseMergeResponse:
    """Tests for the merge response parser (B4 fix with brace-depth counter)."""

    def test_clean_json(self):
        checker, _, _ = _make_checker()
        resp = json.dumps({
            "verb": "lives-in",
            "object": "Chicago (moved from NYC)",
            "status": "active",
            "what": "User relocated to Chicago.",
        })
        result = checker._parse_merge_response(resp)
        assert result is not None
        assert result["verb"] == "lives-in"
        assert result["object"] == "Chicago (moved from NYC)"

    def test_json_with_preamble(self):
        """Handles LLM output with text before the JSON object."""
        checker, _, _ = _make_checker()
        resp = 'Here is the merged fact:\n{"verb": "has", "object": "dog", "status": "active", "what": "User has a dog."}'
        result = checker._parse_merge_response(resp)
        assert result is not None
        assert result["verb"] == "has"

    def test_json_with_thinking_tags(self):
        checker, _, _ = _make_checker()
        resp = '<think>processing merge</think>{"verb": "holds", "object": "degree", "status": "completed", "what": "Finished degree."}'
        result = checker._parse_merge_response(resp)
        assert result is not None
        assert result["verb"] == "holds"

    def test_nested_json_brace_depth(self):
        """Tests the brace-depth counter for nested JSON objects."""
        checker, _, _ = _make_checker()
        # Construct response where json.loads(text[i:]) would fail but
        # the balanced-brace extraction works
        resp = 'Some preamble {"verb": "improved", "object": "5k time to 22:30", "status": "active", "what": "Improved running."} extra garbage {'
        result = checker._parse_merge_response(resp)
        assert result is not None
        assert result["verb"] == "improved"

    def test_missing_verb_key_rejected(self):
        """Response must contain 'verb' key to be accepted."""
        checker, _, _ = _make_checker()
        resp = json.dumps({"object": "something", "status": "active"})
        result = checker._parse_merge_response(resp)
        assert result is None

    def test_completely_malformed(self):
        checker, _, _ = _make_checker()
        result = checker._parse_merge_response("not json at all")
        assert result is None

    def test_empty_string(self):
        checker, _, _ = _make_checker()
        result = checker._parse_merge_response("")
        assert result is None


# ── check_and_supersede integration tests ────────────────────────────────

class TestCheckAndSupersede:
    """Integration tests for the main supersession flow."""

    def test_disabled_config_does_nothing(self):
        cfg = SupersessionConfig(enabled=False)
        checker, store, llm = _make_checker(config=cfg)
        new_fact = _make_fact()
        result = checker.check_and_supersede([new_fact])
        assert result == 0
        llm.complete.assert_not_called()

    def test_empty_facts_list(self):
        checker, store, llm = _make_checker()
        result = checker.check_and_supersede([])
        assert result == 0
        llm.complete.assert_not_called()

    def test_fact_without_subject_skipped(self):
        checker, store, llm = _make_checker()
        fact = _make_fact(subject="")
        result = checker.check_and_supersede([fact])
        assert result == 0
        store.query_facts.assert_not_called()

    def test_no_candidates_no_llm_call(self):
        checker, store, llm = _make_checker()
        store.query_facts.return_value = []
        fact = _make_fact()
        result = checker.check_and_supersede([fact])
        assert result == 0
        llm.complete.assert_not_called()

    def test_contradiction_detected(self):
        """When LLM identifies a superseded fact, it is marked."""
        old_fact = _make_fact(id="old-001", verb="lives-in", object="NYC")
        new_fact = _make_fact(id="new-001", verb="lives-in", object="Chicago")

        checker, store, llm = _make_checker(llm_response="[0]")
        store.query_facts.return_value = [old_fact]

        result = checker.check_and_supersede([new_fact])
        assert result == 1
        store.set_fact_superseded.assert_called_once_with("old-001", "new-001")

    def test_no_contradiction(self):
        """LLM returns empty array => nothing superseded."""
        old_fact = _make_fact(id="old-001", verb="likes", object="pizza")
        new_fact = _make_fact(id="new-001", verb="likes", object="sushi")

        checker, store, llm = _make_checker(llm_response="[]")
        store.query_facts.return_value = [old_fact]

        result = checker.check_and_supersede([new_fact])
        assert result == 0
        store.set_fact_superseded.assert_not_called()

    def test_multiple_facts_multiple_supersessions(self):
        """Multiple new facts can each supersede old ones."""
        old1 = _make_fact(id="old-1", verb="lives-in", object="NYC")
        old2 = _make_fact(id="old-2", verb="works-at", object="Acme")
        new1 = _make_fact(id="new-1", subject="user", verb="lives-in", object="Chicago")
        new2 = _make_fact(id="new-2", subject="user", verb="works-at", object="BigCorp")

        checker, store, llm = _make_checker(llm_response="[0]")

        # Track which new fact we're processing by checking object_contains.
        # The first tag-based call for each fact has object_contains=None;
        # the object-similarity call has a keyword.  We use the keyword to
        # decide which old fact to return as a candidate.
        _fact_idx = [0]  # mutable counter

        def query_side_effect(subject=None, tags=None, limit=20, object_contains=None):
            if subject == "user" and object_contains is None and tags is None:
                # Unfiltered call (e.g., embedding candidates cache) — return all
                return [old1, old2]
            if subject == "user":
                # Tag-based or object-keyword call — return matching candidate
                _fact_idx[0] += 1
                if _fact_idx[0] <= 2:
                    return [old1]
                return [old2]
            return []

        store.query_facts.side_effect = query_side_effect
        result = checker.check_and_supersede([new1, new2])
        assert result == 2

    def test_llm_failure_returns_zero(self):
        """LLM exception during check_batch doesn't crash; returns 0 superseded."""
        old_fact = _make_fact(id="old-001")
        new_fact = _make_fact(id="new-001")

        checker, store, llm = _make_checker()
        store.query_facts.return_value = [old_fact]
        llm.complete.side_effect = RuntimeError("API down")

        result = checker.check_and_supersede([new_fact])
        assert result == 0

    def test_self_not_in_candidates(self):
        """The new fact itself should be excluded from candidates."""
        same_fact = _make_fact(id="fact-same", verb="likes", object="cats")

        checker, store, llm = _make_checker(llm_response="[]")
        # query_facts returns the same fact (id matches new_fact)
        store.query_facts.return_value = [same_fact]

        new_fact = _make_fact(id="fact-same", verb="likes", object="cats")
        result = checker.check_and_supersede([new_fact])
        # LLM should not be called since self was filtered out, leaving no candidates
        assert result == 0


# ── _build_prompt tests ──────────────────────────────────────────────────

class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_prompt_includes_fact_details(self):
        checker, _, _ = _make_checker()
        new_fact = _make_fact(
            subject="user",
            verb="lives-in",
            object="Chicago",
            when_date="2026-01-15",
            what="User moved to Chicago",
        )
        candidate = _make_fact(
            id="old",
            subject="user",
            verb="lives-in",
            object="NYC",
            status="active",
            when_date="2025-06-01",
            what="User lived in NYC",
        )
        prompt = checker._build_prompt(new_fact, [candidate])
        assert "user" in prompt
        assert "lives-in" in prompt
        assert "Chicago" in prompt
        assert "NYC" in prompt
        assert "2026-01-15" in prompt
        assert "[0]" in prompt
