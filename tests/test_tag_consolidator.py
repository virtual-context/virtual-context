"""Tests for virtual_context.core.tag_consolidator — LLM-driven tag clustering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from virtual_context.core.tag_consolidator import (
    ConsolidationGroup,
    ConsolidationResult,
    _merge_transitive_groups,
    _parse_response,
    consolidate_tags,
)
from virtual_context.types import StoredSegment, StoredSummary, TagStats, TagSummary


# ── helpers ──────────────────────────────────────────────────────────────

class _MockLLM:
    """Mock LLM that returns a canned response (tuple[str, dict])."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.response, {}


def _make_store(
    tags: list[TagStats] | None = None,
    tag_summaries: list[TagSummary] | None = None,
    aliases: dict[str, str] | None = None,
    segments_by_tag: dict[str, list[StoredSummary]] | None = None,
    segments: dict[str, StoredSegment] | None = None,
) -> MagicMock:
    """Build a MagicMock that implements the ContextStore methods used by consolidation."""
    store = MagicMock()
    store.get_all_tags.return_value = tags or []
    store.get_all_tag_summaries.return_value = tag_summaries or []
    store.get_tag_aliases.return_value = aliases or {}
    store.get_orphan_tag_snippets.return_value = []

    _segments_by_tag = segments_by_tag or {}
    _segments = segments or {}

    def _get_summaries_by_tags(tags, min_overlap=1, limit=1000):
        result = []
        seen = set()
        for t in tags:
            for s in _segments_by_tag.get(t, []):
                if s.ref not in seen:
                    result.append(s)
                    seen.add(s.ref)
        return result

    store.get_summaries_by_tags.side_effect = _get_summaries_by_tags

    def _get_segment(ref):
        return _segments.get(ref)

    store.get_segment.side_effect = _get_segment
    return store


# ── _parse_response tests ────────────────────────────────────────────────

class TestParseResponse:
    """Tests for the _parse_response helper."""

    def test_valid_single_group(self):
        resp = json.dumps({
            "groups": [{
                "canonical": "scale-model",
                "aliases": ["model-kit", "model-tanks"],
                "reason": "Both relate to scale modeling.",
            }]
        })
        groups = _parse_response(resp)
        assert len(groups) == 1
        assert groups[0].canonical == "scale-model"
        assert groups[0].aliases == ["model-kit", "model-tanks"]
        assert "scale modeling" in groups[0].reason

    def test_valid_multiple_groups(self):
        resp = json.dumps({
            "groups": [
                {"canonical": "cooking", "aliases": ["baking", "recipes"], "reason": "food preparation"},
                {"canonical": "fitness", "aliases": ["gym", "workout"], "reason": "physical exercise"},
            ]
        })
        groups = _parse_response(resp)
        assert len(groups) == 2
        assert {g.canonical for g in groups} == {"cooking", "fitness"}

    def test_empty_groups(self):
        resp = json.dumps({"groups": []})
        groups = _parse_response(resp)
        assert groups == []

    def test_group_missing_canonical(self):
        resp = json.dumps({
            "groups": [{"aliases": ["a", "b"], "reason": "test"}]
        })
        groups = _parse_response(resp)
        assert groups == []  # missing canonical -> skipped

    def test_group_missing_aliases(self):
        resp = json.dumps({
            "groups": [{"canonical": "solo", "reason": "test"}]
        })
        groups = _parse_response(resp)
        assert groups == []  # missing aliases -> skipped

    def test_group_empty_aliases(self):
        resp = json.dumps({
            "groups": [{"canonical": "solo", "aliases": [], "reason": "test"}]
        })
        groups = _parse_response(resp)
        assert groups == []  # empty aliases -> skipped

    def test_malformed_json(self):
        groups = _parse_response("this is not JSON at all")
        assert groups == []

    def test_json_with_markdown_fences(self):
        resp = '```json\n{"groups": [{"canonical": "a", "aliases": ["b"], "reason": "test"}]}\n```'
        groups = _parse_response(resp)
        assert len(groups) == 1
        assert groups[0].canonical == "a"

    def test_json_with_thinking_tags(self):
        resp = '<think>reasoning here</think>{"groups": [{"canonical": "x", "aliases": ["y"], "reason": "test"}]}'
        groups = _parse_response(resp)
        assert len(groups) == 1


# ── _merge_transitive_groups tests ───────────────────────────────────────

class TestMergeTransitiveGroups:
    """Tests for transitive group merging via union-find."""

    def test_no_overlap(self):
        groups = [
            ConsolidationGroup(canonical="a", aliases=["b"]),
            ConsolidationGroup(canonical="c", aliases=["d"]),
        ]
        merged = _merge_transitive_groups(groups)
        assert len(merged) == 2

    def test_transitive_merge(self):
        """Group A: broad <- narrow, Group B: narrow <- specific  =>  Merged: broad <- narrow, specific."""
        groups = [
            ConsolidationGroup(canonical="broad-topic", aliases=["narrow"], reason="r1"),
            ConsolidationGroup(canonical="narrow", aliases=["specific"], reason="r2"),
        ]
        merged = _merge_transitive_groups(groups)
        assert len(merged) == 1
        # All three tags should be in the merged group
        all_tags = {merged[0].canonical} | set(merged[0].aliases)
        assert all_tags == {"broad-topic", "narrow", "specific"}

    def test_shared_alias_merge(self):
        """Two groups share alias 'shared'."""
        groups = [
            ConsolidationGroup(canonical="x", aliases=["shared"]),
            ConsolidationGroup(canonical="y", aliases=["shared"]),
        ]
        merged = _merge_transitive_groups(groups)
        assert len(merged) == 1
        all_tags = {merged[0].canonical} | set(merged[0].aliases)
        assert all_tags == {"x", "y", "shared"}

    def test_singleton_groups_removed(self):
        """A group that ends up with 1 member after merge is dropped."""
        groups = [
            ConsolidationGroup(canonical="solo", aliases=["solo"]),
        ]
        # After merge, solo points to itself, so the group has 1 unique member
        # Actually both are "solo", so set is {"solo"} => 1 member => dropped
        merged = _merge_transitive_groups(groups)
        assert len(merged) == 0


# ── consolidate_tags integration tests ───────────────────────────────────

class TestConsolidateTags:
    """Integration tests for the main consolidate_tags function."""

    def test_empty_store(self):
        store = _make_store()
        llm = _MockLLM('{"groups": []}')
        result = consolidate_tags(store, llm)
        assert isinstance(result, ConsolidationResult)
        assert result.groups == []
        assert result.aliases_written == 0

    def test_no_tags(self):
        store = _make_store(tags=[])
        llm = _MockLLM('{"groups": []}')
        result = consolidate_tags(store, llm)
        assert result.groups == []
        # LLM should not be called when no tags exist
        assert len(llm.calls) == 0

    def test_llm_clustering_writes_aliases(self):
        """LLM identifies a cluster => aliases are written to the store."""
        tags = [
            TagStats(tag="model-kit"),
            TagStats(tag="model-tanks"),
            TagStats(tag="cooking"),
        ]
        tag_summaries = [
            TagSummary(tag="model-kit", summary="Building scale model kits"),
            TagSummary(tag="model-tanks", summary="Painting tank models"),
            TagSummary(tag="cooking", summary="Recipe discussions"),
        ]
        store = _make_store(tags=tags, tag_summaries=tag_summaries)

        llm_response = json.dumps({
            "groups": [{
                "canonical": "scale-model",
                "aliases": ["model-kit", "model-tanks"],
                "reason": "Both refer to scale model hobby projects",
            }]
        })
        llm = _MockLLM(llm_response)

        result = consolidate_tags(store, llm)
        assert len(result.groups) == 1
        assert result.groups[0].canonical == "scale-model"
        assert result.aliases_written == 2
        # Verify store.set_tag_alias was called for each alias
        alias_calls = store.set_tag_alias.call_args_list
        assert len(alias_calls) == 2
        written_aliases = {call.args[0] for call in alias_calls}
        assert written_aliases == {"model-kit", "model-tanks"}

    def test_existing_aliases_not_rewritten(self):
        """Aliases that already exist in the store are not written again."""
        tags = [TagStats(tag="a"), TagStats(tag="b")]
        tag_summaries = [
            TagSummary(tag="a", summary="Tag A"),
            TagSummary(tag="b", summary="Tag B"),
        ]
        store = _make_store(
            tags=tags,
            tag_summaries=tag_summaries,
            aliases={"b": "a"},  # "b" is already aliased
        )

        llm_response = json.dumps({
            "groups": [{"canonical": "a", "aliases": ["b"], "reason": "same"}]
        })
        llm = _MockLLM(llm_response)
        result = consolidate_tags(store, llm)
        assert result.aliases_written == 0
        store.set_tag_alias.assert_not_called()

    def test_dry_run_no_writes(self):
        """dry_run=True computes groups but writes nothing."""
        tags = [TagStats(tag="x"), TagStats(tag="y")]
        tag_summaries = [
            TagSummary(tag="x", summary="X tag"),
            TagSummary(tag="y", summary="Y tag"),
        ]
        store = _make_store(tags=tags, tag_summaries=tag_summaries)

        llm_response = json.dumps({
            "groups": [{"canonical": "x", "aliases": ["y"], "reason": "same"}]
        })
        llm = _MockLLM(llm_response)
        result = consolidate_tags(store, llm, dry_run=True)
        assert len(result.groups) == 1
        assert result.aliases_written == 0
        store.set_tag_alias.assert_not_called()
        store.store_segment.assert_not_called()

    def test_segment_tags_backfill(self):
        """Segments with alias tags get the canonical tag added."""
        tags = [TagStats(tag="canon"), TagStats(tag="alias1")]
        tag_summaries = [
            TagSummary(tag="canon", summary="Canonical tag"),
            TagSummary(tag="alias1", summary="Alias tag"),
        ]

        seg_ref = "seg-001"
        alias_summary = StoredSummary(ref=seg_ref, tags=["alias1"])
        seg = StoredSegment(ref=seg_ref, tags=["alias1"])

        store = _make_store(
            tags=tags,
            tag_summaries=tag_summaries,
            segments_by_tag={"alias1": [alias_summary], "canon": []},
            segments={seg_ref: seg},
        )

        llm_response = json.dumps({
            "groups": [{"canonical": "canon", "aliases": ["alias1"], "reason": "same"}]
        })
        llm = _MockLLM(llm_response)
        result = consolidate_tags(store, llm)
        assert result.segment_tags_added == 1
        # The segment should now have "canon" in its tags
        assert "canon" in seg.tags

    def test_llm_failure_graceful(self):
        """LLM raising an exception doesn't crash consolidation."""
        tags = [TagStats(tag="x")]
        tag_summaries = [TagSummary(tag="x", summary="X")]
        store = _make_store(tags=tags, tag_summaries=tag_summaries)

        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM unavailable")

        result = consolidate_tags(store, llm)
        assert isinstance(result, ConsolidationResult)
        assert result.groups == []
        assert result.aliases_written == 0

    def test_malformed_llm_output_graceful(self):
        """Garbage LLM output is handled without crash."""
        tags = [TagStats(tag="x")]
        tag_summaries = [TagSummary(tag="x", summary="X")]
        store = _make_store(tags=tags, tag_summaries=tag_summaries)

        llm = _MockLLM("I don't know how to JSON sorry!!!")
        result = consolidate_tags(store, llm)
        assert result.groups == []
        assert result.aliases_written == 0
