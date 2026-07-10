"""Channel-scoped quote search over canonical turns.

``find_quote`` keeps no canonical-row haystack of its own: it delegates to
``store.search_canonical_turn_text`` and
``semantic.semantic_canonical_turn_search``. A channel scope must therefore
bite inside each source, before that source's limit and before reranking,
otherwise a global top hit outside the channel starves a lower-ranked
in-channel one.

These tests pin:

* the match rule: exact id, or case-folded label with at most one leading '#'
  removed from both sides;
* lexical filtering in SQL before ORDER BY / LIMIT, with a ``[#channel]``
  outer prefix composed before the reranker reads ``QuoteResult.text``;
* semantic filtering resolved through the PHYSICAL canonical row, so an
  assistant chunk cannot inherit a sibling logical row's channel or excerpt;
* ``matched_side`` never becomes ``"channel"``;
* I3: an unscoped call is byte-identical on rows with and without stored
  channel values.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CanonicalTurnChunkEmbedding,
    channel_excerpt_prefix,
    channel_matches,
    strip_channel_hash,
)


def _store(tmp_path: Path, name: str = "vc.db") -> SQLiteStore:
    store = SQLiteStore(tmp_path / name)
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _row(
    store: SQLiteStore,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    sender: str = "",
    origin_channel_id: str = "",
    origin_channel_label: str = "",
    conv: str = "c",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
        primary_tag="chat",
        tags=["chat"],
        origin_channel_id=origin_channel_id,
        origin_channel_label=origin_channel_label,
    )


# ---------------------------------------------------------------------------
# The match rule
# ---------------------------------------------------------------------------

class TestChannelMatches:
    def test_empty_request_disables_filtering(self):
        assert channel_matches("", "", "") is True
        assert channel_matches("   ", "7", "#a") is True

    def test_exact_id_match(self):
        assert channel_matches("1524974537458974851", "1524974537458974851", "") is True
        assert channel_matches("152", "1524974537458974851", "") is False

    def test_label_match_is_case_folded(self):
        assert channel_matches("#VastTest", "", "#vasttest") is True
        assert channel_matches("vasttest", "", "#VASTTEST") is True

    def test_one_leading_hash_stripped_from_both_sides(self):
        assert channel_matches("#vasttest", "", "vasttest") is True
        assert channel_matches("vasttest", "", "#vasttest") is True
        # Only ONE hash comes off each side.
        assert channel_matches("##vasttest", "", "#vasttest") is False

    def test_request_is_trimmed(self):
        assert channel_matches("  #vasttest  ", "", "#vasttest") is True

    def test_empty_row_provenance_never_matches_a_request(self):
        assert channel_matches("#a", "", "") is False
        assert channel_matches("7", "", "") is False

    def test_id_request_does_not_match_a_label(self):
        assert channel_matches("7", "", "#7x") is False

    def test_prefix_helper_prefers_label_over_id(self):
        assert channel_excerpt_prefix("7", "#a") == "[#a] "
        assert channel_excerpt_prefix("7", "") == "[7] "
        assert channel_excerpt_prefix("", "") == ""

    def test_strip_one_hash(self):
        assert strip_channel_hash("#a") == "a"
        assert strip_channel_hash("##a") == "#a"
        assert strip_channel_hash("a") == "a"
        assert strip_channel_hash("  #a  ") == "a"


# ---------------------------------------------------------------------------
# R1 — lexical filtering
# ---------------------------------------------------------------------------

class TestLexicalChannelFilter:
    def _seed(self, tmp_path: Path) -> SQLiteStore:
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="peptide dosing", origin_channel_id="7",
             origin_channel_label="#vasttest")
        _row(store, ct_id="ct-2", sort_key=2000.0,
             user_content="peptide elsewhere", origin_channel_id="9",
             origin_channel_label="#other")
        _row(store, ct_id="ct-3", sort_key=3000.0,
             user_content="peptide unattributed")
        return store

    def test_filter_by_label_with_hash(self, tmp_path: Path):
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#vasttest",
        )
        assert len(results) == 1
        assert "peptide dosing" in results[0].text

    def test_filter_by_label_without_hash_and_case_folded(self, tmp_path: Path):
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="VASTTEST",
        )
        assert len(results) == 1
        assert "peptide dosing" in results[0].text

    def test_filter_by_id(self, tmp_path: Path):
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="9",
        )
        assert len(results) == 1
        assert "peptide elsewhere" in results[0].text

    def test_unattributed_rows_never_match_a_scoped_search(self, tmp_path: Path):
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#nope",
        )
        assert results == []

    def test_filter_applies_before_the_limit(self, tmp_path: Path):
        """The out-of-channel rows sort first (higher sort_key). With the
        filter applied after LIMIT 1 they would starve the in-channel row.
        """
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#vasttest", limit=1,
        )
        assert len(results) == 1
        assert "peptide dosing" in results[0].text

    def test_scoped_excerpt_composes_channel_then_sender(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes tingle", sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")
        results = store.search_canonical_turn_text(
            "toes", conversation_id="c", channel="#vasttest",
        )
        assert results[0].text.startswith("[#vasttest] BigTex: ")
        assert results[0].matched_side == "user"

    def test_scoped_excerpt_composes_channel_then_user(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes tingle",
             origin_channel_id="7", origin_channel_label="#vasttest")
        results = store.search_canonical_turn_text(
            "toes", conversation_id="c", channel="#vasttest",
        )
        assert results[0].text.startswith("[#vasttest] User: ")

    def test_scoped_assistant_excerpt_is_never_sender_labeled(self, tmp_path: Path):
        """Even a legacy row carrying a sender on both halves."""
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="q", assistant_content="the answer is 42",
             sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")
        results = store.search_canonical_turn_text(
            "answer", conversation_id="c", channel="#vasttest",
        )
        assert results[0].matched_side == "assistant"
        assert results[0].text.startswith("[#vasttest] Assistant: ")

    def test_id_only_row_is_labeled_with_its_id(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="toes tingle", origin_channel_id="1524974537458974851")
        results = store.search_canonical_turn_text(
            "toes", conversation_id="c", channel="1524974537458974851",
        )
        assert results[0].text.startswith("[1524974537458974851] User: ")

    def test_matched_side_both_gets_one_leading_prefix(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="peptide question", assistant_content="peptide answer",
             sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#vasttest",
        )
        qr = results[0]
        assert qr.matched_side == "both"
        assert qr.text.startswith("[#vasttest] BigTex: ")
        assert qr.text.count("[#vasttest]") == 1
        assert "\n\nAssistant: " in qr.text

    def test_channel_never_becomes_a_matched_side(self, tmp_path: Path):
        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#vasttest",
        )
        assert {r.matched_side for r in results} <= {"user", "assistant", "both", "unknown"}

    def test_a_channel_name_is_not_a_text_match(self, tmp_path: Path):
        """Channel is a filter, never another haystack: querying the channel
        name must not surface rows whose text lacks it.
        """
        store = self._seed(tmp_path)
        assert store.search_canonical_turn_text("vasttest", conversation_id="c") == []


class TestLexicalUnscopedByteIdentity:
    def test_unscoped_output_is_identical_with_and_without_stored_channel(
        self, tmp_path: Path,
    ):
        """I3: a stored channel value must not alter unscoped output."""
        plain = _store(tmp_path, "plain.db")
        _row(plain, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes tingle", sender="BigTex")

        tagged = _store(tmp_path, "tagged.db")
        _row(tagged, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes tingle", sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")

        a = plain.search_canonical_turn_text("toes", conversation_id="c")
        b = tagged.search_canonical_turn_text("toes", conversation_id="c")
        assert [r.text for r in a] == [r.text for r in b]
        assert [r.matched_side for r in a] == [r.matched_side for r in b]
        assert a[0].text == "BigTex: my toes tingle"

    def test_unscoped_call_still_returns_out_of_channel_rows(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="peptide a",
             origin_channel_label="#one")
        _row(store, ct_id="ct-2", sort_key=2000.0, user_content="peptide b",
             origin_channel_label="#two")
        assert len(store.search_canonical_turn_text("peptide", conversation_id="c")) == 2


# ---------------------------------------------------------------------------
# R1 — semantic filtering
# ---------------------------------------------------------------------------

class _FakeSemanticStore:
    """Minimal store double: physical rows + chunk embeddings."""

    def __init__(self, rows, chunks):
        self._rows = rows
        self._chunks = chunks
        self.logical_calls = 0

    def get_all_canonical_turn_chunk_embeddings(self, *, conversation_id=None):
        return self._chunks

    def get_all_canonical_turns(self, conversation_id):
        return self._rows

    def get_canonical_turn_rows(self, conversation_id, turn_numbers):
        # The LOGICAL seam. A scoped search must not use it.
        self.logical_calls += 1
        return {}


def _semantic(store):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.core.semantic_search import SemanticSearchManager
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    manager = SemanticSearchManager(store=store, config=config)
    # Hermetic: a deterministic 2-d "embedding" keyed by the text, no model.
    manager._embed_fn = lambda texts: [_vec(t) for t in texts]
    return manager


def _vec(text: str) -> list[float]:
    """Similarity is 1.0 when the marker matches, ~0.6 otherwise."""
    return [1.0, 0.0] if "MATCH" in text else [0.8, 0.6]


def _chunk(ct_id: str, turn_number: int, side: str, text: str):
    return CanonicalTurnChunkEmbedding(
        conversation_id="c",
        side=side,
        chunk_index=0,
        text=text,
        embedding=_vec(text),
        canonical_turn_id=ct_id,
        turn_number=turn_number,
    )


def _physical(ct_id, turn_number, **kw):
    from virtual_context.types import CanonicalTurnRow
    base = dict(
        conversation_id="c",
        canonical_turn_id=ct_id,
        turn_number=turn_number,
        turn_group_number=turn_number // 2,
        primary_tag="chat",
        tags=["chat"],
    )
    base.update(kw)
    return CanonicalTurnRow(**base)


class TestSemanticChannelFilter:
    def test_scoped_search_returns_only_in_channel_rows(self):
        rows = [
            _physical("ct-1", 0, user_content="in channel MATCH",
                      origin_channel_id="7", origin_channel_label="#vasttest"),
            _physical("ct-2", 1, user_content="out of channel MATCH",
                      origin_channel_id="9", origin_channel_label="#other"),
        ]
        chunks = [
            _chunk("ct-1", 0, "user", "in channel MATCH"),
            _chunk("ct-2", 1, "user", "out of channel MATCH"),
        ]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert len(results) == 1
        assert "in channel" in results[0].text
        assert store.logical_calls == 0, "scoped search must not use the logical seam"

    def test_a_global_top_hit_outside_the_channel_does_not_starve_the_result(self):
        """The out-of-channel chunk scores highest. With max_results=1 and a
        post-limit filter, the in-channel chunk would be lost.
        """
        rows = [
            _physical("ct-1", 0, user_content="weak in channel",
                      origin_channel_label="#vasttest"),
            _physical("ct-2", 1, user_content="strong MATCH elsewhere",
                      origin_channel_label="#other"),
        ]
        chunks = [
            _chunk("ct-1", 0, "user", "weak in channel"),
            _chunk("ct-2", 1, "user", "strong MATCH elsewhere"),
        ]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=1, conversation_id="c", channel="#vasttest",
        )
        assert len(results) == 1
        assert "weak in channel" in results[0].text

    def test_assistant_chunk_cannot_inherit_a_sibling_logical_rows_channel(self):
        """Physical ordinal 1 is the assistant half of logical group 0. The
        logical seam would hydrate a different row; the physical row is empty
        of channel, so the scoped search must reject it.
        """
        rows = [
            _physical("ct-user", 0, user_content="question",
                      origin_channel_label="#vasttest"),
            _physical("ct-asst", 1, assistant_content="answer MATCH"),
        ]
        chunks = [_chunk("ct-asst", 1, "assistant", "answer MATCH")]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert results == []
        assert store.logical_calls == 0

    def test_assistant_chunk_with_its_own_channel_is_accepted(self):
        rows = [
            _physical("ct-asst", 1, assistant_content="answer MATCH",
                      origin_channel_label="#vasttest"),
        ]
        chunks = [_chunk("ct-asst", 1, "assistant", "answer MATCH")]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert len(results) == 1
        assert results[0].matched_side == "assistant"
        assert results[0].text == "[#vasttest] Assistant: answer MATCH"

    def test_missing_physical_row_is_rejected_not_guessed(self):
        chunks = [_chunk("ct-gone", 0, "user", "orphan MATCH")]
        store = _FakeSemanticStore([], chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert results == []

    def test_scoped_semantic_user_excerpt_keeps_the_user_label(self):
        """Semantic search never gained sender formatting; scoping must not
        widen it.
        """
        rows = [
            _physical("ct-1", 0, user_content="toes MATCH", sender="BigTex",
                      origin_channel_label="#vasttest"),
        ]
        chunks = [_chunk("ct-1", 0, "user", "toes MATCH")]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert results[0].text == "[#vasttest] User: toes MATCH"

    def test_unscoped_semantic_search_uses_the_logical_seam_unchanged(self):
        """I3: the unscoped path is untouched."""
        rows = [_physical("ct-1", 0, user_content="toes MATCH",
                          origin_channel_label="#vasttest")]
        chunks = [_chunk("ct-1", 0, "user", "toes MATCH")]
        store = _FakeSemanticStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
        )
        assert store.logical_calls == 1
        # No physical row via the logical seam -> chunk-text fallback, and no
        # channel prefix, exactly as before.
        assert results[0].text == "User: toes MATCH"


# ---------------------------------------------------------------------------
# R1 — find_quote end to end
# ---------------------------------------------------------------------------

class _NoSemantic:
    def semantic_canonical_turn_search(self, *a, **kw):
        return []


class TestFindQuoteChannelScoping:
    def _seed(self, tmp_path: Path) -> SQLiteStore:
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="peptide dosing schedule", sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")
        _row(store, ct_id="ct-2", sort_key=2000.0,
             user_content="peptide something else",
             origin_channel_id="9", origin_channel_label="#other")
        return store

    def test_scoped_find_quote_returns_only_matching_rows(self, tmp_path: Path):
        from virtual_context.core.quote_search import find_quote

        store = self._seed(tmp_path)
        result = find_quote(
            store, _NoSemantic(), "peptide",
            max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert result["found"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["excerpt"].startswith("[#vasttest] BigTex: ")

    def test_unscoped_find_quote_returns_both(self, tmp_path: Path):
        from virtual_context.core.quote_search import find_quote

        store = self._seed(tmp_path)
        result = find_quote(
            store, _NoSemantic(), "peptide", max_results=5, conversation_id="c",
        )
        assert len(result["results"]) == 2
        assert all("[#" not in r["excerpt"] for r in result["results"])

    def test_unscoped_find_quote_is_byte_identical_across_stored_channel(
        self, tmp_path: Path,
    ):
        from virtual_context.core.quote_search import find_quote

        plain = _store(tmp_path, "plain.db")
        _row(plain, ct_id="ct-1", sort_key=1000.0,
             user_content="peptide dosing schedule", sender="BigTex")
        tagged = self._seed(tmp_path)
        # Drop the out-of-channel row so the two stores hold the same content.
        conn = tagged._get_conn()
        conn.execute("DELETE FROM canonical_turns WHERE canonical_turn_id = 'ct-2'")
        conn.commit()

        a = find_quote(plain, _NoSemantic(), "peptide",
                       max_results=5, conversation_id="c")
        b = find_quote(tagged, _NoSemantic(), "peptide",
                       max_results=5, conversation_id="c")
        assert a["results"] == b["results"]

    def test_prefix_is_present_during_reranking(self, tmp_path: Path):
        """The reranker scores ``QuoteResult.text``. Adding the prefix only in
        response formatting would be too late.
        """
        from virtual_context.core.quote_search import _search_find_quote_candidates

        store = self._seed(tmp_path)
        candidates = _search_find_quote_candidates(
            store, _NoSemantic(), "peptide",
            limit=5, mode="lookup", conversation_id="c", channel="#vasttest",
        )
        assert candidates
        assert all(qr.text.startswith("[#vasttest] ") for qr in candidates)

    def test_scoped_sender_labeled_excerpt_still_ranks_as_a_user_statement(
        self, tmp_path: Path,
    ):
        """The reranker's ``starts_with_user`` feature reads ``matched_side``,
        not a literal ``User:`` prefix, so the channel prefix cannot demote a
        sender-labeled user match.
        """
        from virtual_context.core.quote_search import _excerpt_starts_with_user_statement

        store = self._seed(tmp_path)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", channel="#vasttest",
        )
        qr = results[0]
        assert qr.text.startswith("[#vasttest] BigTex: ")
        assert _excerpt_starts_with_user_statement(qr.text, qr.matched_side) is True


# ---------------------------------------------------------------------------
# R2 — tool surfaces
# ---------------------------------------------------------------------------

class TestToolSurfaces:
    def test_vc_find_quote_schema_exposes_optional_channel(self):
        from virtual_context.core.tool_loop import vc_tool_definitions

        tools = {t["name"]: t for t in vc_tool_definitions()}
        schema = tools["vc_find_quote"]["input_schema"]
        assert "channel" in schema["properties"]
        assert schema["properties"]["channel"]["type"] == "string"
        assert "channel" not in schema["required"]

    def test_execute_vc_tool_passes_channel_through(self):
        from virtual_context.core import tool_loop

        seen: dict[str, object] = {}

        class _Engine:
            class config:
                class search:
                    find_quote_max_results = 20

            def find_quote(self, **kwargs):
                seen.update(kwargs)
                return {"found": False, "results": []}

        tool_loop.execute_vc_tool(
            _Engine(), "vc_find_quote",
            {"query": "peptide", "mode": "lookup", "channel": "#vasttest"},
        )
        assert seen["channel"] == "#vasttest"
        assert seen["query"] == "peptide"

    def test_execute_vc_tool_defaults_channel_to_empty(self):
        from virtual_context.core import tool_loop

        seen: dict[str, object] = {}

        class _Engine:
            class config:
                class search:
                    find_quote_max_results = 20

            def find_quote(self, **kwargs):
                seen.update(kwargs)
                return {"found": False, "results": []}

        tool_loop.execute_vc_tool(
            _Engine(), "vc_find_quote", {"query": "peptide", "mode": "lookup"},
        )
        assert seen["channel"] == ""

    def test_engine_find_quote_accepts_channel(self):
        import inspect
        from virtual_context.engine import VirtualContextEngine

        sig = inspect.signature(VirtualContextEngine.find_quote)
        assert sig.parameters["channel"].default == ""

    def test_mcp_find_quote_accepts_channel(self):
        import inspect
        from virtual_context.mcp import server

        fn = getattr(server.find_quote, "fn", server.find_quote)
        sig = inspect.signature(fn)
        assert sig.parameters["channel"].default == ""
