"""Sender-aware lexical quote search over canonical turns.

``find_quote`` has no local canonical-row haystack: it delegates lexical
matching to ``store.search_canonical_turn_text``. When a group member's name
lives only in the ``sender`` column (the envelope that carried it is stripped
before the row is hashed), a query for that name matched nothing.

These tests pin the store contract:

* A sender-only match is valid when the row has user content, and reports
  itself as a user-side match rather than a new ``matched_side`` value.
* Any user-side excerpt from a row with a sender uses the sender label,
  whether the query hit ``sender`` or ``user_content``.
* Assistant excerpts stay ``"Assistant: ..."`` even when a legacy row carries
  a logical-turn sender on both halves.
* Text-only matches on rows with no sender render exactly as before (I3).
* The label lands in ``QuoteResult.text``, which is the reranker's haystack,
  so a sender-only candidate is not demoted behind text matches.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "vc.db")
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
    )


class TestSenderOnlyMatch:
    def test_sender_only_match_returns_labeled_user_excerpt(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")

        results = store.search_canonical_turn_text("bigtex", conversation_id="c")
        assert len(results) == 1
        qr = results[0]
        assert qr.matched_side == "user"
        assert qr.text.startswith("BigTex: ")
        assert "toes are tingling" in qr.text

    def test_sender_label_is_in_the_reranker_haystack(self, tmp_path: Path):
        """``_rerank_quote_results`` scores ``qr.text``. The sender must be
        present there, not only in a later formatting pass, or a sender-only
        candidate is scored as if the query were absent.
        """
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")
        qr = store.search_canonical_turn_text("bigtex", conversation_id="c")[0]
        assert "bigtex" in qr.text.lower()

    def test_sender_only_match_requires_user_content(self, tmp_path: Path):
        """An assistant-only row carrying a legacy logical-turn sender must
        not surface as a human-sender match.
        """
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             assistant_content="that sounds neurological", sender="BigTex")
        results = store.search_canonical_turn_text("bigtex", conversation_id="c")
        assert results == []

    def test_sender_match_is_case_insensitive(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="hello", sender="BigTex")
        assert store.search_canonical_turn_text("BIGTEX", conversation_id="c")
        assert store.search_canonical_turn_text("bigtex", conversation_id="c")

    @pytest.mark.parametrize("query", ["Big_ex", "Big%ex"])
    def test_sender_match_treats_like_wildcards_literally(
        self,
        tmp_path: Path,
        query: str,
    ):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="hello", sender="BigTex")

        assert store.search_canonical_turn_text(query, conversation_id="c") == []


class TestSenderLabelOnTextMatches:
    def test_user_text_match_on_sender_row_uses_sender_label(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")
        qr = store.search_canonical_turn_text("tingling", conversation_id="c")[0]
        assert qr.matched_side == "user"
        assert qr.text.startswith("BigTex: ")

    def test_assistant_excerpt_never_takes_a_human_sender_label(self, tmp_path: Path):
        """Legacy rows carry the logical-turn sender on both halves."""
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             assistant_content="neurological perhaps", sender="BigTex")
        qr = store.search_canonical_turn_text("neurological", conversation_id="c")[0]
        assert qr.matched_side == "assistant"
        assert qr.text.startswith("Assistant: ")
        assert "BigTex" not in qr.text

    def test_both_sides_match_labels_user_half_only(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="tingling toes", assistant_content="tingling is odd",
             sender="BigTex")
        qr = store.search_canonical_turn_text("tingling", conversation_id="c")[0]
        assert qr.matched_side == "both"
        assert qr.text.startswith("BigTex: ")
        assert "\n\nAssistant: " in qr.text

    def test_sender_label_keeps_the_user_statement_ranking_signal(self):
        from virtual_context.core.quote_search import _rerank_quote_results
        from virtual_context.types import QuoteResult

        results = [
            QuoteResult(
                text="Assistant: tingling",
                tag="chat",
                segment_ref="assistant",
                match_type="full_text_search",
                source_scope="turn",
                matched_side="assistant",
            ),
            QuoteResult(
                text="An Extremely Long Sender Name: tingling",
                tag="chat",
                segment_ref="user",
                match_type="full_text_search",
                source_scope="turn",
                matched_side="user",
            ),
        ]

        ranked = _rerank_quote_results(
            results,
            "tingling",
            max_results=2,
            mode="lookup",
        )

        assert [result.segment_ref for result in ranked] == ["user", "assistant"]

    def test_sender_label_keeps_the_first_person_value_signal(self):
        from virtual_context.core.quote_search import _build_exact_value_candidates

        candidates = _build_exact_value_candidates(
            [{
                "excerpt": "BigTex: I was using version 1.2.3",
                "matched_side": "user",
                "match_type": "full_text_search",
            }],
            query="what version was I using",
            intent_context="",
        )

        assert candidates[0]["user_statement"] is True


class TestUnchangedWithoutSender:
    def test_user_match_with_empty_sender_renders_as_before(self, tmp_path: Path):
        """I3."""
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="toes tingling")
        qr = store.search_canonical_turn_text("tingling", conversation_id="c")[0]
        assert qr.matched_side == "user"
        assert qr.text.startswith("User: ")

    def test_assistant_match_with_empty_sender_renders_as_before(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, assistant_content="neurological")
        qr = store.search_canonical_turn_text("neurological", conversation_id="c")[0]
        assert qr.text.startswith("Assistant: ")

    def test_non_matching_query_returns_nothing(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="hello", sender="BigTex")
        assert store.search_canonical_turn_text("zzz", conversation_id="c") == []

    def test_result_shape_is_preserved(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="hello", sender="BigTex")
        qr = store.search_canonical_turn_text("bigtex", conversation_id="c")[0]
        assert qr.match_type == "full_text_search"
        assert qr.source_scope == "turn"
        assert qr.segment_ref == "canonical_turn_ct-1"
        assert qr.tag == "chat"
        assert qr.tags == ["chat"]


class TestFindQuoteEndToEnd:
    def test_find_quote_surfaces_a_sender_only_match(self, tmp_path: Path):
        from virtual_context.config import VirtualContextConfig
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.types import (
            RetrieverConfig,
            StorageConfig,
            TagGeneratorConfig,
        )

        config = VirtualContextConfig(
            conversation_id="c",
            tenant_id="t",
            storage=StorageConfig(
                backend="sqlite", sqlite_path=str(tmp_path / "vc.db"),
            ),
            tag_generator=TagGeneratorConfig(type="keyword"),
            retriever=RetrieverConfig(inbound_tagger_type="llm"),
        )
        engine = VirtualContextEngine(config=config)
        store = engine._store
        store.upsert_conversation(tenant_id="t", conversation_id="c")
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")

        result = engine.find_quote("bigtex")
        assert result["found"] is True
        excerpts = [r["excerpt"] for r in result["results"]]
        assert any(e.startswith("BigTex: ") for e in excerpts)
        engine.close()
