"""Quote candidate dedupe identity on the speaker-aware branch.

The legacy branch collapses turn candidates by logical turn number,
byte-identically to the shipped behavior, and forwards no speaker kwarg to
either candidate source. The speaker-aware branch dedupes by the physical
row and role-local lane ``(conversation_id, canonical_turn_id,
source_role)`` instead: duplicate message ids, physical sibling rows, and
logical merging may not collapse or transfer authorship across physical
rows or roles.
"""

from __future__ import annotations

from virtual_context.core.quote_search import (
    _search_find_quote_candidates,
    find_quote,
)
from virtual_context.types import (
    QuoteResult,
    SourceProvenance,
    SpeakerRetrievalContext,
)


def _context() -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="t1",
        owner_conversation_id="c",
        audience_conversation_id="c",
        requester_actor_id="actor:discord:9",
    )


def _qr(
    text,
    turn_number,
    *,
    conversation_id="c",
    canonical_turn_id="",
    source_role="requester",
    actor_id="",
):
    qr = QuoteResult(
        text=text,
        tag="chat",
        segment_ref=f"turn_{turn_number}",
        source_scope="turn",
        turn_number=turn_number,
        matched_side="user",
    )
    if canonical_turn_id:
        qr.provenance = SourceProvenance(
            conversation_id=conversation_id,
            canonical_turn_id=canonical_turn_id,
            source_role=source_role,
            actor_id=actor_id,
        )
    return qr


class _LexStore:
    def __init__(self, results):
        self._results = list(results)
        self.kwargs_seen = []

    def search_canonical_turn_text(
        self, query, limit=5, conversation_id=None, channel="", **kwargs,
    ):
        self.kwargs_seen.append(dict(kwargs))
        return list(self._results)


class _SemanticStub:
    def __init__(self, results=()):
        self._results = list(results)
        self.kwargs_seen = []

    def semantic_canonical_turn_search(
        self, query, *, max_results=5, conversation_id=None, channel="",
        **kwargs,
    ):
        self.kwargs_seen.append(dict(kwargs))
        return list(self._results)


def _candidates(store, semantic, *, speaker_context=None, limit=6):
    return _search_find_quote_candidates(
        store,
        semantic,
        "peptide",
        limit=limit,
        mode="lookup",
        conversation_id="c",
        speaker_context=speaker_context,
    )


class TestLegacyDedupeUnchanged:
    def test_same_turn_number_collapses_and_no_speaker_kwarg_is_sent(self):
        # Duplicate message ids created sibling physical rows that share one
        # logical turn number. The legacy branch collapses them, exactly as
        # shipped, even when provenance happens to be attached.
        store = _LexStore([
            _qr("first sibling", 4, canonical_turn_id="ct-a"),
            _qr("second sibling", 4, canonical_turn_id="ct-b"),
        ])
        semantic = _SemanticStub()
        results = _candidates(store, semantic)
        assert [r.text for r in results] == ["first sibling"]
        # The legacy call shape is byte-identical: no speaker kwarg at all.
        assert "speaker_context" not in store.kwargs_seen[0]
        assert "speaker_context" not in semantic.kwargs_seen[0]

    def test_segment_and_text_fallback_keys_are_untouched(self):
        seg = QuoteResult(text="seg text", tag="chat", segment_ref="seg_1")
        bare = QuoteResult(text="bare text", tag="chat", segment_ref="")
        store = _LexStore([seg, seg, bare, bare])
        results = _candidates(store, _SemanticStub())
        assert [r.text for r in results] == ["seg text", "bare text"]


class TestSpeakerBranchPhysicalDedupe:
    def test_duplicate_message_id_siblings_are_not_collapsed(self):
        # The adversary: identical logical turn number, distinct physical
        # rows. Physical identity keeps both; authorship cannot transfer.
        ctx = _context()
        store = _LexStore([
            _qr("first sibling", 4, canonical_turn_id="ct-a",
                actor_id="actor:discord:1"),
            _qr("second sibling", 4, canonical_turn_id="ct-b",
                actor_id="actor:discord:2"),
        ])
        semantic = _SemanticStub()
        results = _candidates(store, semantic, speaker_context=ctx)
        assert [r.text for r in results] == ["first sibling", "second sibling"]
        # Both candidate sources received the exact context object.
        assert store.kwargs_seen[0]["speaker_context"] is ctx
        assert semantic.kwargs_seen[0]["speaker_context"] is ctx

    def test_same_physical_row_and_role_collapses_across_sources(self):
        ctx = _context()
        lexical_hit = _qr("lexical hit", 4, canonical_turn_id="ct-a")
        semantic_hit = _qr("semantic hit", 4, canonical_turn_id="ct-a")
        store = _LexStore([lexical_hit])
        semantic = _SemanticStub([semantic_hit])
        results = _candidates(store, semantic, speaker_context=ctx)
        assert [r.text for r in results] == ["lexical hit"]

    def test_same_physical_row_with_different_roles_stays_distinct(self):
        ctx = _context()
        store = _LexStore([
            _qr("requester lane", 4, canonical_turn_id="ct-a",
                source_role="requester"),
            _qr("subject lane", 4, canonical_turn_id="ct-a",
                source_role="subject"),
        ])
        results = _candidates(store, _SemanticStub(), speaker_context=ctx)
        assert [r.text for r in results] == ["requester lane", "subject lane"]

    def test_candidates_without_provenance_fall_back_to_the_turn_key(self):
        # A source that projected no physical provenance may only ever
        # collapse MORE, never surface duplicates it cannot distinguish.
        ctx = _context()
        store = _LexStore([
            _qr("first", 4),
            _qr("second", 4),
        ])
        results = _candidates(store, _SemanticStub(), speaker_context=ctx)
        assert [r.text for r in results] == ["first"]


class TestFindQuoteThreading:
    def test_find_quote_forwards_the_exact_context_to_both_sources(self):
        ctx = _context()
        store = _LexStore([
            _qr("peptide dosage answer", 4, canonical_turn_id="ct-a"),
        ])
        semantic = _SemanticStub()
        response = find_quote(
            store, semantic, "peptide",
            conversation_id="c", speaker_context=ctx,
        )
        assert response["found"] is True
        assert store.kwargs_seen[0]["speaker_context"] is ctx
        assert semantic.kwargs_seen[0]["speaker_context"] is ctx

    def test_find_quote_without_context_sends_no_speaker_kwarg(self):
        store = _LexStore([
            _qr("peptide dosage answer", 4, canonical_turn_id="ct-a"),
        ])
        semantic = _SemanticStub()
        response = find_quote(store, semantic, "peptide", conversation_id="c")
        assert response["found"] is True
        assert "speaker_context" not in store.kwargs_seen[0]
        assert "speaker_context" not in semantic.kwargs_seen[0]
