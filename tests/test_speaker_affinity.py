"""Speaker affinity ordering and unresolved-hint parity.

An absent hint and every flavor of bad hint — garbage, malformed, stale
snapshot id, stale lifecycle epoch, off-snapshot handle — run the
byte-identical unconditioned candidate and ranking path; only the
unresolved-hint metadata may differ. A valid hint is a bounded ordering
signal: it never changes the candidate multiset, source limits, or
``found``, and it can move a same-speaker result only inside an
equal-base-relevance bucket. Exact-phrase and rare-term winners cannot be
displaced by a weaker same-speaker hit. With the selection gate off, an
arriving ``speaker`` argument is not consumed at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from virtual_context.core.quote_search import find_quote
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import (
    QuoteResult,
    SearchConfig,
    SourceProvenance,
    SpeakerRetrievalContext,
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
)

OWNER = "conv-g"
AUDIENCE = "conv-g"
ALEX = "actor:discord:alex"
BEA = "actor:discord:bea"
SNAPSHOT_ID = "snap-1"
EPOCH = 7

_METADATA_KEYS = ("speaker_hint", "filter_applied")


def _ctx(**kw) -> SpeakerRetrievalContext:
    defaults = dict(
        tenant_id="t1",
        owner_conversation_id=OWNER,
        audience_conversation_id=AUDIENCE,
        requester_actor_id=ALEX,
        roster_snapshot_id=SNAPSHOT_ID,
        original_active_user_text="what did bea say about the peptide",
    )
    defaults.update(kw)
    return SpeakerRetrievalContext(**defaults)


def _snapshot(*, snapshot_id=SNAPSHOT_ID, epoch=EPOCH) -> SpeakerRosterSnapshot:
    return SpeakerRosterSnapshot(
        snapshot_id=snapshot_id,
        entries=(
            SpeakerRosterEntry(handle="alex", name="Alex", actor_id=ALEX),
            SpeakerRosterEntry(handle="bea", name="Bea", actor_id=BEA),
        ),
        tenant_id="t1",
        audience_conversation_id=AUDIENCE,
        lifecycle_epoch=epoch,
    )


def _qr(text, turn, *, role="requester", actor=""):
    return QuoteResult(
        text=text,
        tag="chat",
        segment_ref=f"canonical_turn_ct{turn}",
        source_scope="turn",
        turn_number=turn,
        matched_side="user" if role == "requester" else "",
        provenance=SourceProvenance(
            conversation_id=OWNER,
            canonical_turn_id=f"ct{turn}",
            source_role=role,
            actor_id=actor,
            audience_conversation_id=AUDIENCE,
            audience_attribution_version=1,
        ),
    )


class LimitStore:
    """Lexical source honoring its ``limit`` over a newest-first corpus."""

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.limits_seen: list[int] = []

    def search_canonical_turn_text(
        self, query, limit=5, conversation_id=None, channel="", **kwargs,
    ):
        self.limits_seen.append(limit)
        return list(self.corpus[:limit])

    def get_lifecycle_epoch(self, conversation_id):
        return EPOCH

    def search_facts(self, query, limit=10, conversation_id=None):
        return []


class SemanticStub:
    def __init__(self, corpus=()):
        self.corpus = list(corpus)
        self.limits_seen: list[int] = []

    def semantic_canonical_turn_search(
        self, query, *, max_results=5, conversation_id=None, channel="",
        **kwargs,
    ):
        self.limits_seen.append(max_results)
        return list(self.corpus[:max_results])


def _engine(store, semantic=None, *, selection=True, annotations=True):
    semantic = semantic if semantic is not None else SemanticStub()
    search = SearchConfig(
        tool_guard_enabled=False,
        speaker_annotations_enabled=annotations,
        speaker_selection_enabled=selection,
    )
    config = SimpleNamespace(search=search, conversation_id=OWNER)

    def _engine_find_quote(
        query, max_results=None, intent_context="", session_filter="",
        mode="lookup", channel="", *, speaker_context=None,
    ):
        # Mirror SearchEngine's gate router for the unconditioned seam.
        if (
            not search.speaker_annotations_enabled
            or speaker_context is None
            or not speaker_context.eligible
        ):
            speaker_context = None
        return find_quote(
            store, semantic, query,
            max_results if max_results is not None else 5,
            intent_context=intent_context,
            session_filter=session_filter,
            mode=mode,
            conversation_id=OWNER,
            channel=channel,
            speaker_context=speaker_context,
        )

    return SimpleNamespace(
        config=config,
        _store=store,
        _semantic=semantic,
        find_quote=_engine_find_quote,
    )


def _run(engine, tool_input, *, ctx=None, snapshot=None):
    out = execute_vc_tool(
        engine, "vc_find_quote", dict(tool_input),
        speaker_context=ctx,
        roster_snapshot=snapshot,
    )
    return json.loads(out)


def _mixed_corpus():
    return [
        _qr("the peptide dose was raised to 10mg", 9, actor=ALEX),
        _qr("bea reviewed the peptide storage notes", 8, actor=BEA),
        _qr("peptide shipment arrived on tuesday", 7, actor=ALEX),
        _qr("we compared peptide vendors last week", 6, actor=BEA),
    ]


class TestUnresolvedHintParity:
    def test_absent_and_every_bad_hint_are_exact_ordered_parity(self):
        baseline = _run(
            _engine(LimitStore(_mixed_corpus())),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert baseline["found"] is True
        assert baseline["conditioning_source"] == "none"
        for key in _METADATA_KEYS:
            assert key not in baseline

        variants = [
            # Garbage: grammatically a handle, not in any snapshot.
            ({"speaker": "zoe"}, _snapshot()),
            # Malformed: fails the bounded handle grammar.
            ({"speaker": "Not A Handle!!"}, _snapshot()),
            # Empty string: arrived but resolvable to nothing.
            ({"speaker": ""}, _snapshot()),
            # Stale snapshot: id differs from the one bound to the request.
            ({"speaker": "bea"}, _snapshot(snapshot_id="snap-OLD")),
            # Stale lifecycle: audience epoch moved after snapshot build.
            ({"speaker": "bea"}, _snapshot(epoch=EPOCH - 1)),
            # Off-snapshot entirely: no snapshot carried by the request.
            ({"speaker": "bea"}, None),
        ]
        for extra, snapshot in variants:
            got = _run(
                _engine(LimitStore(_mixed_corpus())),
                {"query": "peptide", "mode": "lookup", **extra},
                ctx=_ctx(), snapshot=snapshot,
            )
            assert got["speaker_hint"] == "unresolved", extra
            assert got["filter_applied"] is False, extra
            for key in _METADATA_KEYS:
                got.pop(key, None)
            # Byte-identical to the no-hint call: identities, order,
            # scores, limits, found — everything but the metadata above.
            assert got == baseline, extra

    def test_unresolved_hint_sends_identical_source_limits(self):
        plain_store = LimitStore(_mixed_corpus())
        plain_semantic = SemanticStub()
        _run(
            _engine(plain_store, plain_semantic),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        hinted_store = LimitStore(_mixed_corpus())
        hinted_semantic = SemanticStub()
        _run(
            _engine(hinted_store, hinted_semantic),
            {"query": "peptide", "mode": "lookup", "speaker": "zoe"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert hinted_store.limits_seen == plain_store.limits_seen
        assert hinted_semantic.limits_seen == plain_semantic.limits_seen


class TestValidAffinity:
    def test_valid_hint_preserves_multiset_limits_and_found(self):
        plain_store = LimitStore(_mixed_corpus())
        plain_semantic = SemanticStub()
        baseline = _run(
            _engine(plain_store, plain_semantic),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        hinted_store = LimitStore(_mixed_corpus())
        hinted_semantic = SemanticStub()
        hinted = _run(
            _engine(hinted_store, hinted_semantic),
            {"query": "peptide", "mode": "lookup", "speaker": "bea"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert hinted["conditioning_source"] == "explicit_roster"
        for key in _METADATA_KEYS:
            assert key not in hinted
        assert hinted["found"] == baseline["found"]
        # Candidate multiset preserved: same excerpts, possibly reordered.
        assert sorted(r["excerpt"] for r in hinted["results"]) \
            == sorted(r["excerpt"] for r in baseline["results"])
        # Source thresholds and limits identical with and without affinity.
        assert hinted_store.limits_seen == plain_store.limits_seen
        assert hinted_semantic.limits_seen == plain_semantic.limits_seen

    def test_exact_phrase_winner_cannot_be_displaced(self):
        corpus = [
            _qr("bea adjusted the dosage again this morning", 9, actor=BEA),
            _qr("the magnesium glycinate dosage was 400mg", 8, actor=ALEX),
            _qr("bea asked about the dosage schedule", 7, actor=BEA),
        ]
        got = _run(
            _engine(LimitStore(corpus)),
            {"query": "magnesium glycinate dosage", "mode": "lookup",
             "speaker": "bea"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert got["results"][0]["excerpt"] \
            == "the magnesium glycinate dosage was 400mg"

    def test_rare_term_winner_cannot_be_displaced(self):
        corpus = [
            _qr("bea stored the peptide in the fridge", 9, actor=BEA),
            _qr("reconstitution of the peptide vial takes bac water", 8,
                actor=ALEX),
            _qr("bea ordered more peptide on friday", 7, actor=BEA),
        ]
        got = _run(
            _engine(LimitStore(corpus)),
            {"query": "peptide reconstitution", "mode": "lookup",
             "speaker": "bea"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert got["results"][0]["excerpt"] \
            == "reconstitution of the peptide vial takes bac water"

    def test_affinity_reorders_only_inside_an_equal_relevance_bucket(self):
        corpus = [
            _qr("the peptide dose was set", 2, actor=ALEX),
            _qr("the peptide dose was set", 1, actor=BEA),
        ]
        plain = _run(
            _engine(LimitStore(corpus)),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        # Equal base relevance: the shipped stable tie keeps input order.
        assert [r["turn_number"] for r in plain["results"]] == [2, 1]
        hinted = _run(
            _engine(LimitStore(corpus)),
            {"query": "peptide", "mode": "lookup", "speaker": "bea"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        # The same-speaker candidate wins the equal bucket, nothing more.
        assert [r["turn_number"] for r in hinted["results"]] == [1, 2]

    def test_no_actor_id_reaches_the_output(self):
        out = execute_vc_tool(
            _engine(LimitStore(_mixed_corpus())), "vc_find_quote",
            {"query": "peptide", "mode": "lookup", "speaker": "bea"},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        )
        assert "actor:" not in out
        assert ALEX not in out and BEA not in out


class TestSelectionGateOff:
    def test_gate_off_ignores_speaker_and_adds_no_metadata(self):
        plain = _run(
            _engine(LimitStore(_mixed_corpus()), selection=False),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        hinted = _run(
            _engine(LimitStore(_mixed_corpus()), selection=False),
            {"query": "peptide", "mode": "lookup", "speaker": "bea",
             "speaker_only": True},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert hinted == plain
        assert "conditioning_source" not in hinted
        for key in _METADATA_KEYS:
            assert key not in hinted

    def test_annotations_off_disables_the_unit_even_with_selection_on(self):
        plain = _run(
            _engine(LimitStore(_mixed_corpus()), annotations=False),
            {"query": "peptide", "mode": "lookup"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        hinted = _run(
            _engine(LimitStore(_mixed_corpus()), annotations=False),
            {"query": "peptide", "mode": "lookup", "speaker": "bea"},
            ctx=_ctx(), snapshot=_snapshot(),
        )
        assert hinted == plain
        assert "conditioning_source" not in hinted

    def test_ineligible_context_disables_the_unit(self):
        ineligible = _ctx(audience_conversation_id="")
        plain = _run(
            _engine(LimitStore(_mixed_corpus())),
            {"query": "peptide", "mode": "lookup"},
            ctx=ineligible, snapshot=_snapshot(),
        )
        hinted = _run(
            _engine(LimitStore(_mixed_corpus())),
            {"query": "peptide", "mode": "lookup", "speaker": "bea"},
            ctx=ineligible, snapshot=_snapshot(),
        )
        assert hinted == plain
        assert "conditioning_source" not in hinted
