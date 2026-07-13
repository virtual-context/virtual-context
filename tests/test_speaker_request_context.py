"""Requester-intent conditioning and single-target precedence.

First-person intent is classified only over the request's immutable
``original_active_user_text`` — the model can neither create requester
conditioning by rewriting a tool query nor remove it by narrowing one.
Exactly one conditioning target exists per call: a valid explicit roster
handle wins, otherwise the trusted requester on first-person intent,
otherwise none. Missing request roles disable the requester signal, and
the response reports only the conditioning-source class, never an actor.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from virtual_context.core.quote_search import find_quote
from virtual_context.core.tool_loop import (
    _resolve_speaker_conditioning,
    execute_vc_tool,
)
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


def _ctx(**kw) -> SpeakerRetrievalContext:
    defaults = dict(
        tenant_id="t1",
        owner_conversation_id=OWNER,
        audience_conversation_id=AUDIENCE,
        requester_actor_id=ALEX,
        roster_snapshot_id=SNAPSHOT_ID,
        original_active_user_text="What did I say about the dosage?",
    )
    defaults.update(kw)
    return SpeakerRetrievalContext(**defaults)


def _snapshot(*, epoch=EPOCH) -> SpeakerRosterSnapshot:
    return SpeakerRosterSnapshot(
        snapshot_id=SNAPSHOT_ID,
        entries=(
            SpeakerRosterEntry(handle="alex", name="Alex", actor_id=ALEX),
            SpeakerRosterEntry(handle="bea", name="Bea", actor_id=BEA),
        ),
        tenant_id="t1",
        audience_conversation_id=AUDIENCE,
        lifecycle_epoch=epoch,
    )


def _qr(text, turn, *, actor=""):
    return QuoteResult(
        text=text,
        tag="chat",
        segment_ref=f"canonical_turn_ct{turn}",
        source_scope="turn",
        turn_number=turn,
        matched_side="user",
        provenance=SourceProvenance(
            conversation_id=OWNER,
            canonical_turn_id=f"ct{turn}",
            source_role="requester",
            actor_id=actor,
            audience_conversation_id=AUDIENCE,
            audience_attribution_version=1,
        ),
    )


def _tied_corpus():
    """Two candidates with identical base relevance; input order is the tie."""
    return [
        _qr("the dosage was 5mg", 2, actor=BEA),
        _qr("the dosage was 5mg", 1, actor=ALEX),
    ]


class Store:
    def __init__(self, corpus):
        self.corpus = list(corpus)

    def search_canonical_turn_text(
        self, query, limit=5, conversation_id=None, channel="", **kwargs,
    ):
        return list(self.corpus[:limit])

    def get_lifecycle_epoch(self, conversation_id):
        return EPOCH

    def search_facts(self, query, limit=10, conversation_id=None):
        return []


class SemanticStub:
    def semantic_canonical_turn_search(
        self, query, *, max_results=5, conversation_id=None, channel="",
        **kwargs,
    ):
        return []


def _engine(store, *, selection=True):
    semantic = SemanticStub()
    search = SearchConfig(
        tool_guard_enabled=False,
        speaker_annotations_enabled=True,
        speaker_selection_enabled=selection,
    )
    config = SimpleNamespace(search=search, conversation_id=OWNER)

    def _engine_find_quote(
        query, max_results=None, intent_context="", session_filter="",
        mode="lookup", channel="", *, speaker_context=None,
        speaker_handles=None,
    ):
        if speaker_context is None or not speaker_context.eligible:
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
            speaker_handles=(
                speaker_handles if speaker_context is not None else None
            ),
        )

    return SimpleNamespace(
        config=config,
        _store=store,
        _semantic=semantic,
        find_quote=_engine_find_quote,
    )


def _run(tool_input, *, ctx, snapshot=None, corpus=None, selection=True):
    engine = _engine(Store(corpus or _tied_corpus()), selection=selection)
    out = execute_vc_tool(
        engine, "vc_find_quote", dict(tool_input),
        speaker_context=ctx,
        roster_snapshot=snapshot if snapshot is not None else _snapshot(),
    )
    return json.loads(out)


class TestRequesterIntent:
    def test_original_intent_survives_narrow_tool_query(self):
        # The model narrowed the first-person question to just "dosage";
        # the original intent still conditions the trusted requester.
        got = _run({"query": "dosage", "mode": "lookup"}, ctx=_ctx())
        assert got["conditioning_source"] == "requester_intent"
        # The requester's own equal-relevance candidate wins the tie.
        assert [r["turn_number"] for r in got["results"]] == [1, 2]

    def test_model_added_first_person_does_not_condition(self):
        ctx = _ctx(
            original_active_user_text=(
                "Summarize the dosage history for the team."
            ),
        )
        # The tool QUERY says "I" but the original user text does not.
        got = _run(
            {"query": "what did I say about dosage", "mode": "lookup"},
            ctx=ctx,
        )
        assert got["conditioning_source"] == "none"
        # No conditioning target: the stable input-order tie stands.
        assert [r["turn_number"] for r in got["results"]] == [2, 1]

    def test_intent_context_argument_never_conditions(self):
        ctx = _ctx(original_active_user_text="dosage history please")
        engine = _engine(Store(_tied_corpus()))
        got = json.loads(execute_vc_tool(
            engine, "vc_find_quote", {"query": "dosage", "mode": "lookup"},
            intent_context="what did I say about the dosage",
            speaker_context=ctx,
            roster_snapshot=_snapshot(),
        ))
        assert got["conditioning_source"] == "none"


class TestSingleTargetPrecedence:
    def test_explicit_handle_is_the_sole_conditioning_target(self):
        # First-person original text AND an explicit different speaker:
        # the explicit roster selection wins and signals are not combined.
        got = _run(
            {"query": "dosage", "mode": "lookup", "speaker": "bea"},
            ctx=_ctx(),
        )
        assert got["conditioning_source"] == "explicit_roster"
        # Bea's candidate wins the equal bucket — not the requester's.
        assert [r["turn_number"] for r in got["results"]] == [2, 1]

    def test_resolution_object_targets_exactly_one_actor(self):
        engine = _engine(Store(_tied_corpus()))
        conditioning = _resolve_speaker_conditioning(
            engine,
            {"query": "dosage", "speaker": "bea"},
            _ctx(),
            _snapshot(),
        )
        assert conditioning.conditioning_source == "explicit_roster"
        assert conditioning.conditioning_actor_id == BEA
        # And the actor id never leaks through repr.
        assert BEA not in repr(conditioning)

    def test_unresolved_explicit_falls_back_to_requester_intent(self):
        got = _run(
            {"query": "dosage", "mode": "lookup", "speaker": "zoe"},
            ctx=_ctx(),
        )
        # Byte-identical to the no-hint call, which conditions the
        # requester; only the unresolved metadata is added.
        assert got["conditioning_source"] == "requester_intent"
        assert got["speaker_hint"] == "unresolved"
        assert got["filter_applied"] is False
        assert [r["turn_number"] for r in got["results"]] == [1, 2]

    def test_missing_roles_disable_requester_conditioning(self):
        ctx = _ctx(requester_actor_id="")
        got = _run({"query": "dosage", "mode": "lookup"}, ctx=ctx)
        assert got["conditioning_source"] == "none"
        assert [r["turn_number"] for r in got["results"]] == [2, 1]

    def test_gate_off_reports_nothing_and_conditions_nothing(self):
        got = _run(
            {"query": "dosage", "mode": "lookup"}, ctx=_ctx(),
            selection=False,
        )
        assert "conditioning_source" not in got
        assert [r["turn_number"] for r in got["results"]] == [2, 1]
