"""Role-local annotation at the quote-search boundary.

Turn-backed results are annotated where physical provenance still exists:
each entry leaving ``find_quote`` on the speaker-aware branch carries its
audience-scoped label, verification flags, and ``source_role``, and — only
when the request bound a live roster snapshot — the snapshot handle for
in-snapshot actors. Out-of-snapshot actors keep a scoped label with an
empty handle; a stale or missing snapshot empties every handle without
touching labels; the legacy branch emits no annotation key at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from virtual_context.core.quote_search import find_quote
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import (
    CanonicalTurnRow,
    QuoteResult,
    SearchConfig,
    SourceProvenance,
    SpeakerRetrievalContext,
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
)

OWNER = "conv-annot"
AUDIENCE = OWNER
ALEX = "actor:discord:alex"
BEA = "actor:discord:bea"
CARA = "actor:discord:cara"  # attributed but outside the capped snapshot
SNAPSHOT_ID = "snap-1"
EPOCH = 7


def _ctx(**kw) -> SpeakerRetrievalContext:
    defaults = dict(
        tenant_id="t1",
        owner_conversation_id=OWNER,
        audience_conversation_id=AUDIENCE,
        requester_actor_id=ALEX,
        roster_snapshot_id=SNAPSHOT_ID,
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


def _prov(role, actor="", claimed=""):
    return SourceProvenance(
        conversation_id=OWNER,
        canonical_turn_id=f"ct-{role}-{actor or 'none'}",
        source_role=role,
        actor_id=actor,
        audience_conversation_id=AUDIENCE,
        audience_attribution_version=1,
        claimed_subject_label=claimed,
    )


def _corpus() -> list[QuoteResult]:
    def qr(text, turn, prov, side=""):
        return QuoteResult(
            text=text, tag="chat", segment_ref=f"canonical_turn_ct{turn}",
            source_scope="turn", turn_number=turn, matched_side=side,
            provenance=prov,
        )

    return [
        qr("alex booked the launch venue", 9, _prov("requester", ALEX), "user"),
        qr("bea reviewed the launch list", 8, _prov("subject", BEA)),
        qr("cara joined the launch call", 7, _prov("requester", CARA), "user"),
        qr("the launch date is confirmed", 6, _prov("assistant"), "assistant"),
        qr("launch and reply mixed up", 5, _prov("mixed")),
        qr("someone said the launch slipped", 4,
           _prov("subject", "", claimed="Mystery Guest")),
    ]


def _label_rows():
    def row(actor, sort_key, sender):
        return CanonicalTurnRow(
            conversation_id=OWNER,
            canonical_turn_id=f"label-{actor}-{sort_key}",
            sort_key=float(sort_key),
            user_content="hello there",
            sender=sender,
            sender_actor_id=actor,
            audience_conversation_id=AUDIENCE,
            audience_attribution_version=1,
        )

    return [row(ALEX, 3.0, "Alex"), row(BEA, 2.0, "Bea"), row(CARA, 1.0, "Cara")]


class AnnotStore:
    def __init__(self, corpus=None, label_rows=None):
        self.corpus = list(corpus if corpus is not None else _corpus())
        self.label_rows = list(
            label_rows if label_rows is not None else _label_rows()
        )

    def search_canonical_turn_text(
        self, query, limit=5, conversation_id=None, channel="", **kwargs,
    ):
        return list(self.corpus[:limit])

    def get_recent_canonical_turns(self, conversation_id, *, limit):
        return list(self.label_rows)

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


def _engine(store=None, *, selection=True, annotations=True):
    store = store if store is not None else AnnotStore()
    semantic = SemanticStub()
    search = SearchConfig(
        tool_guard_enabled=False,
        speaker_annotations_enabled=annotations,
        speaker_selection_enabled=selection,
    )
    config = SimpleNamespace(search=search, conversation_id=OWNER)

    def _engine_find_quote(
        query, max_results=None, intent_context="", session_filter="",
        mode="lookup", channel="", *, speaker_context=None,
        speaker_handles=None,
    ):
        # Mirror SearchEngine's gate router.
        if (
            not search.speaker_annotations_enabled
            or speaker_context is None
            or not speaker_context.eligible
        ):
            speaker_context = None
        return find_quote(
            store, semantic, query,
            max_results if max_results is not None else 10,
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
        config=config, _store=store, _semantic=semantic,
        find_quote=_engine_find_quote,
    )


def _entries(engine, *, ctx, snapshot, tool_input=None) -> dict[str, dict]:
    got = json.loads(execute_vc_tool(
        engine, "vc_find_quote",
        dict(tool_input or {"query": "launch"}),
        speaker_context=ctx, roster_snapshot=snapshot,
    ))
    assert got["found"] is True
    return {entry["excerpt"]: entry for entry in got["results"]}


class TestBoundaryAnnotation:
    def test_every_lane_is_annotated_at_the_tool_boundary(self):
        by_excerpt = _entries(
            _engine(), ctx=_ctx(), snapshot=_snapshot(),
        )

        alex = by_excerpt["alex booked the launch venue"]
        assert alex["speaker_label"] == "Alex"
        assert alex["speaker_handle"] == "alex"
        assert alex["speaker_actor_known"] is True
        assert alex["speaker_verified"] is True
        assert alex["source_role"] == "requester"

        bea = by_excerpt["bea reviewed the launch list"]
        assert bea["speaker_label"] == "Bea"
        assert bea["speaker_handle"] == "bea"
        assert bea["source_role"] == "subject"

        # Audience-admissible actor outside the capped snapshot: scoped
        # label and known-actor flag, but no handle is minted or revealed.
        cara = by_excerpt["cara joined the launch call"]
        assert cara["speaker_label"] == "Cara"
        assert cara["speaker_handle"] == ""
        assert cara["speaker_actor_known"] is True
        assert cara["speaker_verified"] is True

        assistant = by_excerpt["the launch date is confirmed"]
        assert assistant["speaker_label"] == "assistant"
        assert assistant["speaker_handle"] == ""
        assert assistant["speaker_actor_known"] is False
        assert assistant["speaker_verified"] is True

        mixed = by_excerpt["launch and reply mixed up"]
        assert mixed["speaker_scope"] == "mixed"
        assert "speaker_label" not in mixed

        claim = by_excerpt["someone said the launch slipped"]
        assert claim["speaker_label"] == ""
        assert claim["speaker_verified"] is False
        assert claim["claimed_speaker_label"] == "Mystery Guest"

    def test_selection_gate_off_still_annotates_without_handles_consumed(self):
        # Annotation is a P1 concern: with selection off the same labels and
        # handles appear, and the arriving speaker argument changes nothing.
        by_excerpt = _entries(
            _engine(selection=False), ctx=_ctx(), snapshot=_snapshot(),
            tool_input={"query": "launch", "speaker": "bea"},
        )
        assert by_excerpt["alex booked the launch venue"]["speaker_handle"] \
            == "alex"
        assert by_excerpt["bea reviewed the launch list"]["speaker_label"] \
            == "Bea"

    def test_stale_snapshot_id_keeps_labels_but_empties_handles(self):
        by_excerpt = _entries(
            _engine(), ctx=_ctx(), snapshot=_snapshot(snapshot_id="snap-OLD"),
        )
        alex = by_excerpt["alex booked the launch venue"]
        assert alex["speaker_label"] == "Alex"
        assert alex["speaker_handle"] == ""

    def test_dead_lifecycle_epoch_empties_handles(self):
        by_excerpt = _entries(
            _engine(), ctx=_ctx(), snapshot=_snapshot(epoch=EPOCH - 1),
        )
        alex = by_excerpt["alex booked the launch venue"]
        assert alex["speaker_label"] == "Alex"
        assert alex["speaker_handle"] == ""

    def test_no_snapshot_keeps_labels_and_empty_handles(self):
        # The VCRECALL/command shape: a derived context, no roster snapshot.
        by_excerpt = _entries(_engine(), ctx=_ctx(), snapshot=None)
        alex = by_excerpt["alex booked the launch venue"]
        assert alex["speaker_label"] == "Alex"
        assert alex["speaker_handle"] == ""

    def test_annotations_gate_off_emits_no_annotation_keys(self):
        by_excerpt = _entries(
            _engine(annotations=False, selection=False),
            ctx=_ctx(), snapshot=_snapshot(),
        )
        for entry in by_excerpt.values():
            for key in (
                "speaker_label", "speaker_handle", "speaker_actor_known",
                "speaker_verified", "claimed_speaker_label", "speakers",
                "speaker_scope", "source_role",
            ):
                assert key not in entry

    def test_label_store_failure_fails_open_to_empty_labels(self):
        class NoLabelStore(AnnotStore):
            def get_recent_canonical_turns(self, conversation_id, *, limit):
                raise RuntimeError("label scan down")

        by_excerpt = _entries(
            _engine(NoLabelStore()), ctx=_ctx(), snapshot=_snapshot(),
        )
        alex = by_excerpt["alex booked the launch venue"]
        assert alex["speaker_label"] == ""
        assert alex["speaker_handle"] == "alex"
        assert alex["speaker_actor_known"] is True
