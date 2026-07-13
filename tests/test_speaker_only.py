"""Exact-attribution ``speaker_only`` filtering and exclusion counts.

A valid selection filters role-locally at each candidate source after that
source's ordinary relevance threshold and BEFORE its limit, so a matching
result below an unfiltered source top-N is still found. One classification
pass populates three disjoint, audience-scoped counts; aggregate source
classes and out-of-audience candidates are ineligible before counting and
contribute nothing. An absent or invalid selection with ``speaker_only``
runs the exact unfiltered path, reports ``filter_applied=false`` with the
mandatory warning, and performs no speculative count scan.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from virtual_context.core.quote_search import SpeakerConditioning, find_quote
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

COUNT_KEYS = (
    "pre_filter_matching_count",
    "excluded_other_speakers",
    "excluded_unknown_speaker",
)


def _ctx(**kw) -> SpeakerRetrievalContext:
    defaults = dict(
        tenant_id="t1",
        owner_conversation_id=OWNER,
        audience_conversation_id=AUDIENCE,
        requester_actor_id=ALEX,
        roster_snapshot_id=SNAPSHOT_ID,
        original_active_user_text="what exactly did bea say",
    )
    defaults.update(kw)
    return SpeakerRetrievalContext(**defaults)


def _snapshot() -> SpeakerRosterSnapshot:
    return SpeakerRosterSnapshot(
        snapshot_id=SNAPSHOT_ID,
        entries=(
            SpeakerRosterEntry(handle="alex", name="Alex", actor_id=ALEX),
            SpeakerRosterEntry(handle="bea", name="Bea", actor_id=BEA),
        ),
        tenant_id="t1",
        audience_conversation_id=AUDIENCE,
        lifecycle_epoch=EPOCH,
    )


def _bea_only_conditioning() -> SpeakerConditioning:
    return SpeakerConditioning(
        conditioning_actor_id=BEA,
        conditioning_source="explicit_roster",
        filter_active=True,
        hint_arrived=True,
        speaker_only_requested=True,
    )


def _qr(
    text, turn, *, role="requester", actor="", ctid=None,
    audience=AUDIENCE, version=1, channel="", scope="turn",
    with_provenance=True,
):
    return QuoteResult(
        text=text,
        tag="chat",
        segment_ref=f"canonical_turn_{ctid or f'ct{turn}'}",
        source_scope=scope,
        turn_number=turn,
        matched_side="user" if role == "requester" else "",
        provenance=SourceProvenance(
            conversation_id=OWNER,
            canonical_turn_id=ctid or f"ct{turn}",
            source_role=role,
            actor_id=actor,
            audience_conversation_id=audience,
            audience_attribution_version=version,
            origin_channel_id=channel,
        ) if with_provenance else None,
    )


class LimitStore:
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


def _engine(store, semantic=None, *, selection=True):
    semantic = semantic if semantic is not None else SemanticStub()
    search = SearchConfig(
        tool_guard_enabled=False,
        speaker_annotations_enabled=True,
        speaker_selection_enabled=selection,
    )
    config = SimpleNamespace(search=search, conversation_id=OWNER)

    def _engine_find_quote(
        query, max_results=None, intent_context="", session_filter="",
        mode="lookup", channel="", *, speaker_context=None,
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
        )

    return SimpleNamespace(
        config=config,
        _store=store,
        _semantic=semantic,
        find_quote=_engine_find_quote,
    )


class TestFilterPrecedesSourceLimit:
    def test_match_below_the_unfiltered_lexical_top_n_is_still_found(self):
        # Five newer other-speaker rows fill the unfiltered top-5; the only
        # bea row sits below the source limit.
        corpus = [
            _qr(f"peptide note {i}", 100 - i, actor=ALEX) for i in range(5)
        ] + [
            _qr("bea confirmed the peptide dose herself", 50, actor=BEA),
        ]
        unfiltered = find_quote(
            LimitStore(corpus), SemanticStub(), "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=_ctx(),
        )
        assert all(
            "bea confirmed" not in r["excerpt"]
            for r in unfiltered["results"]
        )

        store = LimitStore(corpus)
        filtered = find_quote(
            store, SemanticStub(), "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=_ctx(),
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert filtered["found"] is True
        assert [r["excerpt"] for r in filtered["results"]] \
            == ["bea confirmed the peptide dose herself"]
        assert filtered["filter_applied"] is True
        assert filtered["pre_filter_matching_count"] == 1
        assert filtered["excluded_other_speakers"] == 5
        assert filtered["excluded_unknown_speaker"] == 0
        # The predicate ran before the limit: the source was overfetched.
        assert store.limits_seen == [100]

    def test_semantic_source_is_filtered_before_its_limit_too(self):
        lexical = [_qr(f"peptide log {i}", 90 - i, actor=ALEX) for i in range(4)]
        semantic_corpus = [
            _qr(f"peptide chat {i}", 40 - i, actor=ALEX, ctid=f"sem{i}")
            for i in range(4)
        ] + [
            _qr("bea wrote down the peptide protocol", 30, actor=BEA,
                ctid="sem-bea"),
        ]
        semantic = SemanticStub(semantic_corpus)
        result = find_quote(
            LimitStore(lexical), semantic, "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=_ctx(),
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert [r["excerpt"] for r in result["results"]] \
            == ["bea wrote down the peptide protocol"]
        assert result["pre_filter_matching_count"] == 1
        assert result["excluded_other_speakers"] == 8
        # Overfetched semantic acceptance, never the plain remainder.
        assert semantic.limits_seen == [100]

    def test_cross_source_duplicates_are_counted_once(self):
        shared = _qr("peptide vendor call", 60, actor=ALEX, ctid="dup-1")
        lexical = [shared, _qr("bea peptide reminder", 59, actor=BEA)]
        semantic = SemanticStub([
            _qr("peptide vendor call", 60, actor=ALEX, ctid="dup-1"),
        ])
        result = find_quote(
            LimitStore(lexical), semantic, "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=_ctx(),
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert result["pre_filter_matching_count"] == 1
        assert result["excluded_other_speakers"] == 1


class TestRoleLocalPredicatesAndDisjointCounts:
    def test_lanes_classify_disjointly_and_scope_ineligible_never_counts(self):
        corpus = [
            # match: requester lane, bea's own statement
            _qr("bea peptide statement", 20, actor=BEA),
            # other: requester lane, different human
            _qr("alex peptide statement", 19, actor=ALEX),
            # match: subject lane, quoted text attributed to bea
            _qr("quoted bea peptide reply", 18, role="subject", actor=BEA),
            # other: subject lane, quoted text attributed to alex
            _qr("quoted alex peptide reply", 17, role="subject", actor=ALEX),
            # other: the reserved assistant identity never matches a human
            _qr("assistant peptide answer", 16, role="assistant"),
            # unknown: mixed excerpt is never one human speaker
            _qr("mixed peptide excerpt", 15, role="mixed"),
            # unknown: requester lane with no durable actor
            _qr("anonymous peptide note", 14, actor=""),
            # unknown: unattributed lane
            _qr("unattributed peptide text", 13, role="unattributed"),
            # ineligible: wrong audience — contributes NO count
            _qr("dm-only bea peptide", 12, actor=BEA, audience="conv-dm"),
            # ineligible: stale attribution version
            _qr("stale-version bea peptide", 11, actor=BEA, version=0),
            # ineligible: aggregate source class (no provenance, not turn)
            _qr("segment peptide rollup", 10, scope="segment",
                with_provenance=False),
        ]
        result = find_quote(
            LimitStore(corpus), SemanticStub(), "peptide",
            max_results=10, conversation_id=OWNER,
            speaker_context=_ctx(),
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert result["filter_applied"] is True
        assert result["pre_filter_matching_count"] == 2
        assert result["excluded_other_speakers"] == 3
        assert result["excluded_unknown_speaker"] == 3
        excerpts = {r["excerpt"] for r in result["results"]}
        assert excerpts == {"bea peptide statement", "quoted bea peptide reply"}
        # The out-of-audience bea row neither counted nor surfaced.
        assert all("dm-only" not in e for e in excerpts)

    def test_empty_stored_channel_fails_closed_under_a_channel_context(self):
        ctx = _ctx(audience_channel_id="chan-9")
        corpus = [
            _qr("bea in-channel peptide", 9, actor=BEA, channel="chan-9"),
            _qr("bea channelless peptide", 8, actor=BEA, channel=""),
            _qr("bea other-channel peptide", 7, actor=BEA, channel="chan-2"),
        ]
        result = find_quote(
            LimitStore(corpus), SemanticStub(), "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=ctx,
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert [r["excerpt"] for r in result["results"]] \
            == ["bea in-channel peptide"]
        assert result["pre_filter_matching_count"] == 1
        assert result["excluded_other_speakers"] == 0
        assert result["excluded_unknown_speaker"] == 0

    def test_zero_matches_still_reports_filter_and_counts(self):
        corpus = [
            _qr("alex peptide note", 5, actor=ALEX),
            _qr("mixed peptide excerpt", 4, role="mixed"),
        ]
        result = find_quote(
            LimitStore(corpus), SemanticStub(), "peptide",
            max_results=5, conversation_id=OWNER,
            speaker_context=_ctx(),
            speaker_conditioning=_bea_only_conditioning(),
        )
        assert result["found"] is False
        assert result["results"] == []
        assert result["filter_applied"] is True
        assert result["pre_filter_matching_count"] == 0
        assert result["excluded_other_speakers"] == 1
        assert result["excluded_unknown_speaker"] == 1


class TestInvalidSelectionWithSpeakerOnly:
    def test_invalid_selection_runs_unfiltered_with_mandatory_warning(self):
        corpus = [
            _qr("alex peptide note", 5, actor=ALEX),
            _qr("bea peptide note", 4, actor=BEA),
        ]
        plain_store = LimitStore(corpus)
        plain = json.loads(execute_vc_tool(
            _engine(plain_store), "vc_find_quote",
            {"query": "peptide", "mode": "lookup"},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        ))
        bad_store = LimitStore(corpus)
        got = json.loads(execute_vc_tool(
            _engine(bad_store), "vc_find_quote",
            {"query": "peptide", "mode": "lookup",
             "speaker": "zoe", "speaker_only": True},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        ))
        # Exact unfiltered result set, mandatory warning, no counts.
        assert got["results"] == plain["results"]
        assert got["found"] == plain["found"]
        assert got["filter_applied"] is False
        assert got["speaker_hint"] == "unresolved"
        assert "attribution filter" in got["warning"]
        for key in COUNT_KEYS:
            assert key not in got
        # No speculative exclusion scan: identical source limits.
        assert bad_store.limits_seen == plain_store.limits_seen

    def test_speaker_only_without_any_selection_also_degrades(self):
        corpus = [_qr("bea peptide note", 4, actor=BEA)]
        got = json.loads(execute_vc_tool(
            _engine(LimitStore(corpus)), "vc_find_quote",
            {"query": "peptide", "mode": "lookup", "speaker_only": True},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        ))
        assert got["filter_applied"] is False
        assert "attribution filter" in got["warning"]
        assert "speaker_hint" not in got
        for key in COUNT_KEYS:
            assert key not in got


class TestEndToEndThroughExecution:
    def test_valid_speaker_only_reports_counts_through_the_tool_result(self):
        corpus = [
            _qr("alex peptide note", 6, actor=ALEX),
            _qr("bea peptide note", 5, actor=BEA),
            _qr("mixed peptide excerpt", 4, role="mixed"),
        ]
        out = execute_vc_tool(
            _engine(LimitStore(corpus)), "vc_find_quote",
            {"query": "peptide", "mode": "lookup",
             "speaker": "bea", "speaker_only": True},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        )
        got = json.loads(out)
        assert got["conditioning_source"] == "explicit_roster"
        assert got["filter_applied"] is True
        assert got["pre_filter_matching_count"] == 1
        assert got["excluded_other_speakers"] == 1
        assert got["excluded_unknown_speaker"] == 1
        assert [r["excerpt"] for r in got["results"]] == ["bea peptide note"]
        # Actor ids never surface in the serialized tool result.
        assert "actor:" not in out

    def test_gate_off_never_filters(self):
        corpus = [
            _qr("alex peptide note", 6, actor=ALEX),
            _qr("bea peptide note", 5, actor=BEA),
        ]
        got = json.loads(execute_vc_tool(
            _engine(LimitStore(corpus), selection=False), "vc_find_quote",
            {"query": "peptide", "mode": "lookup",
             "speaker": "bea", "speaker_only": True},
            speaker_context=_ctx(), roster_snapshot=_snapshot(),
        ))
        assert len(got["results"]) == 2
        assert "filter_applied" not in got
        assert "warning" not in got
