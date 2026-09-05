"""Gate-controlled speaker annotation at the tool, REST, and MCP serializers.

With ``speaker_annotations_enabled=false`` every serialized surface is
byte-identical to the legacy output — same strings, same keys, same order.
With the gate on and a proved audience, fact hits disclose their numeric
attribution version and basis (with audience-scoped labels only for
role-local facts), aggregates disclose an honest ``speaker_scope``, and a
DM-scoped label can never surface through a guild-audience request.
Stateless MCP exposes only the structural annotation subset. Internal actor
ids appear in no output anywhere.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    Fact,
    PagingConfig,
    SearchConfig,
    SegmentMetadata,
    SpeakerRetrievalContext,
    StorageConfig,
    StoredSegment,
    TagGeneratorConfig,
    TagSummary,
    VirtualContextConfig,
)

OWNER = "owner-conv"
GUILD = "aud-guild"
DM = "aud-dm"
ACTOR = "actor:discord:111"
DM_ONLY_ACTOR = "actor:discord:222"
CANARY_ACTORS = (ACTOR, DM_ONLY_ACTOR)


def _ctx(audience: str = GUILD) -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="t",
        owner_conversation_id=OWNER,
        audience_conversation_id=audience,
    )


INELIGIBLE_CTX = SpeakerRetrievalContext(
    tenant_id="t", owner_conversation_id=OWNER,
)


@pytest.fixture
def engine(tmp_path):
    """Real SQLite-backed engine seeded with rows, facts, and a summary."""
    from virtual_context.engine import VirtualContextEngine

    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(
            backend="sqlite", sqlite_path=str(tmp_path / "vc.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
        paging=PagingConfig(enabled=False),
        search=SearchConfig(tool_guard_enabled=False),
    )
    eng = VirtualContextEngine(config=config)
    store = eng._store

    common = dict(
        conversation_id=OWNER,
        assistant_content="Understood.",
        audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
    )
    store.save_canonical_turn(
        turn_number=0,
        user_content="I moved to Boston last spring",
        sender="Sania",
        sender_actor_id=ACTOR,
        audience_conversation_id=GUILD,
        **common,
    )
    # NEWER DM row for the same actor with the private nickname.
    store.save_canonical_turn(
        turn_number=1,
        user_content="dm chatter about Boston",
        sender="SnookieBear",
        sender_actor_id=ACTOR,
        audience_conversation_id=DM,
        **common,
    )
    # A DM-only participant.
    store.save_canonical_turn(
        turn_number=2,
        user_content="private note",
        sender="Hidden Person",
        sender_actor_id=DM_ONLY_ACTOR,
        audience_conversation_id=DM,
        **common,
    )

    ts = datetime(2026, 1, 10, tzinfo=timezone.utc)
    store.store_facts([
        Fact(
            id="fact-v2",
            subject="user",
            verb="visited",
            object="boston",
            what="user visited boston in spring",
            conversation_id=OWNER,
            mentioned_at=ts,
            author_actor_id=ACTOR,
            author_attribution_version=2,
            author_source_role="requester",
        ),
        Fact(
            id="fact-v2-dm-only",
            subject="user",
            verb="visited",
            object="museum",
            what="user visited the museum",
            conversation_id=OWNER,
            mentioned_at=ts.replace(day=9),
            author_actor_id=DM_ONLY_ACTOR,
            author_attribution_version=2,
            author_source_role="subject",
        ),
        Fact(
            id="fact-v1",
            subject="user",
            verb="visited",
            object="harbor",
            what="user visited the harbor",
            conversation_id=OWNER,
            mentioned_at=ts.replace(day=8),
            author_actor_id=ACTOR,
            author_attribution_version=1,
            author_source_role="requester",
        ),
        Fact(
            id="fact-v0",
            subject="user",
            verb="visited",
            object="airport",
            what="user visited the airport",
            conversation_id=OWNER,
            mentioned_at=ts.replace(day=7),
        ),
    ])

    store.save_tag_summary(
        TagSummary(
            tag="boston",
            summary="Trips to Boston and related plans.",
            summary_tokens=8,
            description="Boston travel",
        ),
        conversation_id=OWNER,
    )
    return eng


_TOOL_CALLS = [
    ("vc_query_facts", {"subject": "user"}),
    ("vc_recall_all", {}),
    ("vc_find_quote", {"query": "Boston", "mode": "lookup"}),
    ("vc_search_summaries", {"query": "Boston", "mode": "lookup"}),
]

_SPEAKER_KEYS = (
    "speaker_label", "speaker_handle", "speaker_actor_known",
    "speaker_verified", "claimed_speaker_label", "speaker_scope",
    "speakers", "attribution_basis", "author_attribution_version",
)


class TestGateOffByteIdentity:
    def test_gate_off_parity_excludes_summary_safety_boundary(self, engine):
        assert engine.config.search.speaker_annotations_enabled is False
        for name, tool_input in _TOOL_CALLS:
            legacy = execute_vc_tool(engine, name, dict(tool_input))
            gated = execute_vc_tool(
                engine, name, dict(tool_input), speaker_context=_ctx(),
            )
            if name == "vc_search_summaries":
                legacy_payload = json.loads(legacy)
                gated_payload = json.loads(gated)
                assert legacy_payload["found"] is False
                assert gated_payload["found"] is False
                assert "audience authority is unproved" in legacy_payload["message"]
                assert "No matches" in gated_payload["message"]
            elif name == "vc_query_facts":
                # Fact reads now require proved request authority regardless
                # of the independent annotation display gate.
                legacy_payload = json.loads(legacy)
                gated_payload = json.loads(gated)
                assert legacy_payload["count"] == 0
                assert legacy_payload["request_scope"] == "ineligible"
                assert gated_payload["count"] == 4
                assert gated_payload["fact_content_withheld"] is True
            elif name == "vc_find_quote":
                # Candidate authorization is independent of the annotation
                # display gate.  Missing authority fails closed; proved
                # authority reads the exact audience without adding labels.
                legacy_payload = json.loads(legacy)
                gated_payload = json.loads(gated)
                assert legacy_payload["found"] is False
                assert "authority is unproved" in legacy_payload["message"]
                assert gated_payload["found"] is True
            else:
                assert gated == legacy, name
            for key in _SPEAKER_KEYS:
                assert f'"{key}"' not in legacy, (name, key)

    def test_gate_on_ineligible_context_refuses_summary_search_only(self, engine):
        baseline = {
            name: execute_vc_tool(engine, name, dict(tool_input))
            for name, tool_input in _TOOL_CALLS
        }
        engine.config.search.speaker_annotations_enabled = True
        for name, tool_input in _TOOL_CALLS:
            unproved = execute_vc_tool(
                engine, name, dict(tool_input),
                speaker_context=INELIGIBLE_CTX,
            )
            absent = execute_vc_tool(engine, name, dict(tool_input))
            if name == "vc_search_summaries":
                payload = json.loads(unproved)
                assert payload["found"] is False
                assert "audience authority is unproved" in payload["message"]
            else:
                assert unproved == baseline[name], name
            assert absent == baseline[name], name


class TestFactModelBoundary:
    def _facts_by_id(self, engine, audience=GUILD):
        engine.config.search.speaker_annotations_enabled = True
        out = execute_vc_tool(
            engine, "vc_query_facts", {"subject": "user"},
            speaker_context=_ctx(audience),
        )
        payload = json.loads(out)
        for actor in CANARY_ACTORS:
            assert actor not in out
        return {f["id"]: f for f in payload["facts"]}, payload, out

    def test_fact_rows_expose_only_opaque_structure(self, engine):
        facts, payload, out = self._facts_by_id(engine)

        assert set(facts) == {
            "fact-v2", "fact-v2-dm-only", "fact-v1", "fact-v0",
        }
        assert all(
            set(entry) == {"id", "segment_ref", "tags"}
            for entry in facts.values()
        )
        assert payload["fact_content_withheld"] is True
        for prose in ("boston", "museum", "harbor", "airport"):
            assert prose not in out.lower()
        for key in _SPEAKER_KEYS:
            assert f'"{key}"' not in out

    def test_audience_labels_never_legitimize_generated_fact_prose(self, engine):
        _, _, guild_out = self._facts_by_id(engine, audience=GUILD)
        _, _, dm_out = self._facts_by_id(engine, audience=DM)

        for output in (guild_out, dm_out):
            assert "Sania" not in output
            assert "SnookieBear" not in output
            assert "Hidden Person" not in output
            assert "user visited" not in output


class TestGateOnAggregateAnnotation:
    def test_recall_all_topics_expose_unknown_scope(self, engine):
        engine.config.search.speaker_annotations_enabled = True
        out = execute_vc_tool(
            engine, "vc_recall_all", {}, speaker_context=_ctx(),
        )
        payload = json.loads(out)
        assert payload["summaries"]
        for entry in payload["summaries"]:
            assert entry["speaker_scope"] == "unknown"

    def test_segment_and_tool_output_results_expose_unknown_scope(self):
        engine = MagicMock()
        engine.config.conversation_id = OWNER
        engine.config.search = SearchConfig(
            tool_guard_enabled=False,
            speaker_annotations_enabled=True,
        )
        engine.find_quote.side_effect = lambda **kw: {
            "found": True,
            "results": [
                {"excerpt": "turn text", "source_scope": "turn"},
                {"excerpt": "segment text", "source_scope": "segment"},
                {"excerpt": "tool text", "source_scope": "tool_output"},
                {
                    "excerpt": "already projected",
                    "source_scope": "segment",
                    "speaker_label": "Sania",
                },
            ],
        }
        engine._store.search_facts.return_value = []
        out = execute_vc_tool(
            engine, "vc_find_quote", {"query": "x", "mode": "lookup"},
            speaker_context=_ctx(),
        )
        results = json.loads(out)["results"]
        by_excerpt = {r["excerpt"]: r for r in results}
        # Turn-backed results are the search boundary's singular surface.
        assert "speaker_scope" not in by_excerpt["turn text"]
        assert by_excerpt["segment text"]["speaker_scope"] == "unknown"
        assert by_excerpt["tool text"]["speaker_scope"] == "unknown"
        # Never stamp over an entry another projector already annotated.
        assert "speaker_scope" not in by_excerpt["already projected"]

    def test_expand_topic_result_exposes_unknown_scope(self, tmp_path):
        from virtual_context.engine import VirtualContextEngine
        from tests.paging_sources import seed_paging_sources

        config = VirtualContextConfig(
            conversation_id=OWNER,
            storage=StorageConfig(
                backend="sqlite", sqlite_path=str(tmp_path / "paging.db"),
            ),
            tag_generator=TagGeneratorConfig(type="keyword"),
            paging=PagingConfig(enabled=True),
            search=SearchConfig(tool_guard_enabled=False),
        )
        paging_engine = VirtualContextEngine(config=config)
        context = _ctx()
        seed_paging_sources(paging_engine, "boston", context=context)

        # Reach steady state under one proved request, then pin annotation
        # gate parity without changing the independent source authority.
        execute_vc_tool(paging_engine, "vc_expand_topic", {"tag": "boston"}, speaker_context=context)
        legacy = execute_vc_tool(
            paging_engine, "vc_expand_topic", {"tag": "boston"}, speaker_context=context,
        )
        gated_off = execute_vc_tool(
            paging_engine, "vc_expand_topic", {"tag": "boston"},
            speaker_context=context,
        )
        assert gated_off == legacy
        assert "speaker_scope" not in legacy

        paging_engine.config.search.speaker_annotations_enabled = True
        out = execute_vc_tool(
            paging_engine, "vc_expand_topic", {"tag": "boston"},
            speaker_context=context,
        )
        payload = json.loads(out)
        assert "error" not in payload
        assert payload["speaker_scope"] == "unknown"

    def test_find_quote_withholds_generated_related_fact_prose(self, engine):
        engine.config.search.speaker_annotations_enabled = True
        out = execute_vc_tool(
            engine, "vc_find_quote", {"query": "Boston", "mode": "lookup"},
            speaker_context=_ctx(),
        )
        payload = json.loads(out)
        assert "related_facts" not in payload
        assert "related_facts_count" not in payload
        for actor in CANARY_ACTORS:
            assert actor not in out

    def test_find_quote_never_enriches_with_raw_fact_segment_full_text(self):
        engine = MagicMock()
        engine.config.conversation_id = OWNER
        engine.config.search = SearchConfig(
            tool_guard_enabled=False,
            speaker_annotations_enabled=True,
        )
        engine.find_quote.side_effect = lambda **kw: {
            "found": True,
            "results": [{"excerpt": "turn text", "source_scope": "turn"}],
        }
        fact = Fact(
            id="f1", subject="user", verb="visited", object="boston",
            what="user visited boston", segment_ref="seg-1",
            author_actor_id=ACTOR, author_attribution_version=2,
            author_source_role="requester",
        )
        engine._store.search_facts.return_value = [fact]
        engine._store.get_superseded_facts.return_value = []
        engine._store.get_recent_canonical_turns.return_value = []
        engine._store.get_segment.return_value = StoredSegment(
            ref="seg-1",
            conversation_id=OWNER,
            primary_tag="boston",
            full_text=(
                "BigTex: public Boston note. Kuw9239: private diagnosis. "
                "actor:discord:internal-poison"
            ),
            metadata=SegmentMetadata(turn_count=1, session_date="2026/01/10"),
        )
        out = execute_vc_tool(
            engine, "vc_find_quote", {"query": "Boston", "mode": "lookup"},
            speaker_context=_ctx(),
        )
        payload = json.loads(out)
        segment_entries = [
            r for r in payload["results"]
            if isinstance(r, dict) and r.get("source") == "fact_segment"
        ]
        assert segment_entries == []
        assert payload["results"] == [{
            "excerpt": "turn text",
            "source_scope": "turn",
        }]
        assert "Kuw9239" not in out
        assert "private diagnosis" not in out
        assert "actor:discord:internal-poison" not in out
        engine._store.get_segment.assert_not_called()
        assert "related_facts" not in payload
        assert "related_facts_count" not in payload
        engine._store.search_facts.assert_not_called()
        assert ACTOR not in out


class TestRestCommandSurface:
    def _state(self, find_quote_result):
        engine = MagicMock()
        engine.find_quote.return_value = find_quote_result
        engine.expand_topic.return_value = {"tokens": 10}
        return SimpleNamespace(engine=engine)

    def test_vcrecall_output_carries_topics_only_and_no_actor_ids(self):
        from virtual_context.proxy.handlers import _handle_vcrecall

        state = self._state({
            "found": True,
            "results": [{
                "excerpt": "I moved to Boston",
                "topic": "boston",
                "source_role": "requester",
                "speaker_label": "Sania",
                "speaker_handle": "",
                "speaker_verified": True,
            }],
        })
        text = _handle_vcrecall("boston", state, speaker_context=_ctx())
        assert "boston" in text
        for actor in CANARY_ACTORS:
            assert actor not in text
        # The recall text promotes topics; it never asserts attribution.
        assert "Sania" not in text

    def test_rest_recall_response_json_contains_no_actor_ids(self):
        from virtual_context.proxy.handlers import _handle_vc_command_rest

        state = self._state({
            "found": True,
            "results": [{"excerpt": "x", "topic": "boston"}],
        })
        result = SimpleNamespace(
            vc_command="recall",
            vc_command_arg="boston",
            conversation_id=OWNER,
            speaker_context=_ctx(),
        )
        response = _handle_vc_command_rest(
            result, state, MagicMock(), "tenant-A", OWNER,
        )
        body = bytes(response.body).decode("utf-8")
        for actor in CANARY_ACTORS:
            assert actor not in body


class TestStatelessMCPExposure:
    def _engine(self, *, enabled: bool):
        engine = MagicMock()
        engine.config.search = SearchConfig(speaker_annotations_enabled=enabled)
        return engine

    def test_stateless_mcp_has_structural_annotation_only(self):
        from virtual_context.mcp import server as mcp_server

        engine = self._engine(enabled=True)
        engine.find_quote.return_value = {
            "found": True,
            "results": [{
                "excerpt": "x",
                "source_role": "requester",
                "speaker_label": "Sania",
                "speaker_handle": "sania",
                "speaker_actor_known": True,
                "speaker_verified": True,
                "claimed_speaker_label": "claim",
                "speakers": ["Sania"],
                "attribution_basis": "role_local",
                "author_attribution_version": 2,
            }],
        }
        with patch.object(mcp_server, "_get_engine", return_value=engine):
            out = mcp_server.find_quote("x")
        entry = json.loads(out)["results"][0]
        assert entry["speaker_label"] == ""
        assert entry["speaker_handle"] == ""
        assert "speaker_actor_known" not in entry
        assert "speaker_verified" not in entry
        assert "claimed_speaker_label" not in entry
        assert "speakers" not in entry
        assert entry["source_role"] == "requester"
        assert entry["attribution_basis"] == "role_local"
        assert entry["author_attribution_version"] == 2
        # Stateless MCP has no request authority, so it forwards an explicit
        # ineligible sentinel instead of selecting unscoped history.
        context = engine.find_quote.call_args.kwargs["speaker_context"]
        assert isinstance(context, SpeakerRetrievalContext)
        assert context.eligible is False

    def test_mcp_gate_off_is_byte_identical(self):
        from virtual_context.mcp import server as mcp_server

        raw = {
            "found": True,
            "results": [{"excerpt": "x", "speaker_label": "Sania"}],
        }
        engine = self._engine(enabled=False)
        engine.find_quote.return_value = json.loads(json.dumps(raw))
        with patch.object(mcp_server, "_get_engine", return_value=engine):
            out = mcp_server.find_quote("x")
        assert out == json.dumps(raw)

    def test_mcp_recall_all_marks_aggregates_with_unknown_scope(self):
        from virtual_context.mcp import server as mcp_server

        engine = self._engine(enabled=True)
        engine.recall_all.return_value = {
            "found": True,
            "topics_loaded": 1,
            "total_tokens": 8,
            "summaries": [{"tag": "boston", "summary": "s", "tokens": 8,
                           "description": ""}],
        }
        with patch.object(mcp_server, "_get_engine", return_value=engine):
            out = mcp_server.recall_all()
        payload = json.loads(out)
        assert payload["summaries"][0]["speaker_scope"] == "unknown"
        assert "summary" not in payload["summaries"][0]
        assert "description" not in payload["summaries"][0]

    def test_mcp_recall_all_gate_off_still_strips_unproved_prose(self):
        from virtual_context.mcp import server as mcp_server

        raw = {
            "found": True,
            "topics_loaded": 1,
            "total_tokens": 8,
            "summaries": [{"tag": "boston", "summary": "s", "tokens": 8,
                           "description": ""}],
        }
        engine = self._engine(enabled=False)
        engine.recall_all.return_value = json.loads(json.dumps(raw))
        with patch.object(mcp_server, "_get_engine", return_value=engine):
            out = mcp_server.recall_all()
        payload = json.loads(out)
        assert payload["summaries"] == [{"tag": "boston", "tokens": 8}]
