"""Canary sweep: internal actor ids never reach an observable surface.

Seeds a real SQLite store with actor ids carrying a distinctive canary
substring, then drives every speaker-aware surface a request can touch —
roster construction and rendering, request-local tool schemas, annotated
and filtered quote retrieval, the degraded-warning path, object reprs, and
everything logged while doing all of it — and asserts the canary appears
nowhere. Handles and audience-scoped display labels are the only identity
presentation allowed out.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.search_engine import SearchEngine
from virtual_context.core.speaker_roster import build_speaker_roster
from virtual_context.core.tool_loop import (
    execute_vc_tool,
    vc_tool_definitions_for_runtime,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
)

CANARY = "XyZZy77"
CONV = "conv-secrecy"
ALPHA = f"actor:discord:{CANARY}-alpha"
BRAVO = f"actor:discord:{CANARY}-bravo"


class _NoSemantic:
    def semantic_canonical_turn_search(
        self, query, *, max_results=5, conversation_id=None, channel="",
        **kwargs,
    ):
        return []


@pytest.fixture()
def rig(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "secrecy.db"))
    store.upsert_conversation(tenant_id="t1", conversation_id=CONV)
    rows = [
        (1, "the launch window opens monday", "Alpha", ALPHA),
        (2, "the launch checklist is signed", "Bravo", BRAVO),
        (3, "who owns the launch retro", "Zed", ""),
    ]
    for n, text, sender, actor in rows:
        store.save_canonical_turn(
            CONV, n, text, "",
            canonical_turn_id=f"ct-{n}", turn_hash=f"h-{n}",
            sort_key=float(n), sender=sender, sender_actor_id=actor,
            audience_conversation_id=CONV, audience_attribution_version=1,
        )

    config = VirtualContextConfig(
        conversation_id=CONV,
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    config.assembler.speaker_roster_enabled = True
    config.search.speaker_annotations_enabled = True
    config.search.speaker_selection_enabled = True

    semantic = _NoSemantic()
    search_engine = SearchEngine(
        store=store, semantic=semantic,
        turn_tag_index=MagicMock(), config=config,
    )
    engine = SimpleNamespace(
        config=config, _store=store, _semantic=semantic,
        find_quote=search_engine.find_quote,
    )
    context = SpeakerRetrievalContext(
        tenant_id="t1",
        owner_conversation_id=CONV,
        audience_conversation_id=CONV,
        requester_actor_id=ALPHA,
        original_active_user_text="what did bravo say about the launch",
    )
    build = build_speaker_roster(
        store, speaker_context=context, token_counter=len, max_tokens=2000,
    )
    assert build.snapshot is not None
    return SimpleNamespace(
        store=store, engine=engine, build=build,
        context=build.speaker_context, snapshot=build.snapshot,
    )


def test_canary_actor_absent_from_all_model_and_observability_surfaces(
    rig, caplog,
):
    # Non-vacuous: the canary IS present in the internal request state the
    # surfaces below are built from.
    assert any(CANARY in e.actor_id for e in rig.snapshot.entries)
    assert CANARY in rig.context.requester_actor_id

    surfaces: dict[str, str] = {}

    with caplog.at_level(logging.DEBUG):
        # Roster presentation and the request-local tool schemas.
        surfaces["roster_text"] = rig.build.text
        surfaces["tool_schemas"] = json.dumps(
            vc_tool_definitions_for_runtime(roster_snapshot=rig.snapshot),
        )

        # Annotated retrieval, exact-attribution filtering, and the
        # degraded warning path.
        surfaces["annotated"] = execute_vc_tool(
            rig.engine, "vc_find_quote", {"query": "launch"},
            speaker_context=rig.context, roster_snapshot=rig.snapshot,
        )
        surfaces["filtered"] = execute_vc_tool(
            rig.engine, "vc_find_quote",
            {"query": "launch", "speaker": "bravo", "speaker_only": True},
            speaker_context=rig.context, roster_snapshot=rig.snapshot,
        )
        surfaces["degraded"] = execute_vc_tool(
            rig.engine, "vc_find_quote",
            {"query": "launch", "speaker": "nobody", "speaker_only": True},
            speaker_context=rig.context, roster_snapshot=rig.snapshot,
        )

        # Reprs of every request-owned object a formatter might touch.
        surfaces["context_repr"] = repr(rig.context)
        surfaces["snapshot_repr"] = repr(rig.snapshot)
        surfaces["entries_repr"] = repr(list(rig.snapshot.entries))
        surfaces["build_repr"] = repr(rig.build)

    surfaces["log_records"] = " | ".join(
        record.getMessage() for record in caplog.records
    )

    for name, text in surfaces.items():
        assert CANARY not in text, f"canary leaked through {name}"

    # The sweep exercised real surfaces, not empty strings.
    assert "bravo" in surfaces["roster_text"]
    assert '"enum"' in surfaces["tool_schemas"]
    filtered = json.loads(surfaces["filtered"])
    assert filtered["filter_applied"] is True
    assert filtered["pre_filter_matching_count"] == 1
    degraded = json.loads(surfaces["degraded"])
    assert degraded["filter_applied"] is False
    assert "warning" in degraded


def test_canary_absent_when_identity_checks_fail(rig, caplog):
    """Failure paths sanitize too: stale snapshots and dead stores."""
    import dataclasses

    stale = dataclasses.replace(rig.snapshot, snapshot_id="snap-stale")
    with caplog.at_level(logging.DEBUG):
        degraded = execute_vc_tool(
            rig.engine, "vc_find_quote",
            {"query": "launch", "speaker": "bravo", "speaker_only": True},
            speaker_context=rig.context, roster_snapshot=stale,
        )
        # A label-resolution store failure must fail open without echoing
        # identity into logs.
        broken = SimpleNamespace(
            config=rig.engine.config,
            _store=SimpleNamespace(),  # no store surface at all
            _semantic=rig.engine._semantic,
            find_quote=rig.engine.find_quote,
        )
        execute_vc_tool(
            broken, "vc_find_quote", {"query": "launch"},
            speaker_context=rig.context, roster_snapshot=rig.snapshot,
        )

    joined = degraded + " | ".join(
        record.getMessage() for record in caplog.records
    )
    assert CANARY not in joined
