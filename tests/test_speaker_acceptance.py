"""End-to-end speaker-conditioned retrieval over a real SQLite store.

Replays the production failure shape this feature exists for: a multi-actor
group conversation where the operator asks "what have you discussed with
<member>?". With every speaker gate on, one roster build plus one
``vc_find_quote(speaker=<handle>, speaker_only=true)`` call must return
exactly that member's authored rows — audience-scoped labels, snapshot
handles, verification flags — plus the disjoint exclusion counts, while a
retained DM audience under the same owner contributes nothing: no roster
entry, no result, no count, no label.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.search_engine import SearchEngine
from virtual_context.core.speaker_roster import build_speaker_roster
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
)

TENANT = "t1"
GUILD = "conv-guild-acceptance"      # native guild: audience == owner
DM = "conv-dm-acceptance"            # retained DM audience, same owner rows

OPERATOR = "actor:telegram:yursil"
SANIA = "actor:telegram:sania"
OMAR = "actor:telegram:omar"
PRIVATE = "actor:telegram:privatepal"


class _NoSemantic:
    """Semantic source double: hermetic, no embedding model."""

    def semantic_canonical_turn_search(
        self, query, *, max_results=5, conversation_id=None, channel="",
        **kwargs,
    ):
        return []


def _seed(store: SQLiteStore) -> None:
    import uuid

    store.upsert_conversation(tenant_id=TENANT, conversation_id=GUILD)
    store.upsert_conversation(tenant_id=TENANT, conversation_id=DM)

    def row(conv, n, user, assistant="", *, sender="", actor="",
            audience=GUILD, version=1):
        store.save_canonical_turn(
            conv, n, user, assistant,
            canonical_turn_id=f"ct-{conv}-{n}", turn_hash=f"h-{conv}-{n}",
            sort_key=float(n), sender=sender, sender_actor_id=actor,
            audience_conversation_id=audience,
            audience_attribution_version=version,
        )

    # Guild audience: three attributed humans, one assistant turn, one
    # legacy actorless row.
    row(GUILD, 1, "the trip to boston starts friday", sender="Sania",
        actor=SANIA)
    row(GUILD, 2, "I booked the trip hotel", sender="Yursil", actor=OPERATOR)
    row(GUILD, 3, "can I join the trip", sender="Omar", actor=OMAR)
    row(GUILD, 4, "", "The trip itinerary has three stops.")
    row(GUILD, 5, "packing list for the trip is done", sender="Sania",
        actor=SANIA)
    row(GUILD, 6, "who is driving on the trip", sender="Zed", actor="")

    # A DM conversation with a DM-only participant and a DM row from a
    # guild member carrying a private nickname, then merged under the guild
    # owner. The DM route becomes a retained alias whose rows share the
    # owner, and nothing from it may reach a guild-audience roster, result,
    # count, or label.
    row(DM, 1, "secret trip plans with yursil", sender="PrivatePal",
        actor=PRIVATE, audience=DM)
    row(DM, 2, "dm trip note", sender="Sani-Bear", actor=SANIA, audience=DM)

    merge_id = str(uuid.uuid4())
    reserved = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id=TENANT,
        source_conversation_id=DM, target_conversation_id=GUILD,
        source_label_at_merge="dm",
    )
    assert reserved.status == "reserved"
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id=TENANT,
        source_conversation_id=DM, target_conversation_id=GUILD,
        expected_target_lifecycle_epoch=1, source_label_at_merge="dm",
    )


@pytest.fixture()
def rig(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "acceptance.db"))
    _seed(store)

    config = VirtualContextConfig(
        conversation_id=GUILD,
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    # Every speaker gate on.
    config.assembler.speaker_roster_enabled = True
    config.search.speaker_annotations_enabled = True
    config.search.speaker_selection_enabled = True

    semantic = _NoSemantic()
    search_engine = SearchEngine(
        store=store, semantic=semantic,
        turn_tag_index=MagicMock(), config=config,
    )
    engine = SimpleNamespace(
        config=config,
        _store=store,
        _semantic=semantic,
        find_quote=search_engine.find_quote,
    )

    context = SpeakerRetrievalContext(
        tenant_id=TENANT,
        owner_conversation_id=GUILD,
        audience_conversation_id=GUILD,
        requester_actor_id=OPERATOR,
        original_active_user_text="what have you discussed with Sania?",
    )
    build = build_speaker_roster(
        store, speaker_context=context, token_counter=len, max_tokens=2000,
    )
    assert build.snapshot is not None
    return SimpleNamespace(
        store=store, engine=engine, build=build,
        context=build.speaker_context, snapshot=build.snapshot,
    )


def _find_quote(rig_, tool_input) -> dict:
    out = execute_vc_tool(
        rig_.engine, "vc_find_quote", dict(tool_input),
        speaker_context=rig_.context,
        roster_snapshot=rig_.snapshot,
    )
    return json.loads(out)


def test_roster_presents_guild_members_only(rig):
    handles = [e.handle for e in rig.snapshot.entries]
    names = [e.name for e in rig.snapshot.entries]
    # Most recent admissible guild row first: Sania(5), Omar(3), Yursil(2).
    assert handles == ["sania", "omar", "yursil"]
    assert names == ["Sania", "Omar", "Yursil"]
    assert rig.snapshot.truncated is False
    # The DM-only participant and the DM nickname never surface.
    assert "PrivatePal" not in rig.build.text
    assert "Sani-Bear" not in rig.build.text
    assert "privatepal" not in rig.build.text
    # The snapshot id is bound into the request context.
    assert rig.context.roster_snapshot_id == rig.snapshot.snapshot_id


def test_speaker_only_returns_the_members_rows_with_labels_and_counts(rig):
    got = _find_quote(
        rig, {"query": "trip", "speaker": "sania", "speaker_only": True},
    )

    assert got["found"] is True
    assert got["conditioning_source"] == "explicit_roster"
    assert got["filter_applied"] is True
    assert "warning" not in got

    # Exactly the member's authored guild rows, labeled and verified.
    assert len(got["results"]) == 2
    excerpts = " || ".join(entry["excerpt"] for entry in got["results"])
    assert "boston" in excerpts
    assert "packing list" in excerpts
    for entry in got["results"]:
        assert entry["speaker_label"] == "Sania"
        assert entry["speaker_handle"] == "sania"
        assert entry["speaker_actor_known"] is True
        assert entry["speaker_verified"] is True
        assert entry["source_role"] == "requester"

    # Disjoint audience-scoped counts: two Sania matches; the operator,
    # Omar, and the assistant turn are other speakers; the actorless row is
    # unknown. The DM rows are outside the authorized audience and
    # contribute to no class.
    assert got["pre_filter_matching_count"] == 2
    assert got["excluded_other_speakers"] == 3
    assert got["excluded_unknown_speaker"] == 1

    raw = json.dumps(got)
    assert "secret trip plans" not in raw
    assert "PrivatePal" not in raw
    assert "Sani-Bear" not in raw
    assert "dm trip note" not in raw
    # Actor ids never serialize.
    assert "actor:telegram" not in raw


def test_speaker_only_answers_what_do_you_know_about_this_person(rig):
    """The question a speaker selection exists for.

    "What do you know about X" carries no topic to match on. Every retrieval
    source ranks by the query first and applies the actor predicate to what
    survives its own relevance threshold, so a topicless question surfaced
    nothing about anything, the predicate filtered an empty list, and the
    reader was told there was nothing on record — while the speaker's own
    statements sat in storage, unreached. The speaker IS the query here, so
    their statements are what must come back.
    """
    got = _find_quote(
        rig,
        {
            "query": "what do you know about this person",
            "speaker": "sania",
            "speaker_only": True,
        },
    )

    assert got["found"] is True, (
        "a selected speaker with statements on record must never answer "
        f"'nothing': {got}"
    )
    assert got["filter_applied"] is True
    assert got["conditioning_source"] == "explicit_roster"

    # Her statements, and only hers — same labels and provenance as any
    # other speaker-conditioned path.
    excerpts = " || ".join(entry["excerpt"] for entry in got["results"])
    assert "boston" in excerpts
    assert "packing list" in excerpts
    for entry in got["results"]:
        assert entry["speaker_label"] == "Sania"
        assert entry["speaker_verified"] is True

    # The audience boundary still holds: no DM content, no actor ids.
    raw = json.dumps(got)
    assert "secret trip plans" not in raw
    assert "dm trip note" not in raw
    assert "actor:telegram" not in raw


def test_unfiltered_probe_annotates_every_guild_speaker(rig):
    got = _find_quote(rig, {"query": "trip"})
    assert got["found"] is True
    by_label = {}
    for entry in got["results"]:
        if entry.get("speaker_verified") and entry.get("speaker_label"):
            by_label.setdefault(entry["speaker_label"], entry)
    assert by_label["Sania"]["speaker_handle"] == "sania"
    assert by_label["Yursil"]["speaker_handle"] == "yursil"
    assert by_label["Omar"]["speaker_handle"] == "omar"
    assert "actor:telegram" not in json.dumps(got)


def test_dm_handle_namespace_is_separate(rig):
    # A DM-audience request over the same owner rows builds its own roster:
    # the DM participant and the DM label appear there and only there.
    dm_context = SpeakerRetrievalContext(
        tenant_id=TENANT,
        owner_conversation_id=GUILD,
        audience_conversation_id=DM,
        requester_actor_id=OPERATOR,
    )
    dm_build = build_speaker_roster(
        rig.store, speaker_context=dm_context, token_counter=len,
        max_tokens=2000,
    )
    assert dm_build.snapshot is not None
    dm_names = {e.name for e in dm_build.snapshot.entries}
    assert dm_names == {"PrivatePal", "Sani-Bear"}
    # Sania's guild handle namespace is untouched: the DM assignment is a
    # separate durable row, and the guild snapshot still resolves "sania".
    guild_again = build_speaker_roster(
        rig.store,
        speaker_context=SpeakerRetrievalContext(
            tenant_id=TENANT,
            owner_conversation_id=GUILD,
            audience_conversation_id=GUILD,
            requester_actor_id=OPERATOR,
        ),
        token_counter=len, max_tokens=2000,
    )
    assert [e.handle for e in guild_again.snapshot.entries] \
        == ["sania", "omar", "yursil"]


def test_speaker_only_with_stale_snapshot_degrades_with_warning(rig):
    import dataclasses

    stale = dataclasses.replace(rig.snapshot, snapshot_id="snap-stale")
    out = execute_vc_tool(
        rig.engine, "vc_find_quote",
        {"query": "trip", "speaker": "sania", "speaker_only": True},
        speaker_context=rig.context,
        roster_snapshot=stale,
    )
    got = json.loads(out)
    assert got["found"] is True
    assert got["filter_applied"] is False
    assert "no attribution filter" in got["warning"].lower()
    assert "pre_filter_matching_count" not in got
    assert "excluded_other_speakers" not in got
    assert "excluded_unknown_speaker" not in got
