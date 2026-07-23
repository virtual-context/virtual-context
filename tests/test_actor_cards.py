"""Person-card storage, audience policy, and lifecycle fences.

The worst failure this feature can have is cross-user contamination: one
member's prepare containing another member's card material. Cross-context
leakage (a private DM shaping a public answer) is the same severity. These
tests pin both structurally, at the store, so no prompt wording is load-bearing.
"""
import json
import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CARD_KIND_ACTIVE_GOAL,
    CARD_KIND_COMMUNICATION_PREF,
    CARD_KIND_INTERACTION_STYLE,
    CARD_SCOPE_CROSS_CONTEXT,
    CARD_SCOPE_SAME_CONVERSATION,
    CARD_SENSITIVITY_HIGH,
    ActorCardEntry,
    ActorCardEntrySource,
    SegmentMetadata,
    StoredSegment,
)

OPTICS = "actor:discord:optics"
BIGTEX = "actor:discord:bigtex"


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "cards.db"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _conversation(store, cid, tenant="t1", phase="active", epoch=1):
    conn = store._get_conn()
    now = _now()
    conn.execute(
        """INSERT INTO conversations
               (conversation_id, tenant_id, lifecycle_epoch, phase,
                created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (cid, tenant, epoch, phase, now, now),
    )
    conn.commit()


def _turn(store, ctid, cid, actor, audience, channel="", content="hello"):
    conn = store._get_conn()
    sort_key = conn.execute(
        """SELECT COALESCE(MAX(sort_key), 0) + 1
             FROM canonical_turns
            WHERE conversation_id = ?""",
        (cid,),
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, turn_hash, sort_key,
                user_content, assistant_content, sender_actor_id,
                audience_conversation_id, audience_attribution_version,
                origin_channel_id)
           VALUES (?, ?, ?, ?, ?, '', ?, ?, 1, ?)""",
        (ctid, cid, ctid, sort_key, content, actor, audience, channel),
    )
    conn.commit()


def _segment(store, ref, cid, ctids, complete=True):
    conn = store._get_conn()
    now = _now()
    meta = {"canonical_turn_ids": list(ctids), "source_mapping_complete": complete}
    conn.execute(
        """INSERT INTO segments
               (ref, conversation_id, primary_tag, summary, full_text,
                messages_json, metadata_json, created_at,
                start_timestamp, end_timestamp)
           VALUES (?, ?, 'tag', 's', 'f', '[]', ?, ?, ?, ?)""",
        (ref, cid, json.dumps(meta), now, now, now),
    )
    conn.commit()


def _fact(store, fid, cid, ref, actor, version=1):
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO facts
               (id, subject, verb, object, status, what, who, when_date,
                "where", why, fact_type, tags_json, segment_ref,
                conversation_id, turn_numbers_json, mentioned_at, session_date,
                superseded_by, author_actor_id, author_attribution_version,
                author_source_role, author_source_message_id)
           VALUES (?, 's', 'v', 'o', 'active', 'w', '', '', '', '',
                   'personal', '[]', ?, ?, '[]', ?, '', NULL, ?, ?,
                   'requester', '')""",
        (fid, ref, cid, _now(), actor, version),
    )
    conn.commit()


def _entry(entry_id, kind, body, *, scope=CARD_SCOPE_SAME_CONVERSATION,
           sensitivity="normal", confidence=0.9):
    return ActorCardEntry(
        id=entry_id, kind=kind, body=body, confidence=confidence,
        sensitivity=sensitivity, audience_scope=scope,
    )


def _source(entry_id, owner, audience, fact_id, channel="", tenant="t1"):
    return ActorCardEntrySource(
        entry_id=entry_id, tenant_id=tenant, owner_conversation_id=owner,
        audience_conversation_id=audience, audience_channel_id=channel,
        fact_id=fact_id,
    )


def _turn_source(
    entry_id,
    owner,
    audience,
    canonical_turn_id,
    channel="",
    tenant="t1",
):
    return ActorCardEntrySource(
        entry_id=entry_id,
        tenant_id=tenant,
        owner_conversation_id=owner,
        audience_conversation_id=audience,
        audience_channel_id=channel,
        canonical_turn_id=canonical_turn_id,
    )


def _dm_and_guild(store):
    """One actor speaking in a private DM and in a public guild channel."""
    _conversation(store, "dm")
    _conversation(store, "guild")
    _turn(store, "ct-dm", "dm", OPTICS, "dm", "chan-dm")
    _turn(store, "ct-guild", "guild", OPTICS, "guild", "chan-guild")
    _segment(store, "seg-dm", "dm", ["ct-dm"])
    _segment(store, "seg-guild", "guild", ["ct-guild"])
    _fact(store, "f-dm", "dm", "seg-dm", OPTICS)
    _fact(store, "f-guild", "guild", "seg-guild", OPTICS)
    store.upsert_actor_profile_from_turn("dm", OPTICS, "Optics", seen_at=_now())


def _build_dm_goal_and_cross_pref(store):
    goal = _entry("e-goal", CARD_KIND_ACTIVE_GOAL, "private DM goal")
    pref = _entry("e-pref", CARD_KIND_COMMUNICATION_PREF, "prefers terse answers",
                  scope=CARD_SCOPE_CROSS_CONTEXT)
    return store.replace_actor_card(
        "t1", OPTICS,
        [
            (goal, [_source("e-goal", "dm", "dm", "f-dm", "chan-dm")]),
            (pref, [_source("e-pref", "guild", "guild", "f-guild", "chan-guild")]),
        ],
        input_hash="h1",
        expected_source_epochs={"dm": 1, "guild": 1},
    )


def _bodies(card):
    return sorted(e.body for e in card.entries) if card else None


def _curation(entries, *, substantive=None, coverage_reason=None):
    if substantive is None:
        substantive = bool(entries)
    if coverage_reason is None:
        coverage_reason = "substantive" if substantive else "no_durable_context"
    normalized = [
        {
            **entry,
            "fact_ids": list(entry.get("fact_ids", [])),
            "turn_ids": list(entry.get("turn_ids", [])),
        }
        for entry in entries
    ]
    return {
        "substantive": substantive,
        "coverage_reason": coverage_reason,
        "entries": normalized,
    }


def _admission(decisions, *, substantive=None, coverage_reason=None):
    if substantive is None:
        substantive = any(
            decision.get("admit") is True for decision in decisions
        )
    if coverage_reason is None:
        coverage_reason = "substantive" if substantive else "no_durable_context"
    return {
        "substantive": substantive,
        "coverage_reason": coverage_reason,
        "decisions": decisions,
    }


def _curation_for_visible_fact(kwargs, fact_id, entries):
    visible = {
        item["id"]
        for item in json.loads(kwargs["user"])["facts"]
    }
    return _curation(entries if fact_id in visible else [])


class _AdmitAll:
    """Strict admission stub for tests focused on a later storage boundary."""

    def complete(self, **kwargs):
        prompt = json.loads(kwargs["user"])
        decisions = [{
            "candidate_id": candidate["candidate_id"],
            "admit": True,
            "sensitivity": candidate["proposed_sensitivity"],
            "reason": "durable",
        } for candidate in prompt["candidates"]]
        return json.dumps(_admission(decisions)), {}


def _card_pipeline(
    store,
    curator,
    *,
    admission=None,
    enabled=True,
    admission_model="semantic-model",
):
    from types import SimpleNamespace

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=enabled,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
            actor_card_admission_model=admission_model,
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=curator,
        _parse_response=lambda text: json.loads(text),
    )
    if admission is not None:
        pipeline._actor_card_admission_provider_override = admission
    return pipeline


# ---------------------------------------------------------------------------
# Fact enumeration
# ---------------------------------------------------------------------------

def test_list_actor_facts_derives_audience_from_canonical_rows(store):
    _dm_and_guild(store)
    sources = store.list_actor_facts("t1", OPTICS, limit=10)
    got = {s.fact.id: (s.owner_conversation_id, s.audience_conversation_id,
                       s.audience_channel_id) for s in sources}
    assert got == {
        "f-dm": ("dm", "dm", "chan-dm"),
        "f-guild": ("guild", "guild", "chan-guild"),
    }


def test_list_actor_turn_sources_is_exact_tenant_and_audience_scoped(store):
    _dm_and_guild(store)
    sources = store.list_actor_turn_sources("t1", OPTICS, limit=10)
    got = {
        source.turn.canonical_turn_id: (
            source.owner_conversation_id,
            source.audience_conversation_id,
            source.audience_channel_id,
            source.turn.user_content,
        )
        for source in sources
    }
    assert got == {
        "ct-dm": ("dm", "dm", "chan-dm", "hello"),
        "ct-guild": ("guild", "guild", "chan-guild", "hello"),
    }
    assert store.list_actor_turn_sources(
        "other-tenant", OPTICS, limit=10,
    ) == []
    assert store.list_actor_turn_sources("t1", BIGTEX, limit=10) == []


def test_turn_sourced_card_is_atomic_readable_and_delete_invalidated(store):
    _conversation(store, "guild")
    _turn(
        store,
        "ct-guild",
        "guild",
        OPTICS,
        "guild",
        "chan-guild",
        content="I regularly ask about endurance training.",
    )
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )
    entry = _entry(
        "e-history",
        "relevant_history",
        "Has discussed endurance training with Vast.",
    )

    assert store.replace_actor_card(
        "t1",
        OPTICS,
        [(
            entry,
            [_turn_source(
                "e-history",
                "guild",
                "guild",
                "ct-guild",
                "chan-guild",
            )],
        )],
        input_hash="turn-hash",
        expected_source_epochs={"guild": 1},
    ) == 1
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-guild",
    )) == ["Has discussed endurance training with Vast."]
    row = store._get_conn().execute(
        """SELECT canonical_turn_id
             FROM actor_card_turn_sources
            WHERE entry_id = 'e-history'""",
    ).fetchone()
    assert row[0] == "ct-guild"

    store._get_conn().execute(
        """DELETE FROM canonical_turns
            WHERE canonical_turn_id = 'ct-guild'""",
    )
    store._get_conn().commit()

    profile = store.get_actor_profile("t1", OPTICS)
    assert profile is not None and profile.card_dirty is True
    assert store._get_conn().execute(
        """SELECT 1 FROM actor_card_entries
            WHERE id = 'e-history'""",
    ).fetchone() is None
    assert store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-guild",
    ) is None


def test_segment_source_mapping_survives_store_round_trip(store):
    _conversation(store, "c1")
    now = datetime.now(timezone.utc)
    stored = StoredSegment(
        ref="seg-round-trip", conversation_id="c1", primary_tag="tag",
        tags=["tag"], summary="summary", full_text="full",
        messages=[{
            "role": "user", "content": "hello",
            "metadata": {"_vc_source_canonical_turn_ids": ["ct1"]},
        }],
        metadata=SegmentMetadata(
            canonical_turn_ids=["ct1"], source_mapping_complete=True,
        ),
        created_at=now, start_timestamp=now, end_timestamp=now,
    )

    store.store_segment(stored)
    loaded = store.get_segment("seg-round-trip", conversation_id="c1")

    assert loaded is not None
    assert loaded.metadata.canonical_turn_ids == ["ct1"]
    assert loaded.metadata.source_mapping_complete is True
    assert loaded.messages[0]["metadata"] == {
        "_vc_source_canonical_turn_ids": ["ct1"],
    }


def test_compaction_card_builder_curates_and_skips_unchanged_input(store):
    from types import SimpleNamespace

    _dm_and_guild(store)

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            entries = [{
                "kind": CARD_KIND_ACTIVE_GOAL,
                "body": "finish the migration",
                "confidence": 0.8,
                "sensitivity": "normal",
                "fact_ids": ["f-dm"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-dm", entries),
            ), {}

    llm = LLM()
    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
            actor_card_admission_model="semantic-model",
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=llm,
        _parse_response=lambda text: json.loads(text),
    )
    pipeline._actor_card_admission_provider_override = _AdmitAll()

    assert pipeline._rebuild_actor_card(OPTICS) == 1
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert _bodies(card) == ["finish the migration"]
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "written"
    assert status["source_count"] == 4
    assert status["raw_entry_count"] == 1
    assert status["accepted_entry_count"] == 1
    assert status["written_count"] == 1
    assert status["rejected_counts"] == {}

    assert pipeline._rebuild_actor_card(OPTICS) == 0
    assert llm.calls == 2


def test_turn_only_card_lifecycle_adds_refines_revokes_and_ignores_probe(store):
    """Substantive canonical speech must build and maintain a card without facts."""
    _conversation(store, "guild")
    _turn(
        store,
        "ct-goal",
        "guild",
        OPTICS,
        "guild",
        "chan-a",
        content=(
            "I am leading the Atlas migration to Kubernetes this quarter."
        ),
    )
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    class Curator:
        calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            turns = {
                item["id"]: item["content"]
                for item in json.loads(kwargs["user"])["turns"]
            }
            entries = []
            if "ct-finish" not in turns:
                entries.append({
                    "kind": CARD_KIND_ACTIVE_GOAL,
                    "body": "Is leading the Atlas migration to Kubernetes.",
                    "confidence": 0.95,
                    "sensitivity": "normal",
                    "turn_ids": ["ct-goal"],
                })
            if "ct-pref" in turns and "ct-finish" not in turns:
                entries.append({
                    "kind": CARD_KIND_COMMUNICATION_PREF,
                    "body": "Prefers bullet summaries for the Atlas project.",
                    "confidence": 0.9,
                    "sensitivity": "normal",
                    "turn_ids": ["ct-pref"],
                })
            if "ct-finish" in turns:
                entries.append({
                    "kind": CARD_KIND_COMMUNICATION_PREF,
                    "body": "Prefers brief prose instead of bullet summaries.",
                    "confidence": 0.95,
                    "sensitivity": "normal",
                    "turn_ids": ["ct-finish"],
                })
            return json.dumps(_curation(entries)), {}

    curator = Curator()
    pipeline = _card_pipeline(
        store,
        curator,
        admission=_AdmitAll(),
    )

    assert pipeline._rebuild_actor_card(OPTICS) == 1
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-a",
    )) == ["Is leading the Atlas migration to Kubernetes."]

    _turn(
        store,
        "ct-pref",
        "guild",
        OPTICS,
        "guild",
        "chan-a",
        content=(
            "Going forward, I prefer bullet summaries for the Atlas project."
        ),
    )
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    assert pipeline._rebuild_actor_card(OPTICS) == 2
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-a",
    )) == [
        "Is leading the Atlas migration to Kubernetes.",
        "Prefers bullet summaries for the Atlas project.",
    ]

    _turn(
        store,
        "ct-finish",
        "guild",
        OPTICS,
        "guild",
        "chan-a",
        content=(
            "The Atlas migration is complete. Going forward, use brief prose, "
            "not bullet summaries."
        ),
    )
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    assert pipeline._rebuild_actor_card(OPTICS) == 1
    expected = ["Prefers brief prose instead of bullet summaries."]
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-a",
    )) == expected

    active_rows = store._get_conn().execute(
        """SELECT e.kind, e.body, s.canonical_turn_id
             FROM actor_card_entries e
             JOIN actor_card_turn_sources s
               ON s.entry_id = e.id AND s.tenant_id = e.tenant_id
            WHERE e.tenant_id = ? AND e.actor_id = ?
              AND e.superseded_by IS NULL""",
        ("t1", OPTICS),
    ).fetchall()
    assert [tuple(row) for row in active_rows] == [(
        CARD_KIND_COMMUNICATION_PREF,
        expected[0],
        "ct-finish",
    )]

    _turn(
        store,
        "ct-probe",
        "guild",
        OPTICS,
        "guild",
        "chan-a",
        content="For this test only, begin one reply with Probe:",
    )
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    assert pipeline._rebuild_actor_card(OPTICS) == 1
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-a",
    )) == expected
    assert curator.calls == 4


def test_new_turn_during_model_call_cannot_be_lost_by_card_commit(store):
    _conversation(store, "guild")
    _turn(
        store,
        "ct-before-build",
        "guild",
        OPTICS,
        "guild",
        "chan-a",
        content="I am planning the Atlas migration.",
    )
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    class RacingCurator:
        def complete(self, **_kwargs):
            _turn(
                store,
                "ct-during-build",
                "guild",
                OPTICS,
                "guild",
                "chan-a",
                content="The Atlas migration has now been cancelled.",
            )
            return json.dumps(_curation([{
                "kind": CARD_KIND_ACTIVE_GOAL,
                "body": "Is planning the Atlas migration.",
                "confidence": 0.9,
                "sensitivity": "normal",
                "turn_ids": ["ct-before-build"],
            }])), {}

    pipeline = _card_pipeline(
        store,
        RacingCurator(),
        admission=_AdmitAll(),
    )
    with pytest.raises(
        RuntimeError,
        match="replacement did not commit cleanly",
    ):
        pipeline._rebuild_actor_card(OPTICS)

    profile = store.get_actor_profile("t1", OPTICS)
    assert profile is not None and profile.card_dirty is True
    assert profile.card_input_hash == ""
    assert store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-a",
    ) is None
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "stale_or_rejected_write"


def test_compaction_card_builder_rejects_any_unknown_fact_citation(store):
    from types import SimpleNamespace

    _dm_and_guild(store)

    class LLM:
        def complete(self, **kwargs):
            entries = [{
                "kind": CARD_KIND_ACTIVE_GOAL,
                "body": "unsupported synthesis",
                "confidence": 0.8,
                "sensitivity": "normal",
                "fact_ids": ["f-dm", "invented-fact"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-dm", entries),
            ), {}

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=LLM(),
        _parse_response=lambda text: json.loads(text),
    )

    with pytest.raises(
        RuntimeError, match="rejected every model entry",
    ):
        pipeline._rebuild_actor_card(OPTICS)
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "rejected_all"
    assert status["rejected_counts"] == {
        "unknown_or_cross_audience_fact_id": 1,
    }
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_actor_card_failures_back_off_and_terminally_suppress_same_input(store):
    _dm_and_guild(store)

    class Malformed:
        calls = 0

        def complete(self, **_kwargs):
            self.calls += 1
            return '{"entries":[]}', {}

    curator = Malformed()
    pipeline = _card_pipeline(
        store,
        curator,
        admission=_AdmitAll(),
    )

    with pytest.raises(RuntimeError, match="no valid entries array"):
        pipeline._rebuild_actor_card(OPTICS)
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status["failure_count"] == 1
    assert status["next_retry_at"]

    # Automatic work is bounded during backoff.
    assert pipeline._rebuild_actor_card(OPTICS) == 0
    assert curator.calls == 1

    # An operator-forced rebuild bypasses backoff but still increments the
    # bounded failure counter for this exact immutable input.
    for expected in (2, 3):
        with pytest.raises(RuntimeError, match="no valid entries array"):
            pipeline._rebuild_actor_card(OPTICS, force=True)
        assert (
            store.get_actor_card_rebuild_status(
                "t1", OPTICS,
            )["failure_count"]
            == expected
        )
    assert pipeline._rebuild_actor_card(OPTICS) == 0
    assert curator.calls == 3

    # Changed evidence is a new input and therefore gets one fresh attempt.
    _turn(
        store,
        "ct-new-input",
        "guild",
        OPTICS,
        "guild",
        "chan-guild",
        content="A genuinely new message changes the input hash.",
    )
    with pytest.raises(RuntimeError, match="no valid entries array"):
        pipeline._rebuild_actor_card(OPTICS)
    assert curator.calls == 4
    assert (
        store.get_actor_card_rebuild_status(
            "t1", OPTICS,
        )["failure_count"]
        == 1
    )


def test_transient_card_failure_enters_bounded_due_retry_queue(store):
    _dm_and_guild(store)
    store.mark_actor_card_dirty(
        "t1",
        OPTICS,
        build_input_hash="building:input-hash",
    )
    store.record_actor_card_rebuild_status(
        "t1",
        OPTICS,
        attempted_at=_now(),
        input_hash="input-hash",
        source_count=2,
        raw_entry_count=0,
        accepted_entry_count=0,
        rejected_counts={},
        outcome="model_error",
        response_hash="response-hash",
        written_count=0,
    )

    assert store.list_due_actor_card_rebuilds(
        "t1",
        due_at="2000-01-01T00:00:00+00:00",
    ) == []
    assert store.list_due_actor_card_rebuilds(
        "t1",
        due_at="9999-01-01T00:00:00+00:00",
    ) == [OPTICS]
    assert store.list_due_actor_card_rebuilds(
        "other-tenant",
        due_at="9999-01-01T00:00:00+00:00",
    ) == []

    store._get_conn().execute(
        """UPDATE actor_card_rebuild_status
              SET next_retry_at = '2000-01-01T00:00:00+00:00'
            WHERE tenant_id = 't1' AND actor_id = ?""",
        (OPTICS,),
    )
    store._get_conn().commit()
    pipeline = _card_pipeline(
        store,
        curator=object(),
        admission=_AdmitAll(),
    )
    assert pipeline._due_actor_card_rebuilds() == [OPTICS]

    store._get_conn().execute(
        """UPDATE actor_card_rebuild_status
              SET failure_count = 3
            WHERE tenant_id = 't1' AND actor_id = ?""",
        (OPTICS,),
    )
    store._get_conn().commit()
    assert pipeline._due_actor_card_rebuilds() == []


def test_compaction_card_builder_rejects_boolean_confidence(store):
    from types import SimpleNamespace

    _dm_and_guild(store)

    class LLM:
        def complete(self, **kwargs):
            entries = [{
                "kind": CARD_KIND_ACTIVE_GOAL,
                "body": "not numeric confidence",
                "confidence": True,
                "sensitivity": "normal",
                "fact_ids": ["f-dm"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-dm", entries),
            ), {}

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=LLM(), _parse_response=lambda text: json.loads(text),
    )

    with pytest.raises(
        RuntimeError, match="rejected every model entry",
    ):
        pipeline._rebuild_actor_card(OPTICS)
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "rejected_all"
    assert status["rejected_counts"] == {"invalid_confidence": 1}
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


@pytest.mark.parametrize("sensitivity", ["none", 0, 1])
def test_compaction_card_builder_reports_invalid_sensitivity(store, sensitivity):
    """The malformed values observed in production must not become clean-empty."""
    from types import SimpleNamespace

    _dm_and_guild(store)

    class LLM:
        def complete(self, **kwargs):
            entries = [{
                "kind": CARD_KIND_COMMUNICATION_PREF,
                "body": "prefers concise answers",
                "confidence": 0.8,
                "sensitivity": sensitivity,
                "fact_ids": ["f-guild"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-guild", entries),
            ), {}

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=LLM(), _parse_response=lambda text: json.loads(text),
    )

    with pytest.raises(
        RuntimeError, match="rejected every model entry",
    ):
        pipeline._rebuild_actor_card(OPTICS)

    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "rejected_all"
    assert status["rejected_counts"] == {"invalid_sensitivity": 1}


def test_compaction_card_builder_accepts_explicit_clean_empty_and_records_contract(
    store,
):
    """No durable pattern is valid; malformed output is not."""
    from types import SimpleNamespace

    _dm_and_guild(store)

    class LLM:
        kwargs = None

        def complete(self, **kwargs):
            self.kwargs = kwargs
            return json.dumps(_curation([])), {}

    llm = LLM()
    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=llm, _parse_response=lambda text: json.loads(text),
    )
    pipeline._actor_card_admission_provider_override = _AdmitAll()

    assert pipeline._rebuild_actor_card(OPTICS) == 0
    assert "exactly the string \"normal\" or \"high\"" in llm.kwargs["system"]
    assert "temporary, test-only" in llm.kwargs["system"]
    assert "Every body must be self-contained and unambiguous" in (
        llm.kwargs["system"]
    )
    assert "not a serialization of subject/verb/object fields" in (
        llm.kwargs["system"]
    )
    assert "Preserve every material qualifier from the source" in (
        llm.kwargs["system"]
    )
    assert "Do not turn a qualified statement into a broader" in (
        llm.kwargs["system"]
    )
    prompt = json.loads(llm.kwargs["user"])
    assert prompt["facts"]
    assert all(
        isinstance(item["mentioned_at"], str)
        for item in prompt["facts"]
    )
    assert store.get_actor_profile("t1", OPTICS).card_dirty is False
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "clean_empty"
    assert status["source_count"] == 4
    assert status["raw_entry_count"] == 0
    assert status["accepted_entry_count"] == 0


def test_semantic_admission_rejects_candidate_without_rewriting_card(store):
    """The focused model gate can reject a schema-valid but non-durable entry."""
    from types import SimpleNamespace

    _dm_and_guild(store)

    class Curator:
        def complete(self, **kwargs):
            entries = [{
                "kind": CARD_KIND_COMMUNICATION_PREF,
                "body": "begin every reply with a temporary probe prefix",
                "confidence": 1.0,
                "sensitivity": "normal",
                "fact_ids": ["f-guild"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-guild", entries),
            ), {}

    class Admission:
        prompt = None
        system = None

        def complete(self, **kwargs):
            self.prompt = json.loads(kwargs["user"])
            self.system = kwargs["system"]
            if not self.prompt["candidates"]:
                return json.dumps(_admission([])), {}
            candidate_id = self.prompt["candidates"][0]["candidate_id"]
            return json.dumps(_admission([{
                "candidate_id": candidate_id,
                "admit": False,
                "sensitivity": "normal",
                "reason": "test_probe",
            }])), {}

    admission = Admission()
    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
            actor_card_admission_model="semantic-model",
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=Curator(), _parse_response=lambda text: json.loads(text),
    )
    pipeline._actor_card_admission_provider_override = admission

    with pytest.raises(
        RuntimeError, match="semantic admission failed",
    ):
        pipeline._rebuild_actor_card(OPTICS)
    assert admission.prompt["candidates"][0]["body"] == (
        "begin every reply with a temporary probe prefix"
    )
    assert (
        "candidate body itself must be self-contained and unambiguous"
        in admission.system
    )
    assert "reject with insufficient_evidence" in admission.system
    assert "Compact fact fields and tags" in admission.system
    assert "preserving every material qualifier" in admission.system
    assert "drops a qualifier, broadens the statement" in admission.system
    assert "does not entail the unqualified body" in admission.system
    assert "Reject with not_person_card" in admission.system
    assert any(
        message["content"] == "hello"
        for segment in admission.prompt["evidence_segments"]
        for message in segment["messages"]
    )
    assert {
        segment["segment_ref"]
        for segment in admission.prompt["evidence_segments"]
    } == {"seg-guild"}
    assert {
        fact["id"] for fact in admission.prompt["facts"]
    } == {"f-guild"}
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "coverage_disagreement"
    assert status["accepted_entry_count"] == 1
    assert status["rejected_counts"] == {}
    assert status["failure_count"] == 3


def test_semantic_admission_can_raise_sensitivity_but_not_rewrite_body(store):
    from types import SimpleNamespace

    _dm_and_guild(store)
    body = "Actor has a private medical treatment history."

    class Curator:
        def complete(self, **kwargs):
            entries = [{
                "kind": "relevant_history",
                "body": body,
                "confidence": 0.9,
                "sensitivity": "normal",
                "fact_ids": ["f-dm"],
            }]
            return json.dumps(
                _curation_for_visible_fact(kwargs, "f-dm", entries),
            ), {}

    class Admission:
        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user"])
            if not prompt["candidates"]:
                return json.dumps(_admission([])), {}
            candidate_id = prompt["candidates"][0]["candidate_id"]
            return json.dumps(_admission([{
                "candidate_id": candidate_id,
                "admit": True,
                "sensitivity": "high",
                "reason": "durable",
            }])), {}

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    pipeline._config = SimpleNamespace(
        tenant_id="t1",
        assembler=SimpleNamespace(
            actor_card_enabled=True,
            actor_card_fact_limit=60,
            actor_card_entries_per_kind=3,
            actor_card_admission_model="semantic-model",
        ),
        compactor=SimpleNamespace(max_summary_tokens=500),
    )
    pipeline._compactor = SimpleNamespace(
        llm=Curator(), _parse_response=lambda text: json.loads(text),
    )
    pipeline._actor_card_admission_provider_override = Admission()

    assert pipeline._rebuild_actor_card(OPTICS) == 1
    row = store._get_conn().execute(
        """SELECT body, sensitivity, audience_scope
             FROM actor_card_entries
            WHERE tenant_id = ? AND actor_id = ?
              AND superseded_by IS NULL""",
        ("t1", OPTICS),
    ).fetchone()
    assert tuple(row) == (body, "high", CARD_SCOPE_SAME_CONVERSATION)
    # High-sensitivity material remains structurally non-serving.
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


@pytest.mark.parametrize(
    "variant",
    [
        "extra_top_level",
        "non_boolean",
        "invalid_sensitivity",
        "duplicate_candidate",
        "missing_candidate",
        "hallucinated_candidate",
        "reason_mismatch",
    ],
)
def test_semantic_admission_malformed_output_fails_closed(store, variant):
    _dm_and_guild(store)

    class Curator:
        def complete(self, **_kwargs):
            return json.dumps(_curation([{
                "kind": CARD_KIND_COMMUNICATION_PREF,
                "body": "prefers concise answers",
                "confidence": 0.9,
                "sensitivity": "normal",
                "fact_ids": ["f-guild"],
            }])), {}

    class Admission:
        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user"])
            candidate_id = prompt["candidates"][0]["candidate_id"]
            decision = {
                "candidate_id": candidate_id,
                "admit": True,
                "sensitivity": "normal",
                "reason": "durable",
            }
            if variant == "extra_top_level":
                payload = {**_admission([decision]), "extra": True}
            elif variant == "non_boolean":
                payload = _admission(
                    [{**decision, "admit": "yes"}],
                    substantive=True,
                )
            elif variant == "invalid_sensitivity":
                payload = _admission(
                    [{**decision, "sensitivity": "none"}],
                    substantive=True,
                )
            elif variant == "duplicate_candidate":
                payload = _admission([decision, decision])
            elif variant == "missing_candidate":
                payload = _admission([], substantive=True)
            elif variant == "hallucinated_candidate":
                payload = _admission(
                    [
                        decision,
                        {**decision, "candidate_id": "invented"},
                    ],
                )
            else:
                payload = _admission(
                    [{**decision, "reason": "test_probe"}],
                )
            return json.dumps(payload), {}

    pipeline = _card_pipeline(
        store,
        Curator(),
        admission=Admission(),
    )
    with pytest.raises(
        RuntimeError, match="semantic admission failed",
    ):
        pipeline._rebuild_actor_card(OPTICS)

    profile = store.get_actor_profile("t1", OPTICS)
    assert profile is not None and profile.card_dirty is True
    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "admission_error"
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None


def test_semantic_admission_cannot_lower_sensitivity_or_widen_scope(store):
    _dm_and_guild(store)

    class Curator:
        def complete(self, **_kwargs):
            return json.dumps(_curation([{
                "kind": CARD_KIND_COMMUNICATION_PREF,
                "body": "private communication preference",
                "confidence": 0.9,
                "sensitivity": "high",
                "fact_ids": ["f-dm"],
            }])), {}

    class Admission:
        def complete(self, **kwargs):
            candidate_id = json.loads(
                kwargs["user"],
            )["candidates"][0]["candidate_id"]
            return json.dumps(_admission([{
                "candidate_id": candidate_id,
                "admit": True,
                "sensitivity": "normal",
                "reason": "durable",
            }])), {}

    pipeline = _card_pipeline(
        store,
        Curator(),
        admission=Admission(),
    )
    with pytest.raises(
        RuntimeError, match="semantic admission failed",
    ):
        pipeline._rebuild_actor_card(OPTICS)

    assert store.get_actor_profile("t1", OPTICS).card_dirty is True
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None


def test_forced_rebuild_cannot_write_without_semantic_admission(store):
    _dm_and_guild(store)

    class Curator:
        def complete(self, **_kwargs):
            return json.dumps(_curation([{
                "kind": CARD_KIND_COMMUNICATION_PREF,
                "body": "ungated preference",
                "confidence": 0.9,
                "sensitivity": "normal",
                "fact_ids": ["f-guild"],
            }])), {}

    pipeline = _card_pipeline(
        store,
        Curator(),
        enabled=False,
        admission_model="",
    )
    with pytest.raises(
        RuntimeError, match="semantic admission failed",
    ):
        pipeline._rebuild_actor_card(OPTICS, force=True)

    status = store.get_actor_card_rebuild_status("t1", OPTICS)
    assert status is not None
    assert status["outcome"] == "admission_error"
    assert store.get_actor_profile("t1", OPTICS).card_dirty is True


def test_semantic_evidence_keeps_late_revocation_and_excludes_other_actor(store):
    _conversation(store, "guild")
    canonical_ids = []
    for index in range(9):
        canonical_id = f"ct-optics-{index}"
        canonical_ids.append(canonical_id)
        _turn(
            store,
            canonical_id,
            "guild",
            OPTICS,
            "guild",
            "chan",
            content=f"Optics message {index}",
        )
    canonical_ids.append("ct-bigtex")
    _turn(
        store,
        "ct-bigtex",
        "guild",
        BIGTEX,
        "guild",
        "chan",
        content="Instructions from another actor must never be evidence.",
    )
    canonical_ids.append("ct-optics-stop")
    _turn(
        store,
        "ct-optics-stop",
        "guild",
        OPTICS,
        "guild",
        "chan",
        content=("context " * 200) + "Stop that preference now.",
    )
    _segment(store, "seg-guild", "guild", canonical_ids)
    _fact(store, "f-guild", "guild", "seg-guild", OPTICS)
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    pipeline = object.__new__(CompactionPipeline)
    pipeline._store = store
    sources = list(store.list_actor_facts("t1", OPTICS, limit=60))
    evidence, refs = pipeline._actor_card_evidence_segments(
        OPTICS,
        "guild",
        sources,
        {"f-guild"},
    )

    contents = [
        message["content"]
        for segment in evidence
        for message in segment["messages"]
    ]
    assert len(contents) == 10
    assert len(contents[-1]) <= 1200
    assert contents[-1].endswith("Stop that preference now.")
    assert "...[middle truncated]..." in contents[-1]
    assert all("another actor" not in content for content in contents)
    assert refs == {("guild", "seg-guild")}

    # Simulate a concurrent segment/source mutation after list_actor_facts()
    # proved this fact's audience but before admission reloaded raw rows.
    _conversation(store, "dm")
    store._get_conn().execute(
        """UPDATE canonical_turns
              SET audience_conversation_id = 'dm',
                  user_content = 'CROSS_AUDIENCE_SECRET'
            WHERE canonical_turn_id = ?""",
        (canonical_ids[0],),
    )
    store._get_conn().commit()
    raced, _raced_refs = pipeline._actor_card_evidence_segments(
        OPTICS,
        "guild",
        sources,
        {"f-guild"},
    )
    assert "CROSS_AUDIENCE_SECRET" not in json.dumps(raced)

    bounded, bounded_refs = pipeline._actor_card_evidence_segments(
        OPTICS,
        "guild",
        sources,
        {"f-guild"},
        max_chars=1,
    )
    assert bounded == []
    assert bounded_refs == set()


def test_list_actor_facts_is_tenant_scoped(store):
    """An actor id shared by two tenants must never cross the fact query."""
    _dm_and_guild(store)
    assert store.list_actor_facts("other-tenant", OPTICS, limit=10) == []


def test_incomplete_source_mapping_is_card_ineligible(store):
    _conversation(store, "c1")
    _turn(store, "ct1", "c1", OPTICS, "c1", "chan")
    _segment(store, "seg1", "c1", ["ct1"], complete=False)
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    assert store.list_actor_facts("t1", OPTICS, limit=10) == []


def test_segment_mapping_cannot_borrow_a_row_from_another_owner(store):
    _conversation(store, "c1")
    _conversation(store, "c2")
    _turn(store, "ct-other", "c2", OPTICS, "c2", "chan")
    _segment(store, "seg1", "c1", ["ct-other"], complete=True)
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())

    assert store.list_actor_facts("t1", OPTICS, limit=10) == []


def test_legacy_unversioned_audience_is_card_ineligible(store):
    """A row with no proven audience cannot fall back to the owner."""
    _conversation(store, "c1")
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, turn_hash, sort_key,
                user_content, assistant_content, sender_actor_id,
                audience_conversation_id, audience_attribution_version)
           VALUES ('ct1', 'c1', 'h', 's', 'hi', '', ?, '', 0)""",
        (OPTICS,),
    )
    conn.commit()
    _segment(store, "seg1", "c1", ["ct1"])
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    assert store.list_actor_facts("t1", OPTICS, limit=10) == []


# ---------------------------------------------------------------------------
# Audience policy — the private-to-public leakage boundary
# ---------------------------------------------------------------------------

def test_dm_entry_is_not_served_in_the_guild(store):
    """Influence-only use is NOT an audience boundary: a DM goal can leak by
    shaping a public answer without ever being quoted. It must be structurally
    absent, not merely un-quoted."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)

    in_dm = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert _bodies(in_dm) == ["prefers terse answers", "private DM goal"]

    in_guild = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    )
    assert _bodies(in_guild) == ["prefers terse answers"]
    assert "private DM goal" not in (_bodies(in_guild) or [])


def test_turn_sourced_relevant_history_never_leaks_from_dm_to_guild(store):
    _dm_and_guild(store)
    history = _entry(
        "e-turn-dm-history",
        "relevant_history",
        "Has discussed a private DM topic with Vast.",
    )
    assert store.replace_actor_card(
        "t1",
        OPTICS,
        [(
            history,
            [_turn_source(
                history.id,
                "dm",
                "dm",
                "ct-dm",
                "chan-dm",
            )],
        )],
        input_hash="turn-dm-history",
        expected_source_epochs={"dm": 1},
    ) == 1

    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="dm",
        audience_conversation_id="dm",
        audience_channel_id="chan-dm",
    )) == ["Has discussed a private DM topic with Vast."]
    assert store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-guild",
    ) is None


def test_card_models_never_receive_dm_and_guild_evidence_together(store):
    _conversation(store, "dm")
    _conversation(store, "guild")
    _turn(
        store,
        "ct-private",
        "dm",
        OPTICS,
        "dm",
        "chan-dm",
        content="PRIVATE_ORCHID is the topic of this private discussion.",
    )
    _turn(
        store,
        "ct-public",
        "guild",
        OPTICS,
        "guild",
        "chan-guild",
        content="PUBLIC_CUBE is the topic of this guild discussion.",
    )
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    class Curator:
        prompts = []

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user"])
            self.prompts.append(prompt)
            turn = prompt["turns"][0]
            private = "PRIVATE_ORCHID" in turn["content"]
            return json.dumps(_curation([{
                "kind": "relevant_history",
                "body": (
                    "Has discussed PRIVATE_ORCHID with Vast."
                    if private
                    else "Has discussed PUBLIC_CUBE with Vast."
                ),
                "confidence": 0.9,
                "sensitivity": "normal",
                "turn_ids": [turn["id"]],
            }])), {}

    class Admission:
        prompts = []

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user"])
            self.prompts.append(prompt)
            candidate = prompt["candidates"][0]
            return json.dumps(_admission([{
                "candidate_id": candidate["candidate_id"],
                "admit": True,
                "sensitivity": "normal",
                "reason": "durable",
            }])), {}

    curator = Curator()
    admission = Admission()
    pipeline = _card_pipeline(
        store,
        curator,
        admission=admission,
    )
    assert pipeline._rebuild_actor_card(OPTICS) == 2

    for prompt in [*curator.prompts, *admission.prompts]:
        serialized = json.dumps(prompt)
        assert not (
            "PRIVATE_ORCHID" in serialized
            and "PUBLIC_CUBE" in serialized
        )
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="dm",
        audience_conversation_id="dm",
        audience_channel_id="chan-dm",
    )) == ["Has discussed PRIVATE_ORCHID with Vast."]
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-guild",
    )) == ["Has discussed PUBLIC_CUBE with Vast."]


def test_per_kind_card_limit_is_independent_for_each_audience(store):
    _dm_and_guild(store)

    class Curator:
        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user"])
            audience = prompt["turns"][0]["audience_conversation_id"]
            turn_id = prompt["turns"][0]["id"]
            return json.dumps(_curation([{
                "kind": "relevant_history",
                "body": f"Has substantive history in {audience}.",
                "confidence": 0.9,
                "sensitivity": "normal",
                "turn_ids": [turn_id],
            }])), {}

    pipeline = _card_pipeline(
        store,
        Curator(),
        admission=_AdmitAll(),
    )
    pipeline._config.assembler.actor_card_entries_per_kind = 1
    assert pipeline._rebuild_actor_card(OPTICS) == 2
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="dm",
        audience_conversation_id="dm",
        audience_channel_id="chan-dm",
    )) == ["Has substantive history in dm."]
    assert _bodies(store.get_actor_card(
        "t1",
        OPTICS,
        owner_conversation_id="guild",
        audience_conversation_id="guild",
        audience_channel_id="chan-guild",
    )) == ["Has substantive history in guild."]


def test_interaction_style_crosses_dm_and_guild_for_the_same_actor(store):
    _dm_and_guild(store)
    style = _entry(
        "e-style", CARD_KIND_INTERACTION_STYLE, "likes a playful tone",
        scope=CARD_SCOPE_CROSS_CONTEXT,
    )
    store.replace_actor_card(
        "t1", OPTICS,
        [(style, [_source("e-style", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h-style", expected_source_epochs={"dm": 1},
    )

    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    )

    assert _bodies(card) == ["likes a playful tone"]


def test_conversation_scope_admits_same_conversation_entries_from_all_channels(store):
    _conversation(store, "guild")
    _turn(store, "ct-a", "guild", OPTICS, "guild", "chan-a")
    _turn(store, "ct-b", "guild", OPTICS, "guild", "chan-b")
    _segment(store, "seg-a", "guild", ["ct-a"])
    _segment(store, "seg-b", "guild", ["ct-b"])
    _fact(store, "f-a", "guild", "seg-a", OPTICS)
    _fact(store, "f-b", "guild", "seg-b", OPTICS)
    store.upsert_actor_profile_from_turn("guild", OPTICS, "Optics", seen_at=_now())
    store.replace_actor_card(
        "t1", OPTICS,
        [
            (_entry("e-a", CARD_KIND_ACTIVE_GOAL, "goal from channel a"),
             [_source("e-a", "guild", "guild", "f-a", "chan-a")]),
            (_entry("e-b", CARD_KIND_ACTIVE_GOAL, "goal from channel b"),
             [_source("e-b", "guild", "guild", "f-b", "chan-b")]),
        ],
        input_hash="h-guild", expected_source_epochs={"guild": 1},
    )

    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="",
    )

    assert _bodies(card) == ["goal from channel a", "goal from channel b"]


def test_channel_mismatch_fails_closed(store):
    """A source from channel A is not served to a same_conversation request in
    channel B."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-other",
    )
    assert "private DM goal" not in (_bodies(card) or [])


def test_unknown_source_channel_fails_closed(store):
    """An empty source channel is unknown, not wildcard."""
    _conversation(store, "c1")
    _turn(store, "ct1", "c1", OPTICS, "c1", channel="")  # no durable channel
    _segment(store, "seg1", "c1", ["ct1"])
    _fact(store, "f1", "c1", "seg1", OPTICS)
    store.upsert_actor_profile_from_turn("c1", OPTICS, "Optics", seen_at=_now())
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "goal"),
          [_source("e1", "c1", "c1", "f1", channel="")])],
        input_hash="h", expected_source_epochs={"c1": 1},
    )
    # Request HAS a durable channel; the source's is unknown -> excluded.
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="c1",
        audience_conversation_id="c1", audience_channel_id="chan-real",
    )
    assert card is None


def test_empty_audience_reads_no_card(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="", audience_channel_id="chan-dm",
    ) is None


def test_high_sensitivity_entries_are_never_served(store):
    _dm_and_guild(store)
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "secret", sensitivity=CARD_SENSITIVITY_HIGH),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h", expected_source_epochs={"dm": 1},
    )
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert card is None


# ---------------------------------------------------------------------------
# Cross-actor and cross-tenant isolation — the rubric's worst failures
# ---------------------------------------------------------------------------

def test_no_cross_actor_leakage(store):
    """A prepare for actor A never contains any entry belonging to actor B."""
    _conversation(store, "guild")
    _turn(store, "ct-o", "guild", OPTICS, "guild", "chan")
    _turn(store, "ct-b", "guild", BIGTEX, "guild", "chan")
    _segment(store, "seg-o", "guild", ["ct-o"])
    _segment(store, "seg-b", "guild", ["ct-b"])
    _fact(store, "f-o", "guild", "seg-o", OPTICS)
    _fact(store, "f-b", "guild", "seg-b", BIGTEX)
    store.upsert_actor_profile_from_turn("guild", OPTICS, "Optics", seen_at=_now())
    store.upsert_actor_profile_from_turn("guild", BIGTEX, "BigTex", seen_at=_now())

    store.replace_actor_card(
        "t1", BIGTEX,
        [(_entry("e-b", CARD_KIND_ACTIVE_GOAL, "bigtex protocol claim"),
          [_source("e-b", "guild", "guild", "f-b", "chan")])],
        input_hash="hb", expected_source_epochs={"guild": 1},
    )
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e-o", CARD_KIND_ACTIVE_GOAL, "optics own goal"),
          [_source("e-o", "guild", "guild", "f-o", "chan")])],
        input_hash="ho", expected_source_epochs={"guild": 1},
    )

    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan",
    )
    assert _bodies(card) == ["optics own goal"]
    assert "bigtex protocol claim" not in (_bodies(card) or [])


def test_no_cross_tenant_card_read(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "other-tenant", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


# ---------------------------------------------------------------------------
# Cache semantics and lifecycle fences
# ---------------------------------------------------------------------------

def test_entries_supersede_rather_than_duplicate(store):
    _dm_and_guild(store)
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "first"),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h1", expected_source_epochs={"dm": 1},
    )
    store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e2", CARD_KIND_ACTIVE_GOAL, "second"),
          [_source("e2", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h2", expected_source_epochs={"dm": 1},
    )
    card = store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    )
    assert _bodies(card) == ["second"]

    conn = store._get_conn()
    old = conn.execute(
        "SELECT superseded_by FROM actor_card_entries WHERE id = 'e1'"
    ).fetchone()
    assert old[0] == "e2"  # superseded by its same-kind replacement, not deleted


def test_dirty_card_is_not_served(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE actor_profiles SET card_dirty = 1 WHERE actor_id = ?", (OPTICS,)
    )
    conn.commit()
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_delete_conversation_removes_its_contribution_to_every_card(store):
    """No entry whose fact owner or audience origin is deleted may survive."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)

    store.delete_conversation("dm")

    conn = store._get_conn()
    surviving = {r[0] for r in conn.execute("SELECT id FROM actor_card_entries")}
    assert "e-goal" not in surviving          # the DM-sourced entry is gone
    assert surviving == {"e-pref"}

    # ...and the profile is dirty, so nothing is served until a clean rebuild.
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None


def test_delete_does_not_prune_an_unrelated_tenant_profile(store):
    _conversation(store, "doomed", tenant="t1")
    _conversation(store, "other", tenant="t2")
    _turn(store, "doomed-ct", "doomed", BIGTEX, "doomed")
    assert store.upsert_actor_profile_from_turn(
        "doomed", BIGTEX, "BigTex one", seen_at=_now(),
    )
    assert store.upsert_actor_profile_from_turn(
        "other", BIGTEX, "BigTex", seen_at=_now(),
    )

    store.delete_conversation("doomed")

    profile = store.get_actor_profile("t2", BIGTEX)
    assert profile is not None
    assert profile.display_name == "BigTex"


def test_stale_builder_cannot_resurrect_after_epoch_change(store):
    """A builder that enumerated at epoch 1 cannot write after a resurrect."""
    _dm_and_guild(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE conversations SET lifecycle_epoch = 2 WHERE conversation_id = 'dm'"
    )
    conn.commit()
    written = store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e1", CARD_KIND_ACTIVE_GOAL, "stale"),
          [_source("e1", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="h", expected_source_epochs={"dm": 1},  # stale epoch
    )
    assert written == 0
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_fact_redistillation_epoch_fence_preserves_existing_facts(store):
    _dm_and_guild(store)

    deleted, inserted = store.replace_facts_for_segment(
        "dm", "seg-dm", [], expected_lifecycle_epoch=99,
    )

    assert (deleted, inserted) == (0, 0)
    assert [fact.id for fact in store.get_facts_by_segment("seg-dm")] == ["f-dm"]


def test_card_replacement_rejects_forged_source_provenance(store):
    _dm_and_guild(store)
    written = store.replace_actor_card(
        "t1", OPTICS,
        [(_entry("e-forged", CARD_KIND_COMMUNICATION_PREF, "forged"),
          [_source(
              "e-forged", "guild", "guild", "f-dm", "chan-guild",
          )])],
        input_hash="forged",
        expected_source_epochs={"dm": 1, "guild": 1},
    )

    assert written == 0
    assert store._get_conn().execute(
        "SELECT 1 FROM actor_card_entries WHERE id = 'e-forged'",
    ).fetchone() is None


def test_card_replacement_rejects_source_free_cross_context_entry(store):
    _dm_and_guild(store)
    written = store.replace_actor_card(
        "t1", OPTICS,
        [(_entry(
            "e-unproven", CARD_KIND_COMMUNICATION_PREF, "unproven",
            scope=CARD_SCOPE_CROSS_CONTEXT,
        ), [])],
        input_hash="unproven", expected_source_epochs={},
    )

    assert written == 0
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="guild", audience_channel_id="chan-guild",
    ) is None


def test_card_entry_id_collision_cannot_cross_tenant_or_actor(store):
    _dm_and_guild(store)
    first = _entry("shared-entry", CARD_KIND_ACTIVE_GOAL, "tenant one")
    assert store.replace_actor_card(
        "t1", OPTICS,
        [(first, [_source("shared-entry", "dm", "dm", "f-dm", "chan-dm")])],
        input_hash="one", expected_source_epochs={"dm": 1},
    ) == 1

    _conversation(store, "other-c", tenant="t2")
    _turn(store, "other-ct", "other-c", BIGTEX, "other-c", "other-chan")
    _segment(store, "other-seg", "other-c", ["other-ct"])
    _fact(store, "other-fact", "other-c", "other-seg", BIGTEX)
    store.upsert_actor_profile_from_turn(
        "other-c", BIGTEX, "BigTex", seen_at=_now(),
    )
    second = _entry("shared-entry", CARD_KIND_ACTIVE_GOAL, "tenant two")

    assert store.replace_actor_card(
        "t2", BIGTEX,
        [(second, [_source(
            "shared-entry", "other-c", "other-c", "other-fact", "other-chan",
            tenant="t2",
        )])],
        input_hash="two", expected_source_epochs={"other-c": 1},
    ) == 0
    row = store._get_conn().execute(
        "SELECT tenant_id, actor_id, body FROM actor_card_entries WHERE id = ?",
        ("shared-entry",),
    ).fetchone()
    assert tuple(row) == ("t1", OPTICS, "tenant one")


def test_replace_facts_for_segment_dirties_the_author_card(store):
    """Fact replacement must dirty the card in the same transaction, or stale
    card content stays readable through the crash window."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    # Card is clean and readable to begin with.
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is not None

    store.replace_facts_for_segment("dm", "seg-dm", [])

    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None


def test_card_read_requires_route_to_resolve_to_the_owner(store):
    """An audience route that does not prove owner-or-alias-to-owner reads no
    card, rather than silently falling back to the resolved owner."""
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="guild",
        audience_conversation_id="dm",   # unrelated route, not an alias of guild
        audience_channel_id="chan-guild",
    ) is None


def test_request_audience_resolver_preserves_a_validated_alias_origin(store):
    _conversation(store, "dm", phase="merged")
    _conversation(store, "guild")
    store.save_conversation_alias("dm", "guild")

    assert store.resolve_request_audience("t1", "dm", "guild") == "dm"
    assert store.resolve_request_audience("t1", "guild", "guild") == "guild"
    assert store.resolve_request_audience("other", "dm", "guild") == ""


def test_deleted_audience_route_cannot_read_cross_context_entries(store):
    _dm_and_guild(store)
    _build_dm_goal_and_cross_pref(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE conversations SET phase = 'deleted', deleted_at = ? "
        "WHERE conversation_id = 'dm'",
        (_now(),),
    )
    conn.commit()

    assert store.get_actor_card(
        "t1", OPTICS, owner_conversation_id="dm",
        audience_conversation_id="dm", audience_channel_id="chan-dm",
    ) is None
