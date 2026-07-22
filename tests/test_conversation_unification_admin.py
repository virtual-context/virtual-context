"""Guarded repairs used when channel conversations become one server audience."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore


TENANT = "tenant-men"
SOURCE = "sk:agent:vast:discord:channel:111"
TARGET = "sk:agent:vast:discord:guild:999"
OTHER = "sk:agent:vast:discord:guild:1000"
ACTOR = "actor:discord:42"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conversation(store: SQLiteStore, conversation_id: str) -> None:
    store.activate_conversation(conversation_id)
    store.upsert_conversation(tenant_id=TENANT, conversation_id=conversation_id)
    assert store.set_phase(
        conversation_id=conversation_id, lifecycle_epoch=1, phase="active",
    )


def _merge_source(store: SQLiteStore) -> None:
    merge_id = str(uuid.uuid4())
    reserved = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id,
        tenant_id=TENANT,
        source_conversation_id=SOURCE,
        target_conversation_id=TARGET,
        source_label_at_merge="#source",
    )
    assert reserved.status == "reserved"
    store.merge_conversation_data(
        merge_id=merge_id,
        tenant_id=TENANT,
        source_conversation_id=SOURCE,
        target_conversation_id=TARGET,
        sort_key_offset=10_000.0,
        expected_target_lifecycle_epoch=1,
        expected_source_lifecycle_epoch=1,
        source_label_at_merge="#source",
    )


def _turn(
    store: SQLiteStore,
    turn_id: str,
    *,
    audience_version: int = 1,
    channel: str = "111",
    compacted: bool = False,
    sort_key: float | None = None,
) -> None:
    store.save_canonical_turn(
        SOURCE, -1, f"claim {turn_id}", "",
        canonical_turn_id=turn_id,
        sort_key=sort_key if sort_key is not None else float(len(turn_id) * 100),
        turn_hash=f"hash-{turn_id}",
        sender="BigTex",
        sender_actor_id=ACTOR,
        source_message_id=f"msg-{turn_id}",
        origin_channel_id=channel,
        origin_channel_label="#source" if channel else "",
        audience_conversation_id=SOURCE,
        audience_attribution_version=audience_version,
        tagged_at=_now(),
        compacted_at=_now() if compacted else None,
    )


def test_reattribution_is_dry_run_default_guarded_and_idempotent(tmp_path):
    store = SQLiteStore(tmp_path / "reattribute.db")
    _conversation(store, SOURCE)
    _conversation(store, TARGET)
    _turn(store, "good", compacted=True)
    _turn(store, "stale", audience_version=0)
    _turn(store, "no-channel", channel="")
    _merge_source(store)

    report = store.reattribute_canonical_turn_audience(
        TARGET, SOURCE, TARGET,
        tenant_id=TENANT, expected_lifecycle_epoch=1,
    )
    assert report == {
        "matched_source": 3,
        "eligible": 1,
        "selected": 1,
        "updated": 0,
        "skipped_stale_version": 1,
        "skipped_no_channel": 1,
        "cards_invalidated": 0,
        "dry_run": True,
    }
    before = store.get_all_canonical_turns(TARGET)
    assert {row.audience_conversation_id for row in before} == {SOURCE}

    applied = store.reattribute_canonical_turn_audience(
        TARGET, SOURCE, TARGET,
        tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )
    assert applied["updated"] == 1
    rows = {row.canonical_turn_id: row for row in store.get_all_canonical_turns(TARGET)}
    assert rows["good"].audience_conversation_id == TARGET
    assert rows["good"].origin_channel_id == "111"
    assert rows["good"].origin_conversation_id == SOURCE
    assert rows["stale"].audience_conversation_id == SOURCE
    assert rows["no-channel"].audience_conversation_id == SOURCE

    resolved = store.find_canonical_turn_by_source_message_id(
        TARGET, "msg-good",
        audience_conversation_id=TARGET,
        origin_channel_id="111",
    )
    assert resolved is not None
    assert resolved.sender_actor_id == ACTOR

    replay = store.reattribute_canonical_turn_audience(
        TARGET, SOURCE, TARGET,
        tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )
    assert replay["eligible"] == replay["updated"] == 0


def test_reattribution_refuses_non_alias_and_epoch_mismatch(tmp_path):
    store = SQLiteStore(tmp_path / "reattribute-guards.db")
    _conversation(store, SOURCE)
    _conversation(store, TARGET)

    with pytest.raises(ValueError, match="retained merged alias"):
        store.reattribute_canonical_turn_audience(
            TARGET, SOURCE, TARGET,
            tenant_id=TENANT, expected_lifecycle_epoch=1,
        )

    _turn(store, "good")
    _merge_source(store)
    with pytest.raises(ValueError, match="epoch mismatch"):
        store.reattribute_canonical_turn_audience(
            TARGET, SOURCE, TARGET,
            tenant_id=TENANT, expected_lifecycle_epoch=2,
        )

    for invalid_limit in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            store.reattribute_canonical_turn_audience(
                TARGET, SOURCE, TARGET,
                tenant_id=TENANT, expected_lifecycle_epoch=1,
                limit=invalid_limit,
            )

    group_dm = "sk:agent:vast:discord:group:111"
    with pytest.raises(ValueError, match="channel source and guild owner"):
        store.reattribute_canonical_turn_audience(
            TARGET, group_dm, TARGET,
            tenant_id=TENANT, expected_lifecycle_epoch=1,
        )


def test_reattribution_batches_more_than_sqlite_variable_limit(tmp_path):
    store = SQLiteStore(tmp_path / "reattribute-large.db")
    _conversation(store, SOURCE)
    _conversation(store, TARGET)
    for index in range(1_005):
        _turn(store, f"bulk-{index}", sort_key=float((index + 1) * 100))
    _merge_source(store)

    applied = store.reattribute_canonical_turn_audience(
        TARGET, SOURCE, TARGET,
        tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )
    assert applied["eligible"] == applied["updated"] == 1_005
    assert {
        row.audience_conversation_id
        for row in store.get_all_canonical_turns(TARGET)
    } == {TARGET}


def test_repairs_refuse_owner_that_is_being_merged_away(tmp_path):
    store = SQLiteStore(tmp_path / "merge-race.db")
    _conversation(store, SOURCE)
    _conversation(store, TARGET)
    _conversation(store, OTHER)
    _turn(store, "good")
    _merge_source(store)
    reservation = store.try_reserve_merge_audit_in_progress(
        merge_id=str(uuid.uuid4()),
        tenant_id=TENANT,
        source_conversation_id=TARGET,
        target_conversation_id=OTHER,
        source_label_at_merge="#server",
    )
    assert reservation.status == "reserved"

    with pytest.raises(RuntimeError, match="active merge"):
        store.reattribute_canonical_turn_audience(
            TARGET, SOURCE, TARGET,
            tenant_id=TENANT, expected_lifecycle_epoch=1,
        )
    with pytest.raises(RuntimeError, match="active merge"):
        store.reset_conversation_derived_data(
            TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
        )


def test_derived_reset_preserves_canonical_and_removes_polluted_outputs(tmp_path):
    store = SQLiteStore(tmp_path / "rebuild.db")
    _conversation(store, TARGET)
    now = _now()
    store.save_canonical_turn(
        TARGET, -1, "clean canonical statement", "",
        canonical_turn_id="ct-clean", sort_key=1000.0,
        turn_hash="hash-clean", primary_tag="chat", tags=["chat"],
        sender="BigTex", sender_actor_id=ACTOR,
        origin_channel_id="111", origin_channel_label="#source",
        audience_conversation_id=TARGET, audience_attribution_version=1,
        tagged_at=now, compacted_at=now,
    )
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO segments
           (ref, conversation_id, primary_tag, summary, full_text,
            messages_json, metadata_json, created_at, start_timestamp, end_timestamp)
           VALUES ('seg-bad', ?, 'chat', 'polluted summary', 'clean canonical statement',
                   '[]', '{}', ?, ?, ?)""",
        (TARGET, now, now, now),
    )
    conn.execute("INSERT INTO segment_tags (segment_ref, tag) VALUES ('seg-bad', 'chat')")
    conn.execute(
        """INSERT INTO segment_chunks (segment_ref, chunk_index, text, embedding_json)
           VALUES ('seg-bad', 0, 'polluted', '[]')"""
    )
    conn.execute(
        """INSERT INTO facts
           (id, subject, verb, object, segment_ref, conversation_id)
           VALUES ('fact-bad', 'user', 'said', 'polluted', 'seg-bad', ?)""",
        (TARGET,),
    )
    conn.execute("INSERT INTO fact_tags (fact_id, tag) VALUES ('fact-bad', 'chat')")
    conn.execute(
        """INSERT INTO fact_embeddings (fact_id, conversation_id, model, embedding_json)
           VALUES ('fact-bad', ?, 'test', '[]')""",
        (TARGET,),
    )
    conn.execute(
        """INSERT INTO tag_summaries
           (tag, conversation_id, summary, created_at, updated_at)
           VALUES ('chat', ?, 'polluted history', ?, ?)""",
        (TARGET, now, now),
    )
    conn.execute(
        """INSERT INTO tag_summary_embeddings (tag, conversation_id, embedding_json)
           VALUES ('chat', ?, '[]')""",
        (TARGET,),
    )
    conn.execute(
        """INSERT INTO engine_state
           (conversation_id, compacted_prefix_messages, turn_count, turn_tag_entries, saved_at)
           VALUES (?, 1, 1, '[]', ?)""",
        (TARGET, now),
    )
    conn.commit()

    preview = store.reset_conversation_derived_data(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
    )
    assert preview["canonical_rows"] == 1
    assert preview["segments"] == 1
    assert preview["facts"] == 1
    assert preview["tag_summaries"] == 1
    assert store.get_segment("seg-bad", conversation_id=TARGET) is not None

    applied = store.reset_conversation_derived_data(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )
    assert applied["segments"] == applied["facts"] == applied["tag_summaries"] == 1
    assert store.get_all_segments(conversation_id=TARGET) == []
    assert store.get_all_tag_summaries(conversation_id=TARGET) == []
    assert conn.execute(
        "SELECT COUNT(*) FROM facts WHERE conversation_id = ?", (TARGET,),
    ).fetchone()[0] == 0
    row = store.get_all_canonical_turns(TARGET)[0]
    assert row.user_content == "clean canonical statement"
    assert row.tags == ["chat"]
    assert row.tagged_at == now
    assert row.compacted_at is None
    assert conn.execute(
        "SELECT COUNT(*) FROM engine_state WHERE conversation_id = ?", (TARGET,),
    ).fetchone()[0] == 0


def test_derived_reset_refuses_an_untagged_canonical_backlog(tmp_path):
    store = SQLiteStore(tmp_path / "untagged.db")
    _conversation(store, TARGET)
    store.save_canonical_turn(
        TARGET, -1, "not tagged yet", "",
        canonical_turn_id="ct-untagged", sort_key=1000.0,
        turn_hash="hash-untagged",
        audience_conversation_id=TARGET, audience_attribution_version=1,
        origin_channel_id="111",
    )

    preview = store.reset_conversation_derived_data(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
    )
    assert preview["untagged_rows"] == 1
    with pytest.raises(RuntimeError, match="must be tagged"):
        store.reset_conversation_derived_data(
            TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
            dry_run=False,
        )
    assert len(store.get_all_canonical_turns(TARGET)) == 1


def test_derived_reset_only_touches_rows_with_compaction_state(tmp_path):
    store = SQLiteStore(tmp_path / "selective-reset.db")
    _conversation(store, TARGET)
    now = _now()
    for turn_id in ("plain", "orphan-op"):
        store.save_canonical_turn(
            TARGET, -1, f"statement {turn_id}", "",
            canonical_turn_id=turn_id,
            sort_key=1000.0 if turn_id == "plain" else 2000.0,
            turn_hash=f"hash-{turn_id}",
            audience_conversation_id=TARGET,
            audience_attribution_version=1,
            origin_channel_id="111",
            tagged_at=now,
        )
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET compaction_operation_id = 'orphan' "
        "WHERE canonical_turn_id = 'orphan-op'",
    )
    conn.commit()
    before_plain = conn.execute(
        "SELECT updated_at FROM canonical_turns WHERE canonical_turn_id = 'plain'",
    ).fetchone()[0]

    preview = store.reset_conversation_derived_data(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
    )
    assert preview["canonical_rows"] == 2
    assert preview["canonical_rows_to_reset"] == 1
    store.reset_conversation_derived_data(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )

    plain = conn.execute(
        "SELECT updated_at FROM canonical_turns WHERE canonical_turn_id = 'plain'",
    ).fetchone()[0]
    orphan = conn.execute(
        "SELECT compaction_operation_id FROM canonical_turns "
        "WHERE canonical_turn_id = 'orphan-op'",
    ).fetchone()[0]
    assert plain == before_plain
    assert orphan is None


def test_resequence_interleaves_origins_and_remaps_turn_artifacts(tmp_path):
    store = SQLiteStore(tmp_path / "resequence.db")
    _conversation(store, TARGET)
    source = SOURCE

    def save(
        turn_id: str, *, origin: str, sort_key: float, group: int,
        timestamp: str, user: str = "", assistant: str = "",
    ) -> None:
        store.save_canonical_turn(
            TARGET, -1, user, assistant,
            canonical_turn_id=turn_id,
            sort_key=sort_key,
            turn_hash=f"hash-{turn_id}",
            turn_group_number=group,
            first_seen_at=timestamp,
            last_seen_at=timestamp,
            created_at=timestamp,
            updated_at=timestamp,
            tagged_at=timestamp,
            audience_conversation_id=TARGET,
            audience_attribution_version=1,
        )
        if origin:
            store._get_conn().execute(
                "UPDATE canonical_turns SET origin_conversation_id = ? "
                "WHERE canonical_turn_id = ?",
                (origin, turn_id),
            )

    save(
        "target-u", origin="", sort_key=1000, group=0,
        timestamp="2026-07-20T12:00:00+00:00", user="later",
    )
    save(
        "target-a", origin="", sort_key=2000, group=0,
        timestamp="2026-07-20T12:00:01+00:00", assistant="later reply",
    )
    save(
        "source-u", origin=f" {source} ", sort_key=3000, group=0,
        timestamp="2026-07-19T12:00:00+00:00", user="earlier",
    )
    save(
        "source-a", origin=f" {source} ", sort_key=4000, group=0,
        timestamp="2026-07-19T12:00:01+00:00", assistant="earlier reply",
    )
    conn = store._get_conn()
    conn.execute(
        """INSERT INTO turn_tool_outputs
           (conversation_id, turn_number, tool_output_ref, origin_conversation_id)
           VALUES (?, 0, 'tool-source', ?)""",
        (TARGET, f" {source} "),
    )
    conn.execute(
        """INSERT INTO chain_snapshots
           (ref, conversation_id, turn_number, chain_json, message_count,
            origin_conversation_id)
           VALUES ('chain-target', ?, 0, '{}', 1, '')""",
        (TARGET,),
    )
    conn.execute(
        """INSERT INTO chain_snapshots
           (ref, conversation_id, turn_number, chain_json, message_count,
            origin_conversation_id)
           VALUES ('chain-sentinel', ?, -3000000, '{}', 0, '')""",
        (TARGET,),
    )
    conn.commit()

    preview = store.resequence_canonical_turns(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1,
    )
    assert preview["canonical_rows"] == 4
    assert preview["logical_turns"] == 2
    assert preview["turn_tool_output_unmapped"] == 0
    assert preview["chain_snapshot_unmapped"] == 0
    assert conn.execute(
        "SELECT sort_key FROM canonical_turns WHERE canonical_turn_id = 'target-u'",
    ).fetchone()[0] == 1000

    applied = store.resequence_canonical_turns(
        TARGET, tenant_id=TENANT, expected_lifecycle_epoch=1, dry_run=False,
    )
    assert applied["changed_group_rows"] > 0
    rows = conn.execute(
        """SELECT canonical_turn_id, turn_group_number, sort_key
             FROM canonical_turns WHERE conversation_id = ? ORDER BY sort_key""",
        (TARGET,),
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        ("source-u", 0), ("source-a", 0),
        ("target-u", 1), ("target-a", 1),
    ]
    assert conn.execute(
        "SELECT turn_number FROM turn_tool_outputs WHERE tool_output_ref = 'tool-source'",
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT turn_number FROM chain_snapshots WHERE ref = 'chain-target'",
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT turn_number FROM chain_snapshots WHERE ref = 'chain-sentinel'",
    ).fetchone()[0] == -3_000_000
