"""Regression: stuck 'ingesting' phase self-heals on the next prepare
or per-turn tag call.

Two production paths were observed leaving conversations frozen at
``phase='ingesting'`` with ``ingestion_episode.status='running'`` forever:

1. REST ``/api/v1/context/prepare`` + ``/api/v1/context/ingest`` flow
   (single-turn cron-style POST): ``/ingest`` tags the turn via
   ``fire_turn_complete → _run_tag_turn`` and NEVER enters the legacy
   ingestion thread, so ``_finalize_legacy_ingestion`` never runs.

2. Worker crash mid-ingest with no further work arriving: the legacy
   thread finished tagging before dying; the next POST's
   ``handle_prepare_payload`` sees ``total == done`` and short-circuits
   without finalizing because the existing init→active branch only
   fires on ``phase == 'init'``.

Both are now covered by ``_finalize_ingestion_if_complete``, invoked
from ``handle_prepare_payload`` (total==done branch) and ``_run_tag_turn``
(post-tag path).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Path 1: REST single-turn — handle_prepare_payload on total==done finalizes
# ---------------------------------------------------------------------------

def test_handle_prepare_payload_finalizes_stuck_ingesting_phase(tmp_path: Path):
    """When a conv is stuck at phase='ingesting' with total==done and a
    still-running ingestion_episode, the next handle_prepare_payload
    must flip phase to 'active'. Covers crashed-worker recovery path.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    # Seed the conversation state: phase='ingesting', one tagged canonical
    # turn, running episode whose heartbeat is stale so claim succeeds.
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=120)  # 2 min stale, past 30s TTL

    # Insert a canonical row so total_ingestible == done_ingestible == 1.
    # Use the engine's own persistence path so sort_key / hash / tagged_at
    # are coherent.
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="ingesting")
    with inner._get_conn() as c:
        c.execute(
            """INSERT INTO canonical_turns (
                 canonical_turn_id, conversation_id, sort_key, turn_hash,
                 hash_version, user_content, assistant_content,
                 primary_tag, tags_json, session_date, sender,
                 first_seen_at, last_seen_at, created_at, updated_at,
                 turn_group_number, covered_ingestible_entries, tagged_at
               ) VALUES (?, ?, 0, 'h0', 1, 'u', 'a', 'p', '["p"]',
                         '2026-04-20', 'user', ?, ?, ?, ?, 0, 1, ?)""",
            ("ct-0", conv, now.isoformat(), now.isoformat(),
             now.isoformat(), now.isoformat(), now.isoformat()),
        )
        # Insert the stuck running episode.
        import uuid as _uuid
        c.execute(
            """INSERT INTO ingestion_episode (
                 episode_id, conversation_id, lifecycle_epoch,
                 raw_payload_entries, started_at, status,
                 owner_worker_id, heartbeat_ts
               ) VALUES (?, ?, 1, 1, ?, 'running', ?, ?)""",
            (str(_uuid.uuid4()), conv, stale.isoformat(),
             "old-worker-dead", stale.isoformat()),
        )

    # Fire handle_prepare_payload with an empty ingestible payload (no
    # new work) — it must self-heal.
    from virtual_context.proxy.state import PhaseDecision
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0,
                            "ingestible_entry_count": 0},
    )

    assert decision.phase == "active", (
        f"handle_prepare_payload must finalize stuck ingesting phase; "
        f"got {decision.phase!r}"
    )

    # And the episode row must be status='completed'.
    with inner._get_conn() as c:
        row = c.execute(
            "SELECT status, completed_at FROM ingestion_episode WHERE conversation_id=?",
            (conv,),
        ).fetchone()
    assert row[0] == "completed", f"episode must be completed; got {row[0]!r}"
    assert row[1] is not None, "completed_at must be set"


def test_handle_prepare_payload_skips_finalize_when_untagged_remains(tmp_path: Path):
    """Guard: finalize must NOT fire when there's still untagged work.
    That would complete an episode that isn't actually done.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    now = datetime.now(timezone.utc)

    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="ingesting")
    with inner._get_conn() as c:
        # Tagged row
        c.execute(
            """INSERT INTO canonical_turns (
                 canonical_turn_id, conversation_id, sort_key, turn_hash,
                 hash_version, user_content, assistant_content,
                 primary_tag, tags_json, session_date, sender,
                 first_seen_at, last_seen_at, created_at, updated_at,
                 turn_group_number, covered_ingestible_entries, tagged_at
               ) VALUES (?, ?, 0, 'h0', 1, 'u', 'a', 'p', '["p"]',
                         '2026-04-20', 'user', ?, ?, ?, ?, 0, 1, ?)""",
            ("ct-0", conv, now.isoformat(), now.isoformat(),
             now.isoformat(), now.isoformat(), now.isoformat()),
        )
        # Untagged row (tagged_at NULL)
        c.execute(
            """INSERT INTO canonical_turns (
                 canonical_turn_id, conversation_id, sort_key, turn_hash,
                 hash_version, user_content, assistant_content,
                 primary_tag, tags_json, session_date, sender,
                 first_seen_at, last_seen_at, created_at, updated_at,
                 turn_group_number, covered_ingestible_entries, tagged_at
               ) VALUES (?, ?, 1, 'h1', 1, 'u', 'a', '', '[]',
                         '2026-04-20', 'user', ?, ?, ?, ?, 0, 1, NULL)""",
            ("ct-1", conv, now.isoformat(), now.isoformat(),
             now.isoformat(), now.isoformat()),
        )
        import uuid as _uuid
        c.execute(
            """INSERT INTO ingestion_episode (
                 episode_id, conversation_id, lifecycle_epoch,
                 raw_payload_entries, started_at, status,
                 owner_worker_id, heartbeat_ts
               ) VALUES (?, ?, 1, 2, ?, 'running', ?, ?)""",
            (str(_uuid.uuid4()), conv, now.isoformat(),
             state._worker_id, now.isoformat()),
        )

    # handle_prepare_payload must NOT finalize (untagged row exists).
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0,
                            "ingestible_entry_count": 0},
    )
    assert decision.phase == "ingesting", (
        f"must stay in ingesting while untagged rows remain; got {decision.phase!r}"
    )


# ---------------------------------------------------------------------------
# Path 2: REST _run_tag_turn post-tag opportunistic finalize
# ---------------------------------------------------------------------------

def test_run_tag_turn_finalizes_after_last_tag(tmp_path: Path):
    """After _run_tag_turn tags the last pending turn (total == done),
    it must call _finalize_ingestion_if_complete. Covers the REST
    /prepare+/ingest single-turn flow where no legacy thread runs.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)

    finalize_calls: list[str] = []

    def _capture(conv_id):
        finalize_calls.append(conv_id)
        return False  # return value doesn't matter for this assertion

    # Stub engine.tag_turn + TurnTagIndex access so _run_tag_turn's
    # body runs to the point where it would invoke our hook.
    state.engine.tag_turn = MagicMock(return_value=None)
    state.engine._turn_tag_index.entries.append(
        type("Entry", (), {
            "tags": ["t1"], "primary_tag": "t1",
            "message_hash": "",
        })()
    )

    with patch.object(
        state, "_finalize_ingestion_if_complete", side_effect=_capture,
    ):
        state._run_tag_turn(
            history=[], payload_tokens=0, turn_id="",
            reserved_turn=0, message_hash="",
        )

    assert finalize_calls, (
        "_run_tag_turn must call _finalize_ingestion_if_complete after "
        "successful tagging — REST single-turn flow depends on this"
    )
    assert finalize_calls[0] == state.engine.config.conversation_id


# ---------------------------------------------------------------------------
# Helper idempotency + precondition checks
# ---------------------------------------------------------------------------

def test_finalize_no_op_when_phase_not_ingesting(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Default phase is 'init' — leave it there.
    assert state._finalize_ingestion_if_complete(conv) is False


def test_finalize_no_op_when_no_running_episode(tmp_path: Path):
    """phase='ingesting' but no episode row — finalize must bail quietly."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    conv = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.set_phase(conversation_id=conv, lifecycle_epoch=1, phase="ingesting")
    # No canonical_turns, no episode → total==done==0 but claim fails.
    assert state._finalize_ingestion_if_complete(conv) is False
    # Phase unchanged.
    assert inner.get_conversation_phase(conv) == "ingesting"
