"""Task A34 — Multi-worker sliding-window integration test.

This test exercises the specific scenario the user cared about in planning:

1. Payload 1: Worker A submits an initial batch of canonical turns.
2. Worker A claims the lease and starts tagging.
3. Mid-tagging, Worker B arrives with a sliding-window payload
   (overlap + a few new turns at the tail).
4. Worker B widens ``raw_payload_entries`` (GREATEST) and new canonical
   rows land via IngestReconciler alignment.
5. Worker C arrives with the next sliding window (more tail).
6. Worker A keeps the lease through all of it (the partial unique index
   on ``(conversation_id, lifecycle_epoch) WHERE status='running'``
   prevents any hand-off until the lease lapses) and eventually tags
   everything.

Key invariants:
  - Only Worker A holds the tagger lease.
  - Workers B/C widen ``raw_payload_entries`` but do NOT change ownership.
  - Derived ``total_ingestible`` grows monotonically as new canonical rows
    land; ``done_ingestible`` converges on total as Worker A tags.
  - No "Save rejected" errors at the store layer.

The widenings and new canonical-row inserts that Workers B/C would do
are exercised at the store level in this test: spawning two real
``ProxyState`` instances over the same SQLite file is not supported by
``_make_proxy_state``, and is orthogonal to what this integration test
is checking (the store-level single-owner invariant + monotonic
progress aggregation).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virtual_context.core.canonical_turns import utcnow_iso


def _seed_canonical_row(
    inner,
    conv_id: str,
    canonical_id: str,
    sort_key: float,
    *,
    tagged: bool = False,
) -> None:
    """Insert a single canonical_turns row. Each row counts for one
    ingestible entry (``covered_ingestible_entries = 1``), so seeding
    N rows derives ``total_ingestible = N``.
    """
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
            """,
            (
                canonical_id, conv_id, f"h_{canonical_id}", sort_key,
                now, now, now if tagged else None, now, now,
            ),
        )


def test_sliding_window_three_payloads_monotonic_progress(tmp_path):
    """Scenario:
      1. Worker A submits initial 100 turns -> phase becomes 'ingesting',
         Worker A claims the lease.
      2. Worker A tags 50 rows (simulating mid-flight progress).
      3. Worker B arrives with a sliding-window payload (turns 90-101 =
         12 entries; 10 overlap + 2 new). Worker B widens raw and two new
         canonical rows land. Worker A keeps the lease.
      4. Worker C arrives with the next sliding window (turns 91-102).
         One more canonical row lands. Worker A still owns the lease.
      5. Worker A's tagger runs to completion and clears everything.

    Assertions check monotonic growth of ``total_ingestible``, stable
    ownership, convergence of ``done_ingestible`` on total, and the
    final transition back to ``phase='active'`` with no active episode.
    """
    from tests.test_handle_prepare_payload import _inner_store, _make_proxy_state

    # --- Setup: Worker A owns the conversation. ---
    worker_a = _make_proxy_state(tmp_path)
    try:
        conv_id = worker_a.engine.config.conversation_id
        inner = _inner_store(worker_a.engine)

        # Simulate Payload 1: 100 canonical rows already persisted
        # (untagged). Seeding directly is faster than going through
        # IngestReconciler for this test and lets us control sort keys.
        for i in range(100):
            _seed_canonical_row(inner, conv_id, f"t{i:04d}", float((i + 1) * 1000))

        # Worker A claims the lease (mirrors step 6 of handle_prepare_payload).
        inner.upsert_ingestion_episode(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id=worker_a._worker_id, raw_payload_entries=100,
        )
        claimed = inner.claim_ingestion_lease(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id=worker_a._worker_id, lease_ttl_s=30.0,
        )
        assert claimed is True
        inner.set_phase(
            conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting",
        )

        # Initial state — total=100, done=0.
        snap = inner.read_progress_snapshot(conv_id)
        assert snap.total_ingestible == 100
        assert snap.done_ingestible == 0
        assert snap.phase == "ingesting"
        assert snap.active_episode is not None
        assert snap.active_episode.owner_worker_id == worker_a._worker_id
        assert snap.active_episode.raw_payload_entries == 100

        # --- Worker A tags 50 rows (simulating partial progress). ---
        for i in range(50):
            marked = inner.mark_canonical_row_tagged(
                canonical_turn_id=f"t{i:04d}",
                conversation_id=conv_id,
                expected_lifecycle_epoch=1,
            )
            assert marked is True

        snap_mid = inner.read_progress_snapshot(conv_id)
        assert snap_mid.total_ingestible == 100
        assert snap_mid.done_ingestible == 50
        assert snap_mid.active_episode.owner_worker_id == worker_a._worker_id

        # --- Worker B arrives with sliding-window payload
        # (turns 90-101 = 12 entries; 10 overlap + 2 new). ---
        # Two new canonical rows land (t0100, t0101); the reconciler
        # would dedupe overlap via turn_hash. Worker B then attempts to
        # widen raw (12) — GREATEST keeps it at the existing 100.
        for i in range(100, 102):
            _seed_canonical_row(inner, conv_id, f"t{i:04d}", float((i + 1) * 1000))
        inner.upsert_ingestion_episode(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_b", raw_payload_entries=12,
        )
        # Worker B tries to claim — Worker A still holds a fresh lease,
        # so claim must fail and ownership stays with Worker A.
        b_claim = inner.claim_ingestion_lease(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_b", lease_ttl_s=30.0,
        )
        assert b_claim is False

        snap_after_b = inner.read_progress_snapshot(conv_id)
        assert snap_after_b.total_ingestible == 102  # derived grew from 100 -> 102
        assert snap_after_b.done_ingestible == 50    # unchanged
        assert snap_after_b.active_episode.owner_worker_id == worker_a._worker_id
        # upsert keeps MAX(100, 12) = 100 (sliding-window payload is
        # smaller than the accumulated raw).
        assert snap_after_b.active_episode.raw_payload_entries == 100

        # --- Worker C arrives with next sliding window (turns 91-102). ---
        # One more new canonical row lands (t0102).
        _seed_canonical_row(inner, conv_id, "t0102", float(103 * 1000))
        inner.upsert_ingestion_episode(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_c", raw_payload_entries=12,
        )
        c_claim = inner.claim_ingestion_lease(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_c", lease_ttl_s=30.0,
        )
        assert c_claim is False

        snap_after_c = inner.read_progress_snapshot(conv_id)
        assert snap_after_c.total_ingestible == 103
        assert snap_after_c.done_ingestible == 50
        assert snap_after_c.active_episode.owner_worker_id == worker_a._worker_id

        # Monotonic total check across the three snapshots.
        assert (
            snap.total_ingestible
            <= snap_mid.total_ingestible
            <= snap_after_b.total_ingestible
            <= snap_after_c.total_ingestible
        )

        # --- Worker A's tagger runs synchronously to completion. ---
        worker_a._tagger_run()

        snap_final = inner.read_progress_snapshot(conv_id)
        assert snap_final.total_ingestible == 103
        assert snap_final.done_ingestible == 103
        assert snap_final.phase == "active"
        assert snap_final.active_episode is None  # episode flipped to 'completed'
    finally:
        worker_a.engine.close()


def test_sliding_window_non_owner_widens_but_does_not_claim(tmp_path):
    """Store-level invariant: while Worker A holds a fresh lease, any
    other worker's ``claim_ingestion_lease`` call must fail even though
    its ``upsert_ingestion_episode`` successfully widens raw.

    This is the single-owner guarantee that keeps sliding-window
    payloads from racing Worker A off the tagger role mid-ingestion.
    """
    from tests.test_handle_prepare_payload import _inner_store, _make_proxy_state

    worker_a = _make_proxy_state(tmp_path)
    try:
        conv_id = worker_a.engine.config.conversation_id
        inner = _inner_store(worker_a.engine)

        # Worker A claims via handle_prepare_payload (step 6 spawns tagger).
        decision_a = worker_a.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 10,
                "ingestible_entry_count": 5,
            },
        )
        assert decision_a.started_tagger is True

        # A second worker hitting the same store: widen raw (MAX(raw, 200) = 200)
        # but attempt-claim must fail because Worker A's heartbeat is fresh.
        inner.upsert_ingestion_episode(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_b", raw_payload_entries=200,
        )
        claim_ok = inner.claim_ingestion_lease(
            conversation_id=conv_id, lifecycle_epoch=1,
            worker_id="worker_b", lease_ttl_s=30.0,
        )
        assert claim_ok is False

        snap = inner.read_progress_snapshot(conv_id)
        assert snap.active_episode is not None
        # Worker A still owns.
        assert snap.active_episode.owner_worker_id == worker_a._worker_id
        # Raw widened via GREATEST/MAX.
        assert snap.active_episode.raw_payload_entries == 200
    finally:
        worker_a.engine.close()
