"""Task A23: ProxyState.handle_prepare_payload — steps 1-3 + step 5 metadata.

These tests exercise the minimal shell of the new ingestion flow:

* Entry epoch verify (``verify_epoch``).
* Canonical row persistence via ``engine._ingest_reconciler.ingest_batch``.
* Defense-in-depth epoch verify.
* Per-request metadata update via ``store.update_request_metadata``.
* Return ``PhaseDecision(phase='init', started_tagger=False)``.

Tasks A24-A29 layer on top of this; A30 removes the legacy helpers.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.proxy.formats import (
    detect_format,
    extract_ingestible_messages,
    summarize_payload_accounting,
)
from virtual_context.proxy.state import PhaseDecision, ProxyState


def _make_proxy_state(tmp_path: Path, conversation_id: str = "c") -> ProxyState:
    """Construct a minimal ProxyState backed by a real SQLite engine.

    Mirrors ``tests/test_engine_lifecycle_epoch.py::_make_test_engine`` and
    wraps the resulting engine in a bare ``ProxyState`` (no metrics, no
    upstream URL) so unit assertions touch the real store/epoch paths.
    """
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import (
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )

    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / f"{conversation_id}.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    engine = VirtualContextEngine(config=config)
    return ProxyState(engine)


def _inner_store(engine):
    """Return the concrete SQLite store underneath the wrappers.

    Mirrors the helper in ``tests/test_engine_lifecycle_epoch.py``.
    """
    store = engine._store
    inner = getattr(store, "_store", None)
    if inner is None:
        return store
    segments = getattr(inner, "_segments", None)
    if segments is not None:
        return segments
    return inner


def test_handle_prepare_payload_updates_request_metadata(tmp_path):
    """Step 5 — ``last_raw_payload_entries`` / ``last_ingestible_payload_entries``
    are overwritten on the conversations row."""
    state = _make_proxy_state(tmp_path)
    try:
        state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 500,
                "ingestible_entry_count": 200,
            },
        )
        snap = state.engine._store.read_progress_snapshot(
            state.engine.config.conversation_id,
        )
        assert snap.last_raw_payload_entries == 500
        assert snap.last_ingestible_payload_entries == 200
    finally:
        state.engine.close()


def test_handle_prepare_payload_verifies_epoch_at_entry(tmp_path):
    """Step 2 — entry verify fires BEFORE any DB write.

    External resurrect bumps the DB epoch. The engine still holds the
    stale epoch in memory; the entry ``verify_epoch`` must reject it.
    """
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.mark_conversation_deleted(conv_id)
        inner.increment_lifecycle_epoch_on_resurrect(conv_id)
        with pytest.raises(LifecycleEpochMismatch):
            state.handle_prepare_payload(
                body={},
                payload_accounting={
                    "raw_payload_entry_count": 0,
                    "ingestible_entry_count": 0,
                },
            )
    finally:
        state.engine.close()


def test_handle_prepare_payload_persists_canonical_rows(tmp_path):
    """Step 3 — IngestReconciler writes canonical rows for a fresh payload."""
    state = _make_proxy_state(tmp_path)
    try:
        state.handle_prepare_payload(
            body={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            payload_accounting={
                "raw_payload_entry_count": 2,
                "ingestible_entry_count": 2,
            },
        )
        conv_id = state.engine.config.conversation_id
        rows = state.engine._store.get_all_canonical_turns(conv_id)
        assert len(rows) >= 1, "at least one canonical row was persisted"
    finally:
        state.engine.close()


def test_proxy_ingest_history_keeps_total_fixed_for_single_prepare_payload(tmp_path):
    """A single payload should not widen its own denominator while tagging.

    ``handle_prepare_payload`` persists canonical rows up front. The proxy's
    follow-up bulk ingester must only enrich/tag those rows, not append new
    canonical rows while it works.
    """
    state = _make_proxy_state(tmp_path)
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "part one"},
            {"role": "assistant", "content": "part two"},
        ],
    }
    try:
        fmt = detect_format(body)
        payload_accounting = summarize_payload_accounting(body, fmt)
        messages, _ = extract_ingestible_messages(body, fmt)

        state.handle_prepare_payload(body=body, payload_accounting=payload_accounting)
        conv_id = state.engine.config.conversation_id
        before = state.engine._store.read_progress_snapshot(conv_id)
        assert before.total_ingestible == 3
        assert before.done_ingestible == 0

        ingested = state.engine.ingest_history(
            messages,
            require_existing_canonical=True,
            expected_lifecycle_epoch=state.engine._engine_state.lifecycle_epoch,
        )
        after = state.engine._store.read_progress_snapshot(conv_id)

        assert ingested == 2
        assert after.total_ingestible == before.total_ingestible
        assert after.done_ingestible == before.total_ingestible
    finally:
        state.engine.close()


def test_handle_prepare_payload_returns_phase_decision(tmp_path):
    """The shell returns a ``PhaseDecision``. An empty init conversation
    with no untagged rows transitions to ``active`` via step 5.5."""
    state = _make_proxy_state(tmp_path)
    try:
        decision = state.handle_prepare_payload(
            body={},
            payload_accounting={
                "raw_payload_entry_count": 0,
                "ingestible_entry_count": 0,
            },
        )
        assert isinstance(decision, PhaseDecision)
        # Step 5.5 transitions init → active when total == done == 0.
        assert decision.phase == "active"
        assert decision.started_tagger is False
    finally:
        state.engine.close()


# ---------------------------------------------------------------------------
# Task A24 — phase gate (step 4)
# ---------------------------------------------------------------------------


def test_phase_gate_deleted_resurrects(tmp_path):
    """``phase == 'deleted'`` resurrects the conversation — lifecycle_epoch
    bumps and the engine's in-memory epoch is updated to match. Step 5.5
    then transitions the resurrected-empty conversation to ``active``."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.mark_conversation_deleted(conv_id)
        # Engine's cached epoch is still 1; DB has phase='deleted', epoch=1.
        state.handle_prepare_payload(
            body={},
            payload_accounting={
                "raw_payload_entry_count": 0,
                "ingestible_entry_count": 0,
            },
        )
        # Resurrect bumped to epoch=2; engine cache updated.
        assert state.engine._engine_state.lifecycle_epoch == 2
        assert inner.get_lifecycle_epoch(conv_id) == 2
        snap = inner.read_progress_snapshot(conv_id)
        # Resurrect resets phase to 'init'; step 5.5 sees total==done==0 and
        # moves the empty conversation on to 'active'.
        assert snap.phase == "active"
    finally:
        state.engine.close()


def test_phase_gate_compacting_widens_pending_and_returns(tmp_path):
    """``phase == 'compacting'`` widens pending_raw (epoch-scoped) and
    returns a ``PhaseDecision(phase='compacting', started_tagger=False)``
    without triggering episode creation."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        # Force phase='compacting'.
        inner.set_phase(
            conversation_id=conv_id, lifecycle_epoch=1, phase="compacting",
        )
        decision = state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 1000,
                "ingestible_entry_count": 500,
            },
        )
        assert decision.phase == "compacting"
        assert decision.started_tagger is False
        # pending_raw should reflect the new_raw widening.
        with inner._get_conn() as conn:
            pending = conn.execute(
                "SELECT pending_raw_payload_entries FROM conversations "
                "WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
        assert pending == 1000
        # Per-request metadata still updated (step 5 runs before step 4).
        snap = inner.read_progress_snapshot(conv_id)
        assert snap.last_raw_payload_entries == 1000
        assert snap.last_ingestible_payload_entries == 500
    finally:
        state.engine.close()


def test_phase_gate_ingesting_falls_through(tmp_path):
    """``phase == 'ingesting'`` does not widen pending or resurrect; the
    gate returns a stub ``PhaseDecision`` for now (A25-A29 will extend)."""
    state = _make_proxy_state(tmp_path)
    try:
        conv_id = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        inner.set_phase(
            conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting",
        )
        decision = state.handle_prepare_payload(
            body={"messages": [{"role": "user", "content": "hi"}]},
            payload_accounting={
                "raw_payload_entry_count": 1,
                "ingestible_entry_count": 1,
            },
        )
        # Stub for now — subsequent tasks refine.
        assert decision.phase == "ingesting"
        # pending should NOT have been bumped.
        with inner._get_conn() as conn:
            pending = conn.execute(
                "SELECT pending_raw_payload_entries FROM conversations "
                "WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
        assert pending == 0
    finally:
        state.engine.close()


# ---------------------------------------------------------------------------
# Task A25 — step 5.5 (derive progress + transition phase)
# ---------------------------------------------------------------------------


def test_step_5_5_transitions_init_to_ingesting_when_untagged_work_exists(tmp_path):
    """Empty init conversation receives a payload with content →
    canonical rows land, derived total > done, phase transitions to 'ingesting'."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    decision = state.handle_prepare_payload(
        body={"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]},
        payload_accounting={"raw_payload_entry_count": 2, "ingestible_entry_count": 2},
    )
    # Phase transitioned to ingesting because total > done.
    assert decision.phase == "ingesting"
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "ingesting"


def test_step_5_5_empty_init_transitions_to_active(tmp_path):
    """Init conversation with empty payload (no canonical rows) →
    total == done == 0 → init transitions to active, no ingestion."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    assert decision.phase == "active"
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "active"


def test_step_5_5_active_with_no_new_work_stays_active(tmp_path):
    """Active conversation receives a resend (same content, no new turns) →
    total == done → stays active."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Simulate a fully-tagged conversation.
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute("""
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES ('t0', ?, 'h0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b', ?, ?, 1, ?, ?, ?)
        """, (conv_id, now, now, now, now, now))
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="active")
    decision = state.handle_prepare_payload(
        body={},  # no new content
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    assert decision.phase == "active"
    assert decision.started_tagger is False
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "active"


def test_step_5_5_ingesting_stays_ingesting_when_work_remains(tmp_path):
    """Already-ingesting conversation with more untagged rows → stays ingesting."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed untagged canonical rows + force phase='ingesting'.
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    with inner._get_conn() as conn:
        for i in range(3):
            conn.execute("""
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, NULL, ?, ?)
            """, (f"t{i}", conv_id, f"h{i}", float((i + 1) * 1000), now, now, now, now))
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    assert decision.phase == "ingesting"
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.phase == "ingesting"


# ---------------------------------------------------------------------------
# Task A26 — step 6 (upsert episode + attempt lease claim)
# ---------------------------------------------------------------------------


def test_step_6_creates_episode_and_claims_lease_on_empty(tmp_path):
    """Fresh init conversation with payload content → phase transitions to
    ingesting, episode row created, lease claimed, tagger spawned."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    decision = state.handle_prepare_payload(
        body={"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]},
        payload_accounting={
            "raw_payload_entry_count": 500,
            "ingestible_entry_count": 200,
        },
    )
    assert decision.phase == "ingesting"
    assert decision.started_tagger is True
    # Episode row exists with our worker as owner.
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.active_episode is not None
    assert snap.active_episode.owner_worker_id == state._worker_id
    assert snap.active_episode.raw_payload_entries == 500


def test_step_6_does_not_claim_when_fresh_owner(tmp_path):
    """If a different worker already holds a fresh lease, this worker
    widens raw but does NOT claim."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    # Seed canonical rows so total>done.
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute("""
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES ('t0', ?, 'h0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b', ?, ?, 1, NULL, ?, ?)
        """, (conv_id, now, now, now, now))
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    # Another worker already owns the lease at epoch=1 with raw=100.
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id="other_worker", raw_payload_entries=100,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id="other_worker", lease_ttl_s=30.0,
    )
    # Our call: widens raw, fails to claim (other_worker has fresh lease).
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={
            "raw_payload_entry_count": 200,
            "ingestible_entry_count": 50,
        },
    )
    assert decision.phase == "ingesting"
    assert decision.started_tagger is False  # didn't claim
    # Raw widened via GREATEST/MAX but owner unchanged.
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.active_episode.raw_payload_entries == 200  # widened
    assert snap.active_episode.owner_worker_id == "other_worker"  # unchanged


def test_step_6_reclaims_stale_lease(tmp_path):
    """If previous owner's lease is stale (old heartbeat), this worker takes over."""
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    from virtual_context.core.canonical_turns import utcnow_iso
    now = utcnow_iso()
    with inner._get_conn() as conn:
        conn.execute("""
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at,
                created_at, updated_at
            ) VALUES ('t0', ?, 'h0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b', ?, ?, 1, NULL, ?, ?)
        """, (conv_id, now, now, now, now))
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id="dead_worker", raw_payload_entries=100,
    )
    # Force stale heartbeat.
    with inner._get_conn() as conn:
        conn.execute(
            "UPDATE ingestion_episode SET heartbeat_ts = '2000-01-01T00:00:00+00:00' WHERE conversation_id = ?",
            (conv_id,),
        )
    decision = state.handle_prepare_payload(
        body={},
        payload_accounting={"raw_payload_entry_count": 0, "ingestible_entry_count": 0},
    )
    assert decision.started_tagger is True  # reclaimed
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.active_episode.owner_worker_id == state._worker_id


# ---------------------------------------------------------------------------
# Wiring — server.py::prepare_payload must invoke state.handle_prepare_payload
# ---------------------------------------------------------------------------


def test_prepare_payload_calls_handle_prepare_payload(tmp_path):
    """Verify the integration point: ``server.py::prepare_payload`` calls
    ``state.handle_prepare_payload`` when a valid state is provided, and the
    new flow transitions the conversation into the expected phase."""
    import asyncio

    from virtual_context.proxy.formats import detect_format
    from virtual_context.proxy.metrics import ProxyMetrics
    from virtual_context.proxy.server import prepare_payload

    state = _make_proxy_state(tmp_path)
    try:
        original = state.handle_prepare_payload
        calls: list[tuple[tuple, dict]] = []

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        state.handle_prepare_payload = spy  # type: ignore[assignment]
        body = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        fmt = detect_format(body)
        body_bytes = (
            b'{"model":"claude-opus-4-6",'
            b'"messages":[{"role":"user","content":"hi"},'
            b'{"role":"assistant","content":"hello"}]}'
        )
        asyncio.run(
            prepare_payload(
                body,
                state,
                fmt,
                ProxyMetrics(),
                body_bytes=body_bytes,
                inbound_conversation_id=state.engine.config.conversation_id,
            )
        )
        assert len(calls) == 1, (
            f"expected exactly one handle_prepare_payload call, got {len(calls)}"
        )
        # Verify new flow fired: phase transitioned to ingesting (total>done)
        # or active (empty). A fresh conversation with 2 new messages and no
        # tagged rows should land in 'ingesting'.
        snap = state.engine._store.read_progress_snapshot(
            state.engine.config.conversation_id,
        )
        assert snap.phase in ("ingesting", "active"), (
            f"unexpected phase after wired handle_prepare_payload: {snap.phase}"
        )
    finally:
        state.engine.close()


# ---------------------------------------------------------------------------
# P1 #1 (redux) — handle_prepare_payload stops spawning the tagger thread;
# the legacy ``start_ingestion_if_needed`` path is the authoritative tagger.
# ---------------------------------------------------------------------------


def test_handle_prepare_payload_does_not_spawn_tagger_thread(tmp_path):
    """``handle_prepare_payload`` must NOT call ``_spawn_tagger_thread`` —
    tagger dispatch is delegated to ``start_ingestion_if_needed`` so that
    only one tagger thread races on a given conversation's canonical rows
    (the legacy per-pair ``tag_turn`` path is the authoritative tagger).

    The counter-based instrumentation also exercises the companion path:
    ``start_ingestion_if_needed`` starts its own background thread
    (``_ingestion_thread``) without calling ``_spawn_tagger_thread``.
    """
    from virtual_context.types import Message

    state = _make_proxy_state(tmp_path)
    try:
        # Instrument _spawn_tagger_thread with a counter. Preserve the
        # callable signature so that any accidental invocation is caught
        # — but do not actually spawn a thread.
        tagger_spawn_count = {"count": 0}

        def counting_spawn_tagger_thread() -> None:
            tagger_spawn_count["count"] += 1

        state._spawn_tagger_thread = counting_spawn_tagger_thread  # type: ignore[assignment]

        # Drive handle_prepare_payload through the hottest branch: content
        # present, phase transitions to 'ingesting', lease claimed. Before
        # the P1 #1 (redux) fix this branch unconditionally called
        # ``_spawn_tagger_thread``.
        decision = state.handle_prepare_payload(
            body={"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]},
            payload_accounting={
                "raw_payload_entry_count": 2,
                "ingestible_entry_count": 2,
            },
        )
        assert decision.phase == "ingesting"
        # ``started_tagger=True`` now means "lease was claimed", not
        # "thread was literally spawned". The actual thread is launched
        # by the legacy ``start_ingestion_if_needed`` path downstream.
        assert decision.started_tagger is True
        assert tagger_spawn_count["count"] == 0, (
            "handle_prepare_payload must NOT spawn the tagger thread — "
            "tagger dispatch is delegated to start_ingestion_if_needed "
            f"(got spawn count={tagger_spawn_count['count']})"
        )

        # Stub the background ingestion loop so the thread exits
        # immediately without running the real tagging pipeline.
        state._run_ingestion_with_catchup = lambda *a, **kw: None  # type: ignore[assignment]

        # Now call the legacy path — it must start the ingestion thread
        # (count == 1) and must NOT call _spawn_tagger_thread either.
        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="world"),
        ]
        state.start_ingestion_if_needed(history)
        assert state._ingestion_thread is not None, (
            "start_ingestion_if_needed must create _ingestion_thread"
        )
        # Join the (no-op) thread so we assert on a stable post-start state
        # rather than a liveness race. ``_run_ingestion_with_catchup`` was
        # stubbed to return immediately.
        state._ingestion_thread.join(timeout=2.0)
        # The legacy tagger path does NOT invoke the new per-row tagger
        # spawn helper — the counter stays at zero.
        assert tagger_spawn_count["count"] == 0, (
            "start_ingestion_if_needed must NOT call _spawn_tagger_thread "
            f"(got spawn count={tagger_spawn_count['count']})"
        )
    finally:
        state.engine.close()
