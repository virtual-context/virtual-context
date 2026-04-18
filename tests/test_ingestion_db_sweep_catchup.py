"""Regression test: ingestion must resume after interruption regardless of
whether the next payload re-includes the abandoned rows.

Invariant under test: any canonical row with ``tagged_at IS NULL`` MUST
end up tagged after ``_run_ingestion_with_catchup`` returns — even when
the payload-driven pass only covers a subset of pending rows. The
payload-driven pass handles messages actually present in the incoming
payloads; the ``_tagger_run`` DB-sweep that follows closes the gap for
any canonical rows left untagged from prior interrupted batches.

Without the sweep, a large-batch ingest interrupted by a container
restart (or any other failure) followed by windowed-only follow-ups
would leave an arbitrary middle chunk of canonical rows with
``tagged_at IS NULL`` forever — phase stays ``ingesting`` and the
dashboard progress never reaches 100%.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso


def _seed_canonical_row(inner, conv_id, canonical_id, sort_key, *, tagged=False, user_text="u", assistant_text="a"):
    """Seed a canonical row directly — matches the test_tagger_loop pattern."""
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
            ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, 'b', ?, ?, 1, ?, ?, ?)
        """, (
            canonical_id, conv_id, f"h_{canonical_id}", user_text, assistant_text,
            user_text, assistant_text, sort_key, now, now,
            now if tagged else None, now, now,
        ))


def _prep_episode(inner, conv_id, worker_id):
    """Stand up an ingestion_episode + claim lease + flip phase to ingesting."""
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")


def _count_untagged(inner, conv_id) -> int:
    with inner._get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id=? AND tagged_at IS NULL",
            (conv_id,),
        ).fetchone()
        return int(row[0])


def test_run_ingestion_with_catchup_sweeps_untagged_rows_from_prior_batch(tmp_path: Path, monkeypatch):
    """Scenario: a prior large payload left N untagged rows in the DB (bulk
    ingestion was interrupted by container restart). A new small windowed
    payload arrives; ``_run_ingestion_with_catchup`` runs with only the
    windowed messages. After it returns, **every** untagged row — including
    the ones NOT in the new payload — must be tagged.

    Pre-fix, this test fails: the payload-driven pass only tags the N
    messages in ``initial_messages``; the other rows stay untagged forever.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    # Seed 10 untagged rows simulating the abandoned middle-chunk of a
    # prior interrupted batch. Each row carries user_text + assistant_text
    # that the tagger can generate tags from.
    for i in range(10):
        _seed_canonical_row(
            inner, conv_id, f"orphan_{i}", float((i + 1) * 1000),
            user_text=f"user msg {i}",
            assistant_text=f"assistant reply {i}",
        )
    _prep_episode(inner, conv_id, state._worker_id)

    assert _count_untagged(inner, conv_id) == 10, "precondition: 10 untagged rows"

    # Stub the tag generator so we don't need real LLM calls.
    from virtual_context.types import TagResult

    def _fake_generate_tags(self, combined_text, store_tags, context_turns=None):
        return TagResult(
            tags=["test-tag"], primary="test-tag", source="stub",
        )
    monkeypatch.setattr(
        "virtual_context.core.tag_generator.TagGenerator.generate_tags",
        _fake_generate_tags,
    )

    # Invoke the production path with an EMPTY payload — simulates the
    # resume case where the windowed follow-up has no new content but we
    # still need to sweep the DB for untagged rows from prior batches.
    state._run_ingestion_with_catchup(
        initial_messages=[],
        baseline=0,
        cumulative_total=0,
    )

    remaining = _count_untagged(inner, conv_id)
    assert remaining == 0, (
        f"DB-sweep regression: {remaining} canonical rows are still "
        "untagged after _run_ingestion_with_catchup returned. This means "
        "the payload-driven tagger skipped rows outside the payload and "
        "the _tagger_run DB sweep did not fire — a conversation whose "
        "bulk ingest was interrupted would stay wedged below 100% forever."
    )


def test_db_sweep_runs_when_legacy_pair_tagger_errors(tmp_path: Path, monkeypatch):
    """Scenario: legacy pair-based tagger raises mid-run (e.g. strict
    canonical alignment failure on a conversation with orphan halves from
    a prior crash). The row-based ``_tagger_run`` DB sweep MUST still fire
    and drain the untagged queue — otherwise the episode stays wedged at
    'ingesting' forever and every subsequent POST re-hits the same legacy
    failure without making progress.

    Pre-fix, the sweep only ran when the legacy path completed cleanly, so
    a single strict-canonical RuntimeError would leave the episode stuck
    until the DB state was manually repaired.
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    for i in range(5):
        _seed_canonical_row(
            inner, conv_id, f"orphan_{i}", float((i + 1) * 1000),
            user_text=f"user msg {i}",
            assistant_text=f"assistant reply {i}",
        )
    _prep_episode(inner, conv_id, state._worker_id)

    from virtual_context.types import TagResult

    def _fake_generate_tags(self, combined_text, store_tags, context_turns=None):
        return TagResult(tags=["test-tag"], primary="test-tag", source="stub")
    monkeypatch.setattr(
        "virtual_context.core.tag_generator.TagGenerator.generate_tags",
        _fake_generate_tags,
    )

    def _legacy_boom(self, *args, **kwargs):
        raise RuntimeError(
            "strict canonical tagging could not map payload messages to "
            "existing rows for logical turn 10"
        )
    monkeypatch.setattr(
        type(state), "_ingest_messages_with_progress", _legacy_boom,
    )

    state._run_ingestion_with_catchup(
        initial_messages=[],
        baseline=0,
        cumulative_total=0,
    )

    remaining = _count_untagged(inner, conv_id)
    assert remaining == 0, (
        f"Row-based DB sweep regression: legacy pair-based tagger raised "
        f"but {remaining} canonical rows are still untagged. The sweep "
        "must run unconditionally after the legacy path (unless the pass "
        "was explicitly cancelled) so a strict-canonical alignment failure "
        "cannot wedge the episode forever."
    )


def test_db_sweep_skips_when_payload_pass_was_cancelled(tmp_path: Path):
    """If the payload-driven pass was cancelled by a racing new request,
    the DB sweep MUST NOT fire on this thread — the racing thread will do
    its own sweep. Otherwise two concurrent sweeps race on the same lease.

    Pin: set ``_ingestion_cancel`` before invocation; ensure the sweep
    doesn't run (detected by ``completed_cleanly = False``).
    """
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)

    for i in range(3):
        _seed_canonical_row(
            inner, conv_id, f"row_{i}", float((i + 1) * 1000),
        )
    _prep_episode(inner, conv_id, state._worker_id)

    state._ingestion_cancel.set()  # simulate a racing takeover
    state._run_ingestion_with_catchup(
        initial_messages=[],
        baseline=0,
        cumulative_total=0,
    )

    # Rows must remain untagged — sweep was suppressed by the cancel flag.
    remaining = _count_untagged(inner, conv_id)
    assert remaining == 3, (
        "Cancelled pass must NOT perform a DB sweep — the racing takeover "
        "thread owns the lease now and will run its own sweep. Running "
        "concurrently here would violate single-owner tagging invariant. "
        f"Found {remaining} still untagged (expected 3)."
    )
