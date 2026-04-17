"""Task A27: ProxyState._tagger_run loop with 4 boundary verify_epoch checks.

The tagger runs as a daemon thread. Each iteration re-verifies the lifecycle
epoch before every DB write and passes ``my_epoch`` (captured at spawn) to
every store call so a stale tagger cannot touch a newer lifecycle's rows
even if ``verify_epoch`` races past a delete+resurrect.

These tests exercise ``_tagger_run`` synchronously on the caller's thread so
asserts can observe completion.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import utcnow_iso


def _seed_canonical_row(inner, conv_id, canonical_id, sort_key, tagged=False):
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
            ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, ?, ?, ?)
        """, (canonical_id, conv_id, f"h_{canonical_id}", sort_key, now, now, now if tagged else None, now, now))


def test_tagger_tags_untagged_rows_until_empty(tmp_path: Path):
    """Full happy path: 5 untagged rows → tagger tags all → episode completes."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    for i in range(5):
        _seed_canonical_row(inner, conv_id, f"t{i}", float((i + 1) * 1000))
    # Set up ingestion episode + claim lease ourselves (simulating post-A26 state).
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")

    # Run tagger synchronously.
    state._tagger_run()

    # All rows tagged.
    with inner._get_conn() as conn:
        untagged = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id = ? AND tagged_at IS NULL",
            (conv_id,),
        ).fetchone()[0]
    assert untagged == 0
    # Episode completed, phase transitioned to 'active'.
    snap = inner.read_progress_snapshot(conv_id)
    assert snap.active_episode is None  # completed
    assert snap.phase == "active"


def test_tagger_exits_on_epoch_mismatch(tmp_path: Path):
    """If lifecycle epoch changes mid-loop (resurrect), tagger exits cleanly."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    _seed_canonical_row(inner, conv_id, "t0", 1000.0)
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    # Resurrect: bump DB epoch, but engine's cached epoch is still 1.
    inner.mark_conversation_deleted(conv_id)
    inner.increment_lifecycle_epoch_on_resurrect(conv_id)
    # engine._engine_state.lifecycle_epoch is still 1 — verify_epoch will detect.

    # Running the tagger should NOT raise; should just exit cleanly.
    state._tagger_run()
    # Row should remain untagged (stale tagger couldn't touch new lifecycle).
    with inner._get_conn() as conn:
        row = conn.execute(
            "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id = 't0'"
        ).fetchone()
    assert row[0] is None


def _make_proxy_state_with_keyword_tags(tmp_path: Path):
    """Proxy state where the keyword tagger maps ``court`` → ``legal``.

    Lets us assert real tags were written by ``tag_canonical_row`` rather
    than the default ``_general`` fallback.
    """
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.proxy.state import ProxyState
    from virtual_context.types import (
        KeywordTagConfig,
        StorageConfig,
        TagGeneratorConfig,
        VirtualContextConfig,
    )

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "c.db"),
        ),
        tag_generator=TagGeneratorConfig(
            type="keyword",
            keyword_fallback=KeywordTagConfig(
                tag_keywords={"legal": ["court", "motion"]},
            ),
        ),
    )
    return ProxyState(VirtualContextEngine(config=config))


def test_tagger_writes_real_tags_from_generator(tmp_path: Path):
    """Task F-D: ``_run_tagging_pipeline`` delegates to
    ``TaggingPipeline.tag_canonical_row`` which runs the configured tag
    generator and persists the result. We seed a row whose user_content
    matches the ``legal`` keyword, run the tagger, and assert the row
    carries ``primary_tag='legal'`` + ``tags=['legal']`` afterwards.
    """
    import json

    state = _make_proxy_state_with_keyword_tags(tmp_path)
    conv_id = state.engine.config.conversation_id
    from tests.test_handle_prepare_payload import _inner_store
    inner = _inner_store(state.engine)

    # Seed a single untagged canonical row with content the keyword
    # generator recognizes as ``legal``. Use the raw INSERT helper so the
    # row starts with primary_tag='_general' and tags_json='[]', i.e. the
    # exact shape ``IngestReconciler._prepare_message_row`` produces.
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
            ) VALUES (
                't0', ?, 'h_t0', 1,
                'please file the motion in court', '',
                'please file the motion in court', '',
                1000.0, 'b', ?, ?, 1, NULL, ?, ?
            )
        """, (conv_id, now, now, now, now))

    # Set up the ingestion episode & claim the lease (mirrors live state).
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")

    state._tagger_run()

    # Row must be tagged with ``legal`` (from the keyword mapping), not the
    # default ``_general``, and ``tagged_at`` must have been flipped.
    with inner._get_conn() as conn:
        row = conn.execute(
            "SELECT primary_tag, tags_json, tagged_at"
            " FROM canonical_turns WHERE canonical_turn_id = 't0'"
        ).fetchone()
    assert row["primary_tag"] == "legal"
    assert json.loads(row["tags_json"]) == ["legal"]
    assert row["tagged_at"] is not None


def test_tagger_races_untagged_row_insertion_between_empty_check_and_complete(tmp_path: Path, monkeypatch):
    """Step-7(b) race: tagger fetches empty batch, attempts complete,
    but a new untagged row appears in the window. complete returns False
    (NOT EXISTS guard); tagger loops and tags the new row."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store
    state = _make_proxy_state(tmp_path)
    conv_id = state.engine.config.conversation_id
    inner = _inner_store(state.engine)
    inner.upsert_ingestion_episode(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, raw_payload_entries=0,
    )
    inner.claim_ingestion_lease(
        conversation_id=conv_id, lifecycle_epoch=1,
        worker_id=state._worker_id, lease_ttl_s=30.0,
    )
    inner.set_phase(conversation_id=conv_id, lifecycle_epoch=1, phase="ingesting")
    # Monkeypatch iter_untagged to inject a row on the first empty-batch check.
    orig = state.engine._store.iter_untagged_canonical_rows
    call_count = {"n": 0}
    def spy(**kw):
        call_count["n"] += 1
        result = orig(**kw)
        if call_count["n"] == 1 and not result:
            # First empty batch: inject a row BEFORE returning empty.
            _seed_canonical_row(inner, conv_id, "t_late", 500.0)
        return result
    monkeypatch.setattr(state.engine._store, "iter_untagged_canonical_rows", spy)

    state._tagger_run()
    # The late-inserted row should now be tagged.
    with inner._get_conn() as conn:
        row = conn.execute(
            "SELECT tagged_at FROM canonical_turns WHERE canonical_turn_id = 't_late'"
        ).fetchone()
    assert row[0] is not None
