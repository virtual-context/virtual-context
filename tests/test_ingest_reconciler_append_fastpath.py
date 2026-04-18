"""Tests for the append-path optimisation in IngestReconciler.

Production measurement on 2026-04-18 02:01-02:04:
```
PREP_BREAKDOWN conv=77f110fc-0c0 turn=0 format=openai msgs=5780
  payload=28688.5KB in=15203947t out=23979t total=177193.3ms
  handle_prepare_payload=164392.7ms
  inbound_token_count=342.9ms
```

On a 5780-message append-only payload (5777 unchanged prefix + 3 new tail),
``_ingest_prepared_turns_locked`` was issuing 5780 canonical-row UPDATEs —
one per payload message — even though the hash alignment proves 5777 of
them are canonically correct already. At ~28 ms per UPDATE under DB
contention with the background bulk tagger, that was 164 seconds of
wasted work on every incremental prepare.

Fix: skip ``_write_turn`` for overlap rows in the aligned-merge branch.
Only the real novelty (prefix / suffix) is written. These tests pin:

* **fastpath count** — append of 3 new on top of 100 existing writes 3,
  not 103.
* **exact resend writes zero** — no new turns → no DB writes.
* **rollback is still surgical** — overlap rows keep their ORIGINAL
  ``source_batch_id`` so epoch-mismatch rollback on the new batch
  removes only the 3 new rows, never the unchanged prefix.
* **turns_matched accounting unchanged** — downstream bookkeeping that
  relies on the CanonicalIngestResult counts must still see
  ``turns_matched == overlap_len``.
* **rows_touched still carries the full canonical set** — consumers that
  iterate ``result.rows`` (e.g. the tagger's pre-persisted row lookup)
  still see overlap rows with the correct canonical_turn_id / sort_key.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore


def _fmt():
    from virtual_context.proxy.formats import detect_format
    return detect_format({"messages": []})


def _reconciler(store: SQLiteStore) -> IngestReconciler:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _pair_payload(n_pairs: int) -> dict:
    """Build a stable synthetic payload of ``n_pairs`` user/assistant pairs.

    Content is deterministic so append-only extensions produce identical
    hash alignment against the previously-ingested prefix.
    """
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return {"messages": msgs}


class _WriteCountingStore:
    """Wrap a SQLiteStore and count save_canonical_turn invocations.

    We use delegation (not a MagicMock) so the store's actual logic runs
    and we test the real append-path against a real DB.
    """

    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner
        self.save_count = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def save_canonical_turn(self, *args, **kwargs):
        self.save_count += 1
        return self._inner.save_canonical_turn(*args, **kwargs)


def test_append_path_writes_only_new_suffix_rows(tmp_path: Path):
    """Append of 3 new pairs on top of 100 existing must issue only
    writes for the 3 new rows (6 canonical rows: user+assistant per pair),
    not the full 103 pairs / 206 rows. This is the fix for the 164-second
    incident.
    """
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    wrapped = _WriteCountingStore(store)
    rec = _reconciler(store)
    rec._store = wrapped  # route writes through the counter

    # Seed: 100 pairs.
    seed_payload = _pair_payload(100)
    rec.ingest_batch(
        conversation_id="c", body=seed_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    seed_writes = wrapped.save_count
    assert seed_writes >= 100, (
        "Seed ingest must persist all new rows — the optimisation only "
        f"kicks in for OVERLAP rows, not a fresh ingest. Got {seed_writes}."
    )

    # Incremental: 100 existing + 3 new pairs.
    wrapped.save_count = 0
    incremental_payload = _pair_payload(103)
    result = rec.ingest_batch(
        conversation_id="c", body=incremental_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    incremental_writes = wrapped.save_count

    # We expect writes ONLY for the 3 new pairs. Each pair is either a
    # single canonical row or user+assistant split depending on how the
    # reconciler prepares rows; in either case the count must be
    # "writes for 3 pairs" — not "writes for 103 pairs".
    max_expected_writes = 3 * 2 + 2  # 3 pairs × up to 2 rows/pair + small slack
    assert incremental_writes <= max_expected_writes, (
        "Append-path fast-skip regression: incremental ingest of 3 new "
        f"pairs on 100 existing wrote {incremental_writes} rows. "
        f"Expected <= {max_expected_writes} (only the new tail). This "
        "resurrects the 164-second prepare bottleneck from 2026-04-18."
    )

    # Bookkeeping must still reflect the full overlap.
    assert result.turns_matched >= 100, (
        "turns_matched must still count overlap rows even when writes "
        "are skipped — downstream batch records and progress displays "
        f"depend on it. Got {result.turns_matched}."
    )
    # rows_touched must still carry canonical identity for all 103 pairs
    # so downstream consumers (tagger's pre-persisted row lookup) see a
    # complete canonical set.
    assert len(result.rows) >= 100, (
        f"rows_touched must carry the overlap rows too; got {len(result.rows)}."
    )


def test_exact_resend_writes_zero_rows(tmp_path: Path):
    """Re-ingesting the exact same payload must produce zero canonical
    writes — the alignment ``exact_resend`` merge mode has no prefix and
    no suffix, so no DB work is needed.
    """
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    wrapped = _WriteCountingStore(store)
    rec = _reconciler(store)
    rec._store = wrapped

    payload = _pair_payload(50)
    rec.ingest_batch(
        conversation_id="c", body=payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )

    wrapped.save_count = 0
    result = rec.ingest_batch(
        conversation_id="c", body=payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )

    assert wrapped.save_count == 0, (
        "Exact-resend must write zero canonical rows — incoming fully "
        "equals existing, every row is overlap, no novelty. "
        f"Got {wrapped.save_count} writes."
    )
    assert result.merge_mode == "exact_resend"
    assert result.turns_written == 0


def test_overlap_rows_keep_original_source_batch_id(tmp_path: Path):
    """Overlap rows' stored ``source_batch_id`` must stay on the batch
    that originally wrote them. Rollback on lifecycle-epoch mismatch
    scopes to ``source_batch_id = <this batch_id>``, so the overlap
    rows correctly remain on the prior batch and are not purged when a
    later batch gets rolled back.
    """
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    rec = _reconciler(store)

    seed_payload = _pair_payload(5)
    seed_result = rec.ingest_batch(
        conversation_id="c", body=seed_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    seed_batch_id = seed_result.batch.batch_id

    incremental_payload = _pair_payload(7)
    new_result = rec.ingest_batch(
        conversation_id="c", body=incremental_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    new_batch_id = new_result.batch.batch_id
    assert new_batch_id != seed_batch_id

    # Query the stored rows and assert overlap keeps the seed batch_id,
    # new rows carry the new batch_id.
    with store._get_conn() as conn:
        rows = list(conn.execute(
            "SELECT sort_key, source_batch_id FROM canonical_turns "
            "WHERE conversation_id = 'c' ORDER BY sort_key ASC"
        ))
    # First 10 rows (5 seed pairs × 2 halves) must carry seed batch_id.
    overlap_slice = rows[:10]
    assert all(src == seed_batch_id for _, src in overlap_slice), (
        "Overlap rows must retain their original source_batch_id so "
        "rollback scoping stays surgical to the new batch. "
        f"Rows: {overlap_slice}"
    )
    # Remaining rows (2 new pairs × 2 halves = 4) must carry new batch_id.
    new_slice = rows[10:]
    assert new_slice, "Expected tail rows from the incremental batch."
    assert all(src == new_batch_id for _, src in new_slice), (
        "Newly-appended rows must carry the current batch_id so rollback "
        f"can target them. Rows: {new_slice}"
    )


def test_rollback_by_batch_id_purges_only_new_tail(tmp_path: Path):
    """If the new batch rolls back (epoch mismatch scenario), the
    overlap rows MUST survive — they belong to earlier batches and are
    still canonically correct. The rollback call uses
    ``delete_canonical_turns_by_batch_id`` which scopes by
    ``source_batch_id``; since the optimisation kept overlap rows on the
    prior batch, this test pins that rollback only removes the new tail.
    """
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    rec = _reconciler(store)

    seed_payload = _pair_payload(4)
    rec.ingest_batch(
        conversation_id="c", body=seed_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )

    incremental_payload = _pair_payload(6)
    new_result = rec.ingest_batch(
        conversation_id="c", body=incremental_payload, fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )

    # Simulate rollback of the new batch.
    store.delete_canonical_turns_by_batch_id(
        conversation_id="c",
        batch_id=new_result.batch.batch_id,
    )

    # Only the 4 seed pairs' rows must remain.
    remaining = store.get_all_canonical_turns("c")
    assert len(remaining) == 4 * 2, (
        "Rollback must remove only the new tail rows; overlap rows from "
        f"the prior batch must survive. Got {len(remaining)} remaining."
    )
