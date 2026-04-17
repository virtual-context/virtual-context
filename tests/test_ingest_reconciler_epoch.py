"""Tests for IngestReconciler epoch-safety + CanonicalTurnRow extensions."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
from pathlib import Path

from virtual_context.core.canonical_turns import utcnow_iso
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore


def _fmt():
    # Default OpenAI-style format — detect_format picks openai when there's no
    # provider-specific marker. That's fine for testing ingest: the reconciler
    # passes fmt straight through to extract_ingestible_messages.
    from virtual_context.proxy.formats import detect_format
    return detect_format({"messages": []})


def _reconciler(store: SQLiteStore) -> IngestReconciler:
    # Minimal SemanticSearchManager — no embedding provider, no real embed_fn.
    # embed_and_store_turn is try/except-wrapped in _write_turn, so failures
    # here won't break the test path.
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    # Disable embedding so we don't attempt to load sentence-transformers.
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def test_reconciler_sets_covered_ingestible_entries_and_tagged_at_none(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    rec = _reconciler(s)
    payload = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]}
    result = rec.ingest_batch(
        conversation_id="c",
        body=payload,
        fmt=_fmt(),
        expected_lifecycle_epoch=1,
    )
    assert result.turns_written >= 1
    # Verify rows have covered_ingestible_entries=1 and tagged_at=NULL.
    with s._get_conn() as conn:
        rows = list(conn.execute("""
            SELECT covered_ingestible_entries, tagged_at
              FROM canonical_turns WHERE conversation_id = 'c'
        """))
    assert len(rows) >= 1
    for cov, tagged in rows:
        assert cov == 1
        assert tagged is None


def test_ingest_batch_rejects_stale_epoch_at_entry(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    # Resurrect to bump epoch to 2.
    s.mark_conversation_deleted("c")
    s.increment_lifecycle_epoch_on_resurrect("c")
    assert s.get_lifecycle_epoch("c") == 2
    rec = _reconciler(s)
    with pytest.raises(LifecycleEpochMismatch):
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [{"role": "user", "content": "hi"}]},
            fmt=_fmt(),
            expected_lifecycle_epoch=1,  # stale
        )
    with s._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id='c'"
        ).fetchone()[0]
    assert n == 0  # nothing written


def test_ingest_batch_rolls_back_on_resurrect_race(tmp_path: Path, monkeypatch):
    """Simulate a resurrect landing DURING ingest_prepared_turns — the
    commit-time check must catch it and purge the just-written rows."""
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    rec = _reconciler(s)
    # Patch _ingest_prepared_turns_locked to bump epoch right after its writes land.
    orig = rec._ingest_prepared_turns_locked

    def patched(*args, **kwargs):
        result = orig(*args, **kwargs)
        s.mark_conversation_deleted("c")
        s.increment_lifecycle_epoch_on_resurrect("c")
        return result

    monkeypatch.setattr(rec, "_ingest_prepared_turns_locked", patched)
    with pytest.raises(LifecycleEpochMismatch):
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [{"role": "user", "content": "hi"}]},
            fmt=_fmt(),
            expected_lifecycle_epoch=1,
        )
    # Rows written under the old lifecycle were rolled back.
    with s._get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id='c'"
        ).fetchone()[0]
    assert n == 0


def test_delete_canonical_turns_by_batch_id(tmp_path: Path):
    s = SQLiteStore(tmp_path / "vc.db")
    s.upsert_conversation(tenant_id="t", conversation_id="c")
    now = utcnow_iso()
    with s._get_conn() as conn:
        for i, bid in enumerate(["bA", "bA", "bB"]):
            conn.execute("""
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'u','a','u','a', ?, ?, ?, ?, 1, NULL, ?, ?)
            """, (f"t{i}", "c", f"h{i}", float((i + 1) * 1000), bid, now, now, now, now))
    deleted = s.delete_canonical_turns_by_batch_id(conversation_id="c", batch_id="bA")
    assert deleted == 2
    with s._get_conn() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM canonical_turns WHERE conversation_id='c'"
        ).fetchone()[0]
    assert remaining == 1  # only the bB row
