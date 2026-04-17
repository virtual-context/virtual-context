"""Task A22: Engine carries lifecycle_epoch + verify_epoch() method."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch


def _make_test_engine(tmp_path, conversation_id: str = "test-conv-epoch"):
    """Construct a minimal VirtualContextEngine for testing.

    Mirrors the helper in tests/test_engine_event_bus.py: SQLite backend,
    keyword tag generator, unique conversation_id per engine.
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
    return VirtualContextEngine(config=config)


def _inner_store(engine):
    """Return the concrete SQLite/Postgres store underneath the wrappers."""
    store = engine._store
    inner = getattr(store, "_store", None)
    if inner is None:
        return store
    # CompositeStore wraps segments/facts/etc.; return the segments store.
    segments = getattr(inner, "_segments", None)
    if segments is not None:
        return segments
    return inner


def test_engine_state_carries_lifecycle_epoch(tmp_path):
    """EngineState exposes lifecycle_epoch; fresh engine defaults to 1."""
    eng = _make_test_engine(tmp_path, conversation_id="test-carry-epoch")
    try:
        assert hasattr(eng._engine_state, "lifecycle_epoch")
        assert eng._engine_state.lifecycle_epoch == 1
    finally:
        eng.close()


def test_engine_verify_epoch_passes_when_match(tmp_path):
    """Fresh engine — engine's cached epoch matches the DB's epoch."""
    eng = _make_test_engine(tmp_path, conversation_id="test-match-epoch")
    try:
        eng.verify_epoch()  # no exception
    finally:
        eng.close()


def test_engine_verify_epoch_raises_on_external_bump(tmp_path):
    """External delete+resurrect bumps DB epoch; engine's cached epoch is stale."""
    conv_id = "test-bump-epoch"
    eng = _make_test_engine(tmp_path, conversation_id=conv_id)
    try:
        assert eng._engine_state.lifecycle_epoch == 1
        store = _inner_store(eng)
        # Mark deleted + resurrect bumps DB to epoch 2; engine still has epoch=1.
        store.mark_conversation_deleted(conv_id)
        new_epoch = store.increment_lifecycle_epoch_on_resurrect(conv_id)
        assert new_epoch == 2
        with pytest.raises(LifecycleEpochMismatch):
            eng.verify_epoch()
    finally:
        eng.close()


def test_engine_verify_epoch_raises_on_conversation_row_deleted(tmp_path):
    """If the conversation row has been externally deleted/purged, verify raises."""
    conv_id = "test-row-deleted-epoch"
    eng = _make_test_engine(tmp_path, conversation_id=conv_id)
    try:
        store = _inner_store(eng)
        with store._get_conn() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conv_id,),
            )
        with pytest.raises(LifecycleEpochMismatch):
            eng.verify_epoch()
    finally:
        eng.close()


def test_engine_tolerates_store_without_lifecycle_epoch(tmp_path):
    """If the store doesn't implement get_lifecycle_epoch, verify_epoch no-ops.

    Simulate by swapping the engine's store with a mock that raises
    NotImplementedError (legacy backends like FilesystemStore in principle).
    """
    eng = _make_test_engine(tmp_path, conversation_id="test-unsupported-epoch")
    try:
        class _LegacyStore:
            def get_lifecycle_epoch(self, conversation_id):
                raise NotImplementedError
        eng._store = _LegacyStore()  # type: ignore[assignment]
        # Should not raise — store doesn't support lifecycle_epoch.
        eng.verify_epoch()
    finally:
        # We swapped the store; don't let close() blow up on the stub.
        try:
            eng.close()
        except Exception:
            pass


def test_engine_init_creates_conversation_row_if_missing(tmp_path):
    """Fresh engine on a new conversation creates the conversations row.

    Ensures the Engine.__init__ upsert path fires when the row is absent.
    """
    from virtual_context.storage.sqlite import SQLiteStore

    conv_id = "fresh-conv"
    db_path = tmp_path / f"{conv_id}.db"  # matches _make_test_engine path
    # Pre-create an empty SQLite store (schema initialized on first open)
    # but DO NOT pre-populate the conversations row for conv_id.
    pre_store = SQLiteStore(db_path=str(db_path))
    # Verify the row does not exist.
    with pytest.raises(KeyError):
        pre_store.get_lifecycle_epoch(conv_id)
    pre_store.close()

    # Engine __init__ should upsert the row at epoch=1.
    eng = _make_test_engine(tmp_path, conversation_id=conv_id)
    try:
        # Confirm the row now exists via the engine's store chain.
        inner = _inner_store(eng)
        assert inner.get_lifecycle_epoch(conv_id) == 1
    finally:
        eng.close()
