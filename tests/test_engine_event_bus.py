"""Task A21: Engine owns a ProgressEventBus instance."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virtual_context.core.event_bus import ProgressEventBus


def _make_test_engine(tmp_path, conversation_id: str = "test-conv-bus"):
    """Construct a minimal VirtualContextEngine for testing.

    Mirrors the `engine` fixture in tests/test_engine_sync_turns.py: SQLite
    backend, keyword tag generator, unique conversation_id per engine.
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


def test_engine_has_progress_event_bus(tmp_path):
    eng = _make_test_engine(tmp_path)
    try:
        assert isinstance(eng.progress_event_bus, ProgressEventBus)
    finally:
        eng.close()


def test_two_engines_have_independent_buses(tmp_path):
    """Each Engine instance owns its own bus; subscribers on one don't
    receive events from another."""
    eng_a = _make_test_engine(tmp_path, conversation_id="test-conv-bus-a")
    eng_b = _make_test_engine(tmp_path, conversation_id="test-conv-bus-b")
    try:
        received_a: list = []
        received_b: list = []
        eng_a.progress_event_bus.subscribe(received_a.append)
        eng_b.progress_event_bus.subscribe(received_b.append)
        from virtual_context.core.progress_events import IngestionProgressEvent
        ev = IngestionProgressEvent(
            conversation_id="c",
            lifecycle_epoch=1,
            kind="ingestion",
            timestamp=1.0,
            episode_id="e",
            done=0,
            total=0,
        )
        eng_a.progress_event_bus.publish(ev)
        assert len(received_a) == 1
        assert len(received_b) == 0  # independent
    finally:
        eng_a.close()
        eng_b.close()
