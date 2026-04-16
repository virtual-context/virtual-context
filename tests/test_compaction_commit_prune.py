from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.types import EngineState, EngineStateSnapshot


def _make_pipeline(*, save_ok: bool = True, committed_turn: int = 4):
    store = MagicMock()
    store.load_engine_state.return_value = EngineStateSnapshot(
        conversation_id="conv-1",
        compacted_through=(committed_turn + 1) * 2 if committed_turn >= 0 else 0,
        turn_tag_entries=[],
        turn_count=committed_turn + 1 if committed_turn >= 0 else 0,
        last_compacted_turn=committed_turn,
        last_completed_turn=committed_turn,
        last_indexed_turn=committed_turn,
        checkpoint_version=9,
    )
    pipeline = CompactionPipeline(
        compactor=None,
        segmenter=MagicMock(),
        store=store,
        turn_tag_index=MagicMock(),
        engine_state=EngineState(last_compacted_turn=committed_turn),
        config=SimpleNamespace(conversation_id="conv-1"),
        supersession_checker=None,
        fact_curator=None,
        semantic=MagicMock(),
        telemetry=MagicMock(),
        save_state_callback=MagicMock(return_value=save_ok),
    )
    return pipeline, store


def test_compaction_commits_after_checkpoint_save():
    pipeline, store = _make_pipeline(save_ok=True, committed_turn=4)

    pipeline._commit_compaction_state([])

    pipeline._save_state_callback.assert_called_once()
    store.load_engine_state.assert_not_called()


def test_compaction_returns_early_when_checkpoint_save_fails():
    pipeline, store = _make_pipeline(save_ok=False, committed_turn=4)

    pipeline._commit_compaction_state([])

    store.load_engine_state.assert_not_called()


def test_run_compaction_never_triggers_store_cleanup_from_tag_rules():
    pipeline, store = _make_pipeline(save_ok=True, committed_turn=4)
    pipeline._segmenter.segment = MagicMock(return_value=[])
    pipeline._compact_and_store = MagicMock(return_value=[])
    pipeline._refresh_shared_retrieval_snapshots = MagicMock()
    pipeline._config = SimpleNamespace(
        conversation_id="conv-1",
        tag_rules=[SimpleNamespace(match="debug*", priority=7)],
    )

    pipeline._run_compaction([], [], generated_by_turn_id="turn-123")

    store.cleanup.assert_not_called()
