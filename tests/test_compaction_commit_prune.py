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


def test_compaction_prunes_only_after_committed_checkpoint_save():
    pipeline, store = _make_pipeline(save_ok=True, committed_turn=4)

    pipeline._commit_compaction_state([])

    pipeline._save_state_callback.assert_called_once()
    # No longer reloads from Postgres — trusts in-memory engine_state
    store.load_engine_state.assert_not_called()
    store.prune_turn_messages.assert_called_once_with("conv-1", 5)


def test_compaction_skips_prune_when_checkpoint_save_fails():
    pipeline, store = _make_pipeline(save_ok=False, committed_turn=4)

    pipeline._commit_compaction_state([])

    store.load_engine_state.assert_not_called()
    store.prune_turn_messages.assert_not_called()


def test_compaction_skips_prune_when_committed_checkpoint_lags_expected():
    """Regression guard: if save_state_callback mutates engine_state to a
    lower value, the prune is skipped."""
    pipeline, store = _make_pipeline(save_ok=True, committed_turn=4)

    # Simulate save_state_callback clobbering the checkpoint to a lower value
    def regressing_save(history):
        pipeline._engine_state.last_compacted_turn = 2
        return True

    pipeline._save_state_callback = regressing_save

    pipeline._commit_compaction_state([])

    store.prune_turn_messages.assert_not_called()
