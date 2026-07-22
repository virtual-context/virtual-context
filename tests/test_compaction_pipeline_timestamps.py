from types import SimpleNamespace

from virtual_context.core.compaction_pipeline import CompactionPipeline
from virtual_context.types import CanonicalTurnRow


class _Store:
    def __init__(self, row):
        self.row = row

    def get_uncompacted_canonical_turns(self, conversation_id, *, protected_recent_turns=0):
        return [self.row]

    def get_all_canonical_turns(self, conversation_id):
        return [self.row]


def test_load_compactable_rows_propagates_canonical_timestamp():
    row = CanonicalTurnRow(
        conversation_id="conv",
        canonical_turn_id="ct-1",
        turn_group_number=0,
        user_content="question",
        assistant_content="answer",
        first_seen_at="2026-07-15T23:34:07.584038+00:00",
    )
    pipeline = CompactionPipeline.__new__(CompactionPipeline)
    pipeline._store = _Store(row)
    pipeline._config = SimpleNamespace(
        conversation_id="conv",
        monitor=SimpleNamespace(protected_recent_turns=0),
    )

    _rows, messages = pipeline._load_compactable_rows()

    assert [message.timestamp.isoformat() for message in messages] == [
        "2026-07-15T23:34:07.584038+00:00",
        "2026-07-15T23:34:07.584038+00:00",
    ]
