"""Exact provenance for noncontiguous topic segments.

TopicSegmenter supports A-B-A interleaving.  Compaction must therefore map
signals, tool outputs, ranges, and timestamps from canonical source ids rather
than slicing logical rows with a positional cursor.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import (
    CompactionResult,
    FactSignal,
    Message,
    SegmentMetadata,
    TaggedSegment,
    TurnTagEntry,
)


SOURCE_IDS = "_vc_source_canonical_turn_ids"


def _message(role: str, content: str, canonical_id: str, timestamp: datetime):
    return Message(
        role=role,
        content=content,
        timestamp=timestamp,
        metadata={SOURCE_IDS: [canonical_id]},
    )


def test_noncontiguous_segments_use_exact_canonical_turn_provenance(tmp_path):
    config = load_config(config_dict={
        "tenant_id": "tenant-noncontiguous",
        "conversation_id": "sk:agent:test:discord:guild:999",
        "context_window": 10000,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / "noncontiguous.db")},
        },
        "tag_generator": {"type": "keyword"},
        "compaction": {"merge_lookback": 0},
    })
    engine = VirtualContextEngine(config=config)
    store = engine._store
    conversation_id = engine.config.conversation_id
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

    old_turns = [40, 10, 30]
    for turn in range(3):
        canonical_id = f"ct-{turn}"
        timestamp = now + timedelta(minutes=turn)
        store.save_canonical_turn(
            conversation_id, -1,
            f"user-{turn}", f"assistant-{turn}",
            canonical_turn_id=canonical_id,
            turn_group_number=old_turns[turn],
            sort_key=float((turn + 1) * 1000),
            turn_hash=f"hash-{turn}",
            tagged_at=timestamp.isoformat(),
            primary_tag=f"topic-{turn}",
            tags=[f"topic-{turn}"],
            first_seen_at=timestamp.isoformat(),
            last_seen_at=timestamp.isoformat(),
            created_at=timestamp.isoformat(),
            updated_at=timestamp.isoformat(),
        )
        store.store_tool_output(
            f"tool-{turn}", conversation_id, "test", f"command-{turn}",
            old_turns[turn], f"output-{turn}", len(f"output-{turn}"),
        )
        store.link_turn_tool_output(
            conversation_id, old_turns[turn], f"tool-{turn}",
        )

    assert store.set_phase(
        conversation_id=conversation_id,
        lifecycle_epoch=1,
        phase="active",
    )

    # Exercise the real historical repair boundary before segmentation.  The
    # exact canonical ids must remain authoritative after old local turn
    # numbers and turn-scoped artifacts are rewritten to 0, 1, 2.
    preview = engine.resequence_canonical_turns(conversation_id)
    assert preview["changed_group_rows"] == 3
    applied = engine.resequence_canonical_turns(conversation_id, dry_run=False)
    assert applied["turn_tool_output_unmapped"] == 0
    assert [row.turn_group_number for row in store.get_all_canonical_turns(
        conversation_id,
    )] == [0, 1, 2]

    for turn in range(3):
        engine._turn_tag_index.append(TurnTagEntry(
            turn_number=turn,
            canonical_turn_id=f"ct-{turn}",
            tags=[f"topic-{turn}"],
            primary_tag=f"topic-{turn}",
            fact_signals=[FactSignal(subject=f"signal-{turn}")],
            code_refs=[{"file": f"file-{turn}.py"}],
        ))

    # Segment A contains turns 0 and 2; segment B contains turn 1.  Passing
    # them in library order reproduces the old cursor bug: A used to borrow
    # rows 0+1 and B used to borrow row 2.
    seg_a = TaggedSegment(
        id="seg-a", primary_tag="topic-a", tags=["topic-a"],
        messages=[
            _message("user", "user-0", "ct-0", now),
            _message("assistant", "assistant-0", "ct-0", now),
            _message("user", "user-2", "ct-2", now + timedelta(minutes=2)),
            _message(
                "assistant", "assistant-2", "ct-2",
                now + timedelta(minutes=2),
            ),
        ],
        token_count=20, turn_count=2,
        start_timestamp=now,
        end_timestamp=now + timedelta(minutes=2),
    )
    seg_b = TaggedSegment(
        id="seg-b", primary_tag="topic-b", tags=["topic-b"],
        messages=[
            _message("user", "user-1", "ct-1", now + timedelta(minutes=1)),
            _message(
                "assistant", "assistant-1", "ct-1",
                now + timedelta(minutes=1),
            ),
        ],
        token_count=10, turn_count=1,
        start_timestamp=now + timedelta(minutes=1),
        end_timestamp=now + timedelta(minutes=1),
    )

    captured = {}

    def compact(segments, **kwargs):
        captured.update(kwargs)
        results = []
        for segment in segments:
            results.append(CompactionResult(
                segment_id=segment.id,
                primary_tag=segment.primary_tag,
                tags=list(segment.tags),
                summary=f"summary-{segment.id}",
                summary_tokens=3,
                full_text=" ".join(m.content for m in segment.messages),
                original_tokens=segment.token_count,
                messages=[{
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                } for m in segment.messages],
                metadata=SegmentMetadata(turn_count=segment.turn_count),
                compression_ratio=0.3,
                timestamp=segment.start_timestamp,
                facts=[],
            ))
        return results

    compactor = MagicMock()
    compactor.compact.side_effect = compact
    compactor.model_name = "test-model"
    engine._compaction._compactor = compactor
    engine._tagging._compactor = compactor

    compact_rows = list(store.get_uncompacted_canonical_turns(
        conversation_id, protected_recent_turns=0,
    ))
    results = engine._compaction._compact_and_store(
        [seg_a, seg_b], 6, compact_rows=compact_rows,
    )

    assert [result.segment_id for result in results] == ["seg-a", "seg-b"]
    assert [signal.subject for signal in captured["fact_signals_by_segment"]["seg-a"]] == [
        "signal-0", "signal-2",
    ]
    assert [signal.subject for signal in captured["fact_signals_by_segment"]["seg-b"]] == [
        "signal-1",
    ]
    assert captured["code_refs_by_segment"] == {
        "seg-a": [{"file": "file-0.py"}, {"file": "file-2.py"}],
        "seg-b": [{"file": "file-1.py"}],
    }

    assert sorted(store.get_tool_outputs_for_segment(
        conversation_id, "seg-a",
    )) == ["tool-0", "tool-2"]
    assert store.get_tool_outputs_for_segment(
        conversation_id, "seg-b",
    ) == ["tool-1"]

    stored_a = store.get_segment("seg-a", conversation_id=conversation_id)
    stored_b = store.get_segment("seg-b", conversation_id=conversation_id)
    assert stored_a.metadata.start_turn_number == 0
    assert stored_a.metadata.end_turn_number == 2
    assert stored_a.start_timestamp == now
    assert stored_a.end_timestamp == now + timedelta(minutes=2)
    assert stored_b.metadata.start_turn_number == 1
    assert stored_b.metadata.end_turn_number == 1
    assert stored_b.start_timestamp == now + timedelta(minutes=1)
    assert stored_b.end_timestamp == now + timedelta(minutes=1)
