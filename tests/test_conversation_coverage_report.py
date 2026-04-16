from datetime import datetime, timezone

from virtual_context.core.coverage_report import build_conversation_coverage_report
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SegmentMetadata, StoredSegment, TagSummary


def _make_segment(
    ref: str,
    *,
    conversation_id: str,
    turn_count: int,
    start_turn_number: int,
    end_turn_number: int,
    generated_by_turn_id: str,
    created_at: datetime,
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        conversation_id=conversation_id,
        primary_tag="topic",
        tags=["topic"],
        summary=f"summary {ref}",
        summary_tokens=10,
        full_text=f"full {ref}",
        full_tokens=20,
        messages=[{"role": "user", "content": ref}],
        metadata=SegmentMetadata(
            turn_count=turn_count,
            start_turn_number=start_turn_number,
            end_turn_number=end_turn_number,
            generated_by_turn_id=generated_by_turn_id,
        ),
        created_at=created_at,
        start_timestamp=created_at,
        end_timestamp=created_at,
        compaction_model="test",
        compression_ratio=0.5,
    )


def test_build_conversation_coverage_report_uses_latest_payload_and_exact_ranges(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    conversation_id = "conv-1"
    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)

    store.store_segment(
        _make_segment(
            "seg-a",
            conversation_id=conversation_id,
            turn_count=3,
            start_turn_number=0,
            end_turn_number=2,
            generated_by_turn_id="req-a",
            created_at=now,
        )
    )
    store.store_segment(
        _make_segment(
            "seg-b",
            conversation_id=conversation_id,
            turn_count=3,
            start_turn_number=2,
            end_turn_number=4,
            generated_by_turn_id="req-b",
            created_at=now.replace(hour=13),
        )
    )
    store.store_segment(
        _make_segment(
            "seg-c",
            conversation_id=conversation_id,
            turn_count=4,
            start_turn_number=-1,
            end_turn_number=-1,
            generated_by_turn_id="",
            created_at=now.replace(hour=14),
        )
    )
    store.save_tag_summary(
        TagSummary(
            tag="topic",
            summary="rollup",
            summary_tokens=12,
            covers_through_turn=8,
            generated_by_turn_id="req-b",
            created_at=now.replace(hour=15),
            updated_at=now.replace(hour=15),
        ),
        conversation_id=conversation_id,
    )
    store.save_request_capture(
        {
            "turn": 9,
            "turn_id": "req-latest",
            "ts": "2026-04-10T16:30:00+00:00",
            "api_format": "anthropic",
            "model": "test",
            "stream": False,
            "message_count": 6,
            "client_payload_message_count": 6,
            "client_payload_user_prompt_count": 3,
            "client_payload_timestamped_message_count": 3,
            "client_payload_earliest_timestamp": "2026-03-15T16:10:00+00:00",
            "client_payload_latest_timestamp": "2026-04-10T16:29:00+00:00",
            "conversation_id": conversation_id,
            "inbound_tags": [],
            "response_tags": [],
            "passthrough": False,
            "inbound_tokens": 0,
            "outbound_tokens": 0,
            "inbound_bytes": 0,
            "outbound_bytes": 0,
            "context_tokens": 0,
            "overhead_ms": 0,
            "turns_dropped": 0,
            "turns_stubbed": 0,
            "message_preview": "",
            "upstream_input_tokens": 0,
            "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )

    report = build_conversation_coverage_report(store, conversation_id)

    assert report.conversation_id == conversation_id
    assert report.latest_payload.turn == 9
    assert report.latest_payload.turn_id == "req-latest"
    assert report.latest_payload.message_count == 6
    assert report.latest_payload.user_prompt_count == 3
    assert report.latest_payload.timestamped_message_count == 3
    assert report.latest_payload.earliest_timestamp == "2026-03-15T16:10:00+00:00"
    assert report.latest_payload.latest_timestamp == "2026-04-10T16:29:00+00:00"
    assert report.segment_count == 3
    assert report.summarized_turn_occurrences == 10
    assert report.exact_range_segment_count == 2
    assert report.exact_unique_turn_count == 5
    assert report.exact_start_turn_number == 0
    assert report.exact_end_turn_number == 4
    assert report.tag_summary_count == 1
    assert report.max_tag_summary_turn == 8


def test_build_conversation_coverage_report_prefers_extracted_history_counts(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    conversation_id = "conv-extracted"

    store.save_request_capture(
        {
            "turn": 0,
            "turn_id": "req-large",
            "ts": "2026-04-16T14:08:05+00:00",
            "api_format": "openai",
            "model": "test",
            "stream": False,
            "message_count": 29,
            "client_payload_message_count": 29,
            "client_payload_user_prompt_count": 12,
            "client_payload_timestamped_message_count": 10,
            "client_payload_earliest_timestamp": "2026-04-11T15:05:00+00:00",
            "client_payload_latest_timestamp": "2026-04-16T14:07:00+00:00",
            "ingestible_entry_count": 998,
            "conversation_id": conversation_id,
            "inbound_tags": [],
            "response_tags": [],
            "passthrough": True,
            "inbound_tokens": 0,
            "outbound_tokens": 0,
            "inbound_bytes": 0,
            "outbound_bytes": 0,
            "context_tokens": 0,
            "overhead_ms": 0,
            "turns_dropped": 0,
            "turns_stubbed": 0,
            "message_preview": "",
            "upstream_input_tokens": 0,
            "upstream_output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )

    report = build_conversation_coverage_report(store, conversation_id)

    assert report.latest_payload.message_count == 998
