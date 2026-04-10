"""Read-only helpers for conversation payload span and summary coverage."""

from __future__ import annotations

from datetime import datetime

from .store import ContextStore
from ..types import ConversationCoverageReport, PayloadSpanStats


def _parse_capture_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _int_or_default(value, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _merge_turn_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def build_conversation_coverage_report(
    store: ContextStore,
    conversation_id: str,
) -> ConversationCoverageReport:
    """Summarize the latest payload span and durable summary coverage."""
    segments = store.get_all_segments(conversation_id=conversation_id)
    tag_summaries = store.get_all_tag_summaries(conversation_id=conversation_id)
    captures = store.load_request_captures(limit=50, conversation_id=conversation_id)

    latest_capture = captures[-1] if captures else {}
    latest_payload = PayloadSpanStats(
        turn=_int_or_default(latest_capture.get("turn", -1)),
        turn_id=(latest_capture.get("turn_id", "") or ""),
        captured_at=_parse_capture_timestamp(str(latest_capture.get("ts", "") or "")),
        message_count=_int_or_default(latest_capture.get("client_payload_message_count", latest_capture.get("message_count", 0)), 0),
        pair_count=_int_or_default(latest_capture.get("client_payload_pair_count", 0), 0),
        user_prompt_count=_int_or_default(latest_capture.get("client_payload_user_prompt_count", 0), 0),
        timestamped_message_count=_int_or_default(latest_capture.get("client_payload_timestamped_message_count", 0), 0),
        earliest_timestamp=str(latest_capture.get("client_payload_earliest_timestamp", "") or ""),
        latest_timestamp=str(latest_capture.get("client_payload_latest_timestamp", "") or ""),
    )

    summarized_turn_occurrences = 0
    exact_intervals: list[tuple[int, int]] = []
    oldest_segment_created_at = None
    newest_segment_created_at = None
    for segment in segments:
        summarized_turn_occurrences += _int_or_default(getattr(segment.metadata, "turn_count", 0), 0)
        start = _int_or_default(getattr(segment.metadata, "start_turn_number", -1))
        end = _int_or_default(getattr(segment.metadata, "end_turn_number", -1))
        if start >= 0 and end >= start:
            exact_intervals.append((start, end))
        if oldest_segment_created_at is None or segment.created_at < oldest_segment_created_at:
            oldest_segment_created_at = segment.created_at
        if newest_segment_created_at is None or segment.created_at > newest_segment_created_at:
            newest_segment_created_at = segment.created_at

    merged_intervals = _merge_turn_intervals(exact_intervals)
    exact_unique_turn_count = sum((end - start + 1) for start, end in merged_intervals)

    oldest_tag_summary_created_at = None
    newest_tag_summary_created_at = None
    max_tag_summary_turn = -1
    for summary in tag_summaries:
        if oldest_tag_summary_created_at is None or summary.created_at < oldest_tag_summary_created_at:
            oldest_tag_summary_created_at = summary.created_at
        if newest_tag_summary_created_at is None or summary.created_at > newest_tag_summary_created_at:
            newest_tag_summary_created_at = summary.created_at
        max_tag_summary_turn = max(
            max_tag_summary_turn,
            _int_or_default(summary.covers_through_turn),
        )

    return ConversationCoverageReport(
        conversation_id=conversation_id,
        latest_payload=latest_payload,
        segment_count=len(segments),
        summarized_turn_occurrences=summarized_turn_occurrences,
        exact_range_segment_count=len(exact_intervals),
        exact_unique_turn_count=exact_unique_turn_count,
        exact_start_turn_number=merged_intervals[0][0] if merged_intervals else -1,
        exact_end_turn_number=merged_intervals[-1][1] if merged_intervals else -1,
        tag_summary_count=len(tag_summaries),
        max_tag_summary_turn=max_tag_summary_turn,
        oldest_segment_created_at=oldest_segment_created_at,
        newest_segment_created_at=newest_segment_created_at,
        oldest_tag_summary_created_at=oldest_tag_summary_created_at,
        newest_tag_summary_created_at=newest_tag_summary_created_at,
    )
