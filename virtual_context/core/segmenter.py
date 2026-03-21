"""TopicSegmenter: split conversation by tags using a TagGenerator."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

from .tag_generator import TagGenerator
from ..types import (
    Message,
    SegmenterConfig,
    TaggedSegment,
    TagResult,
    TurnPair,
)

_SESSION_RE = re.compile(r'\[Session from ([^\]]+)\]')


def _parse_session_date(pair: TurnPair) -> str:
    """Extract session date from any message in the pair, or empty string.

    Session headers (``[Session from ...]``) may appear in user *or*
    assistant messages — e.g. when the same speaker has consecutive turns
    across a session boundary.
    """
    for msg in pair.messages:
        m = _SESSION_RE.search(msg.content)
        if m:
            return m.group(1)
    return ""


def _latest_timestamp(pair: TurnPair) -> datetime | None:
    timestamps = [m.timestamp for m in pair.messages if m.timestamp is not None]
    return max(timestamps) if timestamps else None


def _earliest_timestamp(pair: TurnPair) -> datetime | None:
    timestamps = [m.timestamp for m in pair.messages if m.timestamp is not None]
    return min(timestamps) if timestamps else None


class TopicSegmenter:
    """Groups messages into turn pairs, tags each, groups contiguous same-tag pairs."""

    def __init__(
        self,
        tag_generator: TagGenerator,
        config: SegmenterConfig,
        token_counter: Callable[[str], int] | None = None,
        turn_tag_index=None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.tag_generator = tag_generator
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self._turn_tag_index = turn_tag_index
        self._embed_fn = embed_fn

    def segment(
        self, messages: list[Message], turn_offset: int = 0
    ) -> list[TaggedSegment]:
        """Segment a block of conversation into tag-based chunks.

        Args:
            messages: The messages to segment.
            turn_offset: Global turn number of the first pair in *messages*.
                Used to look up correct entries in TurnTagIndex when
                segmenting a slice of conversation (e.g. after compaction
                watermark).
        """
        if not messages:
            return []

        from .tag_scoring import compute_relatedness

        # Pair messages into turns
        pairs = self._pair_turns(messages)

        # Tag each turn pair
        tagged: list[tuple[TurnPair, TagResult]] = []
        for i, pair in enumerate(pairs):
            # Skip empty/whitespace turns (stripped images, media, etc.)
            combined = " ".join(m.content for m in pair.messages)
            if not combined.strip():
                logger.info(
                    "SEGMENT skip_empty turn=%d global=%d content_len=%d",
                    i, turn_offset + i, len(combined),
                )
                continue

            # Check index first — avoid redundant LLM call
            global_turn = turn_offset + i
            entry = self._turn_tag_index.get_tags_for_turn(global_turn) if self._turn_tag_index else None
            if entry:
                tag_result = TagResult(
                    tags=entry.tags, primary=entry.primary_tag,
                    source="index",
                )
            else:
                logger.info(
                    "SEGMENT index_miss turn=%d global=%d index_size=%d — falling back to LLM",
                    i, global_turn,
                    len(self._turn_tag_index.entries) if self._turn_tag_index else 0,
                )
                tag_result = self.tag_generator.generate_tags(combined)
            tagged.append((pair, tag_result))
            preview = " ".join(m.content for m in pair.messages)[:60].replace("\n", " ")
            logger.info(
                "SEGMENT turn=%d global=%d primary=%s tags=%s source=%s preview=\"%s\"",
                i, global_turn, tag_result.primary, sorted(tag_result.tags),
                tag_result.source, preview,
            )

        # Group contiguous same-primary-tag pairs,
        # split on session date change or temporal gap
        segments: list[TaggedSegment] = []
        current_group: list[tuple[TurnPair, TagResult]] = []
        running_session: str = ""  # tracks session date across all pairs
        group_session: str = ""    # session date for the current group

        for turn_idx, (pair, result) in enumerate(tagged):
            parsed = _parse_session_date(pair)
            if parsed:
                running_session = parsed
            if current_group:
                prev_tags = set(current_group[-1][1].tags)
                curr_tags = set(result.tags)
                meaningful_prev = {t for t in prev_tags if t != "_general"}
                meaningful_curr = {t for t in curr_tags if t != "_general"}

                if not meaningful_prev or not meaningful_curr:
                    tag_changed = False  # merge on empty/general-only
                    overlap = 1.0
                else:
                    prev_text = " ".join(m.content for m in current_group[-1][0].messages)
                    curr_text = " ".join(m.content for m in pair.messages)
                    overlap = compute_relatedness(
                        tags_a=meaningful_prev,
                        tags_b=meaningful_curr,
                        text_a=prev_text,
                        text_b=curr_text,
                        embed_fn=self._embed_fn,
                    )
                    tag_changed = overlap < self.config.tag_overlap_threshold
                    logger.info(
                        "SEGMENT relatedness turn=%d prev_tags=%s curr_tags=%s "
                        "score=%.3f threshold=%.1f → %s",
                        turn_offset + turn_idx, sorted(meaningful_prev),
                        sorted(meaningful_curr), overlap,
                        self.config.tag_overlap_threshold,
                        "SPLIT" if tag_changed else "MERGE",
                    )

                # Hard cap on segment size
                max_cap = self.config.max_segment_turns > 0 and len(current_group) >= self.config.max_segment_turns
                if max_cap:
                    tag_changed = True
                session_changed = parsed and parsed != group_session
                temporal_gap = self._has_temporal_gap(current_group[-1][0], pair)

                if tag_changed or session_changed or temporal_gap:
                    if session_changed:
                        reason = f"session_changed session={parsed}"
                    elif temporal_gap:
                        last_ts = _latest_timestamp(current_group[-1][0])
                        new_ts = _earliest_timestamp(pair)
                        gap_min = int((new_ts - last_ts).total_seconds() / 60) if last_ts and new_ts else 0
                        reason = f"temporal_gap minutes={gap_min}"
                    elif max_cap:
                        reason = f"max_segment_turns limit={self.config.max_segment_turns}"
                    else:
                        reason = f"tag_changed overlap={overlap:.3f}"
                    logger.info(
                        "SEGMENT split turn=%d reason=%s",
                        turn_offset + turn_idx, reason,
                    )
                    segments.append(self._build_segment(current_group, group_session))
                    current_group = []
                    # Derive session date from timestamp if no header present
                    if temporal_gap and not parsed:
                        ts = _earliest_timestamp(pair)
                        if ts:
                            running_session = ts.strftime("%Y-%m-%dT%H:%M:%S")
                    group_session = running_session
            if not current_group:
                group_session = running_session
            current_group.append((pair, result))

        if current_group:
            segments.append(self._build_segment(current_group, group_session))

        # Reassign _stub segments to their chronologically nearest neighbor.
        # A stub between segments inherits tags from whichever neighbor is
        # temporally closer — handles both "end of session image" (attach backward)
        # and "morning image to start new conversation" (attach forward).
        segments = self._reassign_stub_segments(segments)

        if self.config.tool_result_segment_threshold > 0:
            segments = self._split_large_tool_results(segments)

        avg_turns = len(tagged) / len(segments) if segments else 0
        logger.info(
            "SEGMENT complete: %d segments from %d turns, avg %.1f turns/segment, "
            "config: threshold=%.1f max_turns=%d",
            len(segments), len(tagged), avg_turns,
            self.config.tag_overlap_threshold, self.config.max_segment_turns,
        )
        return segments

    @staticmethod
    def _split_session_boundaries(messages: list[Message]) -> list[Message]:
        """Split messages that contain a mid-content ``[Session from ...]`` header.

        In some conversation formats (e.g. LoCoMo), the same speaker talks at
        the end of one session and the start of the next, producing a single
        message with an embedded session header.  Splitting ensures the session
        boundary falls between messages so the turn-pairer and segmenter can
        detect it.
        """
        out: list[Message] = []
        for msg in messages:
            parts = _SESSION_RE.split(msg.content, maxsplit=1)
            if len(parts) == 3 and parts[0].strip():
                # parts = [before, captured_date, after]
                out.append(Message(
                    role=msg.role,
                    content=parts[0].strip(),
                    timestamp=msg.timestamp,
                ))
                out.append(Message(
                    role=msg.role,
                    content=f"[Session from {parts[1]}]{parts[2]}",
                    timestamp=msg.timestamp,
                ))
            else:
                out.append(msg)
        return out

    def _pair_turns(self, messages: list[Message]) -> list[TurnPair]:
        """Group messages into (user, assistant) pairs.

        System/tool messages attach to the current pair.
        Messages with mid-content session boundaries are split first.
        """
        messages = self._split_session_boundaries(messages)
        pairs: list[TurnPair] = []
        current_pair: list[Message] = []

        for msg in messages:
            if msg.role == "user":
                if current_pair:
                    pairs.append(TurnPair(messages=current_pair))
                current_pair = [msg]
            elif msg.role == "assistant":
                current_pair.append(msg)
                pairs.append(TurnPair(messages=current_pair))
                current_pair = []
            else:
                # system/tool: attach to current pair
                current_pair.append(msg)

        if current_pair:
            pairs.append(TurnPair(messages=current_pair))

        return pairs

    def _build_segment(
        self, group: list[tuple[TurnPair, TagResult]], session_date: str = "",
    ) -> TaggedSegment:
        all_messages: list[Message] = []
        all_tags: set[str] = set()
        primary_tag = group[0][1].primary

        for pair, result in group:
            all_messages.extend(pair.messages)
            all_tags.update(result.tags)

        text = " ".join(m.content for m in all_messages)
        token_count = self.token_counter(text)

        timestamps = [
            m.timestamp for m in all_messages
            if m.timestamp is not None
        ]
        start_ts = min(timestamps) if timestamps else datetime.now(timezone.utc)
        end_ts = max(timestamps) if timestamps else datetime.now(timezone.utc)

        seg_id = str(uuid.uuid4())
        seg = TaggedSegment(
            id=seg_id,
            primary_tag=primary_tag,
            tags=sorted(all_tags),
            messages=all_messages,
            token_count=token_count,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            turn_count=len(group),
            session_date=session_date,
        )
        logger.info(
            "SEGMENT built ref=%s primary=%s tags=%s turns=%d tokens=%d session=%s",
            seg_id[:8], primary_tag, sorted(all_tags), len(group), token_count,
            session_date or start_ts.strftime("%Y-%m-%dT%H:%M") if start_ts else "",
        )
        return seg

    @staticmethod
    def _reassign_stub_segments(segments: list[TaggedSegment]) -> list[TaggedSegment]:
        """Reassign _stub segments to their chronologically nearest neighbor.

        A stub between two real segments inherits tags from whichever is
        temporally closer. Stubs at the edges inherit from their only neighbor.
        """
        if not segments:
            return segments

        stub_indices = [i for i, s in enumerate(segments) if s.primary_tag == "_stub"]
        if not stub_indices:
            return segments

        # Multiple passes: reassigned stubs become valid donors for adjacent stubs.
        # Max passes bounded by stub count to avoid infinite loops.
        for _pass in range(len(stub_indices) + 1):
            changed = False
            for idx in stub_indices:
                stub = segments[idx]
                if stub.primary_tag != "_stub":
                    continue  # already reassigned in a prior pass
                prev_seg = segments[idx - 1] if idx > 0 else None
                next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

                donor = None
                if prev_seg and prev_seg.primary_tag != "_stub" and next_seg and next_seg.primary_tag != "_stub":
                    # Both neighbors are real — pick the temporally closer one
                    stub_ts = stub.start_timestamp
                    prev_gap = abs((stub_ts - prev_seg.end_timestamp).total_seconds()) if prev_seg.end_timestamp else float("inf")
                    next_gap = abs((next_seg.start_timestamp - stub_ts).total_seconds()) if next_seg.start_timestamp else float("inf")
                    donor = prev_seg if prev_gap <= next_gap else next_seg
                elif prev_seg and prev_seg.primary_tag != "_stub":
                    donor = prev_seg
                elif next_seg and next_seg.primary_tag != "_stub":
                    donor = next_seg

                if donor:
                    stub.primary_tag = donor.primary_tag
                    stub.tags = list(donor.tags)
                    changed = True
                    logger.info(
                        "SEGMENT stub_reassign ref=%s → inherited tags=%s from %s neighbor (pass %d)",
                        stub.id[:8], sorted(stub.tags),
                        "prev" if donor is prev_seg else "next", _pass + 1,
                    )

            if not changed:
                break

        return segments

    def _has_temporal_gap(self, last_pair: TurnPair, new_pair: TurnPair) -> bool:
        """Check if there's a significant time gap between two turn pairs.

        Uses ``config.session_gap_minutes`` as threshold.  Returns False when
        either pair lacks timestamps (graceful no-op for benchmark data).
        """
        threshold = self.config.session_gap_minutes
        if threshold <= 0:
            return False
        last_ts = _latest_timestamp(last_pair)
        new_ts = _earliest_timestamp(new_pair)
        if last_ts is None or new_ts is None:
            return False
        return (new_ts - last_ts).total_seconds() > threshold * 60

    def _split_large_tool_results(
        self, segments: list[TaggedSegment],
    ) -> list[TaggedSegment]:
        """Split out turns with large tool_result blocks into their own segments."""
        threshold = self.config.tool_result_segment_threshold
        result: list[TaggedSegment] = []
        for segment in segments:
            if segment.turn_count <= 1:
                result.append(segment)
                continue
            pairs = self._pair_turns(segment.messages)
            needs_split = False
            for pair in pairs:
                if self._has_large_tool_result(pair.messages, threshold):
                    needs_split = True
                    break
            if not needs_split:
                result.append(segment)
                continue
            # Re-group: isolate large-tool-result turns, keep others together
            current_group: list[TurnPair] = []
            for pair in pairs:
                if self._has_large_tool_result(pair.messages, threshold):
                    if current_group:
                        result.append(self._build_segment_from_pairs(
                            current_group, segment.primary_tag, segment.tags,
                        ))
                        current_group = []
                    result.append(self._build_segment_from_pairs(
                        [pair], segment.primary_tag, segment.tags,
                    ))
                else:
                    current_group.append(pair)
            if current_group:
                result.append(self._build_segment_from_pairs(
                    current_group, segment.primary_tag, segment.tags,
                ))
        return result

    @staticmethod
    def _has_large_tool_result(messages: list[Message], threshold: int) -> bool:
        for m in messages:
            if not m.raw_content:
                continue
            for block in m.raw_content:
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, str):
                    if len(content.encode("utf-8")) >= threshold:
                        return True
                elif isinstance(content, list):
                    total = sum(
                        len(item.get("text", "").encode("utf-8"))
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                    if total >= threshold:
                        return True
        return False

    def _build_segment_from_pairs(
        self,
        pairs: list[TurnPair],
        primary_tag: str,
        tags: list[str],
    ) -> TaggedSegment:
        all_messages: list[Message] = []
        for p in pairs:
            all_messages.extend(p.messages)

        text = " ".join(m.content for m in all_messages)
        token_count = self.token_counter(text)

        timestamps = [
            m.timestamp for m in all_messages
            if m.timestamp is not None
        ]
        start_ts = min(timestamps) if timestamps else datetime.now(timezone.utc)
        end_ts = max(timestamps) if timestamps else datetime.now(timezone.utc)

        return TaggedSegment(
            id=str(uuid.uuid4()),
            primary_tag=primary_tag,
            tags=tags,
            messages=all_messages,
            token_count=token_count,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            turn_count=len(pairs),
        )
