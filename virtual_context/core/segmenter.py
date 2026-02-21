"""TopicSegmenter: split conversation by tags using a TagGenerator."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Callable

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
    """Extract session date from first user message, or empty string."""
    for msg in pair.messages:
        if msg.role == "user":
            m = _SESSION_RE.search(msg.content)
            if m:
                return m.group(1)
            break
    return ""


class TopicSegmenter:
    """Groups messages into turn pairs, tags each, groups contiguous same-tag pairs."""

    def __init__(
        self,
        tag_generator: TagGenerator,
        config: SegmenterConfig,
        token_counter: Callable[[str], int] | None = None,
        turn_tag_index=None,
    ) -> None:
        self.tag_generator = tag_generator
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self._turn_tag_index = turn_tag_index

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

        # Step 1: pair into turns
        pairs = self._pair_turns(messages)

        # Step 2: tag each turn pair
        tagged: list[tuple[TurnPair, TagResult]] = []
        for i, pair in enumerate(pairs):
            # Check index first — avoid redundant LLM call
            global_turn = turn_offset + i
            entry = self._turn_tag_index.get_tags_for_turn(global_turn) if self._turn_tag_index else None
            if entry:
                tag_result = TagResult(
                    tags=entry.tags, primary=entry.primary_tag,
                    source="index",
                )
            else:
                combined = " ".join(m.content for m in pair.messages)
                tag_result = self.tag_generator.generate_tags(combined)
            tagged.append((pair, tag_result))

        # Step 3: group contiguous same-primary-tag pairs, split on session date change
        segments: list[TaggedSegment] = []
        current_group: list[tuple[TurnPair, TagResult]] = []
        running_session: str = ""  # tracks session date across all pairs
        group_session: str = ""    # session date for the current group

        for pair, result in tagged:
            parsed = _parse_session_date(pair)
            if parsed:
                running_session = parsed
            if current_group:
                tag_changed = current_group[0][1].primary != result.primary
                session_changed = parsed and group_session and parsed != group_session
                if tag_changed or session_changed:
                    segments.append(self._build_segment(current_group, group_session))
                    current_group = []
                    group_session = running_session
            if not current_group:
                group_session = running_session
            current_group.append((pair, result))

        if current_group:
            segments.append(self._build_segment(current_group, group_session))

        return segments

    def _pair_turns(self, messages: list[Message]) -> list[TurnPair]:
        """Group messages into (user, assistant) pairs.

        System/tool messages attach to the current pair.
        """
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
        """Build a TaggedSegment from a group of tagged turn pairs."""
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

        return TaggedSegment(
            id=str(uuid.uuid4()),
            primary_tag=primary_tag,
            tags=sorted(all_tags),
            messages=all_messages,
            token_count=token_count,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            turn_count=len(group),
            session_date=session_date,
        )
