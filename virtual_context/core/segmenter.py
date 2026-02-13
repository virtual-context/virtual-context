"""TopicSegmenter: split conversation by domain using classifier pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from ..classifiers.base import ClassifierPipeline
from ..types import (
    ClassificationResult,
    DomainDef,
    DomainSegment,
    Message,
    SegmenterConfig,
    TurnPair,
)


class TopicSegmenter:
    """Groups messages into turn pairs, classifies each, groups contiguous same-domain pairs."""

    def __init__(
        self,
        classifier_pipeline: ClassifierPipeline,
        config: SegmenterConfig,
        domains: list[DomainDef] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.classifier = classifier_pipeline
        self.config = config
        self.domains = domains or []
        self.token_counter = token_counter or (lambda text: len(text) // 4)

    async def segment(self, messages: list[Message]) -> list[DomainSegment]:
        """Segment a block of conversation into domain-tagged chunks."""
        if not messages:
            return []

        # Step 1: pair into turns
        pairs = self._pair_turns(messages)

        # Step 2: classify each turn pair
        classified: list[tuple[TurnPair, ClassificationResult]] = []
        for pair in pairs:
            text = " ".join(m.content for m in pair.messages)
            results = await self.classifier.classify(text, self.domains)
            primary = results[0] if results else ClassificationResult(
                domain="_general", confidence=0.1, source="fallback"
            )
            classified.append((pair, primary))

        # Step 3: group contiguous same-domain pairs
        segments: list[DomainSegment] = []
        current_group: list[tuple[TurnPair, ClassificationResult]] = []

        for pair, result in classified:
            if current_group and current_group[0][1].domain != result.domain:
                segments.append(self._build_segment(current_group))
                current_group = []
            current_group.append((pair, result))

        if current_group:
            segments.append(self._build_segment(current_group))

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
        self, group: list[tuple[TurnPair, ClassificationResult]]
    ) -> DomainSegment:
        """Build a DomainSegment from a group of classified turn pairs."""
        all_messages: list[Message] = []
        confidences: list[float] = []
        secondary: set[str] = set()
        primary_domain = group[0][1].domain

        for pair, result in group:
            all_messages.extend(pair.messages)
            confidences.append(result.confidence)
            if result.domain != primary_domain:
                secondary.add(result.domain)

        text = " ".join(m.content for m in all_messages)
        token_count = self.token_counter(text)

        timestamps = [
            m.timestamp for m in all_messages
            if m.timestamp is not None
        ]
        start_ts = min(timestamps) if timestamps else datetime.now(timezone.utc)
        end_ts = max(timestamps) if timestamps else datetime.now(timezone.utc)

        return DomainSegment(
            id=str(uuid.uuid4()),
            domain=primary_domain,
            secondary_domains=sorted(secondary),
            messages=all_messages,
            token_count=token_count,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            turn_count=len(group),
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )
