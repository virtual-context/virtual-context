"""Immutable, request-local memory rendering and its source dependencies.

These records are accounting artifacts, never a replacement for canonical-row
admission. Every assembly/paging operation rehydrates and validates sources;
records must not be persisted as reusable proof capabilities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from ..types import SpeakerRetrievalContext


@dataclass(frozen=True)
class MemorySourceVersion:
    conversation_id: str
    canonical_turn_id: str
    version: str

    @classmethod
    def from_row(cls, row: object) -> MemorySourceVersion:
        # Include all source and provenance fields, not just the conversation
        # hash: assistant corrections and audience/identity repairs matter too.
        values = asdict(row) if is_dataclass(row) else vars(row)
        digest = hashlib.sha256(json.dumps(
            values, sort_keys=True, ensure_ascii=False, default=str,
            separators=(",", ":"),
        ).encode()).hexdigest()
        return cls(str(getattr(row, "conversation_id", "")),
                   str(getattr(row, "canonical_turn_id", "")), digest)


@dataclass(frozen=True)
class RenderedMemory:
    """One indivisible model-visible topic, with exact measured wrapper cost.

    ``sources`` includes the whole proof dependency snapshot, which can be
    larger than ``presented_source_ids`` for compact structured selections.
    A newer correction in any dependency invalidates the previous selection.
    """

    tag: str
    depth: str
    evidence_kind: str
    sources: tuple[MemorySourceVersion, ...]
    presented_source_ids: tuple[str, ...]
    segment_refs: tuple[str, ...]
    conversation_id: str
    scope: SpeakerRetrievalContext | None = field(repr=False, compare=False)
    text: str
    measured_cost: int


def rendered_memory(
    *, tag: str, depth: str, text: str, renderings: Iterable[str],
    segment_refs: Iterable[str], conversation_id: str,
    scope: SpeakerRetrievalContext | None, token_counter: Callable[[str], int],
) -> RenderedMemory:
    """Freeze metadata from the same proof carriers used to format the text."""
    values = tuple(renderings)
    sources = tuple(dict.fromkeys(
        source for value in values for source in getattr(value, "_source_versions", ())
    ))
    presented = tuple(dict.fromkeys(
        source_id for value in values for source_id in getattr(value, "_presented_source_ids", ())
    ))
    kinds = tuple(sorted(set(
        getattr(value, "_evidence_kind", "quarantine") for value in values
    )))
    return RenderedMemory(
        tag=tag, depth=depth, evidence_kind="+".join(kinds), sources=sources,
        presented_source_ids=presented,
        segment_refs=tuple(dict.fromkeys(ref for ref in segment_refs if ref)),
        conversation_id=conversation_id, scope=scope,
        text=text, measured_cost=token_counter(text),
    )
