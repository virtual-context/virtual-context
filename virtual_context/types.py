"""All dataclasses, Protocols, and type aliases for virtual-context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

DEFAULT_CHAT_MODEL: str = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Fact Extraction (D1)
# ---------------------------------------------------------------------------

class TemporalStatus(str, Enum):
    """Whether the fact is ongoing, completed, or aspirational."""
    ACTIVE = "active"
    COMPLETED = "completed"
    PLANNED = "planned"
    ABANDONED = "abandoned"
    RECURRING = "recurring"


@dataclass
class FactSignal:
    """Lightweight fact signal extracted per-turn by the tagger.
    Cheap to produce, may be noisy/incomplete. Consolidated at compaction."""
    subject: str = ""
    verb: str = ""       # free-form action verb
    object: str = ""
    status: str = ""     # TemporalStatus value
    fact_type: str = "personal"  # personal|experience|world
    what: str = ""               # full-sentence memory with ALL specifics


@dataclass
class Fact:
    """Consolidated queryable fact.
    Produced at compaction from signals + multi-turn context."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Structured queryable fields
    subject: str = ""
    verb: str = ""              # free-form action verb
    object: str = ""
    status: str = "active"      # TemporalStatus value
    # Dimensions
    what: str = ""
    who: str = ""
    when_date: str = ""         # specific date referenced in the fact (ISO or free-form)
    where: str = ""
    why: str = ""
    fact_type: str = "personal"  # personal|experience|world
    # Provenance
    tags: list[str] = field(default_factory=list)
    segment_ref: str = ""
    conversation_id: str = ""
    turn_numbers: list[int] = field(default_factory=list)
    mentioned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_date: str = ""  # original conversation date, e.g. "2023/05/25 (Thu) 10:04"
    # Knowledge update chain
    superseded_by: str | None = None  # fact_id that replaces this fact

    @classmethod
    def from_dict(cls, d: dict, *, dt_parser=None) -> Fact:
        """Build a Fact from a storage row/dict.

        Handles JSON-encoded ``tags_json`` and ``turn_numbers_json``,
        graph-backend ``where_val`` key, and optional datetime parsing
        via *dt_parser* (defaults to ``datetime.fromisoformat``).
        """
        import json as _json

        if dt_parser is None:
            def dt_parser(s):
                dt = datetime.fromisoformat(s)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

        mentioned_raw = d.get("mentioned_at")
        mentioned = dt_parser(mentioned_raw) if mentioned_raw else datetime.now(timezone.utc)

        tags_raw = d.get("tags_json", "[]")
        tags = _json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])

        turns_raw = d.get("turn_numbers_json", "[]")
        turn_numbers = _json.loads(turns_raw) if isinstance(turns_raw, str) else (turns_raw or [])

        sup = d.get("superseded_by")
        if sup == "":
            sup = None

        return cls(
            id=d["id"],
            subject=d.get("subject", ""),
            verb=d.get("verb", ""),
            object=d.get("object", ""),
            status=d.get("status", "active"),
            what=d.get("what", ""),
            who=d.get("who", ""),
            when_date=d.get("when_date", ""),
            where=d.get("where", d.get("where_val", "")),
            why=d.get("why", ""),
            fact_type=d.get("fact_type", "personal"),
            tags=tags,
            segment_ref=d.get("segment_ref", ""),
            conversation_id=d.get("conversation_id", ""),
            turn_numbers=turn_numbers,
            mentioned_at=mentioned,
            session_date=d.get("session_date", ""),
            superseded_by=sup,
        )

    def format_for_prompt(self, include_index: int | None = None) -> str:
        """Canonical one-line rendering for LLM prompts.

        Used by the assembler, curator, supersession checker, and any other
        component that presents facts to an LLM.  All fields are included
        so no consumer silently drops dimensions.
        """
        prefix = f"[{include_index}] " if include_index is not None else "- "
        line = f"{prefix}{self.subject} | {self.verb} | {self.object}"
        if self.what:
            line += f" — {self.what}"
        if self.who and self.who.lower() != self.subject.lower():
            line += f" [who: {self.who}]"
        if self.when_date:
            line += f" [when: {self.when_date}]"
        elif self.session_date:
            line += f" [session: {self.session_date}]"
        if self.where:
            line += f" [where: {self.where}]"
        if self.why:
            line += f" [why: {self.why}]"
        if self.status and self.status != "active":
            line += f" [status: {self.status}]"
        return line


class RelationType(str, Enum):
    """Supported relationship types between facts."""
    SUPERSEDES = "supersedes"
    CAUSED_BY = "caused_by"
    PART_OF = "part_of"
    CONTRADICTS = "contradicts"
    SAME_AS = "same_as"
    RELATED_TO = "related_to"


@dataclass
class FactLink:
    """A directed relationship between two facts."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_fact_id: str = ""
    target_fact_id: str = ""
    relation_type: str = ""         # RelationType value
    confidence: float = 1.0         # 0.0-1.0
    context: str = ""               # one-sentence explanation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "compaction"  # "compaction" | "supersession" | "migration"


@dataclass
class LinkedFact:
    """A Fact returned as part of a link traversal result."""
    fact: Fact
    linked_from_fact_id: str = ""   # which primary fact this is linked to
    relation_type: str = ""
    confidence: float = 1.0
    link_context: str = ""


# ---------------------------------------------------------------------------
# Message & Turn
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime | None = None
    metadata: dict | None = None
    raw_content: list[dict] | None = None


def get_sender_name(metadata: dict | None) -> str | None:
    """Extract sender name from message metadata, with fallbacks.

    Checks (in order):
    1. sender.name / sender.display_name / sender.label (full sender object)
    2. conversation info.sender (string field from conversation metadata)

    Returns None if no sender info is available.
    """
    if not metadata:
        return None
    # Primary: dedicated sender metadata block
    sender = metadata.get("sender")
    if sender and isinstance(sender, dict):
        name = sender.get("name") or sender.get("display_name") or sender.get("label")
        if name:
            return name
    # Fallback: conversation info block often has a "sender" string field
    conv_info = metadata.get("conversation info")
    if conv_info and isinstance(conv_info, dict):
        name = conv_info.get("sender")
        if name and isinstance(name, str):
            return name
    return None


@dataclass
class TurnPair:
    """Atomic unit: a user message and its assistant response (plus any system/tool)."""
    messages: list[Message] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tag Generation (replaces Classification)
# ---------------------------------------------------------------------------

@dataclass
class TagResult:
    """Result of tagging a piece of text."""
    tags: list[str]
    primary: str
    source: str  # "llm", "keyword", "fallback"
    temporal: bool = False  # True when query references a time position ("first thing", "early on")
    related_tags: list[str] = field(default_factory=list)  # semantic alternates for query expansion
    fact_signals: list[FactSignal] = field(default_factory=list)  # D1: per-turn fact signals
    code_refs: list[dict] = field(default_factory=list)  # code artifacts materially discussed in the turn
    query_embedding: list[float] | None = None


# Re-exported for existing callers; canonical definitions in patterns.py
from .patterns import DEFAULT_TEMPORAL_PATTERNS  # noqa: F401


@dataclass
class SplitResult:
    """Result of analyzing a broad tag for splitting."""
    tag: str                           # the broad tag that was analyzed
    splittable: bool
    groups: dict[str, list[int]] = field(default_factory=dict)  # new_tag → [turn_numbers]
    reason: str = ""                   # if not splittable


@dataclass
class TagSplittingConfig:
    """Configuration for automatic splitting of overly-broad tags."""
    enabled: bool = False
    frequency_threshold: int = 15       # min absolute turn count to trigger
    frequency_pct_threshold: float = 0.15  # min relative frequency (count/total)
    max_splits_per_turn: int = 1        # max tags to attempt per on_turn_complete


@dataclass
class TagGeneratorConfig:
    """Configuration for the tag generator."""
    type: str = "keyword"  # "llm" or "keyword"
    provider: str = ""  # provider name for LLM-based tagging
    model: str = ""
    max_tags: int = 10
    min_tags: int = 5
    max_tokens: int = 8192  # LLM max_tokens
    prompt_mode: str = "detailed"  # "detailed" (full rules+examples) or "compact" (minimal)
    keyword_fallback: KeywordTagConfig | None = None
    temporal_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_TEMPORAL_PATTERNS))
    temporal_heuristic_enabled: bool = True
    context_lookback_pairs: int = 5  # Number of recent turn pairs to feed as context to tagger
    context_bleed_threshold: float = 0.1  # Embedding similarity gate: skip context below this (0 = disabled)
    disable_thinking: bool = False  # Prepend /no_think to prompts (for qwen3 models)
    tag_splitting: TagSplittingConfig = field(default_factory=TagSplittingConfig)


@dataclass
class KeywordTagConfig:
    """Keyword/regex-based tag configuration (deterministic fallback)."""
    tag_keywords: dict[str, list[str]] = field(default_factory=dict)
    tag_patterns: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TurnTagEntry:
    """Tag metadata for a single round trip, computed in real time."""
    turn_number: int = -1
    message_hash: str = ""              # sha256[:16] of user+assistant content
    canonical_turn_id: str = ""
    tags: list[str] = field(default_factory=list)
    primary_tag: str = "_general"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_date: str = ""         # e.g. "2023/05/25 (Thu) 10:04" or ISO timestamp
    fact_signals: list[FactSignal] = field(default_factory=list)  # D1: per-turn fact signals
    sender: str = ""                # sender identity from envelope metadata
    code_refs: list[dict] = field(default_factory=list)  # code artifacts carried into compaction


@dataclass
class EngineState:
    """Mutable shared state passed to engine delegates."""
    compacted_prefix_messages: int = 0
    flushed_prefix_messages: int = 0
    last_request_time: float = 0.0
    last_compacted_turn: int = -1
    last_completed_turn: int = -1
    last_indexed_turn: int = -1
    checkpoint_version: int = 0
    conversation_generation: int = 0
    # Progress-bar redesign: lifecycle_epoch tracks the conversations row's
    # current lifecycle. Matches schema default (conversations.lifecycle_epoch
    # DEFAULT 1). Engine.__init__ loads the live value from the store; the
    # epoch bumps only on external delete+resurrect, in which case the in-memory
    # handle is stale and Engine.verify_epoch() raises LifecycleEpochMismatch.
    lifecycle_epoch: int = 1
    tool_tag_counter: int = 0
    split_processed_tags: set[str] = field(default_factory=set)
    trailing_fingerprint: str = ""
    provider: str = ""
    last_tag_ms: float = 0.0
    last_compact_ms: float = 0.0
    last_split_result: SplitResult | None = None

    def history_offset(self, history_len: int, *, total_turns_indexed: int | None = None, watermark: int | None = None) -> int:
        """Effective index into conversation_history for slicing past compacted messages.

        When *total_turns_indexed* is provided (the total number of entries in
        the turn-tag index), the method can distinguish between two cases where
        ``compacted_prefix_messages >= history_len``:

        1. **Session restart** — history was rebuilt from scratch; everything
           in the current history is genuinely new.  Return 0.
        2. **Sliding window** — the proxy trimmed old messages from the
           in-memory history, but the session is continuous.  We must skip
           messages whose turns have already been compacted to avoid
           re-processing.

        Without *total_turns_indexed*, the safe default is to treat rebuilt
        history as fresh and return 0 whenever ``compacted_prefix_messages >= history_len``.

        When *watermark* is provided, it overrides ``self.compacted_prefix_messages``
        as the boundary.  This lets callers pass ``flushed_prefix_messages`` for
        payload assembly while compaction/tagging callers keep using
        ``compacted_prefix_messages`` implicitly.
        """
        ct = watermark if watermark is not None else self.compacted_prefix_messages
        if ct < history_len:
            return ct

        # compacted_prefix_messages >= history_len
        if total_turns_indexed is None:
            # No index coverage information — assume restart / fresh history.
            return 0

        # Sliding-window detection: if the turn-tag index has grown beyond
        # what the trimmed history can represent, we are in a sliding window.
        # Compute the first turn present in the current history and derive
        # how many messages from the front should be skipped.
        history_turns = history_len // 2
        first_turn_in_history = total_turns_indexed - history_turns
        compacted_turn = (ct // 2)  # exclusive: turns < this are compacted

        if compacted_turn <= first_turn_in_history:
            # Watermark is behind the start of the history window —
            # every message in history is un-compacted.
            return 0

        # Some messages at the front of history have already been compacted.
        offset = (compacted_turn - first_turn_in_history) * 2
        return min(offset, history_len)  # safety cap


@dataclass
class EngineStateSnapshot:
    """Serializable snapshot of engine state for persistence across restarts."""
    conversation_id: str
    compacted_prefix_messages: int
    turn_tag_entries: list[TurnTagEntry]
    turn_count: int  # len(conversation_history) // 2
    flushed_prefix_messages: int = 0
    flushed_prefix_messages_present: bool = True
    last_request_time: float = 0.0
    last_compacted_turn: int = -1
    last_completed_turn: int = -1
    last_indexed_turn: int = -1
    checkpoint_version: int = 0
    conversation_generation: int = 0
    saved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    split_processed_tags: list[str] = field(default_factory=list)  # tags already split/summarized
    working_set: list[WorkingSetEntry] = field(default_factory=list)  # paging depth state
    trailing_fingerprint: str = ""  # hash of last N user messages for session matching on restart
    telemetry_rollup: dict = field(default_factory=dict)  # persisted telemetry totals (survives restart)
    request_captures: list[dict] = field(default_factory=list)  # lightweight request summaries for dashboard
    provider: str = ""  # upstream provider (anthropic, openai, gemini, etc.)
    tool_tag_counter: int = 0  # sequential counter for tool_N tags


@dataclass
class TagPromptRule:
    """Per-tag rules for priority and custom summary prompts."""
    match: str  # fnmatch pattern, e.g. "architecture*", "debug*"
    priority: int = 5
    summary_prompt: str | None = None


@dataclass
class StrategyConfig:
    """Retrieval strategy configuration."""
    min_overlap: int = 1
    max_results: int = 10
    max_budget_fraction: float = 0.25
    include_related: bool = True


# ---------------------------------------------------------------------------
# Context Monitoring
# ---------------------------------------------------------------------------

@dataclass
class ContextSnapshot:
    system_tokens: int
    core_context_tokens: int
    retrieved_domain_tokens: int
    conversation_tokens: int
    total_tokens: int
    budget_tokens: int
    turn_count: int
    active_tags: list[str] = field(default_factory=list)


@dataclass
class CompactionSignal:
    priority: Literal["soft", "hard"]
    current_tokens: int
    budget_tokens: int
    overflow_tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class TaggedSegment:
    """A segment of conversation tagged with semantic tags."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    primary_tag: str = "_general"
    tags: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    token_count: int = 0
    start_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    session_date: str = ""         # propagated from constituent turns
    merge_ref: str = ""            # when set, update this existing segment instead of creating new


@dataclass
class SegmentMetadata:
    entities: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    date_references: list[str] = field(default_factory=list)
    code_refs: list[dict] = field(default_factory=list)
    turn_count: int = 0
    canonical_turn_ids: list[str] = field(default_factory=list)
    start_turn_number: int = -1
    end_turn_number: int = -1
    generated_by_turn_id: str = ""
    time_span: tuple[datetime, datetime] | None = None
    session_date: str = ""         # propagated from constituent turns


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    segment_id: str
    primary_tag: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    original_tokens: int = 0
    compression_ratio: float = 0.0
    metadata: SegmentMetadata = field(default_factory=SegmentMetadata)
    full_text: str = ""
    messages: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    facts: list[Fact] = field(default_factory=list)  # D1: extracted facts


@dataclass
class CompactionReport:
    segments_compacted: int
    tokens_freed: int
    tags: list[str] = field(default_factory=list)
    results: list[CompactionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tag_summaries_built: int = 0
    cover_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass
class StoredSegment:
    ref: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    primary_tag: str = "_general"
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    full_text: str = ""
    full_tokens: int = 0
    messages: list[dict] = field(default_factory=list)
    metadata: SegmentMetadata = field(default_factory=SegmentMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    compaction_model: str = ""
    compression_ratio: float = 0.0


@dataclass
class StoredSummary:
    """Lightweight view: summary + metadata, no full text."""
    ref: str = ""
    primary_tag: str = "_general"
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    full_tokens: int = 0
    metadata: SegmentMetadata = field(default_factory=SegmentMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QuoteResult:
    """A passage found by full-text search."""
    text: str          # matching excerpt with surrounding context
    tag: str           # primary_tag of the segment
    segment_ref: str   # segment reference for drill-down
    tags: list[str] = field(default_factory=list)  # all tags on the segment
    score: float = 0.0 # FTS5 rank or match quality
    match_type: str = "fts"  # "fts", "like", or "semantic"
    similarity: float = 0.0  # cosine similarity (semantic matches only)
    session_date: str = ""   # session date from segment metadata or canonical turn
    source_scope: str = "segment"  # "turn", "segment", or "tool_output"
    turn_number: int | None = None
    matched_side: str = ""   # "user", "assistant", "both", or "unknown"


@dataclass
class ChunkEmbedding:
    """A chunk of segment text with its embedding vector."""
    segment_ref: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class CanonicalTurnRow:
    """Canonical archived transcript entry with explicit turn-group metadata."""
    conversation_id: str
    canonical_turn_id: str = ""
    turn_number: int = -1
    turn_group_number: int = -1
    sort_key: float = 0.0
    turn_hash: str = ""
    hash_version: int = 0
    normalized_user_text: str = ""
    normalized_assistant_text: str = ""
    user_content: str = ""
    assistant_content: str = ""
    user_raw_content: str | None = None
    assistant_raw_content: str | None = None
    primary_tag: str = "_general"
    tags: list[str] = field(default_factory=list)
    session_date: str = ""
    sender: str = ""
    fact_signals: list[FactSignal] = field(default_factory=list)
    code_refs: list[dict] = field(default_factory=list)
    tagged_at: str | None = None
    compacted_at: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    source_batch_id: str | None = None
    # Progress-tracking fields (progress-bar redesign).
    # ``covered_ingestible_entries`` is how many ingestible payload entries
    # this canonical row represents. Defaults to 1 for typical 1:1 row->entry
    # ingestion; future grouped-turn ingestion may set it higher.
    # ``tagged_at`` is also above (lifecycle timestamp); we leave it where it
    # is for backwards compat but note that the tagger (A27) owns writes.
    covered_ingestible_entries: int = 1
    created_at: str = ""
    updated_at: str = ""


@dataclass
class CanonicalTurnChunkEmbedding:
    """One embedded chunk from a canonical turn side."""
    conversation_id: str
    side: str
    chunk_index: int
    text: str
    embedding: list[float]
    canonical_turn_id: str = ""
    turn_number: int = -1

@dataclass
class IngestBatchRecord:
    batch_id: str = ""
    conversation_id: str = ""
    received_at: str = ""
    raw_turn_count: int = 0
    merge_mode: str = ""
    turns_matched: int = 0
    turns_appended: int = 0
    turns_prepended: int = 0
    turns_inserted: int = 0
    first_turn_hash: str = ""
    last_turn_hash: str = ""

@dataclass
class TagSummary:
    """Layer-2 summary: one per cover tag, rolls up all segment summaries for that tag."""
    tag: str
    summary: str = ""
    description: str = ""  # 1-line tag description (~15-20 words) for enriched context hints
    summary_tokens: int = 0
    source_segment_refs: list[str] = field(default_factory=list)
    source_turn_numbers: list[int] = field(default_factory=list)
    source_canonical_turn_ids: list[str] = field(default_factory=list)
    code_refs: list[dict] = field(default_factory=list)
    covers_through_turn: int = -1  # highest turn number covered; -1 = never built
    covers_through_canonical_turn_id: str = ""
    generated_by_turn_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TagStats:
    tag: str = ""
    usage_count: int = 0
    total_full_tokens: int = 0
    total_summary_tokens: int = 0
    oldest_segment: datetime | None = None
    newest_segment: datetime | None = None


@dataclass
class ConversationStats:
    """Aggregate statistics for a single conversation."""
    conversation_id: str = ""
    segment_count: int = 0
    total_full_tokens: int = 0
    total_summary_tokens: int = 0
    compression_ratio: float = 0.0
    distinct_tags: list[str] = field(default_factory=list)
    oldest_segment: datetime | None = None
    newest_segment: datetime | None = None
    compaction_model: str = ""
    provider: str = ""


@dataclass
class PayloadSpanStats:
    """Persisted span metadata for one captured client payload."""
    turn: int = -1
    turn_id: str = ""
    captured_at: datetime | None = None
    message_count: int = 0
    user_prompt_count: int = 0
    timestamped_message_count: int = 0
    earliest_timestamp: str = ""
    latest_timestamp: str = ""


@dataclass
class ConversationCoverageReport:
    """Read-only summary of payload span and durable summary coverage."""
    conversation_id: str
    latest_payload: PayloadSpanStats = field(default_factory=PayloadSpanStats)
    segment_count: int = 0
    summarized_turn_occurrences: int = 0
    exact_range_segment_count: int = 0
    exact_unique_turn_count: int = 0
    exact_start_turn_number: int = -1
    exact_end_turn_number: int = -1
    tag_summary_count: int = 0
    max_tag_summary_turn: int = -1
    oldest_segment_created_at: datetime | None = None
    newest_segment_created_at: datetime | None = None
    oldest_tag_summary_created_at: datetime | None = None
    newest_tag_summary_created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalCostReport:
    """Per-retrieval cost metrics."""
    tokens_retrieved: int = 0
    budget_fraction_used: float = 0.0
    strategy_active: str = "default"
    tags_queried: list[str] = field(default_factory=list)
    tags_skipped: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    tags_matched: list[str] = field(default_factory=list)
    summaries: list[StoredSummary] = field(default_factory=list)
    full_detail: list[StoredSegment] = field(default_factory=list)
    total_tokens: int = 0
    retrieval_metadata: dict = field(default_factory=dict)
    cost_report: RetrievalCostReport = field(default_factory=RetrievalCostReport)
    temporal: bool = False  # True when the query references a time position
    facts: list[Fact] = field(default_factory=list)  # D1: matching facts
    retrieval_scores: dict[str, float] = field(default_factory=dict)  # primary_tag → RRF fused score
    query_embedding: list[float] | None = None
    overflow_summaries: list[StoredSummary] = field(default_factory=list)  # relevant but budget-excluded


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paging (Virtual Memory Depth Control)
# ---------------------------------------------------------------------------

class DepthLevel(str, Enum):
    """Depth at which a topic's content is injected into the context window."""
    NONE = "none"           # listed in hint only, nothing injected
    SUMMARY = "summary"     # tag summary (~200t per tag) — default
    SEGMENTS = "segments"   # individual segment summaries (~2,000t per tag)
    FULL = "full"           # StoredSegment.full_text (~8,000t+ per tag)


@dataclass
class WorkingSetEntry:
    """Per-tag depth state in the paging working set."""
    tag: str
    depth: DepthLevel = DepthLevel.SUMMARY
    tokens: int = 0             # current token cost at this depth
    last_accessed_turn: int = 0 # for LRU eviction


@dataclass
class PagingConfig:
    """Configuration for virtual memory paging.

    ``autonomous_models`` lists model-name prefixes (case-insensitive) that
    are trusted to manage their own context via tool calls.  When the request's
    model matches any entry (prefix match), the proxy injects
    ``vc_expand_topic`` / ``vc_collapse_topic`` tools and a budget dashboard.
    Models that don't match run in *supervised* mode: VC manages paging
    silently via ``auto_promote`` / ``auto_evict``.
    """
    enabled: bool = False
    autonomous_models: list[str] = field(default_factory=lambda: [
        "claude-sonnet-4", "claude-opus-4",
        "claude-3-5-sonnet", "claude-3.5-sonnet",
        "claude-3-7-sonnet", "claude-3.7-sonnet",
        "claude-3-opus",
        "gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-5",
        "o1", "o3", "o4-mini",
        "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini-3", "gemini-2.0-flash",
    ])
    auto_promote: bool = True   # auto-expand on strong retrieval match
    auto_evict: bool = True     # auto-collapse coldest when over budget
    max_tool_loops: int = 10    # max continuation rounds in tool loop


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    core_context: str = ""
    tag_sections: dict[str, str] = field(default_factory=dict)
    facts_text: str = ""  # Formatted facts block
    conversation_history: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    budget_breakdown: dict[str, int] = field(default_factory=dict)
    prepend_text: str = ""
    matched_tags: list[str] = field(default_factory=list)
    context_hint: str = ""  # Topic list injected post-compaction
    temporal: bool = False  # True when query references a time position
    presented_segment_refs: set[str] = field(default_factory=set)  # segment refs already shown to reader
    retrieval_metadata: dict = field(default_factory=dict)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    retrieval_summaries: list[StoredSummary] = field(default_factory=list)
    retrieval_full_segments: list[StoredSegment] = field(default_factory=list)
    selected_facts: list[Fact] = field(default_factory=list)
    retrieval_result: RetrievalResult | None = None  # for fill pass overflow access
    presented_tags: set[str] = field(default_factory=set)  # ALL tags visible in rendered output
    assembly_breakdown: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool Loop
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Record of a single VC tool invocation within a tool loop."""
    tool_name: str
    tool_input: dict = field(default_factory=dict)
    result_json: str = ""
    duration_ms: float = 0.0


@dataclass
class ToolLoopResult:
    """Result of a synchronous tool loop."""
    text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    continuation_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"
    raw_requests: list[dict] = field(default_factory=list)
    raw_responses: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    context_window: int = 120_000
    soft_threshold: float = 0.70
    hard_threshold: float = 0.85
    protected_recent_turns: int = 6
    fill_pass_enabled: bool = False
    fill_pass_target: str = "soft"  # "soft", "hard", or float fraction
    fill_pass_summary_ratio: float = 0.60
    store_recovery_threshold: float = 0.70  # trigger store recovery when payload < this fraction of store turns
    defer_payload_mutation: bool = False
    flush_ttl_seconds: int = 300


@dataclass
class SegmenterConfig:
    session_gap_minutes: int = 30  # split segments when messages are >N min apart (0=disabled)
    tag_overlap_threshold: float = 0.5  # min overlap coefficient to keep turns in same segment
    max_segment_turns: int = 20  # hard cap on turns per segment (0=unlimited)
    tool_result_segment_threshold: int = 50000  # bytes; large tool_results get own segment (0=disabled)


@dataclass
class CompactorConfig:
    summary_ratio: float = 0.15
    min_summary_tokens: int = 200
    max_summary_tokens: int = 2000
    max_concurrent_summaries: int = 4
    overflow_buffer: float = 1.2
    llm_token_overhead: int = 2000  # extra tokens for thinking/reasoning overhead
    merge_lookback: int = 5         # check last N segments for merge candidate
    max_segment_tokens: int = 2000  # don't merge if combined full_text would exceed this
    merge_overlap_threshold: float = 0.35  # minimum relatedness score to merge
    code_mode: bool = True  # modifies fact extraction to exclude investigatory actions


@dataclass
class DampeningConfig:
    """Toggleable post-retrieval dampening filters."""
    hub_enabled: bool = True
    hub_penalty_strength: float = 0.6
    hub_min_score: float = 0.2
    gravity_enabled: bool = True
    gravity_threshold: float = 0.5
    gravity_factor: float = 0.5
    resolution_enabled: bool = True
    resolution_boost: float = 1.15


@dataclass
class ScoringConfig:
    """Weights and limits for 3-signal RRF retrieval scoring."""
    idf_weight: float = 0.50
    bm25_weight: float = 0.30
    embedding_weight: float = 0.20
    rrf_k: int = 60
    bm25_limit: int = 20
    embedding_limit: int = 20
    embedding_min_threshold: float = 0.25
    dampening: DampeningConfig = field(default_factory=DampeningConfig)


@dataclass
class RetrieverConfig:
    skip_active_tags: bool = True
    active_tag_lookback: int = 4
    tag_context_max_tokens: int = 30_000
    strategy_configs: dict[str, StrategyConfig] = field(default_factory=lambda: {
        "default": StrategyConfig()
    })
    anchorless_lookback: int = 6           # how many recent turns for working set
    inbound_tagger_type: str = "embedding"  # "embedding" (default) or "llm"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_threshold: float = 0.3
    prefetch_facts: bool = True            # filter facts by query tags instead of fetching all
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass
class AssemblerConfig:
    core_context_max_tokens: int = 18_000
    tag_context_max_tokens: int = 30_000
    facts_max_tokens: int = 20_000
    context_injection_max_tokens: int = -1  # -1 = auto (tag + facts budgets)
    core_files: list[dict] = field(default_factory=list)
    recent_turns_always_included: int = 3
    context_hint_enabled: bool = True
    context_hint_max_tokens: int = 2000
    pre_compaction_filtering: str = "aggressive"  # "off" | "conservative" | "aggressive"

    def __post_init__(self):
        if self.context_injection_max_tokens < 0:
            self.context_injection_max_tokens = self.tag_context_max_tokens + self.facts_max_tokens


@dataclass
class SearchConfig:
    """Configuration for find_quote excerpt/snippet lengths and result limits."""
    # Excerpt/snippet lengths
    excerpt_context_chars: int = 200       # chars of context around a LIKE/manual match
    fts_snippet_chars: int = 500           # FTS5 snippet() max chars for segment search
    tool_output_snippet_chars: int = 100   # FTS5 snippet() max chars for tool output search
    postgres_max_words: int = 100          # ts_headline MaxWords for Postgres FTS
    # Result limits
    find_quote_max_results: int = 20       # max results from vc_find_quote tool calls
    find_quote_default_results: int = 5    # default max_results for engine.find_quote()
    remember_when_max_results: int = 12    # default max_results for vc_remember_when
    semantic_search_max_results: int = 5   # max results from embedding-based search
    query_facts_default_limit: int = 50    # default limit for query_facts()
    search_facts_max_results: int = 10     # max results from FTS fact search


@dataclass
class SummarizationConfig:
    provider: str = ""
    model: str = ""
    max_tokens: int = 1000
    temperature: float = 0.3


@dataclass
class FactsConfig:
    """Configuration for fact features."""
    graph_links: bool = True
    link_types: list[str] = field(default_factory=lambda: [
        "supersedes", "caused_by", "part_of", "contradicts", "same_as", "related_to",
    ])


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    root: str = ".virtualcontext/store"
    sqlite_path: str = ".virtualcontext/store.db"
    # Postgres
    postgres_dsn: str = ""
    # Neo4j
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str = ""


@dataclass
class TelemetryConfig:
    enabled: bool = False
    models_file: str = "models.yaml"


# Deprecated: old configs may reference CostTrackingConfig
CostTrackingConfig = TelemetryConfig


@dataclass
class ProxyInstanceConfig:
    """Configuration for a single proxy listener instance."""
    port: int = 5757
    upstream: str = ""
    label: str = ""
    host: str = "127.0.0.1"
    config: str = ""  # path to per-instance config file; empty = use master config
    upstream_context_limit: int = 0  # 0 = inherit from global/auto-detect


@dataclass
class ProxyConfig:
    request_log_dir: str = ".virtualcontext/request_log"
    request_log_max_files: int = 50
    llm_calls_log: str = ""  # path for JSONL log of all LLM calls (tagger, compactor, etc.)
    upstream_context_limit: int = 0  # 0 = auto-detect from model name
    history_widening_threshold: float = 0.10  # 10% growth + prefix change triggers re-ingest
    passthrough_trim_ratio: float = 0.40  # trim passthrough payloads to upstream_limit * ratio (0=no trim)
    redis_url: str = ""           # empty = disabled, e.g. "redis://127.0.0.1:6379"
    redis_history_cap: int = 600  # safety cap on uncompacted suffix size in snapshot
    instances: list[ProxyInstanceConfig] = field(default_factory=list)


@dataclass
class PreparedPayload:
    """Result of prepare_payload() -- enriched request body ready for forwarding or returning."""
    body: dict
    enriched_body: dict          # may differ from body for active path (has injected context)
    conversation_id: str
    is_passthrough: bool
    turn: int
    request_turn: int
    turn_id: str
    api_format: str
    user_message: str
    is_streaming: bool
    inbound_tokens: int
    outbound_tokens: int
    context_tokens: int
    non_virtualizable_floor: int
    upstream_limit: int
    tags_matched: list[str]
    budget_breakdown: dict
    turns_dropped: int
    turns_stubbed: int
    wait_ms: float
    inbound_ms: float
    overhead_ms: float
    assembled: object | None     # AssembledContext for active path, None for passthrough
    pre_filter_body: dict | None  # body before filtering (for metrics capture)
    paging_enabled: bool
    tool_output_find_quote: bool
    restore_tool_injected: bool
    inbound_bytes: int
    outbound_bytes: int
    metadata: dict = field(default_factory=dict)  # catch-all for anything else
    is_vcattach: bool = False
    vcattach_target_id: str = ""
    vcattach_label: str = ""
    vcattach_old_id: str = ""
    # Generic VC command support (attach, label, status, recall, compact, list, forget)
    vc_command: str = ""
    vc_command_arg: str = ""


@dataclass
class ToolOutputRule:
    match: str = "*"
    truncate_threshold: int | None = None
    head_ratio: float = 0.6
    tail_ratio: float = 0.4
    max_index_bytes: int | None = None


@dataclass
class ToolOutputConfig:
    enabled: bool = False
    default_truncate_threshold: int = 8192
    max_index_bytes: int = 524_288
    default_head_ratio: float = 0.6
    default_tail_ratio: float = 0.4
    rules: list[ToolOutputRule] = field(default_factory=list)


@dataclass
class ToolOutputStats:
    total_intercepted: int = 0
    total_bytes_original: int = 0
    total_bytes_returned: int = 0
    total_bytes_indexed: int = 0
    by_tool: dict[str, dict] = field(default_factory=dict)


@dataclass
class SupersessionConfig:
    """Configuration for fact supersession checking."""
    enabled: bool = False
    provider: str = ""   # provider name from providers dict, or "" to use summarization provider
    model: str = ""      # model override, or "" to use summarization model
    batch_size: int = 20


@dataclass
class CurationConfig:
    """Configuration for LLM-based fact curation."""
    enabled: bool = False
    provider: str = ""   # provider name from providers dict, or "" to use summarization provider
    model: str = ""      # model override, or "" to use summarization model
    max_response_tokens: int = 2048


@dataclass
class VirtualContextConfig:
    version: str = "0.2"
    storage_root: str = ".virtualcontext"
    context_window: int = 120_000
    token_counter: str = "estimate"
    tag_generator: TagGeneratorConfig = field(default_factory=TagGeneratorConfig)
    tag_rules: list[TagPromptRule] = field(default_factory=list)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    compactor: CompactorConfig = field(default_factory=CompactorConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    assembler: AssemblerConfig = field(default_factory=AssemblerConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    paging: PagingConfig = field(default_factory=PagingConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    tool_output: ToolOutputConfig = field(default_factory=ToolOutputConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    facts: FactsConfig = field(default_factory=FactsConfig)
    supersession: SupersessionConfig = field(default_factory=SupersessionConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    providers: dict[str, dict] = field(default_factory=dict)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def cost_tracking(self) -> TelemetryConfig:
        """Deprecated — use ``telemetry`` directly."""
        return self.telemetry


@dataclass(frozen=True)
class CompactionLeaseClaim:
    """Return value of ``claim_compaction_lease``.

    Carries both the claim decision and the previous row's identity so
    the takeover path can scope cleanup DELETEs on ``prev_operation_id``
    without a second round-trip. ``prev_operation_id`` and
    ``prev_owner_worker_id`` are ``None`` when no prior running row
    existed at the caller's lifecycle_epoch.
    """
    claimed: bool
    prev_operation_id: str | None
    prev_owner_worker_id: str | None


class CompactionLeaseLost(Exception):
    """Raised by compaction-scoped store helpers when the per-write
    ownership guard's ``INSERT ... SELECT`` / ``UPDATE ... WHERE EXISTS``
    matches zero rows because the ``compaction_operation`` row has been
    flipped to ``'abandoned'`` by a takeover on another worker (or the
    owner_worker_id no longer matches, or the lifecycle_epoch has moved).

    The compactor pipeline's outer handler catches this to emit
    ``COMPACTION_WRITE_REJECTED`` and exit cleanly via
    ``exit_compaction(success=False)`` without walking the remaining
    phases. Do NOT swallow this in ``except Exception`` — it must
    propagate to the wrapper.
    """

    def __init__(self, operation_id: str, *, write_site: str) -> None:
        super().__init__(
            f"compaction lease lost: operation_id={operation_id} "
            f"write_site={write_site}",
        )
        self.operation_id = operation_id
        self.write_site = write_site


# ---------------------------------------------------------------------------
# VCMERGE typed contracts (T1.1-T1.3 per plan section 3.1)
#
# The merge surface uses three dataclasses across the cloud / engine boundary.
# All three are stable wire-shape types: cloud's REST handler depends on the
# field names + types in `MergeAuditView` and `ReservationResult`; the
# engine's body method returns a `MergeStats` to the caller for response
# shaping. Renames or field-removals here are wire-breaking. Additions are
# safe (default-defaulted in the cloud envelope renderer).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeStats:
    """Returned by `Store.merge_conversation_data` (S1.3 / S1.4) to the
    caller (cloud's REST handler via `Engine.merge_conversation`). Counts
    the rows moved per per-conversation table during the body transaction.

    The dict keys are table names from the 21-table per-conv set
    (segments, segment_tags, canonical_turns, canonical_turn_anchors,
    canonical_turn_chunks, ingest_batches, facts, fact_tags, fact_links,
    tool_outputs, tool_calls, request_captures, request_turn_counters,
    request_context, tag_summaries, tag_summary_embeddings, tag_aliases,
    turn_tool_outputs, segment_tool_outputs, chain_snapshots, media_outputs).
    Cloud surfaces the dict directly in the SSE `conversation_merged`
    event payload as `rows_moved` per spec section 12.2 (the v3 move-
    semantics field name; `rows_copied` was the v2 name and is stale).
    """
    merge_id: str
    source_conversation_id: str
    target_conversation_id: str
    tenant_id: str
    rows_moved: dict[str, int]
    sort_key_offset: float
    request_turn_offset: int
    started_at: datetime
    completed_at: datetime
    # B-D9 (codex iter-2 P2): explicit success + elapsed_seconds. The body
    # method only returns MergeStats on a successful commit, so success is
    # always True here today; the field is reserved for future partial-
    # success states (e.g., post-commit pendings failure that doesn't roll
    # back the body). elapsed_seconds is wall-clock seconds between
    # started_at and completed_at, surfaced for SSE / dashboard rendering
    # without forcing readers to recompute the delta.
    success: bool = True
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class MergeAuditView:
    """Frozen view of a single `merge_audit` row, returned by the lookup
    helpers (S1.5, S1.6) and by `try_reserve_merge_audit_in_progress`
    (S1.1, S1.2) when an idempotent retry collides with an existing row.

    Cloud's REST handler renders the 5-state idempotency envelope from
    this view (per spec section 12.7's discriminator). Frozen so test
    assertions and cloud's response builder can rely on immutability.
    """
    merge_id: str
    tenant_id: str
    source_conversation_id: str
    target_conversation_id: str
    status: Literal["in_progress", "committed", "rolled_back"]
    started_at: datetime
    completed_at: datetime | None
    source_label_at_merge: str
    rows_moved_json: str | None
    error_message: str | None


@dataclass(frozen=True)
class ReservationResult:
    """Returned by `try_reserve_merge_audit_in_progress` (S1.1 / S1.2).
    The 5-state `status` (per codex iter-1 v1.4-3 + spec section 12.7)
    discriminates the action cloud's REST handler should take:

    - "reserved": INSERT succeeded, this caller owns the merge body.
      `merge_id` is the freshly-generated UUID; `existing` is None.
    - "in_progress": prior INSERT succeeded with status='in_progress';
      another caller is mid-body. `existing` is populated; cloud renders
      the in-progress envelope.
    - "committed_match": prior INSERT succeeded with status='committed'
      AND the prior call's source_label_at_merge MATCHES this caller's
      label. Idempotent retry; cloud renders success envelope from
      `existing`.
    - "committed_mismatch": prior INSERT succeeded with status='committed'
      AND the prior call's source_label_at_merge DIFFERS. Cloud renders
      a label-mismatch error envelope referencing `existing`.
    - "race_retry": rare race where the winner row transitioned
      `in_progress -> rolled_back` between this caller's INSERT-fail and
      its SELECT. Cloud retries the reservation flow.
    """
    status: Literal["reserved", "in_progress", "committed_match",
                    "committed_mismatch", "race_retry"]
    merge_id: str
    existing: MergeAuditView | None
