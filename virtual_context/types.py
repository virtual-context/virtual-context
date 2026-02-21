"""All dataclasses, Protocols, and type aliases for virtual-context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Message & Turn
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime | None = None
    metadata: dict | None = None


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
    broad: bool = False  # True when query is vague/retrospective/overview
    temporal: bool = False  # True when query references a time position ("first thing", "early on")
    related_tags: list[str] = field(default_factory=list)  # semantic alternates for query expansion


# Pattern constants — canonical definitions in patterns.py, re-exported here for compat
from .patterns import DEFAULT_BROAD_PATTERNS, DEFAULT_TEMPORAL_PATTERNS  # noqa: F401


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
    broad_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_BROAD_PATTERNS))
    temporal_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_TEMPORAL_PATTERNS))
    broad_heuristic_enabled: bool = True
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
    turn_number: int
    message_hash: str              # sha256[:16] of user+assistant content
    tags: list[str] = field(default_factory=list)
    primary_tag: str = "_general"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_date: str = ""         # e.g. "2023/05/25 (Thu) 10:04" or ISO timestamp


@dataclass
class EngineStateSnapshot:
    """Serializable snapshot of engine state for persistence across restarts."""
    session_id: str
    compacted_through: int
    turn_tag_entries: list[TurnTagEntry]
    turn_count: int  # len(conversation_history) // 2
    saved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    split_processed_tags: list[str] = field(default_factory=list)  # tags already split/summarized
    working_set: list[WorkingSetEntry] = field(default_factory=list)  # paging depth state
    trailing_fingerprint: str = ""  # hash of last N user messages for session matching on restart


@dataclass
class TagPromptRule:
    """Per-tag rules for priority, TTL, and custom summary prompts."""
    match: str  # fnmatch pattern, e.g. "architecture*", "debug*"
    ttl_days: int | None = None
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


@dataclass
class SegmentMetadata:
    entities: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    date_references: list[str] = field(default_factory=list)
    turn_count: int = 0
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
    session_id: str = ""
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
    session_date: str = ""   # session date from segment metadata


@dataclass
class ChunkEmbedding:
    """A chunk of segment text with its embedding vector."""
    segment_ref: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass
class TagSummary:
    """Layer-2 summary: one per cover tag, rolls up all segment summaries for that tag."""
    tag: str
    summary: str = ""
    description: str = ""  # 1-line tag description (~15-20 words) for enriched context hints
    summary_tokens: int = 0
    source_segment_refs: list[str] = field(default_factory=list)
    source_turn_numbers: list[int] = field(default_factory=list)
    covers_through_turn: int = -1  # highest turn number covered; -1 = never built
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
class SessionStats:
    """Aggregate statistics for a single engine session."""
    session_id: str = ""
    segment_count: int = 0
    total_full_tokens: int = 0
    total_summary_tokens: int = 0
    compression_ratio: float = 0.0
    distinct_tags: list[str] = field(default_factory=list)
    oldest_segment: datetime | None = None
    newest_segment: datetime | None = None
    compaction_model: str = ""


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
    broad: bool = False  # True when the query was detected as broad/retrospective
    temporal: bool = False  # True when the query references a time position


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------

@dataclass
class SessionCostSummary:
    """Running session cost totals."""
    total_retrievals: int = 0
    total_compactions: int = 0
    total_tag_generations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


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

    ``autonomous_models`` lists model-name substrings (case-insensitive) that
    are trusted to manage their own context via tool calls.  When the request's
    model matches any entry, the proxy injects ``vc_expand_topic`` /
    ``vc_collapse_topic`` tools and a budget dashboard.  Models that don't
    match run in *supervised* mode: VC manages paging silently via
    ``auto_promote`` / ``auto_evict``.
    """
    enabled: bool = False
    autonomous_models: list[str] = field(default_factory=lambda: [
        "opus", "sonnet", "gpt-4", "gpt-4o",
    ])
    auto_promote: bool = True   # auto-expand on strong retrieval match
    auto_evict: bool = True     # auto-collapse coldest when over budget


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    core_context: str = ""
    tag_sections: dict[str, str] = field(default_factory=dict)
    conversation_history: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    budget_breakdown: dict[str, int] = field(default_factory=dict)
    prepend_text: str = ""
    matched_tags: list[str] = field(default_factory=list)
    context_hint: str = ""  # Topic list injected post-compaction
    broad: bool = False  # True when query is broad — include all history
    temporal: bool = False  # True when query references a time position


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
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    context_window: int = 120_000
    soft_threshold: float = 0.70
    hard_threshold: float = 0.85
    protected_recent_turns: int = 6


@dataclass
class SegmenterConfig:
    pass


@dataclass
class CompactorConfig:
    summary_ratio: float = 0.15
    min_summary_tokens: int = 200
    max_summary_tokens: int = 2000
    max_concurrent_summaries: int = 4
    overflow_buffer: float = 1.2
    llm_token_overhead: int = 8000  # extra tokens for thinking/reasoning overhead


@dataclass
class RetrieverConfig:
    skip_active_tags: bool = True
    active_tag_lookback: int = 4
    tag_context_max_tokens: int = 30_000
    strategy_configs: dict[str, StrategyConfig] = field(default_factory=lambda: {
        "default": StrategyConfig()
    })
    anchorless_lookback: int = 6           # how many recent turns for working set
    inbound_tagger_type: str = "llm"       # "llm" (default) or "embedding"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_threshold: float = 0.3


@dataclass
class AssemblerConfig:
    core_context_max_tokens: int = 18_000
    tag_context_max_tokens: int = 30_000
    core_files: list[dict] = field(default_factory=list)
    recent_turns_always_included: int = 3
    context_hint_enabled: bool = True
    context_hint_max_tokens: int = 2000


@dataclass
class SummarizationConfig:
    provider: str = "ollama"
    model: str = "qwen3:4b-instruct-2507-fp16"
    max_tokens: int = 1000
    temperature: float = 0.3


@dataclass
class StorageConfig:
    backend: str = "sqlite"
    root: str = ".virtualcontext/store"
    sqlite_path: str = ".virtualcontext/store.db"


@dataclass
class CostTrackingConfig:
    enabled: bool = False
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ProxyInstanceConfig:
    """Configuration for a single proxy listener instance."""
    port: int = 5757
    upstream: str = ""
    label: str = ""
    host: str = "127.0.0.1"


@dataclass
class ProxyConfig:
    request_log_dir: str = ".virtualcontext/request_log"
    request_log_max_files: int = 50
    upstream_context_limit: int = 200_000  # max tokens the upstream model accepts
    instances: list[ProxyInstanceConfig] = field(default_factory=list)


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
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)
    paging: PagingConfig = field(default_factory=PagingConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    providers: dict[str, dict] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
