"""All dataclasses, Protocols, and type aliases for virtual-context."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
# Classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    domain: str
    confidence: float  # 0.0 – 1.0
    source: str  # "keyword", "embedding", "llm"


@dataclass
class DomainDef:
    name: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    priority: int = 5
    summary_prompt: str | None = None
    retrieval_limit: int = 3
    retrieval_max_tokens: int = 5000
    ttl_days: int | None = None


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
    domains_in_context: list[str] = field(default_factory=list)


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
class DomainSegment:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = "_general"
    secondary_domains: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    token_count: int = 0
    start_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    confidence: float = 0.0


@dataclass
class SegmentMetadata:
    entities: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    date_references: list[str] = field(default_factory=list)
    turn_count: int = 0
    time_span: tuple[datetime, datetime] | None = None


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    segment_id: str
    domain: str
    summary: str
    summary_tokens: int
    original_tokens: int
    compression_ratio: float
    metadata: SegmentMetadata
    full_text: str
    messages: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CompactionReport:
    segments_compacted: int
    tokens_freed: int
    domains: list[str] = field(default_factory=list)
    results: list[CompactionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass
class StoredSegment:
    ref: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    domain: str = "_general"
    secondary_domains: list[str] = field(default_factory=list)
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
    domain: str = "_general"
    secondary_domains: list[str] = field(default_factory=list)
    summary: str = ""
    summary_tokens: int = 0
    full_tokens: int = 0
    metadata: SegmentMetadata = field(default_factory=SegmentMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DomainStats:
    domain: str = ""
    segment_count: int = 0
    total_full_tokens: int = 0
    total_summary_tokens: int = 0
    oldest_segment: datetime | None = None
    newest_segment: datetime | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    domains_matched: list[str] = field(default_factory=list)
    summaries: list[StoredSummary] = field(default_factory=list)
    full_detail: list[StoredSegment] = field(default_factory=list)
    total_tokens: int = 0
    retrieval_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    core_context: str = ""
    domain_sections: dict[str, str] = field(default_factory=dict)
    conversation_history: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    budget_breakdown: dict[str, int] = field(default_factory=dict)
    prepend_text: str = ""


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
    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...


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
    min_confidence: float = 0.3


@dataclass
class CompactorConfig:
    summary_ratio: float = 0.15
    min_summary_tokens: int = 200
    max_summary_tokens: int = 2000
    max_concurrent_summaries: int = 4
    overflow_buffer: float = 1.2


@dataclass
class RetrieverConfig:
    deep_retrieve_threshold: float = 0.8
    skip_active_domains: bool = True
    active_domain_lookback: int = 4
    domain_context_max_tokens: int = 30_000
    domains: list[DomainDef] = field(default_factory=list)
    velocity_fallback: bool = True
    velocity_lookback: int = 10  # turn pairs to look back
    velocity_threshold: float = 0.3  # min concentration to trigger fallback


@dataclass
class AssemblerConfig:
    core_context_max_tokens: int = 18_000
    domain_context_max_tokens: int = 30_000
    core_files: list[dict] = field(default_factory=list)


@dataclass
class SummarizationConfig:
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 1000
    temperature: float = 0.3


@dataclass
class StorageConfig:
    backend: str = "filesystem"
    root: str = ".virtualcontext/store"


@dataclass
class VirtualContextConfig:
    version: str = "1.0"
    storage_root: str = ".virtualcontext"
    context_window: int = 120_000
    token_counter: str = "estimate"
    domains: dict[str, DomainDef] = field(default_factory=dict)
    classifier_pipeline: list[dict] = field(default_factory=list)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    compactor: CompactorConfig = field(default_factory=CompactorConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    assembler: AssemblerConfig = field(default_factory=AssemblerConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    providers: dict[str, dict] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
