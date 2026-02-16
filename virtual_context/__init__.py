"""virtual-context: OS-style virtual memory for LLM session context management."""

from .config import load_config
from .engine import VirtualContextEngine
from .types import (
    AssembledContext,
    CompactionReport,
    CompactionSignal,
    ContextSnapshot,
    Message,
    TagResult,
    TagStats,
    TagSummary,
    VirtualContextConfig,
)

__version__ = "0.2.1"

__all__ = [
    "VirtualContextEngine",
    "load_config",
    "AssembledContext",
    "CompactionReport",
    "CompactionSignal",
    "ContextSnapshot",
    "Message",
    "TagResult",
    "TagStats",
    "TagSummary",
    "VirtualContextConfig",
]
