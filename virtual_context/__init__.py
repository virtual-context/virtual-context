"""virtual-context: OS-style virtual memory for LLM session context management."""

from .config import load_config
from .engine import VirtualContextEngine
from .types import (
    AssembledContext,
    CompactionReport,
    CompactionSignal,
    ContextSnapshot,
    DomainDef,
    Message,
    VirtualContextConfig,
)

__version__ = "0.1.0"

__all__ = [
    "VirtualContextEngine",
    "load_config",
    "AssembledContext",
    "CompactionReport",
    "CompactionSignal",
    "ContextSnapshot",
    "DomainDef",
    "Message",
    "VirtualContextConfig",
]
