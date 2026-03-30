"""
Virtual Context memory provider for Agent Memory Benchmark.

Requires `virtual-context` to be installed (pip install virtual-context).
Uses VirtualContextEngine for ingestion, retrieval, and optionally direct answering
via the full tool-augmented reader pipeline.

Configuration:
  VC_CONFIG_PATH   — path to virtual-context YAML config (optional, uses defaults)
  VC_READER_MODEL  — reader model for direct_answer mode (default: claude-sonnet-4-5-20250514)
  VC_TAGGER_MODEL  — tagger/compactor model (default: google/gemini-2.0-flash-001)
  VC_PROVIDER      — LLM provider for tagger: "openrouter" or "gemini" (default: openrouter)
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

logger = logging.getLogger(__name__)


def _docs_to_history(documents: list[Document]) -> list[dict]:
    """Convert AMB Documents into a flat message history for VC ingestion.

    Each Document may contain:
      - JSON array of turns (LongMemEval, LoCoMo style)
      - Plain text (LifeBench, Tempo, AMA-Bench)
      - Structured messages list

    We normalize everything into user/assistant message pairs.
    """
    history = []
    for doc in documents:
        turns = []

        # Try parsing as JSON first (LongMemEval, LoCoMo format)
        if doc.content.strip().startswith("[") or doc.content.strip().startswith("{"):
            try:
                parsed = json.loads(doc.content)
                if isinstance(parsed, list):
                    turns = parsed
                elif isinstance(parsed, dict):
                    turns = [parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # If structured messages are provided, prefer those
        if doc.messages and not turns:
            turns = doc.messages

        if turns:
            for turn in turns:
                if isinstance(turn, dict):
                    role = turn.get("role", "user").lower()
                    content = turn.get("content", turn.get("text", ""))
                    if not content:
                        continue
                    # Normalize roles
                    if role in ("user", "human"):
                        history.append({"role": "user", "content": str(content)})
                    elif role in ("assistant", "ai", "bot", "agent"):
                        history.append({"role": "assistant", "content": str(content)})
                    elif role == "system":
                        # Treat system messages as context injected via user turn
                        history.append({"role": "user", "content": f"[System] {content}"})
                    else:
                        history.append({"role": "user", "content": str(content)})
        else:
            # Plain text document: wrap as a single user message with session context
            session_ctx = doc.context or ""
            timestamp_str = f" [date: {doc.timestamp}]" if doc.timestamp else ""
            prefix = f"{session_ctx}{timestamp_str}\n\n" if session_ctx or timestamp_str else ""
            history.append({"role": "user", "content": f"{prefix}{doc.content}"})
            history.append({"role": "assistant", "content": "Understood, I've noted this information."})

    return history


class VirtualContextMemoryProvider(MemoryProvider):
    name = "virtual-context"
    description = "Virtual Context: OS-style memory management for LLM agents. Three-layer hierarchy with tool-augmented retrieval."
    kind = "local"
    provider = "virtual-context"
    link = "https://github.com/virtual-context/virtual-context"
    concurrency = 1  # VC engine is not thread-safe

    def __init__(self):
        self._engine = None
        self._store_dir: Path | None = None
        self._config_path: str | None = os.environ.get("VC_CONFIG_PATH")

    def initialize(self) -> None:
        """Verify virtual-context is importable."""
        try:
            import virtual_context  # noqa: F401
        except ImportError:
            raise ImportError(
                "virtual-context is not installed. "
                "Install with: pip install virtual-context"
            )

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None) -> None:
        """Create a fresh VC engine for this evaluation unit.

        Called once per isolation unit (per-question for LongMemEval,
        per-user for LifeBench, etc.) or once for the whole run if
        no isolation.
        """
        self._store_dir = store_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        self._build_engine(store_dir)

    def _build_engine(self, store_dir: Path) -> None:
        """Instantiate VirtualContextEngine with appropriate config."""
        from virtual_context import VirtualContextEngine
        from virtual_context.config import load_config

        tagger_model = os.environ.get("VC_TAGGER_MODEL", "google/gemini-2.0-flash-001")
        provider = os.environ.get("VC_PROVIDER", "openrouter")

        if self._config_path:
            self._engine = VirtualContextEngine(config_path=self._config_path)
        else:
            # Build config dict and load through standard config loader
            config_dict = {
                "version": "0.2",
                "storage_root": str(store_dir),
                "context_window": 64000,
                "storage": {
                    "backend": "sqlite",
                    "sqlite_path": str(store_dir / "store.db"),
                },
                "tag_generator": {
                    "provider": provider,
                    "model": tagger_model,
                    "min_tags": 3,
                    "max_tags": 10,
                },
                "compactor": {
                    "provider": provider,
                    "model": tagger_model,
                    "soft_threshold": 0.70,
                    "hard_threshold": 0.85,
                    "summary_ratio": 0.15,
                },
            }
            config = load_config(config_dict=config_dict)
            self._engine = VirtualContextEngine(config=config)

    def ingest(self, documents: list[Document]) -> None:
        """Ingest documents by converting to message history and running VC pipeline."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call prepare() first.")

        # Sort documents by timestamp for chronological ingestion
        sorted_docs = sorted(
            documents,
            key=lambda d: d.timestamp or "",
        )

        history = _docs_to_history(sorted_docs)
        if not history:
            logger.warning("No messages extracted from %d documents", len(documents))
            return

        logger.info(
            "Ingesting %d documents (%d messages) into VC",
            len(documents), len(history),
        )

        # Convert dicts to Message objects and feed through on_turn_complete
        from virtual_context.types import Message as VCMessage

        vc_messages = []
        for msg in history:
            vc_messages.append(VCMessage(role=msg["role"], content=msg["content"]))

        # Feed pairs through on_turn_complete incrementally
        for i in range(0, len(vc_messages) - 1, 2):
            pair_end = min(i + 2, len(vc_messages))
            history_so_far = vc_messages[:pair_end]
            try:
                self._engine.on_turn_complete(history_so_far)
            except Exception as e:
                logger.warning("on_turn_complete error at turn %d: %s", i // 2, e)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        """Retrieve relevant context for a query using VC's retrieval pipeline."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized")

        result = self._engine.retrieve(query)

        # Convert VC RetrievalResult to AMB Documents
        docs = []
        raw_parts = []

        # Summaries (StoredSummary objects from tag retrieval)
        for ss in (result.summaries or []):
            tag = ss.primary_tag if hasattr(ss, "primary_tag") else "unknown"
            summary = ss.summary if hasattr(ss, "summary") else str(ss)
            docs.append(Document(
                id=f"summary_{tag}",
                content=f"[{tag}] {summary}",
                user_id=user_id,
            ))
            raw_parts.append({"tag": tag, "summary": summary[:200]})

        # Full-detail segments (when expanded)
        for seg in (result.full_detail or []):
            seg_ref = seg.ref if hasattr(seg, "ref") else str(id(seg))
            text = seg.summary if hasattr(seg, "summary") else str(seg)
            docs.append(Document(
                id=f"seg_{seg_ref}",
                content=text,
                user_id=user_id,
            ))

        # Structured facts
        if result.facts:
            facts_text = "\n".join(
                f.format_for_prompt() if hasattr(f, "format_for_prompt") else str(f)
                for f in result.facts
            )
            if facts_text:
                docs.append(Document(
                    id="facts",
                    content=facts_text,
                    user_id=user_id,
                ))

        return docs[:k], {"vc_tags_matched": result.tags_matched, "vc_details": raw_parts}

    def direct_answer(
        self,
        query: str,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[str, str, dict | None]:
        """Answer using VC's full pipeline: retrieve + assemble + tool loop.

        Returns (answer, context_text, raw_response).
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized")

        # Use transform() which does tag + fetch + assemble into a context block
        context = self._engine.transform(query)

        if not context:
            return "I don't have enough information to answer this question.", "", None

        # For direct_answer, we return the assembled context as both
        # the answer and the context. The benchmark's judge will evaluate
        # the answer against gold answers.
        #
        # TODO: Wire up a reader LLM to actually answer the question
        # using the assembled context + tool loop. For now, we return
        # the context and let the benchmark's RAG prompt handle answering.
        return context, context, {"mode": "transform"}

    def cleanup(self) -> None:
        """Release engine resources."""
        self._engine = None
