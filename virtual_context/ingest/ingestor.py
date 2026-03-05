"""DocumentIngestor: parse → chunk → tag → compact → supersede → store."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from ..core.semantic_search import chunk_segment_text
from ..types import (
    DocumentIngestionResult,
    Message,
    SegmentMetadata,
    StoredSegment,
    TaggedSegment,
)

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Orchestrate document ingestion through the existing compaction pipeline."""

    def __init__(self, engine):
        # Avoid circular import by accepting engine instance (duck-typed)
        self.engine = engine
        self.config = engine.config.document_ingestion
        self._store = engine._store
        self._tag_gen = engine._tag_gen
        self._compactor = engine._compactor
        self._token_counter = engine._token_counter

    def ingest(self, path: Path, source_label: str = "") -> DocumentIngestionResult:
        """Ingest a document: parse → chunk → tag → compact → supersede → store."""
        result = DocumentIngestionResult(source_path=str(path))
        label = source_label or path.stem

        # 1. Parse
        from .parsers import parse_document

        try:
            raw_text = parse_document(path)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        if not raw_text.strip():
            result.errors.append("Document is empty after parsing")
            return result

        # 2. Chunk
        chunks = chunk_segment_text(
            raw_text,
            max_words=self.config.chunk_max_words,
            min_words=self.config.chunk_min_words,
        )
        result.chunks_created = len(chunks)

        if not chunks:
            result.errors.append("No chunks produced from document")
            return result

        # 3. Tag each chunk
        existing_tags = [t.tag for t in self._store.get_all_tags()]
        tagged_segments: list[TaggedSegment] = []
        for i, chunk in enumerate(chunks):
            tag_result = self._tag_gen.generate_tags(chunk, existing_tags=existing_tags)
            segment = TaggedSegment(
                id=f"doc_{uuid4().hex[:8]}_{i}",
                primary_tag=tag_result.tags[0] if tag_result.tags else "document",
                tags=tag_result.tags or ["document"],
                messages=[Message(role="user", content=chunk)],
                token_count=self._token_counter(chunk),
            )
            tagged_segments.append(segment)

        # 4. Compact each segment (summary + fact extraction)
        all_facts = []
        comp_results = self._compactor.compact(tagged_segments)
        for comp_result in comp_results:
            stored = StoredSegment(
                ref=comp_result.segment_id,
                session_id=label,
                primary_tag=comp_result.primary_tag,
                tags=comp_result.tags,
                summary=comp_result.summary,
                summary_tokens=comp_result.summary_tokens,
                full_text=comp_result.full_text,
                full_tokens=comp_result.original_tokens,
                messages=comp_result.messages,
                metadata=comp_result.metadata,
                compaction_model=self._compactor.model_name,
                compression_ratio=comp_result.compression_ratio,
            )
            self._store.store_segment(stored)
            self.engine._embed_and_store_chunks(stored)
            result.segments_stored += 1

            if comp_result.facts:
                for fact in comp_result.facts:
                    fact.segment_ref = stored.ref
                    fact.session_id = label
                self._store.store_facts(comp_result.facts)
                all_facts.extend(comp_result.facts)

        result.facts_extracted = len(all_facts)

        # 5. Supersession check
        if all_facts and self.config.supersession.enabled:
            checker = self._build_supersession_checker()
            if checker:
                result.facts_superseded = checker.check_and_supersede(all_facts)

        return result

    def _build_supersession_checker(self):
        """Build checker using configured or default provider."""
        sc = self.config.supersession
        provider_name = sc.provider or self.engine.config.summarization.provider
        model = sc.model or self.engine.config.summarization.model
        provider_config = self.engine.config.providers.get(provider_name, {})
        llm = self.engine._build_provider(provider_name, provider_config)
        if not llm:
            return None
        from .supersession import FactSupersessionChecker

        return FactSupersessionChecker(
            llm_provider=llm,
            model=model,
            store=self._store,
            config=sc,
        )
