"""ContextRetriever: tag inbound message and fetch relevant summaries by tag overlap."""

from __future__ import annotations

import logging
import math
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .store import ContextStore, tag_summary_embedding_vectors
from .tag_generator import TagGenerator, detect_temporal_heuristic
from .turn_tag_index import TurnTagIndex
from ..patterns import DEFAULT_TEMPORAL_PATTERNS
from ..types import (
    RetrievalCostReport,
    RetrievalResult,
    RetrieverConfig,
    SegmentMetadata,
    StoredSegment,
    StoredSummary,
    TagResult,
    TagSummary,
)

from .tag_canonicalizer import flatten_alias_map

logger = logging.getLogger(__name__)

_RETRIEVAL_BREAKDOWN_LOG_THRESHOLD_MS = 500.0



# Gathers a request's facts while its summaries are being chosen.
_FACTS_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="vc-facts")

class ContextRetriever:
    judgment_runtime = None  # engine-owned JudgmentRuntime; None = module default


    def __init__(
        self,
        tag_generator: TagGenerator,
        store: ContextStore,
        config: RetrieverConfig,
        turn_tag_index: TurnTagIndex | None = None,
        inbound_tagger: TagGenerator | None = None,
        conversation_id: str | None = None,
        session_state_provider=None,
        query_embed_fn=None,
        judgment_runtime=None,
    ) -> None:
        self.tag_generator = tag_generator
        self.judgment_runtime = judgment_runtime
        self.store = store
        self.config = config
        self._turn_tag_index = turn_tag_index
        self._inbound_tagger = inbound_tagger
        self._conversation_id = conversation_id
        self._session_state_provider = session_state_provider
        # Shared local embed function for dense fact retrieval. Independent of
        # ``inbound_tagger`` so ``fact_dense_retrieval`` works under any
        # ``inbound_tagger_type``. Signature: ``list[str] -> list[list[float]]``.
        self._query_embed_fn = query_embed_fn
        # Pre-compile heuristic patterns for embedding-based inbound tagger
        self._temporal_patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_TEMPORAL_PATTERNS]

    def _retrieval_memo_key(
        self, lookup_text: str, message: str, active_tags, context_turns, post_compaction: bool, all_tags,
    ) -> str:
        """Fingerprint of everything inbound tagging and candidate scoring read."""
        vocab = hashlib.sha256()
        for ts in all_tags:
            vocab.update(f"{ts.tag}\x1f{getattr(ts, 'usage_count', 0)}\x1e".encode("utf-8"))
        scoring = self.config.scoring
        payload = json.dumps({
            "q": lookup_text,
            "m": message,
            "active": sorted(str(t) for t in (active_tags or [])),
            "ctx": [str(t) for t in (context_turns or [])],
            "post": bool(post_compaction),
            "vocab": vocab.hexdigest(),
            "skip_active": bool(self.config.skip_active_tags),
            "ctx_turns": getattr(scoring, "embedding_context_turns", 0),
            "ctx_guard": bool(getattr(scoring, "embedding_context_guard", True)),
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _retrieval_memo_parts(lookup_text, message, active_tags, context_turns, post_compaction, all_tags) -> str:
        """Short per-component fingerprints, so a memo miss shows which input changed."""
        def short(value) -> str:
            return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:6]
        return ",".join([
            f"q={short(lookup_text)}", f"m={short(message)}",
            f"active={short(sorted(str(t) for t in (active_tags or [])))}",
            f"ctx={short([str(t) for t in (context_turns or [])])}",
            f"post={int(bool(post_compaction))}",
            f"names={short(sorted(ts.tag for ts in all_tags))}",
            f"counts={short([(ts.tag, getattr(ts, 'usage_count', 0)) for ts in all_tags])}",
        ])

    def _load_all_tags_snapshot(self) -> list:
        if self._session_state_provider is not None and self._conversation_id:
            cached = self._session_state_provider.load_tag_stats_snapshot(
                self._conversation_id,
            )
            if cached is not None:
                return cached
        all_tags = self.store.get_all_tags(
            conversation_id=self._conversation_id,
        )
        if self._session_state_provider is not None and self._conversation_id:
            self._session_state_provider.save_tag_stats_snapshot(
                self._conversation_id,
                all_tags,
            )
        return all_tags

    def _load_tag_summary_embedding_snapshot(self) -> dict[str, list[float]] | None:
        if self._session_state_provider is not None and self._conversation_id:
            cached = self._session_state_provider.load_tag_summary_embedding_snapshot(
                self._conversation_id,
            )
            if cached is not None:
                return cached
        stored = tag_summary_embedding_vectors(self.store, self._conversation_id)
        if self._session_state_provider is not None and self._conversation_id:
            self._session_state_provider.save_tag_summary_embedding_snapshot(
                self._conversation_id,
                stored,
            )
        return stored

    def _build_context_query_embedding(
        self, message: str, context_turns: list[str] | None,
    ) -> tuple[list[float] | None, str, float | None]:
        """Embed recent-turn context + the current message for the guard signal.

        Returns ``(embedding, mode, embed_ms)``. ``embedding`` is None (and mode
        ``"bare"``) whenever context blending is disabled, no context is
        available, or the inbound tagger cannot embed — in which case retrieval
        scores the bare query embedding exactly as before. ``mode`` is
        ``"guard"`` or ``"concat"`` per ``scoring.embedding_context_guard`` when
        a context vector is produced.
        """
        scoring = self.config.scoring
        n_ctx = getattr(scoring, "embedding_context_turns", 0)
        if n_ctx <= 0 or not context_turns:
            return None, "bare", None
        embed_text = getattr(self._inbound_tagger, "embed_text", None)
        if embed_text is None:
            return None, "bare", None
        recent = [t for t in context_turns[-n_ctx:] if t]
        if not recent:
            return None, "bare", None
        concat_text = (" ".join(recent) + " " + message).strip()
        _stage = time.monotonic()
        ctx_embedding = embed_text(concat_text)
        embed_ms = round((time.monotonic() - _stage) * 1000, 1)
        if not ctx_embedding:
            return None, "bare", embed_ms
        mode = "guard" if getattr(scoring, "embedding_context_guard", True) else "concat"
        return ctx_embedding, mode, embed_ms

    def _compute_idf_weights(self, all_tags: list | None = None) -> dict[str, float]:
        """Compute IDF weights from tag usage counts in the store.

        Returns a dict mapping tag → log(1 + total_segments / usage_count).
        Rare tags get higher weights than common ones.
        """
        if all_tags is None:
            all_tags = self.store.get_all_tags(
                conversation_id=self._conversation_id,
            )
        if not all_tags:
            return {}
        total = sum(ts.usage_count for ts in all_tags)
        if total == 0:
            return {}
        return {
            ts.tag: math.log(1 + total / max(ts.usage_count, 1))
            for ts in all_tags
        }

    def _fetch_all_facts(self) -> list:
        try:
            return self.store.query_facts(limit=9999, conversation_id=self._conversation_id)
        except Exception:
            return []

    def _fetch_facts_by_tags(self, tags: list[str], limit: int = 500) -> list:
        try:
            return self.store.query_facts(tags=tags, limit=limit, conversation_id=self._conversation_id)
        except Exception:
            return []

    def _fetch_facts_dense(
        self,
        message: str,
        context_turns: list[str] | None,
        expanded_tags: list[str],
        retrieval_metadata: dict,
    ) -> list:
        """Augment the legacy tag-gated fact fetch with a dense kNN ranking.

        Runs the exact gate-off legacy fetch first to preserve the
        non-evictable tag-gated floor (INV-6), then ranks the conversation's
        current-model fact vectors by cosine similarity to the (optionally
        recent-turn-blended) query embedding and unions the top
        ``fact_dense_top_n`` in front of the floor. A fact present in both
        sets appears once at its dense position. Attaches dense rank/score,
        the legacy floor ids, and per-fact source to *retrieval_metadata* so
        the assembler can preserve dense ordering without re-scoring.
        """
        from .math_utils import cosine_similarity

        _t0 = time.monotonic()
        # 1. Legacy floor — the exact gate-off fetch, order preserved.
        if self.config.prefetch_facts and expanded_tags:
            legacy_facts = self._fetch_facts_by_tags(expanded_tags)
        else:
            legacy_facts = self._fetch_all_facts()
        floor_ids = [f.id for f in legacy_facts]
        floor_set = set(floor_ids)

        embed_fn = self._query_embed_fn
        if embed_fn is None:
            logger.warning(
                "FACT_DENSE_BREAKDOWN dense retrieval enabled but no query "
                "embedder configured; falling back to legacy floor only",
            )
            return legacy_facts

        # 2. Query embeddings on the shared local provider.
        try:
            bare_vec = embed_fn([message])[0]
        except Exception:
            logger.warning("FACT_DENSE_BREAKDOWN query embed failed; legacy floor only")
            return legacy_facts

        scoring = self.config.scoring
        n_ctx = getattr(scoring, "embedding_context_turns", 0)
        ctx_vec = None
        if n_ctx > 0 and context_turns:
            recent = [t for t in context_turns[-n_ctx:] if t]
            if recent:
                concat_text = (" ".join(recent) + " " + message).strip()
                try:
                    ctx_vec = embed_fn([concat_text])[0]
                except Exception:
                    ctx_vec = None
        guard = getattr(scoring, "embedding_context_guard", True)

        # 3. Rank current-model fact vectors against the query. A store that
        # can search does it in place and returns only the winners.
        model = getattr(self.config, "embedding_model", "")
        top_n = getattr(self.config, "fact_dense_top_n", 20)
        if ctx_vec is None:
            queries = [bare_vec]
        else:
            queries = [bare_vec, ctx_vec] if guard else [ctx_vec]
        top: list[tuple[float, str, object]] | None = None
        search = getattr(self.store, "search_fact_embeddings", None)
        if callable(search):
            try:
                hits = search(self._conversation_id, model, queries, limit=top_n)
            except Exception:
                # A store that can search must not fall back to loading every vector.
                logger.error("FACT_DENSE_BREAKDOWN store search failed; legacy floor only", exc_info=True)
                hits = []
            if hits is not None:
                top = [(score, fact.id, fact) for fact, score in hits]
        if top is None:
            try:
                loaded = self.store.load_fact_embeddings(
                    self._conversation_id, model, expected_dim=len(bare_vec),
                )
            except Exception:
                loaded = {}
            scored: list[tuple[float, str, object]] = []
            for fid, (fact, vec) in loaded.items():
                score = max(cosine_similarity(q, vec) for q in queries)
                scored.append((score, fid, fact))
            # Cosine DESC, tie-break fact.id ASC (deterministic union order).
            scored.sort(key=lambda x: (-x[0], x[1]))
            top = scored[:top_n]

        # 4. Deterministic union by fact.id: dense first, floor-only after.
        dense_rank_by_id: dict[str, int] = {}
        dense_score_by_id: dict[str, float] = {}
        source_by_id: dict[str, str] = {}
        union: list = []
        seen: set[str] = set()
        for rank, (score, fid, fact) in enumerate(top):
            dense_rank_by_id[fid] = rank
            dense_score_by_id[fid] = score
            source_by_id[fid] = "both" if fid in floor_set else "dense"
            union.append(fact)
            seen.add(fid)
        for fact in legacy_facts:
            if fact.id in seen:
                continue
            union.append(fact)
            seen.add(fact.id)
            source_by_id[fact.id] = "tag"

        # 5. Metadata for the assembler dense-priority branch.
        retrieval_metadata["fact_dense_rank_by_id"] = dense_rank_by_id
        retrieval_metadata["fact_dense_score_by_id"] = dense_score_by_id
        retrieval_metadata["fact_tag_floor_ids"] = floor_ids
        retrieval_metadata["fact_source_by_id"] = source_by_id

        overlap = len(set(dense_rank_by_id) & floor_set)
        top1 = top[0][0] if top else 0.0
        logger.info(
            "FACT_DENSE_BREAKDOWN conv=%s dense=%d floor=%d overlap=%d union=%d "
            "top1_cos=%.4f load_score_ms=%.1f",
            (self._conversation_id or "none")[:12],
            len(dense_rank_by_id), len(floor_ids), overlap, len(union),
            top1, round((time.monotonic() - _t0) * 1000, 1),
        )
        return union

    def _load_alias_map(self) -> dict[str, str]:
        """alias -> canonical for this conversation; empty when the store has none."""
        getter = getattr(self.store, "get_tag_aliases", None)
        if not callable(getter):
            return {}
        try:
            try:
                aliases = getter(conversation_id=self._conversation_id)
            except TypeError:
                aliases = getter()
        except Exception:
            logger.debug("alias map unavailable", exc_info=True)
            return {}
        return flatten_alias_map(aliases or {})

    @staticmethod
    def _tag_summary_item(ts: TagSummary) -> StoredSummary:
        """A tag summary shaped as a retrievable summary, keeping its sources."""
        return StoredSummary(
            ref=f"tag-summary-{ts.tag}",
            primary_tag=ts.tag,
            tags=[ts.tag],
            summary=ts.summary,
            summary_tokens=ts.summary_tokens,
            metadata=SegmentMetadata(
                code_refs=list(getattr(ts, "code_refs", []) or []),
                canonical_turn_ids=list(getattr(ts, "source_canonical_turn_ids", None) or []),
            ),
            created_at=ts.updated_at,
            start_timestamp=ts.created_at,
            end_timestamp=ts.updated_at,
        )

    def _judgment_runtime(self):
        if self.judgment_runtime is not None:
            return self.judgment_runtime
        from .judgment import current
        return current()

    def _embedding_pool_live(self) -> bool:
        """Topic selection runs through Jev."""
        from .judgment import JudgmentMode

        rt = self._judgment_runtime()
        return rt.enabled_for("topic_select") and rt.mode_for("topic_select") is JudgmentMode.JEV

    def _embedding_pool_enabled(self) -> bool:
        from .judgment import JudgmentMode

        rt = self._judgment_runtime()
        return (
            rt.enabled_for("topic_select")
            and rt.mode_for("topic_select") is JudgmentMode.JEV
            and getattr(rt.config, "topic_pool_source", "fused") == "embedding"
        )

    def _embedding_pool(self, query_embedding: list[float]) -> dict[str, float] | None:
        """The ``topic_pool_size`` topics most similar to the query, by cosine similarity."""
        import numpy as np

        loaded = None
        provider = self._session_state_provider
        if (
            provider is not None and self._conversation_id
            and hasattr(provider, "load_tag_summary_embedding_matrix")
        ):
            loaded = provider.load_tag_summary_embedding_matrix(self._conversation_id)
        if loaded is None:
            stored = tag_summary_embedding_vectors(self.store, self._conversation_id)
            tags = list(stored)
            if not tags:
                return None
            rows = np.asarray([stored[tag] for tag in tags], dtype=np.float32)
            norms = np.linalg.norm(rows, axis=1)
            keep = norms > 0.0
            loaded = ([t for t, ok in zip(tags, keep.tolist()) if ok], rows[keep] / norms[keep][:, None])
        tags, matrix = loaded
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if not tags or norm == 0.0:
            return None
        similarities = matrix @ (query / norm)
        size = max(1, int(self._judgment_runtime().config.topic_pool_size))
        top = np.argsort(similarities)[::-1][:size]
        return {tags[i]: float(similarities[i]) for i in top}

    def _segment_candidates(self, query_embedding: list[float] | None) -> list[StoredSummary]:
        """The segments whose chunks are most similar to the query, best first."""
        rt = self._judgment_runtime()
        size = int(getattr(rt.config, "segment_pool_size", 0) or 0)
        if size <= 0 or query_embedding is None or not self._conversation_id:
            return []
        native = getattr(self.store, "search_segment_chunks_by_embedding", None)
        if getattr(self.config, "vector_search_enabled", False) and callable(native):
            chosen = self._native_segment_refs(native, query_embedding, size)
        else:
            chosen = self._matrix_segment_refs(query_embedding, size)
        segments: list[StoredSummary] = []
        for ref in chosen:
            try:
                segment = self.store.get_segment(ref, conversation_id=self._conversation_id)
            except Exception:
                segment = None
            if segment is not None and (segment.summary or "").strip():
                segments.append(self._segment_item(segment))
        return segments

    def _native_segment_refs(self, search, query_embedding: list[float], size: int) -> list[str]:
        """Distinct segments in the order the database ranks their chunks."""
        chosen: list[str] = []
        after = None
        while len(chosen) < size:
            page = search(query_embedding, conversation_id=self._conversation_id, limit=200, after=after)
            if not page:
                break
            for row in page:
                if row["segment_ref"] not in chosen:
                    chosen.append(row["segment_ref"])
                    if len(chosen) >= size:
                        break
            after = page[-1]["cursor"]
        return chosen

    def _matrix_segment_refs(self, query_embedding: list[float], size: int) -> list[str]:
        provider = self._session_state_provider
        if provider is None or not hasattr(provider, "load_segment_chunk_matrix"):
            return []
        import numpy as np

        loaded = provider.load_segment_chunk_matrix(self._conversation_id)
        if loaded is None:
            return []
        refs, matrix = loaded
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        similarities = matrix @ (query / norm)
        chosen: list[str] = []
        for index in np.argsort(similarities)[::-1]:
            ref = refs[index]
            if ref not in chosen:
                chosen.append(ref)
                if len(chosen) >= size:
                    break
        return chosen

    @staticmethod
    def _segment_item(segment: StoredSegment) -> StoredSummary:
        return StoredSummary(
            ref=segment.ref,
            primary_tag=segment.primary_tag,
            tags=list(segment.tags),
            summary=segment.summary,
            summary_tokens=segment.summary_tokens,
            full_tokens=segment.full_tokens,
            metadata=segment.metadata,
            created_at=segment.created_at,
            start_timestamp=segment.start_timestamp,
            end_timestamp=segment.end_timestamp,
        )

    def _select_topics(
        self,
        lookup_text: str,
        ranked_tags: list[str],
        max_results: int,
        provider: object | None,
        memo_key: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> tuple[list[str], bool, list[StoredSummary]]:
        """The topics to retrieve, whether Jev chose them, and the items it chose in order.

        A turn's continuation calls resend the same request, so a Jev choice
        is kept in the shared retrieval memo next to the scores it was made
        from.
        """
        from .judgment import select_topics

        loaded: dict[str, TagSummary] = {}

        def load(tags: list[str]) -> dict[str, str]:
            for tag in tags:
                if tag in loaded:
                    continue
                try:
                    ts = self.store.get_tag_summary(tag, conversation_id=self._conversation_id or "")
                except Exception:
                    ts = None
                if ts is not None and (ts.summary or "").strip():
                    loaded[tag] = ts
            return {tag: loaded[tag].summary for tag in tags if tag in loaded}

        def items(ordered: list[str], segments: dict[str, StoredSummary]) -> list[StoredSummary]:
            out: list[StoredSummary] = []
            for key in ordered:
                kind, name = key[:2], key[2:]
                if kind == "T:" and name in loaded:
                    out.append(self._tag_summary_item(loaded[name]))
                elif kind == "S:" and name in segments:
                    out.append(segments[name])
            return out

        topics_key = f"{memo_key}:topics" if memo_key else ""
        can_memo = bool(topics_key and provider is not None and self._conversation_id)
        if can_memo:
            cached = provider.load_retrieval_memo(self._conversation_id, topics_key)
            ordered = cached.get("ordered") if isinstance(cached, dict) else None
            if isinstance(ordered, list) and ordered and all(isinstance(k, str) for k in ordered):
                tags = [k[2:] for k in ordered if k.startswith("T:")]
                load(tags)
                segments: dict[str, StoredSummary] = {}
                for key in ordered:
                    if key.startswith("S:"):
                        try:
                            segment = self.store.get_segment(key[2:], conversation_id=self._conversation_id)
                        except Exception:
                            segment = None
                        if segment is not None:
                            segments[key[2:]] = self._segment_item(segment)
                return tags, True, items(ordered, segments)
        segment_pool = self._segment_candidates(query_embedding) if self._embedding_pool_live() else []
        tags, chosen_by_jev, ordered = select_topics(
            lookup_text, ranked_tags, load,
            max_results=max_results, runtime=self.judgment_runtime,
            segment_candidates=[(seg.ref, seg.summary) for seg in segment_pool],
        )
        if chosen_by_jev and can_memo:
            provider.save_retrieval_memo(self._conversation_id, topics_key, {"ordered": ordered})
        return tags, chosen_by_jev, items(ordered, {seg.ref: seg for seg in segment_pool}) if chosen_by_jev else []

    def _load_all_tag_summaries(self, token_budget: int) -> tuple[list[StoredSummary], int]:
        """Load all tag summaries within *token_budget*.

        Returns ``(selected_summaries, total_tokens)``.  Used by the
        post-compaction summary floor.
        """
        tag_summaries = self.store.get_all_tag_summaries(conversation_id=self._conversation_id)
        if not tag_summaries:
            return [], 0

        selected: list[StoredSummary] = []
        total_tokens = 0
        for ts in tag_summaries:
            if total_tokens + ts.summary_tokens > token_budget:
                break
            selected.append(self._tag_summary_item(ts))
            total_tokens += ts.summary_tokens
        return selected, total_tokens

    def _detect_temporal(self, message: str) -> bool:
        from .judgment import judge_temporal_intent
        return judge_temporal_intent(
            message, lambda: detect_temporal_heuristic(message, self._temporal_patterns),
            runtime=self.judgment_runtime,
        )

    def retrieve(
        self,
        message: str,
        current_active_tags: list[str] | None = None,
        current_utilization: float = 0.0,
        post_compaction: bool = False,
        context_turns: list[str] | None = None,
        *,
        entries_snapshot: list | None = None,
        query_text: str | None = None,
        facts_transform: Callable[[list], list] | None = None,
    ) -> RetrievalResult:
        """Tag inbound message, fetch relevant summaries by tag overlap.

        Args:
            message: The inbound user message.
            current_active_tags: Tags from recent conversation turns to skip.
            current_utilization: Current context window usage ratio (0.0-1.0).
            context_turns: Recent user/assistant text for context-aware tagging.
            query_text: Optional contextual lookup text. The original message
                stays separate and alone determines temporal intent. Query
                topics still drive relevance and working-set/history selection;
                they are not canonical actor claims or ingest tags.
            entries_snapshot: optional bounded view of
                ``TurnTagIndex.entries`` captured by the caller at the
                inbound method entry. When provided, the retriever's
                fallback paths (working-set tags on tagger failure,
                inherit-from-previous on _general) read tags from the
                snapshot so a concurrent legacy-tagger append does not
                give one inbound call mixed views of the index. Default
                None preserves backward compat for non-proxy callers.
        """
        start_time = time.monotonic()
        _breakdown: dict[str, float] = {}

        def _note(stage: str, started_at: float) -> None:
            _breakdown[stage] = round((time.monotonic() - started_at) * 1000, 1)

        active_tags = set(current_active_tags or [])
        overflow: list[StoredSummary] = []
        lookup_text = message if query_text is None else query_text

        # Tag the inbound message — scope vocabulary to current conversation
        _load_tags_stage = time.monotonic()
        all_tags = self._load_all_tags_snapshot()
        _note("load_all_tags", _load_tags_stage)
        store_tags = [ts.tag for ts in all_tags]
        # Continuation calls of one turn resend the same request; the tagging
        # and scoring they need is identical, so it is kept for a few minutes
        # in shared state keyed by everything those stages read.
        _memo: dict | None = None
        _memo_key = ""
        _provider = self._session_state_provider
        if (
            _provider is not None and self._conversation_id
            and hasattr(_provider, "load_retrieval_memo")
        ):
            _memo_key = self._retrieval_memo_key(
                lookup_text, message, active_tags, context_turns, post_compaction, all_tags,
            )
            _memo = _provider.load_retrieval_memo(self._conversation_id, _memo_key)
            if _memo is not None and not isinstance(_memo.get("scores"), dict):
                _memo = None
            if _memo is None:
                logger.info(
                    "RETRIEVAL_MEMO miss conv=%s key=%s parts=%s",
                    self._conversation_id[:12], _memo_key[:12],
                    self._retrieval_memo_parts(
                        lookup_text, message, active_tags, context_turns, post_compaction, all_tags,
                    ),
                )
        _tag_stage = time.monotonic()
        if _memo is not None:
            _tr = _memo.get("tag_result") or {}
            tag_result = TagResult(
                tags=list(_tr.get("tags") or []),
                primary=str(_tr.get("primary") or "_general"),
                source=str(_tr.get("source") or "fallback"),
                temporal=bool(_tr.get("temporal")),
                related_tags=list(_tr.get("related_tags") or []),
                query_embedding=_tr.get("query_embedding"),
            )
        elif self._inbound_tagger is not None:
            # Embedding-based: match against existing vocabulary (no hallucination)
            # Store tags are conversation-scoped via get_all_tags().
            # TurnTagIndex is inherently per-conversation (loaded from that
            # conversation's engine_state), so all its tags are in scope.
            vocab_tags = list(set(store_tags))
            if self._turn_tag_index:
                vocab_tags = list(set(vocab_tags) | self._turn_tag_index.all_tags())
            tag_result = self._inbound_tagger.generate_tags(
                lookup_text, vocab_tags, context_turns=context_turns,
            )
            # Apply temporal heuristic (embedding tagger doesn't detect these)
            if not tag_result.temporal:
                tag_result.temporal = self._detect_temporal(message)
            logger.debug("Inbound embedding match: tags=%s temporal=%s",
                         tag_result.tags, tag_result.temporal)
        else:
            # LLM-based: open-ended tag generation (pass known tags for reuse)
            tag_result = self.tag_generator.generate_tags(
                lookup_text, store_tags, context_turns=context_turns,
            )
        if query_text is not None:
            # A date or temporal request inside a quotation is not the
            # current requester's time filter, even if an LLM tagger says so.
            tag_result.temporal = self._detect_temporal(message)
        _note("tag_generate", _tag_stage)

        query_embedding = getattr(tag_result, 'query_embedding', None)

        retrieval_metadata: dict = {}
        if _memo_key:
            retrieval_metadata["memo"] = "hit" if _memo is not None else "miss"
        if query_text is not None:
            retrieval_metadata["query_context"] = "explicit"
        retrieval_scores: dict[str, float] = {}

        # Temporal detection is now advisory; time-scoped recall is tool-driven
        # via vc_remember_when rather than an automatic retriever branch.
        if tag_result.temporal:
            retrieval_metadata["temporal_hint"] = True

        if tag_result.source == "fallback":
            # Tagging failed — fall back to working set tags. When the
            # caller supplied a bounded entries_snapshot (proxy mode),
            # the working-set computation uses it to stay consistent
            # with the rest of the inbound call's view of the index.
            if self._turn_tag_index:
                working_set = list(self._turn_tag_index.get_active_tags(
                    lookback=self.config.anchorless_lookback,
                    entries_snapshot=entries_snapshot,
                ))
                query_tags = [t for t in working_set if t != "_general"]
                retrieval_metadata["fallback"] = "working_set"
            else:
                query_tags = [t for t in tag_result.tags if t != "_general"]
        else:
            # Normal flow — filter out active tags (already in recent context)
            query_tags = [
                t for t in tag_result.tags
                if not (self.config.skip_active_tags and t in active_tags)
                and t != "_general"
            ]

        skipped_tags = [t for t in tag_result.tags if t in active_tags]

        if not query_tags:
            # _general fallback: use previous turn's tags for focused
            # retrieval. Honor the entries_snapshot when supplied so
            # the inherited tags match the same bounded view as the
            # rest of this inbound call.
            if self._turn_tag_index:
                prev = self._turn_tag_index.latest_meaningful_tags(
                    entries_snapshot=entries_snapshot,
                )
                if prev:
                    query_tags = [t for t in prev.tags if t not in self._turn_tag_index._NON_INHERITABLE_TAGS]
                    if query_tags:
                        logger.info(
                            "Retriever: _general fallback → using previous turn tags %s (T%d)",
                            query_tags, prev.turn_number,
                        )
                        retrieval_metadata["general_fallback"] = "previous_turn"

        if not query_tags:
            logger.info(
                "Retriever: no query tags after filtering (message_tags=%s, active_tags=%s, skipped=%s)",
                tag_result.tags, list(active_tags), skipped_tags,
            )
            # Summary floor: post-compaction, no query tags — inject all tag summaries
            if post_compaction:
                token_budget = self.config.tag_context_max_tokens
                retrieval_metadata["tag_token_budget"] = token_budget
                floor_summaries, floor_tokens = self._load_all_tag_summaries(token_budget)
                if floor_summaries:
                    elapsed = time.monotonic() - start_time
                    return RetrievalResult(
                        tags_matched=[],
                        summaries=floor_summaries,
                        total_tokens=floor_tokens,
                        facts=self._fetch_all_facts(),
                        overflow_summaries=overflow,
                        retrieval_metadata={
                            **(retrieval_metadata if query_text is not None else {}),
                            "elapsed_ms": round(elapsed * 1000, 1),
                            "tags_from_message": tag_result.tags,
                            "tags_skipped_active": skipped_tags,
                            "summary_floor": True,
                            "tag_token_budget": token_budget,
                        },
                        cost_report=RetrievalCostReport(
                            tags_queried=[],
                            tags_skipped=skipped_tags,
                            strategy_active="summary_floor",
                        ),
                    )

            elapsed = time.monotonic() - start_time
            return RetrievalResult(
                tags_matched=[],
                summaries=[],
                total_tokens=0,
                facts=self._fetch_all_facts(),
                overflow_summaries=overflow,
                retrieval_metadata={
                    **(retrieval_metadata if query_text is not None else {}),
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "tags_from_message": tag_result.tags,
                    "tags_skipped_active": skipped_tags,
                },
                cost_report=RetrievalCostReport(
                    tags_queried=[],
                    tags_skipped=skipped_tags,
                    strategy_active="default",
                ),
            )

        # Expand query with related tags from tagger
        related_query_tags = [
            t for t in tag_result.related_tags
            if t not in set(query_tags) and t != "_general"
        ]
        expanded_tags = list(set(query_tags) | set(related_query_tags))
        query_tag_set = set(query_tags)

        # Facts depend only on the expanded tags, not on topic selection or
        # rerank, so they are gathered, and passed through *facts_transform*,
        # while the summaries are chosen. The job runs in a fresh context: the
        # store's context variables scope write transactions, never reads.
        def _gather_facts() -> list:
            _facts_stage = time.monotonic()
            if getattr(self.config, "fact_dense_retrieval", False):
                gathered = self._fetch_facts_dense(
                    lookup_text, context_turns, expanded_tags, retrieval_metadata,
                )
            elif self.config.prefetch_facts and expanded_tags:
                gathered = self._fetch_facts_by_tags(expanded_tags)
                logger.info("Retriever: facts=%d (prefetch tags=%s)", len(gathered), expanded_tags)
            else:
                gathered = self._fetch_all_facts()
                logger.info("Retriever: facts=%d (all, prefetch=%s)", len(gathered),
                            "off" if not self.config.prefetch_facts else "no-tags")
            _note("fetch_facts", _facts_stage)
            if facts_transform is not None:
                _transform_stage = time.monotonic()
                gathered = facts_transform(gathered)
                _note("facts_transform", _transform_stage)
            return gathered

        _facts_future = _FACTS_POOL.submit(_gather_facts)

        from .retrieval_scoring import score_candidates

        _idf_stage = time.monotonic()
        idf_weights = self._compute_idf_weights(all_tags)
        tag_stats = {ts.tag: ts.usage_count for ts in all_tags}
        _note("idf_prepare", _idf_stage)
        _score_stage = time.monotonic()
        if _memo is not None:
            scores = {str(k): float(v) for k, v in _memo["scores"].items()}
            breakdowns = dict(_memo.get("breakdowns") or {})
            logger.info(
                "RETRIEVAL_MEMO hit conv=%s key=%s tags=%d",
                (self._conversation_id or "")[:12], _memo_key[:12], len(scores),
            )
        stored_embeddings = None
        query_embedding_context = None
        # A judged topic choice can draw its pool from query-to-summary
        # similarity alone; the other signals and the context vector are then
        # not computed.
        _pool_scores = None
        if _memo is None and query_embedding is not None and self._embedding_pool_enabled():
            _pool_scores = self._embedding_pool(query_embedding)
        if _memo is None and query_embedding is not None and _pool_scores is None:
            stored_embeddings = self._load_tag_summary_embedding_snapshot()
        if _memo is None and query_embedding is not None and _pool_scores is None:
            query_embedding_context, embed_mode, ctx_embed_ms = (
                self._build_context_query_embedding(lookup_text, context_turns)
            )
            if query_embedding_context is not None:
                if ctx_embed_ms is not None:
                    _breakdown["context_embed"] = ctx_embed_ms
                logger.info(
                    "Retriever: embedding-context mode=%s turns=%d embed_ctx=%sms",
                    embed_mode, self.config.scoring.embedding_context_turns,
                    ctx_embed_ms,
                )
        # Resolve the strategy up front — its max_results is K for the
        # embedding reserved-seats pass inside score_candidates.
        strategy = self.config.strategy_configs.get("default")
        if strategy is None:
            from ..types import StrategyConfig
            strategy = StrategyConfig()

        if _memo is None:
            if _pool_scores is not None:
                scores, breakdowns = _pool_scores, {}
            else:
                scores, breakdowns = score_candidates(
                    query_tags=query_tags,
                    related_tags=related_query_tags,
                    query_text=lookup_text,
                    query_embedding=query_embedding,
                    store=self.store,
                    idf_weights=idf_weights,
                    conversation_id=self._conversation_id,
                    config=self.config.scoring,
                    tag_stats=tag_stats,
                    stored_embeddings=stored_embeddings,
                    query_embedding_context=query_embedding_context,
                    top_k=strategy.max_results,
                )
            if _memo_key and _provider is not None:
                _provider.save_retrieval_memo(self._conversation_id, _memo_key, {
                    "tag_result": {
                        "tags": list(tag_result.tags), "primary": tag_result.primary,
                        "source": tag_result.source, "temporal": bool(tag_result.temporal),
                        "related_tags": list(tag_result.related_tags or []),
                        "query_embedding": (
                            [float(v) for v in query_embedding] if query_embedding is not None else None
                        ),
                    },
                    "scores": {str(k): float(v) for k, v in scores.items()},
                    "breakdowns": breakdowns,
                })
        _note("score_candidates", _score_stage)
        # Alias groups compete as one topic: fold every aliased tag's score into
        # its canonical so max_results counts distinct topics, then keep the
        # alias keys so sections rendered under an alias still find their score.
        alias_map = self._load_alias_map()
        if alias_map:
            folded: dict[str, float] = {}
            for tag, score in scores.items():
                canonical = alias_map.get(tag, tag)
                folded[canonical] = max(folded.get(canonical, float("-inf")), score)
            scores = folded
            retrieval_scores = dict(folded)
            for alias, canonical in alias_map.items():
                if canonical in folded:
                    retrieval_scores[alias] = folded[canonical]
        else:
            retrieval_scores = scores

        # Apply token budget

        budget_fraction = strategy.max_budget_fraction
        if current_utilization > 0.5:
            scale = max(0.1, 1.0 - current_utilization)
            budget_fraction *= scale
        token_budget = int(self.config.tag_context_max_tokens * budget_fraction)
        # The assembler renders these summaries as larger structured blocks;
        # it bounds the rendered sections by the same budget selected here.
        retrieval_metadata["tag_token_budget"] = token_budget

        ranked_tags = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        _topic_stage = time.monotonic()
        top_tags, jev_topics, jev_items = self._select_topics(
            lookup_text, ranked_tags, strategy.max_results, _provider, _memo_key,
            query_embedding=query_embedding,
        )
        _note("select_topics", _topic_stage)
        retrieval_metadata["top_tags"] = list(top_tags)
        retrieval_metadata["topics_chosen_by_jev"] = jev_topics

        # Fetch summaries for all top-scored tags at once, then rank by RRF score
        _summary_fetch_stage = time.monotonic()
        all_summaries = self.store.get_summaries_by_tags(
            tags=top_tags, min_overlap=1,
            limit=strategy.max_results * 3,
            conversation_id=self._conversation_id,
        )
        if alias_map:
            # Each chosen topic's aliases are fetched in their own query, so
            # one large alias group cannot crowd another topic out of a
            # shared result limit.
            seen_refs = {s.ref for s in all_summaries}
            for canonical in top_tags:
                alias_tags = [alias for alias, target in alias_map.items() if target == canonical]
                if not alias_tags:
                    continue
                for extra in self.store.get_summaries_by_tags(
                    tags=alias_tags, min_overlap=1,
                    limit=strategy.max_results * 3,
                    conversation_id=self._conversation_id,
                ):
                    if extra.ref not in seen_refs:
                        seen_refs.add(extra.ref)
                        all_summaries.append(extra)
        _note("fetch_ranked_summaries", _summary_fetch_stage)
        # Sort by (RRF fused score of best matching tag, IDF query-tag overlap) descending;
        # retrieval_scores carries alias keys so alias-tagged summaries rank with their topic.
        def _summary_sort_key(s: StoredSummary) -> tuple[float, float]:
            best_rrf = max((retrieval_scores.get(t, 0.0) for t in s.tags), default=0.0)
            idf_overlap = sum(idf_weights.get(t, 1.0) for t in s.tags if t in query_tag_set)
            return (best_rrf, idf_overlap)
        all_summaries.sort(key=_summary_sort_key, reverse=True)
        from .judgment import rerank_summaries as _rerank_summaries
        all_summaries = _rerank_summaries(lookup_text, all_summaries, runtime=self.judgment_runtime)
        if jev_topics:
            # The judged topics' own summaries and segments lead, most relevant
            # first; a topic can hold the answer without segments of its own,
            # and a segment can hold it without its topic being chosen.
            all_summaries = jev_items + all_summaries

        selected: list[StoredSummary] = []
        selected_refs: set[str] = set()
        total_tokens = 0
        _overflow_cap = 50
        for summary in all_summaries:
            if summary.ref in selected_refs:
                continue
            if total_tokens + summary.summary_tokens > token_budget:
                if len(overflow) < _overflow_cap:
                    overflow.append(summary)
                    logger.debug(
                        "Retriever: '%s' OVERFLOW (%dt, %d candidates)",
                        summary.primary_tag, summary.summary_tokens, len(overflow),
                    )
                continue
            selected.append(summary)
            selected_refs.add(summary.ref)
            total_tokens += summary.summary_tokens
            logger.info(
                "Retriever: '%s' INCLUDE (%dt, budget %d/%dt used)",
                summary.primary_tag, summary.summary_tokens, total_tokens, token_budget,
            )

        # Alias ride-along: if a query tag made it into the selected results,
        # find additional segments via its aliases that didn't make max_results.
        # These ride free — they don't count against max_results.
        if selected:
            alias_extras = _alias_ride_along(
                self.store, query_tag_set, selected, selected_refs,
                token_budget, total_tokens,
                conversation_id=self._conversation_id,
            )
            if alias_extras:
                selected.extend(alias_extras)
                selected_refs.update(s.ref for s in alias_extras)
                total_tokens += sum(s.summary_tokens for s in alias_extras)
                logger.info(
                    "Alias ride-along: +%d segments (%d tokens)",
                    len(alias_extras),
                    sum(s.summary_tokens for s in alias_extras),
                )

        # FTS fallback: tag overlap missed — try full-text search on stored segments
        # Convert tags to FTS query terms: "cook-mode" → "cook mode"
        if not selected and expanded_tags:
            fts_terms = " OR ".join(
                f'"{tag.replace("-", " ")}"' for tag in expanded_tags
            )
            fts_results = self.store.search(
                query=fts_terms,
                limit=strategy.max_results,
                conversation_id=self._conversation_id,
            )
            for summary in fts_results:
                if total_tokens + summary.summary_tokens > token_budget:
                    break
                selected.append(summary)
                total_tokens += summary.summary_tokens
            if fts_results:
                retrieval_metadata["fts_fallback"] = True
                logger.debug(
                    "FTS fallback: tag overlap empty, text search found %d results",
                    len(selected),
                )

        # Summary floor: post-compaction, no results — inject all tag summaries
        if not selected and post_compaction:
            selected, total_tokens = self._load_all_tag_summaries(token_budget)
            if selected:
                retrieval_metadata["summary_floor"] = True

        # Deep retrieval: fetch full segments when we have matches
        full_detail: list[StoredSegment] = []
        if selected:
            _full_detail_stage = time.monotonic()
            for summary in selected[:3]:  # limit to top 3
                segment = self.store.get_segment(summary.ref, conversation_id=self._conversation_id)
                if segment:
                    full_detail.append(segment)
            _note("fetch_full_detail", _full_detail_stage)

        # Collect matched tags (include related tag matches too)
        expanded_set = set(expanded_tags)
        matched_tags = list({tag for s in selected for tag in s.tags if tag in expanded_set})

        elapsed = time.monotonic() - start_time

        retrieval_metadata.update({
            "elapsed_ms": round(elapsed * 1000, 1),
            "tags_from_message": tag_result.tags,
            "tags_queried": query_tags,
            "tags_skipped_active": skipped_tags,
            "candidates_found": len(all_summaries),
            "summaries_returned": len(selected),
            "strategy": "default",
            "budget_fraction": round(budget_fraction, 3),
            "idf_reranked": bool(idf_weights),
            "related_tags_used": related_query_tags,
            "query_expanded": len(related_query_tags) > 0,
        })

        # Facts: dense-augmented (gated), tag prefetch, or fetch all, gathered above.
        _facts_wait_stage = time.monotonic()
        facts = _facts_future.result()
        _note("facts_wait", _facts_wait_stage)
        if facts_transform is not None:
            retrieval_metadata["facts_prepared"] = True

        total_ms = round((time.monotonic() - start_time) * 1000, 1)
        if total_ms >= _RETRIEVAL_BREAKDOWN_LOG_THRESHOLD_MS:
            stages = " ".join(
                f"{stage}={ms:.1f}ms"
                for stage, ms in sorted(_breakdown.items(), key=lambda item: item[1], reverse=True)
                if ms > 0
            )
            logger.info(
                "RETRIEVE_BREAKDOWN conv=%s total=%sms query_tags=%d expanded=%d selected=%d facts=%d %s",
                (self._conversation_id or "")[:12] or "none",
                total_ms,
                len(query_tags),
                len(expanded_tags),
                len(selected),
                len(facts),
                stages or "no-stages",
            )

        return RetrievalResult(
            tags_matched=matched_tags,
            summaries=selected,
            full_detail=full_detail,
            total_tokens=total_tokens,
            facts=facts,
            overflow_summaries=overflow,
            retrieval_metadata=retrieval_metadata,
            retrieval_scores=retrieval_scores,
            query_embedding=query_embedding,
            cost_report=RetrievalCostReport(
                tokens_retrieved=total_tokens,
                budget_fraction_used=total_tokens / token_budget if token_budget > 0 else 0.0,
                strategy_active="default",
                tags_queried=query_tags,
                tags_skipped=skipped_tags,
            ),
        )


def _alias_ride_along(
    store: ContextStore,
    query_tag_set: set[str],
    selected: list[StoredSummary],
    selected_refs: set[str],
    token_budget: int,
    tokens_used: int,
    conversation_id: str | None = None,
) -> list[StoredSummary]:
    """Find extra segments via aliases for tags that already made the cut.

    If query tag X qualified (appears in selected results), find all tags
    in X's alias group and retrieve segments that match those aliases but
    weren't already selected. These ride free — no max_results cap.
    """
    try:
        aliases = store.get_tag_aliases(conversation_id=conversation_id)
    except TypeError:
        aliases = store.get_tag_aliases()
    if not aliases:
        return []

    # Build reverse: canonical → {aliases}
    reverse: dict[str, set[str]] = {}
    for alias, canonical in aliases.items():
        reverse.setdefault(canonical, set()).add(alias)

    # Which query tags actually matched in the selected results?
    qualified_tags: set[str] = set()
    for s in selected:
        qualified_tags.update(set(s.tags) & query_tag_set)

    if not qualified_tags:
        return []

    # For each qualified tag, collect its full alias group
    alias_tags: set[str] = set()
    for qt in qualified_tags:
        # qt might be an alias → get its canonical + siblings
        if qt in aliases:
            canon = aliases[qt]
            alias_tags.add(canon)
            alias_tags.update(reverse.get(canon, set()))
        # qt might be a canonical → get all its aliases
        if qt in reverse:
            alias_tags.update(reverse[qt])

    # Remove tags we already queried with
    alias_tags -= query_tag_set
    if not alias_tags:
        return []

    # Fetch segments matching alias tags
    alias_summaries = store.get_summaries_by_tags(
        tags=list(alias_tags),
        min_overlap=1,
        limit=50,
        conversation_id=conversation_id,
    )

    # Add new segments within token budget
    extras: list[StoredSummary] = []
    remaining = token_budget - tokens_used
    for s in alias_summaries:
        if s.ref in selected_refs:
            continue
        if s.summary_tokens > remaining:
            continue
        extras.append(s)
        selected_refs.add(s.ref)
        remaining -= s.summary_tokens

    return extras
