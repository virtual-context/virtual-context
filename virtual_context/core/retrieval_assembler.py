"""RetrievalAssembler: retrieval, assembly, and hint building.

Extracted from engine.py -- handles on_message_inbound (tag, retrieve, assemble),
reassemble_context, retrieve, transform, filter_history, and context hint building.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .engine_utils import get_recent_context
from .hint_builder import build_autonomous_hint, build_supervised_hint, build_default_hint
from .store import ContextStore
from .turn_tag_index import TurnTagIndex

if TYPE_CHECKING:
    from .assembler import ContextAssembler
    from .monitor import ContextMonitor
    from .paging_manager import PagingManager
    from .retriever import ContextRetriever
    from .semantic_search import SemanticSearchManager
    from ..types import (
        AssembledContext,
        DepthLevel,
        EngineState,
        Message,
        RetrievalResult,
        VirtualContextConfig,
        WorkingSetEntry,
    )

logger = logging.getLogger(__name__)


class RetrievalAssembler:
    """Retrieval, assembly, hint building, and history filtering.

    Owns the inbound pipeline: tag the query, retrieve relevant summaries,
    build context hints, assemble the enriched context block. Also owns
    reassemble (post-paging), retrieve/transform (headless), and
    filter_history (tag-based turn dropping).
    """

    def __init__(
        self,
        retriever: ContextRetriever,
        assembler: ContextAssembler,
        monitor: ContextMonitor,
        paging: PagingManager,
        store: ContextStore,
        turn_tag_index: TurnTagIndex,
        engine_state: EngineState,
        fact_curator,
        config: VirtualContextConfig,
        token_counter,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._monitor = monitor
        self._paging = paging
        self._store = store
        self._turn_tag_index = turn_tag_index
        self._engine_state = engine_state
        self._fact_curator = fact_curator
        self.config = config
        self._token_counter = token_counter

        # Internal state
        self._last_retrieval_result: RetrievalResult | None = None
        self._last_conversation_history: list[Message] | None = None
        self._last_model_name: str = ""
        self._presented_segment_refs: set[str] = set()

    # ------------------------------------------------------------------
    # Semantic helper (needed for context bleed gate in _get_recent_context)
    # ------------------------------------------------------------------

    def _set_semantic(self, semantic: SemanticSearchManager) -> None:
        """Inject semantic search manager reference (set by engine after init)."""
        self._semantic = semantic

    def _get_recent_context(
        self, history: list[Message], n_pairs: int, exclude_last: int = 2,
        current_text: str | None = None,
    ) -> list[str] | None:
        """Collect up to *n_pairs* recent user+assistant text strings."""
        return get_recent_context(
            history,
            n_pairs,
            semantic=self._semantic,
            bleed_threshold=self.config.tag_generator.context_bleed_threshold,
            exclude_last=exclude_last,
            current_text=current_text,
        )

    # ------------------------------------------------------------------
    # on_message_inbound
    # ------------------------------------------------------------------

    def on_message_inbound(
        self,
        message: str,
        conversation_history: list[Message],
        model_name: str = "",
        max_context_tokens: int | None = None,
    ) -> AssembledContext:
        """Before sending to LLM: tag, retrieve, assemble enriched context."""
        # Determine active tags from recent tag results
        # Post-compaction: don't suppress retrieval -- stored summaries are needed
        # since raw turns have been compacted away
        if self._engine_state.compacted_through > 0:
            active_tags = []
        else:
            active_tags = self._get_active_tags(conversation_history)

        # Compute current utilization (only count un-compacted history)
        _offset = self._engine_state.history_offset(len(conversation_history))
        snapshot = self._monitor.build_snapshot(
            conversation_history[_offset:]
        )
        utilization = snapshot.total_tokens / snapshot.budget_tokens if snapshot.budget_tokens > 0 else 0.0

        # Build context for inbound tagger.
        # Include recent turns even post-compaction so query-time tagging can
        # use immediate conversational cues.
        n_context = self.config.tag_generator.context_lookback_pairs
        # For inbound, the current message is not yet in history -- no need to exclude
        context = self._get_recent_context(
            conversation_history, n_context, exclude_last=0,
        )

        # Retrieve relevant tag summaries
        retrieval_result = self._retriever.retrieve(
            message=message,
            current_active_tags=active_tags,
            current_utilization=utilization,
            post_compaction=(self._engine_state.compacted_through > 0),
            context_turns=context,
        )

        # D2: Curate facts down to query-relevant subset before assembly
        if self._fact_curator and retrieval_result.facts:
            retrieval_result.facts = self._fact_curator.curate(
                retrieval_result.facts,
                question=message,
            )

        # Build context awareness hint (post-compaction only)
        _paging_mode = self._resolve_paging_mode(model_name) if self.config.paging.enabled else None
        context_hint = self._build_context_hint(paging_mode=_paging_mode)

        # Load core context
        core_context = self._assembler.load_core_context()

        # Paging: load content at working set depth levels
        ws_param, full_segments_param = self._load_working_set_segments()
        if ws_param:
            # Update last_accessed_turn for tags matched by current query
            query_tags = retrieval_result.retrieval_metadata.get("tags_from_message", [])
            for tag, entry in self._paging.working_set.items():
                if tag in query_tags:
                    entry.last_accessed_turn = len(self._turn_tag_index.entries)

        # Assemble enriched context -- only pass uncompacted messages
        uncompacted = conversation_history[_offset:]
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=retrieval_result,
            conversation_history=uncompacted,
            token_budget=self.config.context_window,
            context_hint=context_hint,
            working_set=ws_param,
            full_segments=full_segments_param,
            max_context_tokens=max_context_tokens,
        )

        # Expose the message's own tags for downstream use (e.g. history filtering).
        # Use tags_from_message (what the tag generator produced for this message)
        # rather than tags_matched (which only includes tags found in the store).
        message_tags = retrieval_result.retrieval_metadata.get(
            "tags_from_message", retrieval_result.tags_matched
        )

        # Retry with expanded context if only _general was produced.
        # Applies both pre- and post-compaction.
        if message_tags == ["_general"]:
            expanded = self._get_recent_context(
                conversation_history, n_context * 2, exclude_last=0,
            )
            if expanded:
                retry_result = self._retriever.retrieve(
                    message=message,
                    current_active_tags=active_tags,
                    current_utilization=utilization,
                    post_compaction=(self._engine_state.compacted_through > 0),
                    context_turns=expanded,
                )
                retry_tags = retry_result.retrieval_metadata.get(
                    "tags_from_message", retry_result.tags_matched
                )
                if retry_tags != ["_general"]:
                    message_tags = retry_tags
                    retrieval_result = retry_result
                    # Re-assemble with the improved retrieval result so
                    # prepend_text includes the newly matched summaries.
                    if self._fact_curator and retrieval_result.facts:
                        retrieval_result.facts = self._fact_curator.curate(
                            retrieval_result.facts, question=message,
                        )
                    assembled = self._assembler.assemble(
                        core_context=core_context,
                        retrieval_result=retrieval_result,
                        conversation_history=uncompacted,
                        token_budget=self.config.context_window,
                        context_hint=context_hint,
                        working_set=ws_param,
                        full_segments=full_segments_param,
                        max_context_tokens=max_context_tokens,
                    )

        # Final fallback: inherit from most recent meaningful turn in the index
        if message_tags == ["_general"]:
            prev = self._turn_tag_index.latest_meaningful_tags()
            if prev:
                message_tags = list(prev.tags)

        assembled.matched_tags = message_tags
        assembled.context_hint = context_hint
        assembled.retrieval_metadata = dict(retrieval_result.retrieval_metadata or {})
        assembled.retrieval_scores = dict(retrieval_result.retrieval_scores or {})
        assembled.retrieval_summaries = list(retrieval_result.summaries or [])
        assembled.retrieval_full_segments = list(retrieval_result.full_detail or [])

        # Cache for reassemble_context() -- used after paging tool execution
        self._last_retrieval_result = retrieval_result
        self._last_conversation_history = conversation_history
        self._last_model_name = model_name
        self._presented_segment_refs = set(assembled.presented_segment_refs)

        return assembled

    # ------------------------------------------------------------------
    # reassemble_context
    # ------------------------------------------------------------------

    def reassemble_context(self) -> str:
        """Re-assemble context with the current working set.

        Call after ``expand_topic()`` / ``collapse_topic()`` to get an
        updated ``prepend_text`` that reflects the new depth levels.
        Reuses the retrieval result from the most recent
        ``on_message_inbound()`` call -- no re-tagging or re-retrieval.

        Returns the updated prepend_text, or "" if no prior inbound call.
        """
        rr = self._last_retrieval_result
        history = self._last_conversation_history
        if rr is None:
            return ""

        model_name = self._last_model_name
        _pm = self._resolve_paging_mode(model_name) if self.config.paging.enabled else None
        context_hint = self._build_context_hint(paging_mode=_pm)
        core_context = self._assembler.load_core_context()

        ws_param, full_segments_param = self._load_working_set_segments()

        _hist = history or []
        uncompacted = _hist[self._engine_state.history_offset(len(_hist)):]
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=rr,
            conversation_history=uncompacted,
            token_budget=self.config.context_window,
            context_hint=context_hint,
            working_set=ws_param,
            full_segments=full_segments_param,
        )
        return assembled.prepend_text

    # ------------------------------------------------------------------
    # retrieve / transform
    # ------------------------------------------------------------------

    def retrieve(self, message: str, active_tags: list[str] | None = None) -> RetrievalResult:
        return self._retriever.retrieve(message, current_active_tags=active_tags or [])

    def transform(self, message: str, active_tags: list[str] | None = None, budget: int | None = None) -> str:
        result = self.retrieve(message, active_tags)
        if not result.summaries:
            return ""
        core_context = self._assembler.load_core_context()
        assembled = self._assembler.assemble(
            core_context=core_context,
            retrieval_result=result,
            conversation_history=[],
            token_budget=budget or self.config.context_window,
        )
        return assembled.prepend_text

    # ------------------------------------------------------------------
    # filter_history
    # ------------------------------------------------------------------

    def filter_history(
        self,
        conversation_history: list[Message],
        current_tags: list[str],
        recent_turns: int | None = None,
    ) -> list[Message]:
        """Filter conversation history by tag relevance.

        Pre-compaction behaviour is governed by
        ``config.assembler.pre_compaction_filtering``:

        * ``"off"``          -- no tag-based drops (all turns pass through)
        * ``"conservative"`` -- tag-based drops with doubled protection window
                               (``monitor.protected_recent_turns * 2``)
        * ``"aggressive"``   -- tag-based drops with standard protection window
                               (``monitor.protected_recent_turns``)

        Post-compaction (``_compacted_through > 0``): always uses standard
        ``monitor.protected_recent_turns`` regardless of mode setting.

        Returns a new list -- the original is not mutated.
        """
        mode = getattr(
            getattr(self.config, "assembler", None),
            "pre_compaction_filtering", "aggressive",
        )
        watermark = self._engine_state.compacted_through
        pre_compaction = watermark == 0

        # Determine protection window
        if recent_turns is not None:
            # Explicit override (e.g. from tests)
            protected = recent_turns
        elif pre_compaction and mode == "off":
            return list(conversation_history)
        elif pre_compaction and mode == "conservative":
            protected = self.config.monitor.protected_recent_turns * 2
        else:
            # aggressive, or post-compaction
            protected = self.config.monitor.protected_recent_turns

        total = len(conversation_history)
        protected_count = protected * 2  # each turn = 2 messages

        if total <= protected_count:
            return list(conversation_history)

        # Skip compacted messages -- their content is in stored summaries
        offset = self._engine_state.history_offset(total)
        older = conversation_history[offset:-protected_count]
        recent = conversation_history[-protected_count:]

        current_tag_set = set(current_tags)
        filtered: list[Message] = []

        # Walk older messages in pairs (user, assistant)
        i = 0
        while i < len(older):
            if i + 1 < len(older):
                pair = [older[i], older[i + 1]]
                step = 2
            else:
                filtered.append(older[i])
                break

            turn_idx = (watermark + i) // 2
            entry = self._turn_tag_index.get_tags_for_turn(turn_idx)

            if entry is None:
                # No tag data -- conservatively include
                filtered.extend(pair)
            elif "rule" in entry.tags or set(entry.tags) & current_tag_set:
                filtered.extend(pair)
            # else: no overlap -- drop this turn

            i += step

        filtered.extend(recent)
        return filtered

    # ------------------------------------------------------------------
    # _get_active_tags
    # ------------------------------------------------------------------

    def _get_active_tags(self, history: list[Message]) -> list[str]:
        lookback = self.config.retriever.active_tag_lookback
        return list(self._turn_tag_index.get_active_tags(lookback=lookback))

    # ------------------------------------------------------------------
    # Context hint building
    # ------------------------------------------------------------------

    def _build_context_hint(self, paging_mode: str | None = None) -> str:
        """Build a topic list for post-compaction prompts.

        *paging_mode* overrides the resolved mode (``"autonomous"`` or
        ``"supervised"``).  The proxy passes this from the per-request model
        check; headless/MCP callers omit it and the method resolves from config.

        When paging is enabled, builds a richer hint with depth info and budget.
        Mode determines detail level:
        - supervised: topic list with depth, "call expand_topic for detail"
        - autonomous: full budget dashboard with token costs

        Returns empty string if compaction hasn't occurred or the feature is disabled.
        """
        if not self.config.assembler.context_hint_enabled:
            return ""
        if self._engine_state.compacted_through == 0:
            return ""

        tag_summaries = self._store.get_all_tag_summaries(
            conversation_id=self.config.conversation_id,
        )
        if not tag_summaries:
            return ""

        # Determine paging mode
        paging_enabled = self.config.paging.enabled
        if paging_mode is None and paging_enabled:
            paging_mode = self._resolve_paging_mode()

        if paging_enabled and paging_mode == "autonomous":
            hint = self._build_autonomous_hint(tag_summaries)
        elif paging_enabled and paging_mode == "supervised":
            hint = self._build_supervised_hint(tag_summaries)
        else:
            hint = self._build_default_hint(tag_summaries)

        return hint

    def _build_autonomous_hint(self, tag_summaries: list) -> str:
        return build_autonomous_hint(
            tag_summaries=tag_summaries,
            working_set=self._paging.working_set,
            budget=self.config.assembler.tag_context_max_tokens,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
            calculate_depth_tokens=self._paging.calculate_depth_tokens,
            fact_counts=self._store.get_fact_count_by_tags(conversation_id=self.config.conversation_id),
            max_tool_rounds=self.config.paging.max_tool_loops,
        )

    def _build_supervised_hint(self, tag_summaries: list) -> str:
        return build_supervised_hint(
            tag_summaries=tag_summaries,
            working_set=self._paging.working_set,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
            max_tool_rounds=self.config.paging.max_tool_loops,
        )

    def _build_default_hint(self, tag_summaries: list) -> str:
        return build_default_hint(
            tag_summaries=tag_summaries,
            max_hint_tokens=self.config.assembler.context_hint_max_tokens,
            token_counter=self._token_counter,
        )

    # ------------------------------------------------------------------
    # _resolve_paging_mode
    # ------------------------------------------------------------------

    def _resolve_paging_mode(self, model_name: str = "") -> str:
        """Check if *model_name* is trusted for autonomous paging.

        Uses prefix matching against the config's ``autonomous_models``
        entries (which default to a sensible built-in set).  Returns
        ``"autonomous"`` if matched, ``"supervised"`` otherwise.
        """
        model = model_name.lower()
        if not model:
            return "supervised"
        for pattern in self.config.paging.autonomous_models:
            if model.startswith(pattern.lower()):
                return "autonomous"
        return "supervised"

    # ------------------------------------------------------------------
    # _load_working_set_segments
    # ------------------------------------------------------------------

    def _load_working_set_segments(self) -> tuple[dict | None, dict | None]:
        """Load full segments for working-set tags at SEGMENTS or FULL depth.

        Returns ``(working_set_dict, full_segments_dict)`` for the assembler,
        or ``(None, None)`` if paging is disabled or the working set is empty.
        """
        from ..types import DepthLevel

        if not self.config.paging.enabled or not self._paging.working_set:
            return None, None
        ws_param = self._paging.working_set
        full_segments_param: dict = {}
        seen_refs: set[str] = set()
        for tag, entry in self._paging.working_set.items():
            if entry.depth in (DepthLevel.SEGMENTS, DepthLevel.FULL):
                segs = self._store.get_segments_by_tags(
                    tags=[tag], min_overlap=1, limit=500,
                    conversation_id=self.config.conversation_id,
                )
                deduped = [s for s in segs if s.ref not in seen_refs]
                seen_refs.update(s.ref for s in deduped)
                if deduped:
                    full_segments_param[tag] = deduped
        return ws_param, full_segments_param

    # ------------------------------------------------------------------
    # recall_all
    # ------------------------------------------------------------------

    def recall_all(self) -> dict:
        tag_summaries = self._store.get_all_tag_summaries(
            conversation_id=self.config.conversation_id,
        )
        if not tag_summaries:
            return {"found": False, "message": "No stored summaries yet."}
        budget = self.config.assembler.tag_context_max_tokens
        selected = []
        total_tokens = 0
        for ts in tag_summaries:
            if total_tokens + ts.summary_tokens > budget:
                break
            selected.append({
                "tag": ts.tag,
                "summary": ts.summary,
                "tokens": ts.summary_tokens,
                "description": ts.description or "",
            })
            total_tokens += ts.summary_tokens
        return {
            "found": True,
            "topics_loaded": len(selected),
            "total_tokens": total_tokens,
            "summaries": selected,
        }
