"""CompactionPipeline: segmentation, compaction, and storage.

Extracted from engine.py — handles Phase 2 of turn processing (compact_if_needed),
manual compaction (compact_manual), and the shared compaction core (_run_compaction).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from .store import ContextStore
from .structured_summary import validate_tag_rollup_inputs
from .turn_tag_index import TurnTagIndex

if TYPE_CHECKING:
    from .compactor import DomainCompactor
    from .segmenter import TopicSegmenter
    from .semantic_search import SemanticSearchManager
    from .telemetry import TelemetryLedger
    from ..types import (
        ActorCardEntry,
        ActorCardEntrySource,
        ActorRoster,
        CompactionReport,
        CompactionResult,
        CompactionSignal,
        EngineState,
        CanonicalTurnRow,
        Message,
        VirtualContextConfig,
    )

# Compatibility exports retained for callers and policy-version overrides.
from .community.actor_card_policy import (  # noqa: F401
    _ACTOR_CARD_CITATION_LIMIT,
    _ACTOR_CARD_POLICY_VERSION,
    _ACTOR_CARD_SEMANTIC_CONTRACT,
    _ACTOR_CARD_JUDGMENT_RULES,
    _ACTOR_CARD_CONFIDENCE_SCALE,
    _ACTOR_CARD_SINGLE_SOURCE_CONFIDENCE_CAP,
    _ActorCardAdmissionError,
    _ActorCardCoverageError,
    _EmptyResponseFallbackProvider,
    _format_rejection_counts,
)
from .community.actor_card_rebuild import ActorCardRebuildService
from .community.actor_card_curation import ActorCardCurationService
from .community.actor_card_admission import ActorCardAdmissionService
from .community.actor_card_evidence import ActorCardEvidenceService
from .community.attribution import CommunityAttributionService
from .community.canonical_sources import (
    physical_rows_by_group, physical_rows_by_id, reply_target_rows,
)

logger = logging.getLogger(__name__)

# Lazy-import for _is_stub_content from engine to avoid circular imports.
_is_stub_content_fn: Callable[[str], bool] | None = None


def _ensure_engine_imports() -> None:
    """Lazy-import module-level symbols from engine to avoid circular imports."""
    global _is_stub_content_fn
    if _is_stub_content_fn is None:
        from ..engine import _is_stub_content as _stub
        _is_stub_content_fn = _stub


class CompactionPipeline:
    """Segmentation, compaction, storage, and tag summary building.

    Owns the ``compact_if_needed`` and ``compact_manual`` entry points as well
    as the shared ``_run_compaction`` core that both call.

    Constructor dependencies mirror what the engine previously wired internally.
    """

    def __init__(
        self,
        compactor: DomainCompactor | None,
        segmenter: TopicSegmenter,
        store: ContextStore,
        turn_tag_index: TurnTagIndex,
        engine_state: EngineState,
        config: VirtualContextConfig,
        supersession_checker,
        fact_curator,
        semantic: SemanticSearchManager,
        telemetry: TelemetryLedger,
        save_state_callback: Callable,
        session_state_provider=None,
        worker_id: str | None = None,
        prewarm_context_hint_callback: Callable[[], str] | None = None,
    ) -> None:
        self._compactor = compactor
        self._segmenter = segmenter
        self._store = store
        self._turn_tag_index = turn_tag_index
        self._engine_state = engine_state
        self._config = config
        self._supersession_checker = supersession_checker
        self._fact_curator = fact_curator
        self._semantic = semantic
        self._telemetry = telemetry
        self._save_state_callback = save_state_callback
        self._session_state_provider = session_state_provider
        self._prewarm_context_hint_callback = prewarm_context_hint_callback
        # Per-write ownership guard: the worker identity seeded at construction
        # (or set post-construction by the caller). ProxyState writes its own
        # self._worker_id here after construction so store_segment guards can
        # scope every write to the live compaction_operation row.
        self._worker_id: str | None = worker_id

    def _compaction_guard_kwargs(
        self, operation_id: str | None, *, include_conversation_id: bool = False,
    ) -> dict[str, object]:
        """Return guard kwargs forming an all-or-nothing tuple.

        Per fencing plan §5.6 and the storage-side
        ``_validate_compaction_guard_kwargs`` contract, every fenced
        write must receive either all guard kwargs as ``None`` (legacy
        unguarded path) or all as non-``None`` (fenced path with active
        op). Mixed partial kwargs are rejected as programming errors.

        ``operation_id`` and ``self._worker_id`` are the gate: when
        both are set we emit the full guard tuple; otherwise every
        kwarg is ``None`` so the storage method takes the legacy path.

        ``include_conversation_id`` adds the conversation kwarg for the
        two methods whose contract carries it
        (``store_chunk_embeddings``, ``store_fact_links``,
        ``FactLinkChecker.check_and_link``).
        """
        is_guarded = operation_id is not None and self._worker_id is not None
        kwargs: dict[str, object] = {
            "operation_id": operation_id if is_guarded else None,
            "owner_worker_id": self._worker_id if is_guarded else None,
            "lifecycle_epoch": (
                int(self._engine_state.lifecycle_epoch) if is_guarded else None
            ),
        }
        if include_conversation_id:
            kwargs["conversation_id"] = (
                self._config.conversation_id if is_guarded else None
            )
        return kwargs

    def _embed_and_store_fact_embeddings(
        self, facts, *, operation_id: str | None, guard_kwargs: dict,
    ) -> None:
        """Compute and persist dense embeddings for freshly-written facts.

        Mirrors the tag-summary embedding posture: ``CompactionLeaseLost``
        propagates (fail-closed) so the outer wrapper can emit
        ``COMPACTION_WRITE_REJECTED``; any other embedding/store failure
        is logged and swallowed so a degraded embedder never blocks a
        compaction. Model versioning rides ``retriever.embedding_model``.
        """
        from ..types import CompactionLeaseLost as _CLL
        embed_fn = self._semantic.get_embed_fn() if self._semantic else None
        if not embed_fn or not facts:
            return
        conv_id = self._config.conversation_id
        # A vector row is per-conversation; an empty conversation_id would
        # write an unscoped row the read path can never target.
        assert conv_id, "conversation_id must be non-empty before embedding facts"
        model = self._config.retriever.embedding_model
        for fact in facts:
            try:
                text = fact.embed_text()
                if not text:
                    continue
                emb = embed_fn([text])[0]
                self._store.store_fact_embeddings(
                    fact.id, conv_id, model, emb, **guard_kwargs,
                )
            except _CLL:
                raise
            except Exception as e:
                logger.warning("Failed to embed fact %s: %s", fact.id, e)

    # Services are constructed from current dependencies at each boundary.
    # Engines can rebind stores/config, and admin/tests can replace callbacks;
    # neither a cached service nor an ambient owner object should hide that.
    def _actor_card_rebuild_service(self) -> ActorCardRebuildService:
        return ActorCardRebuildService(
            store=getattr(self, "_store", None), config=getattr(self, "_config", None), compactor=getattr(self, "_compactor", None),
            curate_partition=self._curate_actor_card_partition,
            admit_entries=self._admit_actor_card_entries,
            policy_version=_ACTOR_CARD_POLICY_VERSION,
            evidence_records=self._actor_card_evidence_service().fingerprint_records,
        )

    def _actor_card_curation_service(self) -> ActorCardCurationService:
        return ActorCardCurationService(
            config=getattr(self, "_config", None), compactor=getattr(self, "_compactor", None),
            prompt_turns=self._actor_card_prompt_turns,
            curation_provider=self._actor_card_curation_provider,
            provider_for_model=self._actor_card_provider_for_model,
            curation_override=getattr(self, "_actor_card_curation_provider_override", None),
            admission_override=getattr(self, "_actor_card_admission_provider_override", None),
        )

    def _actor_card_admission_service(self) -> ActorCardAdmissionService:
        return ActorCardAdmissionService(
            config=getattr(self, "_config", None), compactor=getattr(self, "_compactor", None),
            admission_provider=self._actor_card_admission_provider,
            evidence_segments=self._actor_card_evidence_segments,
            prompt_turns=self._actor_card_prompt_turns,
        )

    def _actor_card_evidence_service(self) -> ActorCardEvidenceService:
        return ActorCardEvidenceService(
            store=getattr(self, "_store", None), paired_agent_replies=self._paired_agent_replies,
        )

    def _attribution_service(self) -> CommunityAttributionService:
        counts = getattr(self, "_agent_quote_counts", None)
        if counts is None:
            counts = {
                self.QUOTE_AGENT_AUTHORED: 0,
                self.QUOTE_NOT_AGENT: 0,
                self.QUOTE_IDENTITY_UNKNOWN: 0,
            }
            self._agent_quote_counts = counts
        return CommunityAttributionService(
            store=getattr(self, "_store", None), config=getattr(self, "_config", None), quote_outcomes=counts,
            quote_is_agent_output=self._quote_is_agent_output,
            record_quote_outcome=self._record_quote_outcome,
            segment_source_ids=self._segment_source_ids,
        )

    def _due_actor_card_rebuilds(self, *, limit: int = 25) -> list[str]:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_rebuild_service().due_rebuilds(limit=limit)

    def _consolidate_actor_cards_after_compaction(
        self,
        actor_ids: set[str],
        *,
        disable_replacement_passes: bool = False,
    ) -> int:
        """Run dedicated person-card work at the successful compaction boundary.

        Ordinary canonical ingestion only dirties a card.  The model-backed
        curator/admission workflow belongs here, after the compaction's
        segments, facts, canonical markers, tag summaries, and retrieval
        snapshots have all committed.  Actor ids are coalesced across every
        segment so one compaction performs at most one rebuild attempt per
        affected person.

        Recovery/backlog compactions run this phase too.  They still rewrite
        derived facts and can therefore invalidate a card's provenance; if
        they skipped consolidation, that card would remain unavailable until
        some unrelated future compaction.  A card failure is isolated from the
        already-successful compaction and remains dirty for a later retry.
        """
        candidates = {
            (actor_id or "").strip()
            for actor_id in actor_ids
            if (actor_id or "").strip()
        }
        dispatch = (
            "recovery" if disable_replacement_passes else "ordinary"
        )
        # A previous transient provider failure must not strand a since-silent
        # actor forever.  Any successful compaction services a bounded
        # tenant-local retry queue after its stored backoff expires.
        candidates.update(self._due_actor_card_rebuilds(limit=25))
        attempted = 0
        for actor_id in sorted(candidates):
            attempted += 1
            started = time.monotonic()
            logger.info(
                "ACTOR_CARD_COMPACTION_CONSOLIDATION actor=%s "
                "status=begin dispatch=%s",
                actor_id[:24],
                dispatch,
            )
            try:
                written = self._rebuild_actor_card(actor_id)
            except Exception:
                logger.warning(
                    "ACTOR_CARD_COMPACTION_CONSOLIDATION actor=%s "
                    "status=failed elapsed_ms=%d",
                    actor_id[:24],
                    int((time.monotonic() - started) * 1000),
                    exc_info=True,
                )
                continue
            logger.info(
                "ACTOR_CARD_COMPACTION_CONSOLIDATION actor=%s "
                "status=complete written=%d elapsed_ms=%d",
                actor_id[:24],
                written,
                int((time.monotonic() - started) * 1000),
            )
        return attempted

    def _rebuild_actor_card(self, actor_id: str, *, force: bool = False) -> int:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_rebuild_service().rebuild(actor_id, force=force)

    def _paired_agent_replies(self, turn_sources: list) -> dict:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_evidence_service().paired_agent_replies(turn_sources)

    def _actor_card_prompt_turns(
        self,
        turn_sources: list,
        *,
        max_chars: int = 96_000,
    ) -> list[dict]:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_evidence_service().prompt_turns(turn_sources, max_chars=max_chars)

    def _curate_actor_card_partition(
        self,
        fact_sources: list,
        turn_sources: list,
    ) -> tuple[str, bool, str, list, set[str]]:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_curation_service().curate_partition(fact_sources, turn_sources)

    def _actor_card_provider_for_model(self, selected_model: str):
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_curation_service().provider_for_model(selected_model)

    def _actor_card_curation_provider(self):
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_curation_service().curation_provider()

    def _actor_card_admission_provider(self):
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_curation_service().admission_provider()

    def _actor_card_evidence_segments(
        self,
        actor_id: str,
        audience_conversation_id: str,
        sources: list,
        candidate_fact_ids: set[str],
        *,
        required_fact_ids: set[str] | None = None,
        max_chars: int = 64_000,
    ) -> tuple[list[dict], set[tuple[str, str]]]:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_evidence_service().evidence_segments(actor_id, audience_conversation_id, sources, candidate_fact_ids, required_fact_ids=required_fact_ids, max_chars=max_chars)

    def _admit_actor_card_entries(
        self,
        actor_id: str,
        audience_conversation_id: str,
        fact_sources: list,
        turn_sources: list,
        normalized: list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
        *,
        curator_substantive: bool,
        existing_entry_ids: set[str] | None = None,
    ) -> tuple[
        list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
        str,
        Counter[str],
        bool,
    ]:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._actor_card_admission_service().admit_entries(actor_id, audience_conversation_id, fact_sources, turn_sources, normalized, curator_substantive=curator_substantive, existing_entry_ids=existing_entry_ids)

    def _physical_rows_by_group(
        self, groups: list[int] | None = None,
    ) -> dict[int, list["CanonicalTurnRow"]]:
        """Hydrate only selected logical groups, preserving physical siblings."""
        return physical_rows_by_group(
            self._store, self._config.conversation_id, groups or (),
        )

    @staticmethod
    def _segment_source_ids(segment) -> tuple[list[str], bool]:
        """Delegate to the explicit community service; retain the caller seam."""
        return CommunityAttributionService.segment_source_ids(segment)

    @staticmethod
    def _source_human_identity_keys(
        source_ids: list[str],
        physical_by_id: dict[str, "CanonicalTurnRow"],
    ) -> set[tuple[str, ...]] | None:
        """Delegate to the explicit community service; retain the caller seam."""
        return CommunityAttributionService.source_human_identity_keys(source_ids, physical_by_id)

    # Outcomes of the agent-authored quote check. THREE states, not two:
    # "this is not the agent" and "I cannot tell" both decline to suppress and
    # are NOT the same answer. Collapsing them makes an unconfigured guard
    # indistinguishable from a configured one that never matches, which is how
    # an inert layer reads as a healthy one.
    QUOTE_AGENT_AUTHORED = "agent_authored"
    QUOTE_NOT_AGENT = "not_agent"
    QUOTE_IDENTITY_UNKNOWN = "agent_identity_unknown"

    def _validated_agent_actor_ids(self, physical_by_id: dict) -> dict:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._attribution_service().validated_agent_actor_ids(physical_by_id)

    def _record_quote_outcome(self, outcome: str) -> None:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._attribution_service().record_quote_outcome(outcome)

    def _log_quote_outcomes(self) -> None:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._attribution_service().log_quote_outcomes()

    def _quote_is_agent_output(
        self, *, channel_id: str, target_message_id: str,
        reply_subject_actor_id: str = "", agent_actor_ids: dict | None = None,
    ) -> str:
        """Delegate to the explicit community service; retain the caller seam."""
        return self._attribution_service().quote_is_agent_output(channel_id=channel_id, target_message_id=target_message_id, reply_subject_actor_id=reply_subject_actor_id, agent_actor_ids=agent_actor_ids)

    def _build_actor_roster(
        self, segment, physical_by_id: dict, agent_actor_ids: dict | None = None,
    ) -> "ActorRoster":
        """Delegate to the explicit community service; retain the caller seam."""
        return self._attribution_service().build_actor_roster(segment, physical_by_id, agent_actor_ids)

    def _load_compactable_rows(self) -> tuple[list["CanonicalTurnRow"], list["Message"]]:
        from ..types import SOURCE_CANONICAL_TURN_IDS_KEY, Message

        rows = list(
            self._store.get_uncompacted_canonical_turns(
                self._config.conversation_id,
                protected_recent_turns=self._config.monitor.protected_recent_turns,
            )
        )
        by_group = self._physical_rows_by_group([
            row.turn_group_number for row in rows
            if type(row.turn_group_number) is int and row.turn_group_number >= 0
        ])
        messages: list[Message] = []
        seen_groups: set[int] = set()

        def _timestamp_for(source_row) -> datetime | None:
            for raw_timestamp in (
                getattr(source_row, "first_seen_at", None),
                getattr(source_row, "last_seen_at", None),
                getattr(source_row, "created_at", None),
                getattr(source_row, "updated_at", None),
            ):
                if not raw_timestamp:
                    continue
                try:
                    return datetime.fromisoformat(
                        str(raw_timestamp).replace("Z", "+00:00")
                    )
                except (TypeError, ValueError):
                    continue
            return None

        for row in rows:
            raw_group = getattr(row, "turn_group_number", -1)
            group = int(raw_group if raw_group is not None else -1)
            if group >= 0:
                if group in seen_groups:
                    continue
                seen_groups.add(group)

            # The store's uncompacted seam returns one LOGICAL row per group.
            # That representation is lossy for authorship: one merged string
            # can be backed by multiple physical human rows. Reconstruct each
            # message from its exact physical row so topic grouping sees one
            # durable actor and one source id at a time. A store without the
            # physical seam falls back to the logical row but carries no false
            # multi-row proof.
            backing = (
                list(by_group.get(group, []))
                if group >= 0 else []
            ) or [row]
            backing.sort(key=lambda source: (
                float(getattr(source, "sort_key", 0.0) or 0.0),
                int(
                    getattr(source, "turn_number", -1)
                    if getattr(source, "turn_number", None) is not None
                    else -1
                ),
                str(getattr(source, "canonical_turn_id", "") or ""),
            ))
            physical_proof = group >= 0 and bool(by_group.get(group))
            for source in backing:
                timestamp = _timestamp_for(source)
                canonical_id = str(
                    getattr(source, "canonical_turn_id", "") or ""
                ).strip()
                source_ids = [canonical_id] if physical_proof and canonical_id else []

                if (source.user_content or "").strip():
                    user_metadata: dict = {}
                    if (source.sender or "").strip():
                        user_metadata["sender"] = {"name": source.sender}
                    if source_ids:
                        user_metadata[SOURCE_CANONICAL_TURN_IDS_KEY] = source_ids
                    messages.append(Message(
                        role="user",
                        content=source.user_content,
                        timestamp=timestamp,
                        metadata=user_metadata or None,
                        source_actor_id=(source.sender_actor_id or "").strip(),
                        source_logical_turn_number=int(
                            getattr(row, "turn_number", -1) or 0
                        ),
                        source_audience_conversation_id=str(getattr(
                            source, "audience_conversation_id", "",
                        ) or ""),
                        source_origin_channel_id=str(getattr(
                            source, "origin_channel_id", "",
                        ) or ""),
                        source_audience_attribution_version=int(getattr(
                            source, "audience_attribution_version", 0,
                        ) or 0),
                    ))

                if (source.assistant_content or "").strip():
                    assistant_metadata = (
                        {SOURCE_CANONICAL_TURN_IDS_KEY: source_ids}
                        if source_ids else None
                    )
                    messages.append(Message(
                        role="assistant",
                        content=source.assistant_content,
                        timestamp=timestamp,
                        metadata=assistant_metadata,
                        source_logical_turn_number=int(
                            getattr(row, "turn_number", -1) or 0
                        ),
                        source_audience_conversation_id=str(getattr(
                            source, "audience_conversation_id", "",
                        ) or ""),
                        source_origin_channel_id=str(getattr(
                            source, "origin_channel_id", "",
                        ) or ""),
                        source_audience_attribution_version=int(getattr(
                            source, "audience_attribution_version", 0,
                        ) or 0),
                    ))
        return rows, messages

    def _refresh_compaction_watermark(self) -> None:
        """Read the exact prefix from scalar storage metadata, without bodies."""
        compacted_messages, last_prefix_turn = self._store.get_compaction_watermark(
            self._config.conversation_id,
        )
        self._engine_state.compacted_prefix_messages = compacted_messages
        self._engine_state.last_compacted_turn = last_prefix_turn

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def compact_if_needed(
        self,
        conversation_history: list[Message],
        signal: CompactionSignal,
        progress_callback: Callable[..., None] | None = None,
        turn_id: str = "",
        operation_id: str | None = None,
        *,
        preexisting_operation_id: str | None = None,
        disable_replacement_passes: bool = False,
    ) -> CompactionReport | None:
        """Phase 2 of turn processing: run compaction.

        Slow (~10s with LLM summarizer). Can run in background after
        tag_turn() completes — the next request only needs the tag index.

        *signal*: the CompactionSignal returned by tag_turn().
        *operation_id*: the compaction_operation PK for the per-write ownership
        guard.  When provided (along with ``self._worker_id``), every
        ``store_segment`` call is scoped to the active compaction row — stale
        writes raise ``CompactionLeaseLost`` instead of inserting silently.
        *preexisting_operation_id*: when set by the takeover path, overrides
        *operation_id* so all downstream guarded writes use the pre-inserted
        row's id rather than a freshly generated one.
        *disable_replacement_passes*: when True, the compaction dispatch
        forces insert-only behavior at every gated call site
        (merge-into-existing-segment route, ``replace_facts_for_segment``,
        ``store_chunk_embeddings``, ``save_tag_summary``,
        ``store_tag_summary_embedding``, and the
        ``FactLinkChecker.check_and_link`` /
        ``FactSupersessionChecker.check_and_supersede`` mutation passes).
        Backlog-sweeper dispatches set this to True so a recovery
        compaction cannot overwrite content owned by other operations.
        Per fencing plan §7 / spec v1.4 §1.4.
        """
        if preexisting_operation_id is not None:
            operation_id = preexisting_operation_id
        _t_compact = time.monotonic()

        if self._compactor is None:
            logger.warning(
                "Compaction triggered but no LLM provider configured. "
                "Configure a provider in the providers section."
            )
            return None

        logger.info(
            f"Compaction triggered ({signal.priority}): "
            f"{signal.current_tokens}/{signal.budget_tokens} tokens, "
            f"overflow={signal.overflow_tokens}"
        )

        compact_rows, compact_messages = self._load_compactable_rows()

        if not compact_messages:
            logger.info(
                "Compaction skipped: no uncompacted canonical turns outside protected zone "
                "(history=%d msgs, protected=%d turns, compacted_prefix_messages=%d)",
                len(conversation_history),
                self._config.monitor.protected_recent_turns,
                self._engine_state.compacted_prefix_messages,
            )
            return None

        logger.info(
            "Compacting %d canonical turns (%d messages, first_turn=%d, last_turn=%d, watermark=%d)",
            len(compact_rows),
            len(compact_messages),
            compact_rows[0].turn_number if compact_rows else -1,
            compact_rows[-1].turn_number if compact_rows else -1,
            self._engine_state.compacted_prefix_messages,
        )
        report = self._run_compaction(
            conversation_history,
            compact_messages,
            compact_rows=compact_rows,
            progress_callback=progress_callback,
            generated_by_turn_id=turn_id,
            operation_id=operation_id,
            disable_replacement_passes=disable_replacement_passes,
        )

        self._engine_state.last_compact_ms = round((time.monotonic() - _t_compact) * 1000, 1)
        self._commit_compaction_state(conversation_history)
        return report

    def compact_manual(
        self,
        conversation_history: list[Message],
        turn_id: str = "",
        operation_id: str | None = None,
        *,
        disable_replacement_passes: bool = False,
    ) -> CompactionReport | None:
        """Trigger manual compaction regardless of thresholds.

        Uses the same pipeline as on_turn_complete: respects the compaction
        watermark, protected recent turns, advances the watermark, stores
        segments, and rebuilds tag summaries for affected tags.
        *operation_id*: see ``compact_if_needed`` for ownership-guard semantics.
        *disable_replacement_passes*: see ``compact_if_needed`` for the
        C2R gate semantics.
        """
        if self._compactor is None:
            logger.warning("No LLM provider configured for compaction")
            return None

        if not conversation_history:
            return None

        compact_rows, compact_messages = self._load_compactable_rows()
        if not compact_messages:
            return None

        report = self._run_compaction(
            conversation_history,
            compact_messages,
            compact_rows=compact_rows,
            generated_by_turn_id=turn_id,
            operation_id=operation_id,
            disable_replacement_passes=disable_replacement_passes,
        )

        self._commit_compaction_state(conversation_history)
        return report

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _propagate_tool_output_links(
        self, segment_ref: str, turn_start: int, turn_end: int,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        """Copy turn-level tool output links to the segment join table.

        Iterates turns in ``[turn_start, turn_end)`` and for each turn that
        has ``turn_tool_outputs`` entries, writes a corresponding
        ``segment_tool_outputs`` row.  Non-critical -- failures are
        silenced EXCEPT ``CompactionLeaseLost``, which must propagate
        per fencing plan §5.6 fail-closed exception handling so the
        compactor's outer handler can emit ``COMPACTION_WRITE_REJECTED``
        and exit cleanly without walking the remaining phases.
        """
        from ..types import CompactionLeaseLost
        try:
            for t in range(turn_start, turn_end):
                refs = self._store.get_tool_outputs_for_turn(
                    self._config.conversation_id, t,
                )
                for ref in refs:
                    self._store.link_segment_tool_output(
                        self._config.conversation_id, segment_ref, ref,
                        operation_id=operation_id,
                        owner_worker_id=owner_worker_id,
                        lifecycle_epoch=lifecycle_epoch,
                    )
        except CompactionLeaseLost:
            raise
        except Exception:
            pass  # non-critical

    def _propagate_tool_output_links_for_turns(
        self, segment_ref: str, turn_numbers,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        """Copy tool links for an exact, potentially noncontiguous turn set.

        Topic segmentation deliberately supports A-B-A interleaving, so a
        segment's turns are not necessarily a positional slice or a numeric
        range.  Callers that have canonical source provenance must use this
        exact form; the range helper remains for legacy call sites whose input
        is genuinely contiguous.
        """
        from ..types import CompactionLeaseLost
        try:
            for turn_number in sorted({int(turn) for turn in turn_numbers}):
                refs = self._store.get_tool_outputs_for_turn(
                    self._config.conversation_id, turn_number,
                )
                for ref in refs:
                    self._store.link_segment_tool_output(
                        self._config.conversation_id, segment_ref, ref,
                        operation_id=operation_id,
                        owner_worker_id=owner_worker_id,
                        lifecycle_epoch=lifecycle_epoch,
                    )
        except CompactionLeaseLost:
            raise
        except Exception:
            pass  # non-critical

    def _run_compaction(
        self,
        conversation_history: list[Message],
        compact_messages: list[Message],
        *,
        compact_rows: list["CanonicalTurnRow"] | None = None,
        progress_callback: Callable[..., None] | None = None,
        generated_by_turn_id: str = "",
        operation_id: str | None = None,
        preexisting_operation_id: str | None = None,
        disable_replacement_passes: bool = False,
    ) -> CompactionReport:
        """Shared compaction core: segment, compact, store, build tag summaries.

        Called by both ``compact_if_needed`` (threshold-triggered) and
        ``compact_manual`` (explicit) after their respective guard checks
        have selected *compact_messages*.

        *operation_id*: when provided alongside ``self._worker_id``, every
        ``store_segment`` call carries the ownership guard kwargs so a stale
        write raises ``CompactionLeaseLost`` before it persists.
        *preexisting_operation_id*: takeover path override; takes precedence
        over *operation_id* when set.

        Returns a CompactionReport (never None — callers handle None guards).
        """
        if preexisting_operation_id is not None:
            operation_id = preexisting_operation_id
        from ..types import CompactionReport

        compact_rows = list(compact_rows or [])

        turn_offset = compact_rows[0].turn_number if compact_rows else (self._engine_state.compacted_prefix_messages // 2)

        def _emit_weighted_progress(
            done: int,
            total: int,
            result,
            *,
            phase: str,
            phase_name: str,
            base_percent: int,
            span_percent: int,
            **kwargs,
        ) -> None:
            if not progress_callback:
                return
            progress_fraction = kwargs.pop("progress_fraction", 0.0)
            bounded_total = max(total, 1)
            bounded_done = max(0, min(done, bounded_total))
            if progress_fraction:
                bounded_done = min(
                    float(bounded_total),
                    float(bounded_done) + max(0.0, min(float(progress_fraction), 0.999)),
                )
            overall_percent = base_percent + int(span_percent * bounded_done / bounded_total)
            progress_callback(
                done,
                total,
                result,
                phase=phase,
                overall_percent=overall_percent,
                phase_name=phase_name,
                **kwargs,
            )

        _segmenter_phase_ranges = {
            "segment_tagging": (0, 12),
            "segment_grouping": (12, 10),
            "segment_postprocess": (22, 3),
        }

        def _segmenter_progress(done: int, total: int, result, **kwargs) -> None:
            phase_name = str(kwargs.pop("phase_name", "segment_tagging"))
            base_percent, span_percent = _segmenter_phase_ranges.get(
                phase_name, (0, 25),
            )
            _emit_weighted_progress(
                done,
                total,
                result,
                phase=phase_name,
                phase_name=phase_name,
                base_percent=base_percent,
                span_percent=span_percent,
                **kwargs,
            )

        # Phase 1: Segmenter (0-25%)
        segments = self._segmenter.segment(
            compact_messages,
            turn_offset=turn_offset,
            progress_callback=_segmenter_progress,
        )
        logger.info(
            "Segmented %d messages into %d segments (watermark=%d)",
            len(compact_messages), len(segments), self._engine_state.compacted_prefix_messages,
        )

        # Phase 2+3: Compact + Store (25-75%)
        actor_card_candidates: set[str] = set()
        results = self._compact_and_store(
            segments,
            len(compact_messages),
            compact_rows=compact_rows,
            progress_callback=progress_callback,
            generated_by_turn_id=generated_by_turn_id,
            operation_id=operation_id,
            disable_replacement_passes=disable_replacement_passes,
            actor_card_candidates=actor_card_candidates,
        )

        compacted_turn_ids = [
            row.canonical_turn_id
            for row in compact_rows
            if getattr(row, "canonical_turn_id", "")
        ]
        if compacted_turn_ids:
            self._store.mark_canonical_turns_compacted(
                self._config.conversation_id,
                compacted_turn_ids,
                **self._compaction_guard_kwargs(operation_id),
            )
        if compact_rows:
            self._refresh_compaction_watermark()

        tokens_freed = sum(r.original_tokens - r.summary_tokens for r in results)
        tags = list({tag for r in results for tag in r.tags})

        # Build/update tag summaries — only for tags in newly compacted segments
        tag_summaries_built, cover_tags = self._build_tag_summaries(
            results=results,
            compact_rows=compact_rows,
            operation_id=operation_id,
            generated_by_turn_id=generated_by_turn_id,
            progress_callback=progress_callback,
            disable_replacement_passes=disable_replacement_passes,
        )

        report = CompactionReport(
            segments_compacted=len(results),
            tokens_freed=tokens_freed,
            tags=tags,
            results=results,
            tag_summaries_built=tag_summaries_built,
            cover_tags=cover_tags,
        )

        self._refresh_shared_retrieval_snapshots()
        self._prewarm_context_hint(operation_id)
        self._consolidate_actor_cards_after_compaction(
            actor_card_candidates,
            disable_replacement_passes=disable_replacement_passes,
        )

        return report

    def _build_tag_summaries(
        self,
        *,
        results: list,
        compact_rows: list | None,
        operation_id: str | None,
        generated_by_turn_id: str = "",
        progress_callback: Callable[..., None] | None = None,
        disable_replacement_passes: bool = False,
    ) -> tuple[int, list[str]]:
        """Build and persist tag summaries for the just-compacted segments.

        Returns ``(count_built, cover_tags)`` so callers (``_run_compaction``)
        can populate the resulting ``CompactionReport``.

        Cover-tag derivation:

        * Every non-``_general`` tag carried by ``results`` (the
          just-compacted segments), plus the primary-tag guarantee so
          every result's ``primary_tag`` is included even when absent
          from the tag lists. The tag-summary table must stay complete
          for the read paths that consume it directly (context hint,
          broad/recall-all floor, tag-summary-embedding scoring); the
          staleness check inside ``compact_tag_summaries`` bounds the
          LLM cost of the wide set.

        Turn-data sourcing for ``compact_tag_summaries`` (``tag_to_turns`` +
        ``tag_to_canonical_turn_ids`` + ``max_turn``):

        * Prefer the in-memory ``_turn_tag_index.entries`` (normal
          request-driven path). The index carries the same per-turn tags
          the tagger produced.
        * Fall back to deriving the maps from ``compact_rows`` when the
          index is empty (cold-start / takeover compactions). Each
          ``CanonicalTurnRow`` carries its own ``turn_number`` +
          ``canonical_turn_id`` + ``tags`` + ``primary_tag``, so the data
          is equivalent for the compactor's per-tag summary builder. The
          fallback closes a gap where takeover compactions with an empty
          in-memory index silently skipped tag-summary building even
          though ``cover_tags`` was correctly populated.

        Caller contract: invoke once per compaction pass with the
        ``results`` and ``compact_rows`` produced upstream.
        """
        if not (results and self._compactor):
            return 0, []

        # Every non-``_general`` tag carried by the just-compacted
        # segments gets a tag summary. Historically this intersected the
        # greedy set-cover with the compacted tags (plus a primary-tag
        # guarantee), which structurally omitted every non-primary
        # secondary tag outside the cover — those tags landed in
        # ``segment_tags`` with no ``tag_summaries`` row on every
        # compaction. The read side assumes completeness: the
        # context-hint topic list, the broad/recall-all summary floor,
        # and tag-summary-embedding scoring all read the
        # ``tag_summaries`` table directly, so an omitted tag was
        # invisible there, and a row materialized by an external repair
        # sweep went permanently stale because later compactions kept
        # skipping the tag. The existing staleness check inside
        # ``compact_tag_summaries`` keeps the widened set cheap: fresh
        # summaries are skipped, only new/stale ones burn LLM budget.
        cover_tags: list[str] = sorted({
            tag
            for r in results
            for tag in r.tags
            if tag and tag != "_general"
        })
        # Primary tag guarantee (unchanged): every segment's primary_tag
        # gets a summary even when it is absent from the tag lists.
        cover_set = set(cover_tags)
        for r in results:
            if r.primary_tag and r.primary_tag not in cover_set:
                cover_tags.append(r.primary_tag)
                cover_set.add(r.primary_tag)
        if not cover_tags:
            return 0, []

        # Gather segment summaries per cover tag (input to the compactor's
        # per-tag summary builder).
        tag_to_summaries: dict[str, list] = {}
        for tag in cover_tags:
            summaries = self._store.get_summaries_by_tags(
                tags=[tag], min_overlap=1, limit=50,
                conversation_id=self._config.conversation_id,
            )
            if summaries:
                tag_to_summaries[tag] = summaries

        # Structured claims cross a trust boundary here. The compactor is a
        # pure model-facing component and cannot tell whether a stored segment
        # dropped a newer correction, changed its canonical ids, or forged a
        # claim provenance digest. Rehydrate the exact physical rows once and
        # pass only an opaque proof for envelopes that still validate. Invalid
        # segment prose remains useful to the retrieval-only tag synopsis.
        tag_rollup_sources = [
            summary
            for summaries in tag_to_summaries.values()
            for summary in summaries
        ]
        requested_source_ids = list(dict.fromkeys(
            canonical_id
            for summary in tag_rollup_sources
            for canonical_id in (
                getattr(
                    getattr(summary, "metadata", None),
                    "canonical_turn_ids",
                    (),
                ) or ()
            )
            if type(canonical_id) is str
            and bool(canonical_id)
            and canonical_id == canonical_id.strip()
        ))
        try:
            physical_rows = self._store.get_canonical_turn_rows_by_id(
                [
                    (self._config.conversation_id, canonical_id)
                    for canonical_id in requested_source_ids
                ],
                internal_validation=True,
            )
            physical_by_id = {
                str(getattr(row, "canonical_turn_id", "") or ""): row
                for row in physical_rows.values()
                if str(getattr(row, "canonical_turn_id", "") or "")
            }
            validated_tag_rollup_inputs = validate_tag_rollup_inputs(
                tag_rollup_sources,
                physical_by_id,
                conversation_id=self._config.conversation_id,
            )
        except Exception as exc:
            logger.warning(
                "Tag structured-input validation failed closed for conv=%s "
                "(%s)",
                self._config.conversation_id[:12],
                type(exc).__name__,
            )
            validated_tag_rollup_inputs = validate_tag_rollup_inputs(
                (), {}, conversation_id=self._config.conversation_id,
            )
        unique_source_refs = {
            str(getattr(summary, "ref", "") or "")
            for summary in tag_rollup_sources
            if str(getattr(summary, "ref", "") or "")
        }
        rejected_structured_count = len(
            unique_source_refs - validated_tag_rollup_inputs.segment_refs
        )
        if rejected_structured_count:
            logger.warning(
                "Tag rollup withheld structured envelopes for %d of %d "
                "source segments in conv=%s",
                rejected_structured_count,
                len(unique_source_refs),
                self._config.conversation_id[:12],
            )

        # Gather turn numbers + canonical_turn_ids per cover tag, plus
        # ``max_turn``. Prefer the in-memory index; fall back to the
        # compact_rows source when the index is empty.
        tag_to_turns: dict[str, list[int]] = {}
        tag_to_canonical_turn_ids: dict[str, list[str]] = {}
        if self._turn_tag_index.entries:
            for entry in self._turn_tag_index.entries:
                for tag in entry.tags:
                    if tag in cover_tags:
                        tag_to_turns.setdefault(tag, []).append(entry.turn_number)
                        if entry.canonical_turn_id:
                            tag_to_canonical_turn_ids.setdefault(tag, []).append(
                                entry.canonical_turn_id,
                            )
            max_turn = max(e.turn_number for e in self._turn_tag_index.entries)
        else:
            for row in compact_rows or []:
                row_tags = set(getattr(row, "tags", None) or [])
                row_primary = getattr(row, "primary_tag", "") or ""
                if row_primary:
                    row_tags.add(row_primary)
                # ``turn_number`` is a real int (0 is valid, -1 means
                # "unset"); avoid ``or`` because ``0 or -1`` evaluates
                # to -1 and corrupts the cover-tag → turn-number map.
                _raw_turn = getattr(row, "turn_number", -1)
                row_turn = int(_raw_turn if _raw_turn is not None else -1)
                row_cid = getattr(row, "canonical_turn_id", "") or ""
                for tag in row_tags:
                    if tag in cover_tags:
                        tag_to_turns.setdefault(tag, []).append(row_turn)
                        if row_cid:
                            tag_to_canonical_turn_ids.setdefault(tag, []).append(row_cid)
            max_turn = max(
                (
                    int(
                        getattr(r, "turn_number", -1)
                        if getattr(r, "turn_number", -1) is not None
                        else -1
                    )
                    for r in (compact_rows or [])
                ),
                default=0,
            )

        # Load existing tag summaries for the compactor's staleness check.
        existing_tag_summaries: dict = {}
        for tag in cover_tags:
            ts = self._store.get_tag_summary(
                tag, conversation_id=self._config.conversation_id,
            )
            if ts:
                existing_tag_summaries[tag] = ts

        new_tag_summaries = self._compactor.compact_tag_summaries(
            cover_tags=cover_tags,
            tag_to_summaries=tag_to_summaries,
            tag_to_turns=tag_to_turns,
            tag_to_canonical_turn_ids=tag_to_canonical_turn_ids,
            existing_tag_summaries=existing_tag_summaries,
            max_turn=max_turn,
            generated_by_turn_id=generated_by_turn_id,
            validated_tag_rollup_inputs=validated_tag_rollup_inputs,
        )

        for ts_i, ts in enumerate(new_tag_summaries):
            # C2R gate (fencing plan §7.2 #5 + #6): backlog-sweeper
            # dispatches skip both ``save_tag_summary`` and
            # ``store_tag_summary_embedding`` when a row already
            # exists for ``(tag, conversation_id)`` so the recovery
            # compaction cannot UPSERT over content owned by another
            # operation. The two writes share the lockstep invariant
            # (the tag-summary row gates the embedding row) so a
            # single existence probe via ``get_tag_summary`` covers
            # both.
            _skip_ts = False
            if disable_replacement_passes:
                _existing_ts = self._store.get_tag_summary(
                    ts.tag, conversation_id=self._config.conversation_id,
                )
                if _existing_ts is not None:
                    logger.info(
                        "  C2R gate: skipping tag summary write for "
                        "tag %s (pre-existing row)", ts.tag,
                    )
                    _skip_ts = True
            if not _skip_ts:
                self._store.save_tag_summary(
                    ts,
                    conversation_id=self._config.conversation_id,
                    **self._compaction_guard_kwargs(operation_id),
                )
            # Compute and store tag summary embedding for RRF scoring.
            try:
                from ..types import CompactionLeaseLost as _CLL
                embed_fn = self._semantic.get_embed_fn() if self._semantic else None
                if embed_fn and ts.summary and not _skip_ts:
                    emb = embed_fn([ts.summary[:2000]])[0]
                    self._store.store_tag_summary_embedding(
                        ts.tag, self._config.conversation_id, emb,
                        **self._compaction_guard_kwargs(operation_id),
                    )
            except _CLL:
                # Fail-closed: lease loss must propagate per fencing
                # plan §5.6 so the outer wrapper can emit
                # COMPACTION_WRITE_REJECTED.
                raise
            except Exception as e:
                logger.debug("Failed to embed tag summary '%s': %s", ts.tag, e)
            if progress_callback:
                try:
                    _pct = 95 + int(5 * (ts_i + 1) / max(len(new_tag_summaries), 1))
                    progress_callback(
                        ts_i + 1, len(new_tag_summaries), None,
                        phase="tag_summary_built",
                        overall_percent=_pct,
                        phase_name="tag_summaries",
                        tag=ts.tag,
                    )
                except Exception:
                    pass

        return len(new_tag_summaries), cover_tags

    #: Ownership-probe TTL for the pre-warm fence check. Deliberately huge
    #: so ``claim_compaction_lease``'s stale-heartbeat takeover branch can
    #: never trigger — the call degenerates to a pure "do I still own the
    #: active operation row" probe (claimed=True iff the caller already
    #: owns it).
    _PREWARM_OWNERSHIP_PROBE_TTL_S = 1e9

    def _prewarm_context_hint(self, operation_id: str | None) -> None:
        """Warm the context-hint cache at compaction commit.

        Compaction changes the engine-state fields the hint cache key
        hashes, so the first post-compaction request would rebuild the
        hint from every tag summary inside the request hot path. The
        callback rebuilds and caches it now instead (both cache layers).

        Fencing: on the guarded path (operation_id + worker_id set) the
        warm only runs while this worker still owns the active
        compaction operation — a worker that lost its lease mid-commit
        must not publish a hint built from its stale view. When
        ownership cannot be verified, the warm is skipped (degrading to
        the old first-request rebuild), never the other way around.

        Failure is isolated: a pre-warm error is logged and swallowed —
        it must never fail the compaction commit.
        """
        if self._prewarm_context_hint_callback is None:
            return
        try:
            if operation_id is not None and self._worker_id is not None:
                claim = self._store.claim_compaction_lease(
                    conversation_id=self._config.conversation_id,
                    lifecycle_epoch=int(self._engine_state.lifecycle_epoch),
                    worker_id=self._worker_id,
                    lease_ttl_s=self._PREWARM_OWNERSHIP_PROBE_TTL_S,
                )
                if not getattr(claim, "claimed", False):
                    logger.warning(
                        "CONTEXT_HINT_PREWARM_SKIPPED conv=%s op=%s: "
                        "compaction lease no longer held",
                        (self._config.conversation_id or "")[:12],
                        operation_id,
                    )
                    return
            self._prewarm_context_hint_callback()
        except Exception:
            logger.warning(
                "CONTEXT_HINT_PREWARM_FAILED conv=%s op=%s: first "
                "post-compaction request will rebuild the hint instead",
                (self._config.conversation_id or "")[:12],
                operation_id,
                exc_info=True,
            )

    def _refresh_shared_retrieval_snapshots(self) -> None:
        if self._session_state_provider is None or not self._config.conversation_id:
            return
        try:
            self._session_state_provider.refresh_tag_stats_snapshot(
                self._config.conversation_id,
            )
        except Exception:
            logger.warning(
                "Tag-stats snapshot refresh failed for %s",
                self._config.conversation_id[:12],
                exc_info=True,
            )
        try:
            self._session_state_provider.refresh_tag_summary_embedding_snapshot(
                self._config.conversation_id,
            )
        except Exception:
            logger.warning(
                "Tag-summary embedding snapshot refresh failed for %s",
                self._config.conversation_id[:12],
                exc_info=True,
            )

    def _commit_compaction_state(self, conversation_history: list[Message]) -> None:
        """Persist the committed compaction checkpoint."""
        saved = self._save_state_callback(conversation_history)
        if not saved:
            logger.warning(
                "Compaction checkpoint save failed for conversation %s",
                self._config.conversation_id[:12],
            )

    def _compact_and_store(
        self, segments: list, compact_messages_len: int,
        *,
        compact_rows: list["CanonicalTurnRow"] | None = None,
        progress_callback: Callable[..., None] | None = None,
        generated_by_turn_id: str = "",
        operation_id: str | None = None,
        disable_replacement_passes: bool = False,
        actor_card_candidates: set[str] | None = None,
    ) -> list[CompactionResult]:
        """Two-pass compact and store.

        Pass 1 (sequential, no LLM): handle stubs, check store for merge
        candidates, combine turns where matches are found.

        Pass 2 (batch, LLM): compact all prepared segments, then store results.
        """
        from datetime import datetime, timezone

        from ..types import (
            SOURCE_CANONICAL_TURN_IDS_KEY,
            CompactionResult,
            FactSignal,
            Message,
            SegmentMetadata,
            StoredSegment,
        )
        from .tag_scoring import compute_relatedness

        _ensure_engine_imports()
        compact_rows = list(compact_rows or [])

        # Source ids, not a conversation-sized history map, define evidence.
        # Historical merge sources are added on demand below; corrected old
        # source rows are read afresh instead of trusting cached segment text.
        source_ids = dict.fromkeys(
            source_id for segment in segments
            for source_id in self._segment_source_ids(segment)[0]
        )
        physical_by_id = {
            source_id: row
            for (_owner, source_id), row in physical_rows_by_id(
                self._store,
                ((self._config.conversation_id, source_id) for source_id in source_ids),
            ).items()
        }

        all_results: list[CompactionResult] = []

        def _emit_progress(
            done: int,
            total: int,
            result,
            *,
            phase: str,
            phase_name: str,
            base_percent: int,
            span_percent: int,
            **kwargs,
        ) -> None:
            if not progress_callback:
                return
            bounded_total = max(total, 1)
            bounded_done = max(0, min(done, bounded_total))
            overall_percent = base_percent + int(span_percent * bounded_done / bounded_total)
            progress_callback(
                done,
                total,
                result,
                phase=phase,
                overall_percent=overall_percent,
                phase_name=phase_name,
                **kwargs,
            )

        # D1: Gather fact signals from TurnTagIndex scoped per segment.
        # Topic segments may be noncontiguous (A-B-A interleaving), so the
        # segment's own canonical source ids are the only safe mapping back to
        # logical turns.  A positional cursor here previously attached fact
        # signals, range metadata, and tool outputs from unrelated segments.
        logical_rows_by_turn = {
            int(row.turn_number): row
            for row in compact_rows
            if getattr(row, "turn_number", None) is not None
            and int(row.turn_number) >= 0
        }
        segment_signals: dict[str, list[FactSignal]] = {}
        segment_code_refs: dict[str, list[dict]] = {}
        segment_turn_ranges: dict[str, tuple[int, int]] = {}  # seg.id -> (start, end_exclusive)
        segment_turn_numbers: dict[str, list[int]] = {}
        segment_canonical_turn_ids: dict[str, list[str]] = {}
        merged_existing_exact_ranges: dict[str, tuple[int, int] | None] = {}
        for seg in segments:
            exact_ids, _mapping_complete = self._segment_source_ids(seg)
            exact_turns = sorted({
                int(getattr(physical_by_id[cid], "turn_group_number", -1))
                for cid in exact_ids
                if cid in physical_by_id
                and getattr(physical_by_id[cid], "turn_group_number", None)
                is not None
                and int(getattr(
                    physical_by_id[cid], "turn_group_number", -1,
                )) >= 0
            })
            seg_rows = [
                logical_rows_by_turn[turn]
                for turn in exact_turns
                if turn in logical_rows_by_turn
            ]
            segment_turn_numbers[seg.id] = list(exact_turns)
            segment_canonical_turn_ids[seg.id] = list(exact_ids)
            if exact_turns:
                segment_turn_ranges[seg.id] = (
                    exact_turns[0],
                    exact_turns[-1] + 1,
                )
            signals: list[FactSignal] = []
            code_refs: list[dict] = []
            for row in seg_rows:
                entry = self._turn_tag_index.get_tags_for_canonical_turn(row.canonical_turn_id)
                if entry is None:
                    entry = self._turn_tag_index.bind_canonical_turn_id(
                        row.turn_number,
                        row.canonical_turn_id,
                    )
                if entry is None:
                    logger.debug(
                        "Missing canonical turn tag entry during compaction for conv=%s turn=%d canonical=%s",
                        self._config.conversation_id[:12],
                        row.turn_number,
                        row.canonical_turn_id[:12] if row.canonical_turn_id else "",
                    )
                    continue
                if entry and entry.fact_signals:
                    signals.extend(entry.fact_signals)
                if entry and getattr(entry, "code_refs", None):
                    code_refs.extend(entry.code_refs)
            if signals:
                segment_signals[seg.id] = signals
            if code_refs:
                segment_code_refs[seg.id] = code_refs

        merge_lookback = self._config.compactor.merge_lookback
        max_seg_tokens = self._config.compactor.max_segment_tokens
        merge_threshold = self._config.compactor.merge_overlap_threshold

        # ==================================================================
        # Pass 1: Sequential pre-pass — stubs + merge check (no LLM calls)
        # ==================================================================
        compactable: list = []  # segments ready for LLM compaction
        merged_mapping_prereqs: dict[str, bool] = {}
        now = datetime.now(timezone.utc)

        # P1: pre-load embeddings and embed_fn once (not per-segment)
        stored_embeddings = self._store.load_tag_summary_embeddings(
            conversation_id=self._config.conversation_id,
        )
        embed_fn = self._semantic.get_embed_fn() if self._semantic else None

        for seg in segments:
            # --- Stub passthrough (no LLM) ---
            text = " ".join(m.content for m in seg.messages)
            if _is_stub_content_fn(text):
                text = text.strip()
                turn_range = segment_turn_ranges.get(seg.id)
                exact_ids, mapping_complete = self._segment_source_ids(seg)
                mapping_complete = bool(
                    mapping_complete
                    and exact_ids
                    and all(cid in physical_by_id for cid in exact_ids)
                )
                logger.info(
                    "SEGMENT passthrough_stub ref=%s tokens=%d primary=%s",
                    seg.id[:8], seg.token_count, seg.primary_tag,
                )
                result = CompactionResult(
                    segment_id=seg.id,
                    primary_tag=seg.primary_tag,
                    tags=seg.tags,
                    summary=text or f"[empty turn: {seg.primary_tag}]",
                    summary_tokens=seg.token_count,
                    full_text=text,
                    original_tokens=seg.token_count,
                    messages=[
                        {
                            "role": m.role,
                            "content": m.content,
                            **({"metadata": m.metadata} if m.metadata else {}),
                        }
                        for m in seg.messages
                    ],
                    metadata=SegmentMetadata(
                        code_refs=segment_code_refs.get(seg.id, []),
                        turn_count=seg.turn_count,
                        canonical_turn_ids=(
                            list(exact_ids)
                            if exact_ids
                            else list(segment_canonical_turn_ids.get(seg.id, []))
                        ),
                        start_turn_number=turn_range[0] if turn_range else -1,
                        end_turn_number=(turn_range[1] - 1) if turn_range and turn_range[1] > turn_range[0] else -1,
                        generated_by_turn_id=generated_by_turn_id,
                        session_date=getattr(seg, "session_date", ""),
                        source_mapping_complete=mapping_complete,
                    ),
                    compression_ratio=1.0,
                    timestamp=seg.start_timestamp,
                )
                stored = StoredSegment(
                    ref=result.segment_id,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model="passthrough",
                    compression_ratio=1.0,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                )
                self._store.store_segment(
                    stored,
                    **self._compaction_guard_kwargs(operation_id),
                )
                # Propagate turn -> segment tool output links
                turn_numbers = segment_turn_numbers.get(seg.id, [])
                if turn_numbers:
                    self._propagate_tool_output_links_for_turns(
                        stored.ref, turn_numbers,
                        **self._compaction_guard_kwargs(operation_id),
                    )
                all_results.append(result)
                continue

            # --- Merge check: find best existing segment to merge with ---
            # C2R gate (fencing plan §7.2 #1): backlog-sweeper dispatches
            # force pure-insert behavior by skipping merge candidate
            # selection entirely. Without this, a recovery compaction
            # could merge into an existing segment and overwrite
            # content owned by other operations.
            if merge_lookback > 0 and not disable_replacement_passes:
                new_ids, new_mapping_complete = self._segment_source_ids(seg)
                new_identity_keys = (
                    self._source_human_identity_keys(new_ids, physical_by_id)
                    if new_mapping_complete
                    else None
                )
                merge_identity_eligible = bool(
                    new_mapping_complete
                    and new_ids
                    and new_identity_keys is not None
                    and len(new_identity_keys) == 1
                )
                candidates = (
                    self._store.get_segments_by_tags(
                        tags=seg.tags, min_overlap=1, limit=merge_lookback,
                        conversation_id=self._config.conversation_id,
                    )
                    if merge_identity_eligible
                    else []
                )
                seg_tags = set(seg.tags)
                seg_text = " ".join(m.content for m in seg.messages)[:2000]
                # B4: Pre-compute segment embedding once (not per-candidate)
                seg_embedding = None
                if embed_fn and seg_text:
                    try:
                        seg_embedding = embed_fn([seg_text])[0]
                    except Exception:
                        pass
                best_score = 0.0
                best_candidate = None
                best_candidate_source_ids: list[str] = []

                for candidate in candidates:
                    candidate_meta = candidate.metadata
                    candidate_source_ids = list(
                        getattr(candidate_meta, "canonical_turn_ids", []) or []
                    )
                    if not (
                        candidate_meta
                        and getattr(
                            candidate_meta, "source_mapping_complete", False,
                        )
                        and candidate_source_ids
                    ):
                        continue
                    physical_by_id.update({
                        source_id: row
                        for (_owner, source_id), row in physical_rows_by_id(
                            self._store,
                            ((self._config.conversation_id, source_id)
                             for source_id in candidate_source_ids
                             if source_id not in physical_by_id),
                        ).items()
                    })
                    candidate_identity_keys = self._source_human_identity_keys(
                        candidate_source_ids, physical_by_id,
                    )
                    if (
                        candidate_identity_keys is None
                        or len(candidate_identity_keys) > 1
                        or candidate_identity_keys != new_identity_keys
                    ):
                        continue

                    combined_tokens = candidate.full_tokens + seg.token_count
                    if combined_tokens > max_seg_tokens:
                        continue
                    # Multi-signal relatedness: tag overlap + embedding + keyword
                    cand_embedding = stored_embeddings.get(candidate.primary_tag)
                    relatedness = compute_relatedness(
                        tags_a=seg_tags,
                        tags_b=set(candidate.tags),
                        text_a=seg_text,
                        text_b=candidate.summary[:2000] if candidate.summary else "",
                        embedding_a=seg_embedding,
                        embedding_b=cand_embedding,
                    )
                    if relatedness < merge_threshold:
                        continue
                    try:
                        age_days = (now - candidate.created_at).days
                    except (TypeError, AttributeError):
                        age_days = 30
                    recency = max(0.5, 1.0 - age_days / 60)
                    combined_score = relatedness * recency
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate
                        best_candidate_source_ids = candidate_source_ids

                if best_candidate is not None:
                    # Combine turns: prepend existing segment's messages
                    candidate_messages = []
                    for cid in best_candidate_source_ids:
                        old_row = physical_by_id[cid]
                        if (old_row.user_content or "").strip():
                            metadata = {
                                SOURCE_CANONICAL_TURN_IDS_KEY: [cid],
                            }
                            if (old_row.sender or "").strip():
                                metadata["sender"] = {"name": old_row.sender}
                            candidate_messages.append(Message(
                                role="user",
                                content=old_row.user_content,
                                metadata=metadata,
                                source_actor_id=(
                                    old_row.sender_actor_id or ""
                                ).strip(),
                                source_logical_turn_number=int(
                                    getattr(old_row, "turn_number", -1) or 0
                                ),
                                source_audience_conversation_id=(
                                    old_row.audience_conversation_id or ""
                                ).strip(),
                                source_origin_channel_id=(
                                    old_row.origin_channel_id or ""
                                ).strip(),
                                source_audience_attribution_version=int(
                                    old_row.audience_attribution_version or 0
                                ),
                            ))
                        if (old_row.assistant_content or "").strip():
                            candidate_messages.append(Message(
                                role="assistant",
                                content=old_row.assistant_content,
                                metadata={
                                    SOURCE_CANONICAL_TURN_IDS_KEY: [cid],
                                },
                                source_logical_turn_number=int(
                                    getattr(old_row, "turn_number", -1) or 0
                                ),
                                source_audience_conversation_id=(
                                    old_row.audience_conversation_id or ""
                                ).strip(),
                                source_origin_channel_id=(
                                    old_row.origin_channel_id or ""
                                ).strip(),
                                source_audience_attribution_version=int(
                                    old_row.audience_attribution_version or 0
                                ),
                            ))
                    merged_mapping_prereqs[seg.id] = True
                    seg.messages = candidate_messages + list(seg.messages)
                    seg.merge_ref = best_candidate.ref
                    seg.token_count += best_candidate.full_tokens
                    start_candidates = [
                        value for value in (
                            best_candidate.start_timestamp,
                            seg.start_timestamp,
                        ) if value is not None
                    ]
                    end_candidates = [
                        value for value in (
                            best_candidate.end_timestamp,
                            seg.end_timestamp,
                        ) if value is not None
                    ]
                    if start_candidates:
                        seg.start_timestamp = min(start_candidates)
                    if end_candidates:
                        seg.end_timestamp = max(end_candidates)
                    old_tc = best_candidate.metadata.turn_count if best_candidate.metadata else len(best_candidate.messages) // 2
                    seg.turn_count += old_tc
                    seg.tags = list(set(best_candidate.tags) | seg_tags)
                    old_start = getattr(best_candidate.metadata, "start_turn_number", -1)
                    old_end = getattr(best_candidate.metadata, "end_turn_number", -1)
                    merged_existing_exact_ranges[seg.id] = (
                        (old_start, old_end)
                        if old_start >= 0 and old_end >= old_start
                        else None
                    )
                    logger.info(
                        "MERGE PREP: segment '%s' (%s) merging with stored %s "
                        "(%s, %d existing turns, relatedness=%.2f)",
                        seg.id[:8], seg.primary_tag,
                        best_candidate.ref[:8], best_candidate.primary_tag,
                        old_tc, best_score,
                    )

            compactable.append(seg)

        if not compactable:
            if all_results:
                _emit_progress(
                    len(all_results),
                    len(all_results),
                    all_results[-1],
                    phase="segment_stored",
                    phase_name="store",
                    base_percent=80,
                    span_percent=15,
                )
            return all_results

        logger.info("Pass 1 complete: %d stubs stored, %d segments ready for compaction (%d merges)",
                    len(all_results), len(compactable),
                    sum(1 for s in compactable if s.merge_ref))

        # ==================================================================
        # Pass 2: Batch LLM compaction + store
        # ==================================================================
        fact_signals_by_segment = {
            seg.id: segment_signals[seg.id]
            for seg in compactable if seg.id in segment_signals
        } or None
        code_refs_by_segment = {
            seg.id: segment_code_refs[seg.id]
            for seg in compactable if seg.id in segment_code_refs
        } or None

        def _compactor_progress(done: int, total: int, result, **kwargs) -> None:
            kwargs.pop("phase", None)  # avoid double-passing phase
            _emit_progress(
                done,
                total,
                result,
                phase="segment_compacting",
                phase_name=str(kwargs.pop("phase_name", "compactor")),
                base_percent=25,
                span_percent=55,
                **kwargs,
            )

        # Rosters come from the physical rows the segment's own messages name,
        # not from the positional cursor above: the cursor walks logical merged
        # rows and cannot survive noncontiguous topic grouping or a session
        # split, so it is not a safe basis for deciding who authored a fact.
        # Validate the configured agent identity ONCE against every physical
        # row this run holds, before any suppression decision is taken.
        self._agent_quote_counts = {
            self.QUOTE_AGENT_AUTHORED: 0,
            self.QUOTE_NOT_AGENT: 0,
            self.QUOTE_IDENTITY_UNKNOWN: 0,
        }
        # A historical canonical target already owns its requester facts.
        # Preserve that suppression, including ambiguous targets, with a
        # bounded lookup of only the quoted source-message ids.
        physical_by_id.update(reply_target_rows(
            self._store, self._config.conversation_id, tuple(physical_by_id.values()),
        ))
        _agent_actor_ids = self._validated_agent_actor_ids(physical_by_id)
        actor_rosters_by_segment = {
            seg.id: self._build_actor_roster(seg, physical_by_id, _agent_actor_ids)
            for seg in compactable
        }
        self._log_quote_outcomes()
        exact_source_ids = {
            seg.id: self._segment_source_ids(seg) for seg in compactable
        }

        results = self._compactor.compact(
            compactable,
            fact_signals_by_segment=fact_signals_by_segment,
            code_refs_by_segment=code_refs_by_segment,
            actor_rosters_by_segment=actor_rosters_by_segment,
            progress_callback=_compactor_progress,
        )

        # Coalesce person-card work across every segment in this compaction.
        # One actor can appear in many segments; rebuilding after each one
        # wastes model calls and lets an early rebuild observe only part of the
        # just-written evidence.
        card_actors_to_rebuild: set[str] = set()
        for seg_idx, result in enumerate(results):
            seg = compactable[seg_idx]
            new_turn_range = segment_turn_ranges.get(seg.id)
            exact_start = -1
            exact_end = -1
            if new_turn_range and new_turn_range[1] > new_turn_range[0]:
                new_start = new_turn_range[0]
                new_end = new_turn_range[1] - 1
                if seg.merge_ref:
                    existing_range = merged_existing_exact_ranges.get(seg.id)
                    if existing_range is not None:
                        exact_start = min(existing_range[0], new_start)
                        exact_end = max(existing_range[1], new_end)
                else:
                    exact_start = new_start
                    exact_end = new_end
            result.metadata.start_turn_number = exact_start
            result.metadata.end_turn_number = exact_end
            result.metadata.generated_by_turn_id = generated_by_turn_id
            # Prefer the exact per-message provenance. A legacy or synthesized
            # segment with no source ids remains incomplete; it must not borrow
            # a positional row mapping from an unrelated topic segment.
            exact_ids, mapping_complete = exact_source_ids.get(seg.id, ([], False))
            mapping_complete = bool(
                mapping_complete
                and exact_ids
                and all(cid in physical_by_id for cid in exact_ids)
                and (
                    not seg.merge_ref
                    or merged_mapping_prereqs.get(seg.id, False)
                )
            )
            if exact_ids:
                result.metadata.canonical_turn_ids = list(exact_ids)
                result.metadata.source_mapping_complete = bool(mapping_complete)
            else:
                result.metadata.canonical_turn_ids = list(
                    segment_canonical_turn_ids.get(seg.id, [])
                )
                result.metadata.source_mapping_complete = False

            # Store or update
            if seg.merge_ref:
                stored = StoredSegment(
                    ref=seg.merge_ref,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=seg.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model=self._compactor.model_name,
                    compression_ratio=result.compression_ratio,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                )
                self._store.update_segment(
                    stored,
                    **self._compaction_guard_kwargs(operation_id),
                )
                self._semantic.embed_and_store_chunks(
                    stored,
                    **self._compaction_guard_kwargs(
                        operation_id, include_conversation_id=True,
                    ),
                    disable_replacement_passes=disable_replacement_passes,
                )
                result.segment_id = seg.merge_ref
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  COMPACT MERGED %d/%d: %s (session_date=%s, %dt→%dt, %d turns)",
                    seg_idx + 1, len(results), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens, seg.turn_count,
                )
            else:
                stored = StoredSegment(
                    ref=result.segment_id,
                    conversation_id=self._config.conversation_id,
                    primary_tag=result.primary_tag,
                    tags=result.tags,
                    summary=result.summary,
                    summary_tokens=result.summary_tokens,
                    full_text=result.full_text,
                    full_tokens=result.original_tokens,
                    messages=result.messages,
                    metadata=result.metadata,
                    compaction_model=self._compactor.model_name,
                    compression_ratio=result.compression_ratio,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                )
                self._store.store_segment(
                    stored,
                    **self._compaction_guard_kwargs(operation_id),
                )
                self._semantic.embed_and_store_chunks(
                    stored,
                    **self._compaction_guard_kwargs(
                        operation_id, include_conversation_id=True,
                    ),
                    disable_replacement_passes=disable_replacement_passes,
                )
                session_date = getattr(result.metadata, 'session_date', '') if result.metadata else ''
                logger.info(
                    "  COMPACT NEW %d/%d: %s (session_date=%s, %dt→%dt, %d turns)",
                    seg_idx + 1, len(results), result.primary_tag,
                    session_date or 'none',
                    result.original_tokens, result.summary_tokens, seg.turn_count,
                )

            # Propagate turn -> segment tool output links
            turn_numbers = segment_turn_numbers.get(seg.id, [])
            if turn_numbers:
                self._propagate_tool_output_links_for_turns(
                    stored.ref, turn_numbers,
                    **self._compaction_guard_kwargs(operation_id),
                )

            all_results.append(result)
            stored_done = seg_idx + 1
            card_actors_to_rebuild.update(
                (physical_by_id[canonical_id].sender_actor_id or "").strip()
                for canonical_id in (
                    stored.metadata.canonical_turn_ids or []
                )
                if canonical_id in physical_by_id
                and (
                    physical_by_id[canonical_id].sender_actor_id or ""
                ).strip()
            )

            _emit_progress(
                stored_done,
                len(results),
                result,
                phase="segment_stored",
                phase_name="store",
                base_percent=80,
                span_percent=15,
            )

            _seg_ref = stored.ref
            _existing_facts_before = self._store.get_facts_by_segment(_seg_ref)
            if result.facts or _existing_facts_before:
                card_actors_to_rebuild.update({
                    (fact.author_actor_id or "").strip()
                    for fact in [*_existing_facts_before, *result.facts]
                    if (fact.author_actor_id or "").strip()
                })
                for fact in result.facts:
                    fact.segment_ref = _seg_ref
                    fact.conversation_id = self._config.conversation_id
                # C2R gate (fencing plan §7.2 #3): backlog-sweeper
                # dispatches skip ``replace_facts_for_segment`` when
                # the segment already has facts so the recovery
                # compaction cannot DELETE-then-INSERT facts owned by
                # other operations. The new-segment path
                # (no pre-existing facts) is a pure insert and runs
                # normally.
                _skip_facts = False
                if disable_replacement_passes:
                    _existing = self._store.get_facts_by_segment(_seg_ref)
                    if _existing:
                        logger.info(
                            "  C2R gate: skipping fact replacement for "
                            "segment %s (%d pre-existing facts)",
                            result.primary_tag, len(_existing),
                        )
                        _skip_facts = True
                if _skip_facts:
                    _deleted, _inserted = 0, 0
                else:
                    _deleted, _inserted = self._store.replace_facts_for_segment(
                        self._config.conversation_id, _seg_ref, result.facts,
                        **self._compaction_guard_kwargs(operation_id),
                    )
                    if _deleted:
                        # Name the REF as well as the tag. Eviction is keyed on
                        # segment_ref and one tag maps to many refs, so a reader
                        # counting segments from the tag alone counts tags.
                        logger.info(
                            "  Replaced %d old facts with %d new for segment %s ref=%s",
                            _deleted, _inserted, result.primary_tag, _seg_ref,
                        )
                    else:
                        logger.info(
                            "  Stored %d facts for segment %s ref=%s",
                            _inserted, result.primary_tag, _seg_ref,
                        )
                    # Embed-on-write: only for facts actually inserted. The
                    # DELETE half of replace_facts_for_segment cascades old
                    # vectors via the FK. A (0, 0) return (guard mismatch at
                    # OBSERVE) or a raised CompactionLeaseLost never reaches
                    # here with rows to embed.
                    if _inserted:
                        self._embed_and_store_fact_embeddings(
                            result.facts,
                            operation_id=operation_id,
                            guard_kwargs=self._compaction_guard_kwargs(operation_id),
                        )
                _superseded_count = 0
                _links_count = 0
                # C2R gate (fencing plan §7.2 #7/#8): backlog-sweeper
                # dispatches skip the supersession + fact-link mutation
                # passes entirely. ``promote_planned_facts`` ->
                # ``update_fact_fields`` and ``set_fact_superseded``
                # are both replacement-shaped writes that a recovery
                # compaction must not perform. V1 takes the simplest
                # path and skips ``check_and_link`` /
                # ``check_and_supersede`` outright; any pure-insert
                # ``store_fact_links`` write that would have followed
                # is also skipped to keep the gate behavior uniform.
                if self._supersession_checker and not disable_replacement_passes:
                    from ..types import CompactionLeaseLost
                    _full_guard = self._compaction_guard_kwargs(
                        operation_id, include_conversation_id=True,
                    )
                    _triple_guard = self._compaction_guard_kwargs(operation_id)
                    try:
                        if hasattr(self._supersession_checker, 'check_and_link'):
                            _links_count, _superseded_count = self._supersession_checker.check_and_link(
                                result.facts, **_full_guard,
                            )
                        else:
                            _superseded_count = self._supersession_checker.check_and_supersede(
                                result.facts, **_triple_guard,
                            ) or 0
                        if _superseded_count:
                            logger.info("  Superseded %d facts for segment %s", _superseded_count, result.primary_tag)
                        if _links_count:
                            logger.info("  Linked %d facts for segment %s", _links_count, result.primary_tag)
                    except CompactionLeaseLost:
                        # Fencing plan §5.6 fail-closed handling: the
                        # outer compaction wrapper catches this and
                        # emits COMPACTION_WRITE_REJECTED, exiting the
                        # operation cleanly without walking the rest
                        # of the phases.
                        raise
                    except Exception as e:
                        logger.warning("Supersession/linking failed: %s", e)
                _emit_progress(
                    stored_done,
                    len(results),
                    result,
                    phase="facts_extracted",
                    phase_name="store",
                    base_percent=80,
                    span_percent=15,
                    fact_count=len(result.facts),
                    superseded_count=_superseded_count,
                    links_count=_links_count,
                )

        if actor_card_candidates is not None:
            actor_card_candidates.update(card_actors_to_rebuild)

        return all_results
