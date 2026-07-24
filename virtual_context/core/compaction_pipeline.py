"""CompactionPipeline: segmentation, compaction, and storage.

Extracted from engine.py — handles Phase 2 of turn processing (compact_if_needed),
manual compaction (compact_manual), and the shared compaction core (_run_compaction).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from .engine_utils import extract_turn_pairs
from .store import ContextStore
from .turn_tag_index import TurnTagIndex
from ..types import LLMProviderError

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
        SegmentMetadata,
        StoredSegment,
        VirtualContextConfig,
    )

logger = logging.getLogger(__name__)

_ACTOR_CARD_CITATION_LIMIT = 16


class _ActorCardAdmissionError(RuntimeError):
    """Validation failure that preserves a hashable, non-logged response."""

    def __init__(self, message: str, response_text: str = "") -> None:
        super().__init__(message)
        self.response_text = response_text


class _ActorCardCoverageError(_ActorCardAdmissionError):
    """A deterministic curator/admission judgment disagreement."""


class _EmptyResponseFallbackProvider:
    """Use a second model for refusal fallback and coverage adjudication."""

    def __init__(
        self,
        primary,
        fallback,
        *,
        primary_model: str,
        fallback_model: str,
        stage: str = "admission",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._stage = stage

    def complete(self, **kwargs):
        text, usage, _source = self.complete_with_source(**kwargs)
        return text, usage

    def complete_with_source(self, **kwargs):
        """Complete and report which independent model supplied the result."""
        try:
            text, usage = self._primary.complete(**kwargs)
        except LLMProviderError as exc:
            logger.warning(
                "ACTOR_CARD_%s_FALLBACK primary_model=%s "
                "fallback_model=%s reason=provider_error status=%s",
                self._stage.upper(),
                self._primary_model,
                self._fallback_model,
                exc.status_code,
            )
            fallback_text, fallback_usage = self._fallback.complete(**kwargs)
            return fallback_text, fallback_usage, "fallback"
        if isinstance(text, str) and text.strip():
            return text, usage, "primary"
        logger.warning(
            "ACTOR_CARD_%s_FALLBACK primary_model=%s "
            "fallback_model=%s reason=empty_response",
            self._stage.upper(),
            self._primary_model,
            self._fallback_model,
        )
        fallback_text, fallback_usage = self._fallback.complete(**kwargs)
        return fallback_text, fallback_usage, "fallback"

    def complete_fallback(self, **kwargs):
        """Call the independent fallback directly for a coverage tiebreak."""
        text, usage = self._fallback.complete(**kwargs)
        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "ACTOR_CARD_%s_FALLBACK_EMPTY model=%s",
                self._stage.upper(),
                self._fallback_model,
            )
        return text, usage


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

    def _due_actor_card_rebuilds(self, *, limit: int = 25) -> list[str]:
        """Read the bounded retry queue for transient card-build failures."""
        if not getattr(
            self._config.assembler,
            "actor_card_enabled",
            False,
        ):
            return []
        getter = getattr(
            self._store,
            "list_due_actor_card_rebuilds",
            None,
        )
        if not callable(getter):
            return []
        from datetime import datetime, timezone

        try:
            return list(getter(
                self._config.tenant_id,
                due_at=datetime.now(timezone.utc).isoformat(),
                limit=max(0, int(limit)),
            ))
        except Exception:
            logger.warning(
                "actor card retry queue read failed",
                exc_info=True,
            )
            return []

    def _rebuild_actor_card(self, actor_id: str, *, force: bool = False) -> int:
        """Curate and atomically replace one actor's rebuildable card cache.

        Facts are useful compact evidence, but they are not the membership
        criterion for a person card. Exact actor-authored canonical turns are
        also supplied so a substantive contributor can receive a meaningful
        card even when the fact extractor emitted nothing. A separate model
        independently judges substantive coverage and semantically admits each
        immutable candidate before the atomic replacement.
        """
        if (
            not force
            and not getattr(self._config.assembler, "actor_card_enabled", False)
        ):
            return 0
        if self._compactor is None or not actor_id:
            return 0
        from datetime import datetime, timezone

        from ..types import (
            CARD_CROSS_CONTEXT_KINDS,
            CARD_ENTRY_BODY_MAX_CHARS,
            CARD_KINDS,
            CARD_SCOPE_CROSS_CONTEXT,
            CARD_SCOPE_SAME_CONVERSATION,
            CARD_SENSITIVITY_NORMAL,
            ActorCardEntry,
            ActorCardEntrySource,
        )

        tenant_id = self._config.tenant_id

        def _read_inputs() -> tuple[list, list, list, str]:
            configured_curation_model = (
                getattr(
                    self._config.assembler,
                    "actor_card_curation_model",
                    "",
                )
                or ""
            ).strip()
            curation_fallback_model = (
                getattr(
                    self._config.assembler,
                    "actor_card_curation_fallback_model",
                    "",
                )
                or ""
            ).strip()
            admission_model = (
                getattr(
                    self._config.assembler,
                    "actor_card_admission_model",
                    "",
                )
                or ""
            ).strip()
            admission_fallback_model = (
                getattr(
                    self._config.assembler,
                    "actor_card_admission_fallback_model",
                    "",
                )
                or ""
            ).strip()
            facts = list(self._store.list_actor_facts(
                tenant_id,
                actor_id,
                limit=int(self._config.assembler.actor_card_fact_limit),
            ))
            turns = list(self._store.list_actor_turn_sources(
                tenant_id,
                actor_id,
                limit=int(getattr(
                    self._config.assembler,
                    "actor_card_turn_limit",
                    500,
                )),
            ))
            carryover_getter = getattr(
                self._store,
                "list_actor_card_carryovers",
                None,
            )
            carryovers = (
                list(carryover_getter(tenant_id, actor_id))
                if callable(carryover_getter)
                else []
            )
            fact_payload = [
                {
                    "id": source.fact.id,
                    "subject": source.fact.subject,
                    "verb": source.fact.verb,
                    "object": source.fact.object,
                    "what": source.fact.what,
                    "status": source.fact.status,
                    "superseded_by": source.fact.superseded_by,
                    "fact_type": source.fact.fact_type,
                    "mentioned_at": source.fact.mentioned_at.isoformat(),
                    "session_date": source.fact.session_date,
                    "author_version": (
                        source.fact.author_attribution_version
                    ),
                    "author_role": source.fact.author_source_role,
                }
                for source in facts
            ]
            turn_payload = [
                {
                    "id": source.turn.canonical_turn_id,
                    "owner": source.owner_conversation_id,
                    "audience": source.audience_conversation_id,
                    "channel": source.audience_channel_id,
                    "content": source.turn.user_content,
                    "created_at": (
                        source.turn.created_at
                        or source.turn.first_seen_at
                        or ""
                    ),
                    "owner_epoch": source.owner_lifecycle_epoch,
                    "audience_epoch": source.audience_lifecycle_epoch,
                }
                for source in turns
            ]
            carryover_payload = [
                {
                    "entry": {
                        "id": entry.id,
                        "kind": entry.kind,
                        "body": entry.body,
                        "confidence": entry.confidence,
                        "scope": entry.audience_scope,
                    },
                    "sources": sorted([
                        {
                            "owner": source.owner_conversation_id,
                            "audience": source.audience_conversation_id,
                            "channel": source.audience_channel_id,
                            "fact_id": source.fact_id,
                            "turn_id": source.canonical_turn_id,
                        }
                        for source in sources
                    ], key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                    )),
                }
                for entry, sources in carryovers
            ]
            curation_model = configured_curation_model or (
                getattr(self._compactor, "model_name", "")
                or getattr(
                    getattr(self._compactor, "llm", None),
                    "model",
                    "",
                )
                or type(getattr(self._compactor, "llm", None)).__name__
            )
            digest = hashlib.sha256(json.dumps(
                {
                    "policy": 12,
                    "curation_model": curation_model,
                    "curation_fallback_model": curation_fallback_model,
                    "admission_model": admission_model,
                    "admission_fallback_model": admission_fallback_model,
                    "prompt_max_chars": int(getattr(
                        self._config.assembler,
                        "actor_card_prompt_max_chars",
                        192_000,
                    )),
                    "facts": fact_payload,
                    "turns": turn_payload,
                    "carryovers": carryover_payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            return facts, turns, carryovers, digest

        (
            fact_sources,
            turn_sources,
            carryover_entries,
            input_hash,
        ) = _read_inputs()
        profile = self._store.get_actor_profile(tenant_id, actor_id)
        if profile is None:
            return 0
        if not force:
            status_getter = getattr(
                self._store,
                "get_actor_card_rebuild_status",
                None,
            )
            status = (
                status_getter(tenant_id, actor_id)
                if callable(status_getter)
                else None
            )
            failed_outcomes = {
                "model_error",
                "invalid_response",
                "rejected_all",
                "admission_error",
                "coverage_disagreement",
                "coverage_gap",
                "stale_or_rejected_write",
            }
            if (
                status
                and (status.get("input_hash") or "") == input_hash
                and status.get("outcome") in failed_outcomes
            ):
                failures = int(status.get("failure_count") or 0)
                if failures >= 3:
                    logger.error(
                        "ACTOR_CARD_REBUILD_SUPPRESSED actor=%s "
                        "input_hash=%s failures=%d reason=terminal",
                        actor_id[:24],
                        input_hash[:16],
                        failures,
                    )
                    return 0
                retry_raw = status.get("next_retry_at") or ""
                try:
                    retry_at = datetime.fromisoformat(
                        str(retry_raw).replace("Z", "+00:00")
                    )
                except (TypeError, ValueError):
                    retry_at = None
                if retry_at is not None and retry_at > datetime.now(timezone.utc):
                    logger.info(
                        "ACTOR_CARD_REBUILD_SUPPRESSED actor=%s "
                        "input_hash=%s failures=%d reason=backoff",
                        actor_id[:24],
                        input_hash[:16],
                        failures,
                    )
                    return 0
        if profile.card_input_hash == input_hash and not profile.card_dirty:
            return 0
        build_marker = (
            f"building:{input_hash}:{time.time_ns()}:{id(self)}"
        )
        if not self._store.mark_actor_card_dirty(
            tenant_id,
            actor_id,
            build_input_hash=build_marker,
        ):
            return 0
        # Re-enumerate only after the unique build marker is installed. A
        # mutation before the marker is therefore included; a mutation after
        # it clears the marker and makes the transactional replacement fail.
        (
            fact_sources,
            turn_sources,
            carryover_entries,
            input_hash,
        ) = _read_inputs()

        fact_source_by_audience_id = {
            (source.audience_conversation_id, source.fact.id): source
            for source in fact_sources
        }
        turn_source_by_audience_id = {
            (
                source.audience_conversation_id,
                source.turn.canonical_turn_id,
            ): source
            for source in turn_sources
        }
        raw_entries: list[tuple[str, dict, set[str]]] = []
        curator_substantive_by_audience: dict[str, bool] = {}
        fact_sources_by_audience: dict[str, list] = {}
        turn_sources_by_audience: dict[str, list] = {}
        for source in fact_sources:
            fact_sources_by_audience.setdefault(
                source.audience_conversation_id,
                [],
            ).append(source)
        for source in turn_sources:
            turn_sources_by_audience.setdefault(
                source.audience_conversation_id,
                [],
            ).append(source)
        audience_ids = sorted(
            set(fact_sources_by_audience)
            | set(turn_sources_by_audience)
            | {
                (source.audience_conversation_id or "").strip()
                for _entry, sources in carryover_entries
                for source in sources
                if (source.audience_conversation_id or "").strip()
            }
        )
        response_text = ""
        admission_response_text = ""
        parsed_entries = True
        model_exception: Exception | None = None
        admission_exception: Exception | None = None
        curation_responses: list[str] = []
        try:
            for audience_id in audience_ids:
                (
                    partition_response,
                    partition_substantive,
                    _coverage_reason,
                    partition_entries,
                    visible_turn_ids,
                ) = self._curate_actor_card_partition(
                    fact_sources_by_audience.get(audience_id, []),
                    turn_sources_by_audience.get(audience_id, []),
                )
                curation_responses.append(partition_response)
                curator_substantive_by_audience[audience_id] = (
                    partition_substantive
                )
                raw_entries.extend(
                    (audience_id, item, visible_turn_ids)
                    for item in partition_entries
                )
            response_text = json.dumps(
                curation_responses,
                separators=(",", ":"),
            )
        except Exception as exc:
            parsed_entries = False
            response_text = getattr(exc, "response_text", "")
            if not isinstance(exc, _ActorCardAdmissionError):
                model_exception = exc

        now = datetime.now(timezone.utc).isoformat()
        # Every audience is curated independently and receives the configured
        # per-kind budget in its own prompt. Enforce the same boundary here:
        # a busy DM must not consume a guild's quota (or vice versa) and turn
        # an otherwise substantive partition into a terminal coverage gap.
        per_audience_kind: dict[tuple[str, str], int] = {}
        normalized: list[tuple[ActorCardEntry, list[ActorCardEntrySource]]] = []
        normalized_by_audience: dict[
            str,
            list[tuple[ActorCardEntry, list[ActorCardEntrySource]]],
        ] = {}
        normalized_entries_by_key: dict[tuple, ActorCardEntry] = {}
        rejected: Counter[str] = Counter()
        for audience_id, item, prompt_turn_ids in raw_entries:
            if not isinstance(item, dict):
                rejected["entry_not_object"] += 1
                continue
            kind = item.get("kind")
            body = item.get("body")
            confidence = item.get("confidence")
            fact_ids = item.get("fact_ids")
            turn_ids = item.get("turn_ids")
            if set(item) != {
                "kind",
                "body",
                "confidence",
                "fact_ids",
                "turn_ids",
            }:
                rejected["invalid_entry_shape"] += 1
                continue
            if kind not in CARD_KINDS:
                rejected["invalid_kind"] += 1
                continue
            quota_key = (audience_id, kind)
            if not isinstance(body, str) or not body.strip():
                rejected["invalid_body"] += 1
                continue
            body = body.strip()
            if (
                len(body) > CARD_ENTRY_BODY_MAX_CHARS
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in body)
            ):
                rejected["invalid_body"] += 1
                continue
            if isinstance(confidence, bool) or not isinstance(
                confidence, (int, float)
            ):
                rejected["invalid_confidence"] += 1
                continue
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                rejected["invalid_confidence"] += 1
                continue
            if not isinstance(fact_ids, list):
                rejected["invalid_fact_ids"] += 1
                continue
            if not isinstance(turn_ids, list):
                rejected["invalid_turn_ids"] += 1
                continue
            if any(not isinstance(fid, str) for fid in fact_ids):
                rejected["invalid_fact_ids"] += 1
                continue
            if any(not isinstance(turn_id, str) for turn_id in turn_ids):
                rejected["invalid_turn_ids"] += 1
                continue

            # Some otherwise schema-compliant curators copy a canonical turn
            # id into both citation arrays. IDs are opaque, so repair only the
            # case that is structurally provable from this exact partition:
            # an id placed in the wrong namespace must resolve as a visible
            # source in the other namespace under the same audience. Truly
            # unknown or cross-audience ids still reject the whole entry.
            normalized_fact_ids: list[str] = []
            normalized_turn_ids: list[str] = []

            def _valid_fact_id(source_id: str) -> bool:
                return (
                    audience_id,
                    source_id,
                ) in fact_source_by_audience_id

            def _valid_turn_id(source_id: str) -> bool:
                return bool(
                    source_id in prompt_turn_ids
                    and (
                        audience_id,
                        source_id,
                    ) in turn_source_by_audience_id
                )

            unknown_fact_id = False
            for source_id in dict.fromkeys(fact_ids):
                if _valid_fact_id(source_id):
                    normalized_fact_ids.append(source_id)
                elif _valid_turn_id(source_id):
                    normalized_turn_ids.append(source_id)
                else:
                    unknown_fact_id = True
                    break
            if unknown_fact_id:
                rejected["unknown_or_cross_audience_fact_id"] += 1
                continue

            unknown_turn_id = False
            for source_id in dict.fromkeys(turn_ids):
                if _valid_turn_id(source_id):
                    normalized_turn_ids.append(source_id)
                elif _valid_fact_id(source_id):
                    normalized_fact_ids.append(source_id)
                else:
                    unknown_turn_id = True
                    break
            if unknown_turn_id:
                rejected["unknown_or_cross_audience_turn_id"] += 1
                continue

            fact_ids = list(dict.fromkeys(normalized_fact_ids))
            turn_ids = list(dict.fromkeys(normalized_turn_ids))
            # Citation order has no semantics. Canonicalize it before both
            # bounding and identity so even over-limit reordered copies retain
            # the same exact source subset and cannot evade deduplication.
            fact_ids = sorted(fact_ids)
            turn_ids = sorted(turn_ids)
            if (
                len(fact_ids) + len(turn_ids)
                > _ACTOR_CARD_CITATION_LIMIT
            ):
                remaining = _ACTOR_CARD_CITATION_LIMIT
                fact_ids = fact_ids[:remaining]
                remaining -= len(fact_ids)
                turn_ids = turn_ids[:remaining]
                rejected["citations_trimmed"] += 1
            if not fact_ids and not turn_ids:
                rejected["missing_citations"] += 1
                continue

            scope = (
                CARD_SCOPE_CROSS_CONTEXT
                if kind in CARD_CROSS_CONTEXT_KINDS
                else CARD_SCOPE_SAME_CONVERSATION
            )
            digest = hashlib.sha256(json.dumps(
                [actor_id, kind, body, fact_ids, turn_ids],
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:32]
            entry = ActorCardEntry(
                id=f"card-{digest}", tenant_id=tenant_id, actor_id=actor_id,
                kind=kind, body=body, confidence=confidence,
                # Retained only for schema compatibility. Sensitivity is not
                # part of curation, admission, scoping, or serving policy.
                sensitivity=CARD_SENSITIVITY_NORMAL, audience_scope=scope,
                created_at=now, updated_at=now,
            )
            semantic_key = (
                audience_id,
                kind,
                body,
                tuple(fact_ids),
                tuple(turn_ids),
            )
            existing_entry = normalized_entries_by_key.get(semantic_key)
            if existing_entry is not None:
                # Duplicate model candidates are one immutable claim. Keep the
                # strongest confidence regardless of curator output order.
                existing_entry.confidence = max(
                    existing_entry.confidence,
                    entry.confidence,
                )
                rejected["duplicate_entry"] += 1
                continue
            if per_audience_kind.get(quota_key, 0) >= int(
                self._config.assembler.actor_card_entries_per_kind
            ):
                rejected["per_kind_limit"] += 1
                continue
            entry_sources = [
                ActorCardEntrySource(
                    entry_id=entry.id,
                    tenant_id=tenant_id,
                    owner_conversation_id=(
                        fact_source_by_audience_id[
                            (audience_id, fid)
                        ].owner_conversation_id
                    ),
                    audience_conversation_id=(
                        fact_source_by_audience_id[
                            (audience_id, fid)
                        ].audience_conversation_id
                    ),
                    audience_channel_id=(
                        fact_source_by_audience_id[
                            (audience_id, fid)
                        ].audience_channel_id
                    ),
                    fact_id=fid,
                )
                for fid in fact_ids
            ]
            entry_sources.extend(
                ActorCardEntrySource(
                    entry_id=entry.id,
                    tenant_id=tenant_id,
                    owner_conversation_id=(
                        turn_source_by_audience_id[
                            (audience_id, turn_id)
                        ].owner_conversation_id
                    ),
                    audience_conversation_id=(
                        turn_source_by_audience_id[
                            (audience_id, turn_id)
                        ].audience_conversation_id
                    ),
                    audience_channel_id=(
                        turn_source_by_audience_id[
                            (audience_id, turn_id)
                        ].audience_channel_id
                    ),
                    canonical_turn_id=turn_id,
                )
                for turn_id in turn_ids
            )
            normalized_entries_by_key[semantic_key] = entry
            normalized.append((entry, entry_sources))
            normalized_by_audience.setdefault(audience_id, []).append(
                (entry, entry_sources)
            )
            per_audience_kind[quota_key] = (
                per_audience_kind.get(quota_key, 0) + 1
            )

        # A fresh curator is allowed to propose better identity/style entries,
        # but omission is not a deletion decision.  Re-submit every currently
        # active cross-context entry to semantic admission with its immutable
        # body and exact sources.  Same-conversation goals/history deliberately
        # retain replacement semantics: they are the rotating working set.
        existing_entry_ids_by_audience: dict[str, set[str]] = {}
        fresh_entry_ids = {
            entry.id for entry, _sources in normalized
        }
        for entry, entry_sources in carryover_entries:
            if (
                entry.kind not in CARD_CROSS_CONTEXT_KINDS
                or entry.audience_scope != CARD_SCOPE_CROSS_CONTEXT
                or not entry_sources
            ):
                logger.error(
                    "ACTOR_CARD_CARRYOVER_INVALID actor=%s entry=%s "
                    "kind=%s scope=%s sources=%d",
                    actor_id[:24],
                    entry.id,
                    entry.kind,
                    entry.audience_scope,
                    len(entry_sources),
                )
                raise RuntimeError(
                    "actor card carryover violated the cross-context boundary"
                )
            audiences = {
                (source.audience_conversation_id or "").strip()
                for source in entry_sources
            }
            if "" in audiences or len(audiences) != 1:
                # Never put evidence from two privacy audiences in one model
                # prompt.  Failing the refresh leaves the last-good card
                # served; silently dropping it would recreate the bug this
                # path exists to prevent.
                logger.error(
                    "ACTOR_CARD_CARRYOVER_AUDIENCE_INVALID actor=%s "
                    "entry=%s audience_count=%d",
                    actor_id[:24],
                    entry.id,
                    len(audiences),
                )
                raise RuntimeError(
                    "actor card carryover has ambiguous source audience"
                )
            audience_id = next(iter(audiences))
            existing_entry_ids_by_audience.setdefault(
                audience_id,
                set(),
            ).add(entry.id)
            if entry.id in fresh_entry_ids:
                continue
            normalized.append((entry, entry_sources))
            normalized_by_audience.setdefault(audience_id, []).append(
                (entry, entry_sources)
            )

        basic_accepted_count = len(normalized)
        independently_substantive = False
        coverage_gap = False
        if parsed_entries and (
            fact_sources or turn_sources or carryover_entries
        ):
            try:
                admitted_entries: list[
                    tuple[ActorCardEntry, list[ActorCardEntrySource]]
                ] = []
                admission_responses: list[str] = []
                for audience_id in audience_ids:
                    (
                        partition_admitted,
                        partition_response,
                        admission_rejections,
                        partition_substantive,
                    ) = self._admit_actor_card_entries(
                        actor_id,
                        audience_id,
                        fact_sources_by_audience.get(audience_id, []),
                        turn_sources_by_audience.get(audience_id, []),
                        normalized_by_audience.get(audience_id, []),
                        curator_substantive=(
                            curator_substantive_by_audience[audience_id]
                        ),
                        existing_entry_ids=(
                            existing_entry_ids_by_audience.get(
                                audience_id,
                                set(),
                            )
                        ),
                    )
                    # Carryovers do not bypass the configured per-kind cap.
                    # When admission leaves more than the cap, retain an
                    # already-admitted stable entry before a fresh equivalent;
                    # the prompt asks the model to reject redundant candidates,
                    # so this is only the deterministic last line of defense.
                    cap = int(
                        self._config.assembler.actor_card_entries_per_kind
                    )
                    limited: list[
                        tuple[ActorCardEntry, list[ActorCardEntrySource]]
                    ] = []
                    by_kind: dict[
                        str,
                        list[
                            tuple[
                                ActorCardEntry,
                                list[ActorCardEntrySource],
                            ]
                        ],
                    ] = {}
                    for item in partition_admitted:
                        by_kind.setdefault(item[0].kind, []).append(item)
                    existing_ids = existing_entry_ids_by_audience.get(
                        audience_id,
                        set(),
                    )
                    for kind in sorted(by_kind):
                        ranked = sorted(
                            by_kind[kind],
                            key=lambda item: (
                                0 if item[0].id in existing_ids else 1,
                                -float(item[0].confidence or 0.0),
                                item[0].updated_at or "",
                                item[0].id,
                            ),
                        )
                        limited.extend(ranked[:max(0, cap)])
                        if len(ranked) > max(0, cap):
                            rejected["post_admission_per_kind_limit"] += (
                                len(ranked) - max(0, cap)
                            )
                    partition_admitted = limited
                    admitted_entries.extend(partition_admitted)
                    admission_responses.append(partition_response)
                    rejected.update(admission_rejections)
                    independently_substantive = (
                        independently_substantive
                        or partition_substantive
                    )
                    privacy_only_rejection = bool(
                        normalized_by_audience.get(audience_id)
                    ) and (
                        admission_rejections.get(
                            "semantic_explicit_privacy_request",
                            0,
                        )
                        == len(normalized_by_audience[audience_id])
                        == sum(admission_rejections.values())
                    )
                    coverage_gap = (
                        coverage_gap
                        or (
                            partition_substantive
                            and not partition_admitted
                            and not privacy_only_rejection
                        )
                    )
                normalized = admitted_entries
                admission_response_text = json.dumps(
                    admission_responses,
                    separators=(",", ":"),
                )
            except Exception as exc:
                admission_exception = exc
                admission_response_text = getattr(
                    exc, "response_text", "",
                )

        response_hash = (
            hashlib.sha256(json.dumps(
                {
                    "curation": response_text,
                    "admission": admission_response_text,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if response_text or admission_response_text
            else ""
        )

        def _record_status(outcome: str, *, written_count: int = 0) -> None:
            recorder = getattr(
                self._store, "record_actor_card_rebuild_status", None,
            )
            if callable(recorder):
                try:
                    recorder(
                        tenant_id,
                        actor_id,
                        attempted_at=now,
                        input_hash=input_hash,
                        source_count=len(fact_sources) + len(turn_sources),
                        raw_entry_count=len(raw_entries),
                        accepted_entry_count=len(normalized),
                        rejected_counts=dict(sorted(rejected.items())),
                        outcome=outcome,
                        response_hash=response_hash,
                        written_count=written_count,
                    )
                except Exception:
                    logger.warning(
                        "actor card rebuild status write failed actor=%s",
                        actor_id[:24],
                        exc_info=True,
                    )

        if not parsed_entries:
            outcome = "model_error" if model_exception is not None else "invalid_response"
            _record_status(outcome)
            logger.warning(
                "ACTOR_CARD_REBUILD actor=%s sources=%d outcome=%s "
                "response_hash=%s error_type=%s",
                actor_id[:24],
                len(fact_sources) + len(turn_sources),
                outcome,
                response_hash[:16],
                type(model_exception).__name__ if model_exception is not None else "",
            )
            raise RuntimeError(
                "actor card curation failed"
                if model_exception is not None
                else "actor card curation response has no valid entries array"
            ) from model_exception

        if raw_entries and not basic_accepted_count:
            _record_status("rejected_all")
            logger.warning(
                "ACTOR_CARD_REBUILD actor=%s sources=%d raw=%d accepted=0 "
                "outcome=rejected_all rejected=%s response_hash=%s",
                actor_id[:24],
                len(fact_sources) + len(turn_sources),
                len(raw_entries),
                json.dumps(dict(sorted(rejected.items())), separators=(",", ":")),
                response_hash[:16],
            )
            raise RuntimeError("actor card curation rejected every model entry")

        if admission_exception is not None:
            admission_outcome = (
                "coverage_disagreement"
                if isinstance(
                    admission_exception,
                    _ActorCardCoverageError,
                )
                else "admission_error"
            )
            _record_status(admission_outcome)
            logger.warning(
                "ACTOR_CARD_REBUILD actor=%s sources=%d raw=%d "
                "basic_accepted=%d outcome=%s error_type=%s "
                "response_hash=%s",
                actor_id[:24],
                len(fact_sources) + len(turn_sources),
                len(raw_entries),
                basic_accepted_count,
                admission_outcome,
                type(admission_exception).__name__,
                response_hash[:16],
            )
            raise RuntimeError("actor card semantic admission failed") from (
                admission_exception
            )
        if coverage_gap:
            _record_status("coverage_gap")
            raise RuntimeError(
                "actor card coverage gate found a substantive actor "
                "without an admitted entry"
            )

        expected_epochs: dict[str, int] = {}
        for source in [*fact_sources, *turn_sources]:
            expected_epochs[source.owner_conversation_id] = (
                source.owner_lifecycle_epoch
            )
            expected_epochs[source.audience_conversation_id] = (
                source.audience_lifecycle_epoch
            )
        epoch_getter = getattr(self._store, "get_lifecycle_epoch", None)
        if carryover_entries and not callable(epoch_getter):
            raise RuntimeError(
                "actor card carryover cannot prove lifecycle epochs"
            )
        for _entry, entry_sources in carryover_entries:
            for source in entry_sources:
                for conversation_id in (
                    source.owner_conversation_id,
                    source.audience_conversation_id,
                ):
                    if conversation_id not in expected_epochs:
                        expected_epochs[conversation_id] = int(
                            epoch_getter(conversation_id)
                        )
        written = self._store.replace_actor_card(
            tenant_id,
            actor_id,
            normalized,
            input_hash=input_hash,
            expected_source_epochs=expected_epochs,
            expected_build_marker=build_marker,
        )
        refreshed = self._store.get_actor_profile(tenant_id, actor_id)
        if refreshed is None or refreshed.card_dirty or (
            refreshed.card_input_hash != input_hash
        ):
            _record_status("stale_or_rejected_write", written_count=written)
            raise RuntimeError("actor card replacement did not commit cleanly")
        outcome = (
            (
                "clean_empty_filtered"
                if basic_accepted_count and not normalized
                else "clean_empty"
            )
            if not normalized
            else ("partial" if rejected else "written")
        )
        _record_status(outcome, written_count=written)
        logger.info(
            "ACTOR_CARD_REBUILD actor=%s sources=%d raw=%d accepted=%d "
            "written=%d outcome=%s rejected=%s response_hash=%s",
            actor_id[:24],
            len(fact_sources) + len(turn_sources),
            len(raw_entries),
            len(normalized),
            written,
            outcome,
            json.dumps(dict(sorted(rejected.items())), separators=(",", ":")),
            response_hash[:16],
        )
        return written

    @staticmethod
    def _actor_card_prompt_turns(
        turn_sources: list,
        *,
        max_chars: int = 96_000,
    ) -> list[dict]:
        """Render a bounded, deterministic set of actor-authored messages.

        Discord messages are normally small, but canonical ingestion is also
        used by API callers. Individual and aggregate bounds prevent one actor
        from turning card curation into an unbounded model call. A truncated
        message remains visibly marked so neither model can treat it as exact
        evidence for a dropped qualifier.
        """
        rendered: list[dict] = []
        used = 0
        for source in turn_sources:
            content = (source.turn.user_content or "").strip()
            if not content:
                continue
            truncated = len(content) > 4_000
            if truncated:
                content = (
                    content[:1_940]
                    + "\n...[middle omitted; do not infer omitted text]...\n"
                    + content[-1_940:]
                )
            item = {
                "id": source.turn.canonical_turn_id,
                "timestamp": (
                    source.turn.created_at
                    or source.turn.first_seen_at
                    or ""
                ),
                "audience_conversation_id": (
                    source.audience_conversation_id
                ),
                "audience_channel_id": source.audience_channel_id,
                "content": content,
                "truncated": truncated,
            }
            cost = len(json.dumps(item, separators=(",", ":")))
            if used + cost > max(0, int(max_chars)):
                break
            rendered.append(item)
            used += cost
        return rendered

    def _curate_actor_card_partition(
        self,
        fact_sources: list,
        turn_sources: list,
    ) -> tuple[str, bool, str, list, set[str]]:
        """Curate one audience partition without exposing another audience."""
        from ..types import CARD_ENTRY_BODY_MAX_CHARS

        prompt_facts = [
            {
                "id": source.fact.id,
                "fact": source.fact.format_for_prompt(),
                "author_role": source.fact.author_source_role,
                "status": source.fact.status,
                "fact_type": source.fact.fact_type,
                "mentioned_at": source.fact.mentioned_at.isoformat(),
                "session_date": source.fact.session_date,
            }
            for source in fact_sources
        ]
        prompt_turns = self._actor_card_prompt_turns(
            turn_sources,
            max_chars=int(getattr(
                self._config.assembler,
                "actor_card_prompt_max_chars",
                192_000,
            )),
        )
        prompt_turn_ids = {item["id"] for item in prompt_turns}
        system = (
            "Curate a compact person card from exact messages and facts "
            "authored by one actor in one policy audience. The card is for "
            "durable interaction continuity, not a fact scrapbook or a "
            "transcript. Independently decide whether the actor contributed "
            "substantive interaction. Substantive means at least one "
            "informative message that reveals a useful ongoing goal, durable "
            "preference/style, or a meaningful topic the actor has discussed "
            "with the agent. Greetings, bot invocation checks, memory/"
            "preference probes, and isolated trivia questions are not "
            "substantive. Return JSON only with exactly: substantive, "
            "coverage_reason, and entries. substantive must be boolean. "
            "coverage_reason must be exactly one of \"substantive\", "
            "\"greeting_only\", \"one_off_trivia\", \"bot_meta_or_test\", "
            "\"no_durable_context\", or \"insufficient_evidence\". It must be "
            "\"substantive\" exactly when substantive is true. entries must "
            "be an array. A substantive actor must receive at least one entry; "
            "a non-substantive actor must receive none. Each entry must contain "
            "exactly kind, body, confidence, fact_ids, and turn_ids. kind must "
            "be exactly one of \"communication_pref\", \"active_goal\", "
            "\"relevant_history\", or \"interaction_style\". confidence must "
            "be a number from 0 through 1. fact_ids "
            "and turn_ids must be arrays, may individually be empty, and "
            "together must cite at least one provided id that fully supports "
            f"the body. Use at most {_ACTOR_CARD_CITATION_LIMIT} citation ids "
            "total per entry. Put "
            "fact ids only in fact_ids and turn ids only in turn_ids; never "
            "copy an id into both arrays. Obey entries_per_kind as a hard "
            "maximum. Use a neutral concise body and do not invent identity "
            "or intent. Every body must be self-contained and unambiguous when "
            "read without the surrounding transcript; include essential "
            "referents such as the specific medication, goal, preference, or "
            "discussion topic. Write natural person-facing language, not a "
            "serialization of subject/verb/object fields, ontology names, or "
            "tag labels. Preserve every material qualifier from the source, "
            "including exceptions, exclusions, uncertainty, frequency, "
            "timing, and scope. Do not turn a qualified statement into a "
            "broader or more certain claim. Do not promote temporary, "
            "test-only, one-turn, session-only, or channel-only instructions "
            "into communication_pref or interaction_style. Do not retain a "
            "preference or goal that later evidence stopped, replaced, "
            "completed, or contradicted. Use message timestamps, mentioned_at, "
            "and status to resolve conflicts, with the newest applicable "
            "evidence winning. A communication preference or interaction "
            "style is durable only when explicitly stated as lasting or "
            "consistently supported by repeated natural interactions. Use "
            "relevant_history for concise, useful continuity about a "
            "meaningful topic this actor actually discussed with the agent "
            "when no narrower durable preference or goal is justified. "
            "An isolated, underspecified follow-up whose missing referent "
            "cannot be recovered from the supplied evidence is insufficient "
            "for relevant_history, even if it sounds important. Subject matter "
            "must never determine admission: medical, sexual, financial, "
            "location, credential, and other topics are evaluated by the same "
            "durability and evidence rules as every other topic. Do not omit "
            "or soften a candidate because of its subject. If the actor "
            "explicitly and unambiguously asks that particular information "
            "not be retained or reused, do not propose it for the card. Do not "
            "infer such a request from the topic, from a DM, or from context."
        )
        user = json.dumps({
            "facts": prompt_facts,
            "turns": prompt_turns,
            "limits": {
                "entries_per_kind": int(
                    self._config.assembler.actor_card_entries_per_kind
                ),
                "body_chars": CARD_ENTRY_BODY_MAX_CHARS,
            },
        }, separators=(",", ":"))
        valid_coverage_reasons = {
            "substantive",
            "greeting_only",
            "one_off_trivia",
            "bot_meta_or_test",
            "no_durable_context",
            "insufficient_evidence",
        }

        def _parse_curation(text: str) -> dict:
            try:
                parsed = self._compactor._parse_response(text)
            except Exception as exc:
                raise _ActorCardAdmissionError(
                    "actor card curation response is not valid JSON",
                    text,
                ) from exc
            if (
                not isinstance(parsed, dict)
                or set(parsed)
                != {"substantive", "coverage_reason", "entries"}
                or not isinstance(parsed.get("substantive"), bool)
                or not isinstance(parsed.get("coverage_reason"), str)
                or not isinstance(parsed.get("entries"), list)
                or parsed["coverage_reason"] not in valid_coverage_reasons
                or parsed["substantive"]
                != (parsed["coverage_reason"] == "substantive")
                or parsed["substantive"] != bool(parsed["entries"])
            ):
                raise _ActorCardAdmissionError(
                    "actor card curation response has invalid coverage shape",
                    text,
                )
            return parsed

        request_kwargs = {
            "system": system,
            "user": user,
            "max_tokens": max(
                2000,
                min(
                    4000,
                    2 * int(self._config.compactor.max_summary_tokens),
                ),
            ),
        }
        provider = self._actor_card_curation_provider()
        complete_with_source = getattr(provider, "complete_with_source", None)
        if callable(complete_with_source):
            response_text, _usage, curation_source = complete_with_source(
                **request_kwargs,
            )
        else:
            response_text, _usage = provider.complete(**request_kwargs)
            curation_source = "provider"

        try:
            parsed = _parse_curation(response_text)
        except _ActorCardAdmissionError as primary_exc:
            complete_fallback = getattr(provider, "complete_fallback", None)
            if (
                curation_source == "fallback"
                or not callable(complete_fallback)
            ):
                raise
            logger.warning(
                "ACTOR_CARD_CURATION_FALLBACK reason=invalid_response "
                "response_hash=%s",
                hashlib.sha256(
                    response_text.encode("utf-8")
                ).hexdigest()[:16],
            )
            fallback_text = ""
            try:
                fallback_text, _usage = complete_fallback(**request_kwargs)
                parsed = _parse_curation(fallback_text)
            except Exception as fallback_exc:
                combined = json.dumps(
                    {
                        "primary": primary_exc.response_text,
                        "fallback": (
                            getattr(fallback_exc, "response_text", "")
                            or fallback_text
                        ),
                    },
                    separators=(",", ":"),
                )
                raise _ActorCardAdmissionError(
                    "actor card curation primary and fallback responses "
                    "were invalid",
                    combined,
                ) from fallback_exc
            response_text = fallback_text
        return (
            response_text,
            parsed["substantive"],
            parsed["coverage_reason"],
            parsed["entries"],
            prompt_turn_ids,
        )

    def _actor_card_provider_for_model(self, selected_model: str):
        """Create a zero-temperature provider through the configured gateway."""
        base = self._compactor.llm
        from ..providers.anthropic import AnthropicProvider
        from ..providers.generic_openai import GenericOpenAIProvider

        if isinstance(base, GenericOpenAIProvider):
            return GenericOpenAIProvider(
                base_url=base.base_url,
                model=selected_model,
                temperature=0.0,
                api_key=base.api_key,
                reasoning_effort="low",
            )
        if isinstance(base, AnthropicProvider):
            return AnthropicProvider(
                api_key=base.api_key,
                model=selected_model,
                temperature=0.0,
            )
        raise RuntimeError(
            "actor-card model override is unsupported by "
            f"{type(base).__name__}"
        )

    def _actor_card_curation_provider(self):
        """Build the optional dedicated curator and malformed-response fallback."""
        override = getattr(
            self, "_actor_card_curation_provider_override", None,
        )
        if override is not None:
            return override
        model = (
            getattr(
                self._config.assembler,
                "actor_card_curation_model",
                "",
            )
            or ""
        ).strip()
        if not model:
            return self._compactor.llm
        fallback_model = (
            getattr(
                self._config.assembler,
                "actor_card_curation_fallback_model",
                "",
            )
            or ""
        ).strip()
        primary = self._actor_card_provider_for_model(model)
        if fallback_model and fallback_model != model:
            return _EmptyResponseFallbackProvider(
                primary,
                self._actor_card_provider_for_model(fallback_model),
                primary_model=model,
                fallback_model=fallback_model,
                stage="curation",
            )
        return primary

    def _actor_card_admission_provider(self):
        """Build the dedicated semantic admission provider.

        The curation model may be deliberately cheap. Admission is a separate
        safety boundary over immutable candidates and actor-authored evidence;
        it cannot invent or rewrite card bodies.
        """
        override = getattr(
            self, "_actor_card_admission_provider_override", None,
        )
        if override is not None:
            return override
        model = (
            getattr(
                self._config.assembler,
                "actor_card_admission_model",
                "",
            )
            or ""
        ).strip()
        if not model:
            return None
        fallback_model = (
            getattr(
                self._config.assembler,
                "actor_card_admission_fallback_model",
                "",
            )
            or ""
        ).strip()
        primary = self._actor_card_provider_for_model(model)
        if fallback_model and fallback_model != model:
            return _EmptyResponseFallbackProvider(
                primary,
                self._actor_card_provider_for_model(fallback_model),
                primary_model=model,
                fallback_model=fallback_model,
            )
        return primary

    def _actor_card_evidence_segments(
        self,
        actor_id: str,
        audience_conversation_id: str,
        sources: list,
        candidate_fact_ids: set[str],
        *,
        max_chars: int = 64_000,
    ) -> tuple[list[dict], set[tuple[str, str]]]:
        """Return bounded actor-authored turns from candidate-cited segments.

        Selection is provenance-based: canonical actor ids and segment source
        mappings decide which messages are evidence. Message text is never
        regex-classified. Uncited segments remain available to the admission
        model as compact facts, not as unrelated raw conversation text.
        """
        from ..types import AUDIENCE_ATTRIBUTION_VERSION

        source_by_id = {
            source.fact.id: source
            for source in sources
            if source.audience_conversation_id == audience_conversation_id
        }
        candidate_refs = {
            (
                source_by_id[fact_id].owner_conversation_id,
                source_by_id[fact_id].fact.segment_ref,
            )
            for fact_id in candidate_fact_ids
            if fact_id in source_by_id
            and source_by_id[fact_id].fact.segment_ref
        }
        owner_rows: dict[str, dict[str, object]] = {}
        for owner in sorted({
            source.owner_conversation_id for source in sources
        }):
            owner_rows[owner] = {
                row.canonical_turn_id: row
                for row in self._store.get_all_canonical_turns(owner)
            }

        by_ref: dict[tuple[str, str], dict] = {}
        for source in sources:
            ref = source.fact.segment_ref
            ref_key = (source.owner_conversation_id, ref)
            if (
                not ref
                or ref_key not in candidate_refs
                or ref_key in by_ref
            ):
                continue
            segment = self._store.get_segment(
                ref,
                conversation_id=source.owner_conversation_id,
            )
            if segment is None:
                continue
            messages: list[dict] = []
            try:
                newest_time = float(segment.end_timestamp.timestamp())
            except (AttributeError, TypeError, ValueError, OSError):
                newest_time = float("-inf")
            for canonical_id in list(
                segment.metadata.canonical_turn_ids or []
            ):
                row = owner_rows.get(
                    source.owner_conversation_id, {},
                ).get(canonical_id)
                content = (row.user_content or "").strip() if row else ""
                if (
                    row is None
                    or row.sender_actor_id != actor_id
                    or row.audience_conversation_id
                    != audience_conversation_id
                    or int(row.audience_attribution_version or 0)
                    != AUDIENCE_ATTRIBUTION_VERSION
                    or not content
                ):
                    continue
                if len(content) > 1200:
                    content = (
                        content[:580]
                        + "\n...[middle truncated]...\n"
                        + content[-580:]
                    )
                messages.append({
                    "turn": row.turn_number,
                    "timestamp": (
                        row.created_at or row.first_seen_at or ""
                    ),
                    "content": content,
                })
            if messages:
                by_ref[ref_key] = {
                    "owner_conversation_id": source.owner_conversation_id,
                    "segment_ref": ref,
                    "messages": messages,
                    "_newest_time": newest_time,
                }

        ordered = sorted(
            by_ref.values(),
            key=lambda item: (
                -item["_newest_time"],
                item["owner_conversation_id"],
                item["segment_ref"],
            ),
        )
        admitted: list[dict] = []
        admitted_refs: set[tuple[str, str]] = set()
        used = 0
        for item in ordered:
            public = {
                "owner_conversation_id": item["owner_conversation_id"],
                "segment_ref": item["segment_ref"],
                "messages": item["messages"],
            }
            cost = len(json.dumps(public, separators=(",", ":")))
            if used + cost > max_chars:
                continue
            admitted.append(public)
            admitted_refs.add((
                item["owner_conversation_id"],
                item["segment_ref"],
            ))
            used += cost
        return admitted, admitted_refs

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
        """Independently check coverage and admit immutable candidates."""

        provider = self._actor_card_admission_provider()
        if provider is None:
            raise RuntimeError(
                "actor-card semantic admission model is not configured"
            )
        existing_entry_ids = set(existing_entry_ids or ())

        candidate_fact_ids = {
            source.fact_id
            for _entry, entry_sources in normalized
            for source in entry_sources
            if source.fact_id
        }
        evidence_segments, evidence_refs = (
            self._actor_card_evidence_segments(
                actor_id,
                audience_conversation_id,
                fact_sources,
                candidate_fact_ids,
            )
        )
        fact_source_by_id = {
            source.fact.id: source for source in fact_sources
        }
        turn_source_by_id = {
            source.turn.canonical_turn_id: source
            for source in turn_sources
        }
        actor_turns = self._actor_card_prompt_turns(
            turn_sources,
            max_chars=int(getattr(
                self._config.assembler,
                "actor_card_prompt_max_chars",
                192_000,
            )),
        )
        visible_turn_ids = {turn["id"] for turn in actor_turns}
        candidates: list[dict] = []
        eligible: dict[
            str, tuple["ActorCardEntry", list["ActorCardEntrySource"]]
        ] = {}
        rejection_counts: Counter[str] = Counter()
        for entry, entry_sources in normalized:
            fact_ids = [
                source.fact_id for source in entry_sources
                if source.fact_id
            ]
            turn_ids = [
                source.canonical_turn_id for source in entry_sources
                if source.canonical_turn_id
            ]
            refs = {
                (
                    fact_source_by_id[fact_id].owner_conversation_id,
                    fact_source_by_id[fact_id].fact.segment_ref,
                )
                for fact_id in fact_ids
                if fact_id in fact_source_by_id
            }
            is_existing = entry.id in existing_entry_ids
            if (
                not is_existing
                and refs
                and not refs.issubset(evidence_refs)
            ):
                rejection_counts["evidence_unavailable"] += 1
                continue
            if any(
                turn_id not in turn_source_by_id
                or turn_id not in visible_turn_ids
                for turn_id in turn_ids
            ) and not is_existing:
                rejection_counts["evidence_unavailable"] += 1
                continue
            if not fact_ids and not turn_ids:
                rejection_counts["evidence_unavailable"] += 1
                continue
            eligible[entry.id] = (entry, entry_sources)
            candidates.append({
                "candidate_id": entry.id,
                "origin": "existing" if is_existing else "fresh",
                "kind": entry.kind,
                "body": entry.body,
                "proposed_confidence": entry.confidence,
                "fact_ids": fact_ids,
                "turn_ids": turn_ids,
                "source_segments": [
                    {
                        "owner_conversation_id": owner,
                        "segment_ref": ref,
                    }
                    for owner, ref in sorted(refs)
                ],
            })

        compact_facts = [{
            "id": source.fact.id,
            "owner_conversation_id": source.owner_conversation_id,
            "segment_ref": source.fact.segment_ref,
            "fact": source.fact.format_for_prompt(),
            "status": source.fact.status,
            "mentioned_at": source.fact.mentioned_at.isoformat(),
        } for source in fact_sources]
        system = (
            "You are the conservative semantic admission gate for a person "
            "card. Candidate bodies are immutable: you may admit or reject "
            "them, but you may not invent, rewrite, or merge candidates. Use "
            "only actor-authored facts and source "
            "messages. All bounded actor turns and compact facts are supplied "
            "so later evidence can revoke or replace a candidate. "
            "Independently decide whether this actor contributed substantive "
            "interaction; do not defer to the curator's claim. Substantive "
            "means at least one informative message that reveals a useful "
            "ongoing goal, durable preference/style, or a meaningful topic "
            "the actor discussed with the agent. Greetings, bot invocation "
            "checks, memory/preference probes, and isolated trivia questions "
            "are not substantive. A substantive actor must finish with at "
            "least one admitted card entry; relevant_history is appropriate "
            "for useful topic continuity when no narrower entry is justified. "
            "Return JSON only with exactly substantive, coverage_reason, and "
            "decisions. substantive must be boolean. coverage_reason must be "
            "exactly one of \"substantive\", \"greeting_only\", "
            "\"one_off_trivia\", \"bot_meta_or_test\", "
            "\"no_durable_context\", or \"insufficient_evidence\", and must "
            "be \"substantive\" exactly when substantive is true. "
            "Return exactly one decision for every candidate, with "
            "exactly candidate_id, admit, and reason. admit must be a boolean. "
            "reason must be exactly one of "
            "\"durable\", \"temporary\", \"test_probe\", "
            "\"stopped_or_replaced\", \"completed\", \"contradicted\", "
            "\"insufficient_evidence\", \"not_durable\", "
            "\"not_person_card\", \"redundant\", or "
            "\"explicit_privacy_request\". Use reason "
            "\"durable\" if and only if admit is true. "
            "Candidate origin is either fresh or existing. An existing "
            "candidate is an immutable entry that a prior independent "
            "admission accepted from its cited evidence. Curator omission is "
            "not evidence against it. Re-admit an existing candidate unless "
            "later actor-authored evidence explicitly stops, replaces, "
            "completes, or contradicts it, or a materially better fresh "
            "candidate makes it redundant. When fresh and existing candidates "
            "substantially overlap, prefer the existing candidate for "
            "continuity unless the fresh one materially corrects, updates, or "
            "better preserves the evidence; reject the other as redundant or "
            "stopped_or_replaced. Do not admit redundant copies. "
            "Reject temporary, test/probe, one-turn, session-only, "
            "channel-only, stopped, replaced, completed, or contradicted "
            "material. Later source messages revoke or replace earlier "
            "material. Requested answer prefixes and memory-system tests are "
            "not durable identity preferences. A communication preference or "
            "interaction style is admissible only when a source message "
            "explicitly establishes durability beyond the current test, "
            "session, and channel, or when consistent natural evidence appears "
            "across distinct actor-authored messages or interactions, ideally "
            "spread over time. Repeated test instructions do not establish a "
            "pattern. The immutable candidate body itself "
            "must be self-contained and unambiguous without relying on the "
            "surrounding segment; reject with insufficient_evidence when an "
            "essential referent (such as which medication, goal, or "
            "preference) is omitted. The body must be fully entailed by the "
            "cited actor-authored messages. Compact fact fields and tags help "
            "locate evidence but cannot independently justify body text. "
            "Full entailment requires preserving every material qualifier in "
            "the source, including exceptions, exclusions, uncertainty, "
            "frequency, timing, and scope. Reject with insufficient_evidence "
            "if the immutable body drops a qualifier, broadens the statement, "
            "or increases its certainty. For example, source text like "
            "\"good blood pressure, excluding stressful events\" does not "
            "entail the unqualified body \"has good blood pressure.\" "
            "Reject with not_person_card when a body exposes internal "
            "ontology/tag language or serializes a machine fact triple rather "
            "than stating a natural person fact. Subject matter must never "
            "determine admission: medical, sexual, financial, location, "
            "credential, and other topics are evaluated by the same durability "
            "and evidence rules as every other topic. Do not omit, soften, or "
            "reject a candidate because of its subject. Use "
            "explicit_privacy_request only when actor-authored evidence "
            "explicitly and unambiguously asks that the cited information not "
            "be retained or reused; never infer privacy from the topic, from a "
            "DM, or from context. A visibly truncated turn cannot prove a "
            "claim whose qualifier may be in omitted text. When uncertain, "
            "reject."
        )
        user = json.dumps({
            "curator_substantive_claim": curator_substantive,
            "candidates": candidates,
            "facts": compact_facts,
            "actor_turns": actor_turns,
            "evidence_segments": evidence_segments,
        }, separators=(",", ":"))
        request_kwargs = {
            "system": system,
            "user": user,
            "max_tokens": max(
                800,
                min(
                    4000,
                    300 + 250 * len(candidates),
                ),
            ),
        }
        complete_with_source = getattr(provider, "complete_with_source", None)
        if callable(complete_with_source):
            response_text, _usage, admission_source = complete_with_source(
                **request_kwargs,
            )
        else:
            response_text, _usage = provider.complete(**request_kwargs)
            admission_source = "provider"

        def _parse_admission(
            text: str,
        ) -> tuple[bool, dict[str, dict]]:
            try:
                parsed = self._compactor._parse_response(text)
            except Exception as exc:
                raise _ActorCardAdmissionError(
                    "actor-card admission response is not valid JSON",
                    text,
                ) from exc
            if (
                not isinstance(parsed, dict)
                or set(parsed)
                != {"substantive", "coverage_reason", "decisions"}
                or not isinstance(parsed["substantive"], bool)
                or not isinstance(parsed["coverage_reason"], str)
                or not isinstance(parsed["decisions"], list)
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission response has invalid coverage shape",
                    text,
                )
            independently_substantive = parsed["substantive"]
            valid_coverage_reasons = {
                "substantive",
                "greeting_only",
                "one_off_trivia",
                "bot_meta_or_test",
                "no_durable_context",
                "insufficient_evidence",
            }
            if (
                parsed["coverage_reason"] not in valid_coverage_reasons
                or independently_substantive
                != (parsed["coverage_reason"] == "substantive")
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission response has invalid coverage decision",
                    text,
                )
            decisions: dict[str, dict] = {}
            valid_reasons = {
                "durable",
                "temporary",
                "test_probe",
                "stopped_or_replaced",
                "completed",
                "contradicted",
                "insufficient_evidence",
                "not_durable",
                "not_person_card",
                "redundant",
                "explicit_privacy_request",
            }
            for decision in parsed["decisions"]:
                if (
                    not isinstance(decision, dict)
                    or set(decision) != {"candidate_id", "admit", "reason"}
                    or not isinstance(decision.get("candidate_id"), str)
                    or not isinstance(decision.get("admit"), bool)
                    or decision.get("reason") not in valid_reasons
                    or (
                        bool(decision.get("admit"))
                        != (decision.get("reason") == "durable")
                    )
                    or decision["candidate_id"] in decisions
                ):
                    raise _ActorCardAdmissionError(
                        "actor-card admission response contains an invalid decision",
                        text,
                    )
                decisions[decision["candidate_id"]] = decision
            if set(decisions) != set(eligible):
                raise _ActorCardAdmissionError(
                    "actor-card admission response does not cover every candidate",
                    text,
                )
            return independently_substantive, decisions

        try:
            independently_substantive, decisions = _parse_admission(
                response_text,
            )
        except _ActorCardAdmissionError as primary_exc:
            complete_fallback = getattr(provider, "complete_fallback", None)
            if (
                admission_source == "fallback"
                or not callable(complete_fallback)
            ):
                raise
            logger.warning(
                "ACTOR_CARD_ADMISSION_FALLBACK reason=invalid_response "
                "response_hash=%s",
                hashlib.sha256(
                    response_text.encode("utf-8")
                ).hexdigest()[:16],
            )
            fallback_text = ""
            try:
                fallback_text, _usage = complete_fallback(**request_kwargs)
                independently_substantive, decisions = _parse_admission(
                    fallback_text,
                )
            except Exception as fallback_exc:
                combined = json.dumps(
                    {
                        "primary": primary_exc.response_text,
                        "fallback": (
                            getattr(fallback_exc, "response_text", "")
                            or fallback_text
                        ),
                    },
                    separators=(",", ":"),
                )
                raise _ActorCardAdmissionError(
                    "actor-card admission primary and fallback responses "
                    "were invalid",
                    combined,
                ) from fallback_exc
            response_text = json.dumps(
                {
                    "primary": response_text,
                    "fallback": fallback_text,
                    "selected": "fallback",
                },
                separators=(",", ":"),
            )
            admission_source = "fallback"
        if independently_substantive != curator_substantive:
            primary_substantive = independently_substantive
            complete_fallback = getattr(provider, "complete_fallback", None)
            if (
                admission_source == "fallback"
                or not callable(complete_fallback)
            ):
                raise _ActorCardCoverageError(
                    "actor-card curator and admission coverage decisions disagree",
                    response_text,
                )
            adjudication_text = ""
            try:
                adjudication_text, _usage = complete_fallback(**request_kwargs)
                adjudicated_substantive, adjudicated_decisions = (
                    _parse_admission(adjudication_text)
                )
            except Exception as exc:
                combined = json.dumps(
                    {
                        "primary": response_text,
                        "adjudicator": (
                            getattr(exc, "response_text", "")
                            or adjudication_text
                        ),
                        "selected": "error",
                        "error_type": type(exc).__name__,
                    },
                    separators=(",", ":"),
                )
                raise _ActorCardAdmissionError(
                    "actor-card coverage adjudicator failed",
                    combined,
                ) from exc
            # With boolean coverage and an initial disagreement, the third
            # judgment necessarily agrees with either the curator or the
            # primary admission model. Select that two-of-three result and
            # its internally consistent candidate decisions.
            selected = "primary"
            if adjudicated_substantive == curator_substantive:
                independently_substantive = adjudicated_substantive
                decisions = adjudicated_decisions
                selected = "curator_fallback"
            logger.warning(
                "ACTOR_CARD_COVERAGE_ADJUDICATED curator=%s primary=%s "
                "fallback=%s selected=%s",
                curator_substantive,
                primary_substantive,
                adjudicated_substantive,
                selected,
            )
            response_text = json.dumps(
                {
                    "primary": response_text,
                    "adjudicator": adjudication_text,
                    "selected": selected,
                },
                separators=(",", ":"),
            )

        admitted: list[
            tuple["ActorCardEntry", list["ActorCardEntrySource"]]
        ] = []
        for candidate_id, (entry, entry_sources) in eligible.items():
            decision = decisions[candidate_id]
            if candidate_id in existing_entry_ids:
                logger.info(
                    "ACTOR_CARD_CARRYOVER_DECISION actor=%s audience=%s "
                    "entry=%s admit=%s reason=%s",
                    actor_id[:24],
                    audience_conversation_id[:48],
                    candidate_id,
                    bool(decision["admit"]),
                    decision["reason"],
                )
            if not decision["admit"]:
                rejection_counts[
                    f"semantic_{decision['reason']}"
                ] += 1
                continue
            admitted.append((entry, entry_sources))
        if not independently_substantive and admitted:
            raise _ActorCardAdmissionError(
                "non-substantive actor cannot have an admitted card entry",
                response_text,
            )
        return (
            admitted,
            response_text,
            rejection_counts,
            independently_substantive,
        )

    def _physical_rows_by_group(self) -> dict[int, list["CanonicalTurnRow"]]:
        """Physical canonical rows of this conversation, grouped by turn group.

        ``get_uncompacted_canonical_turns`` returns LOGICAL rows: both stores
        run ``_merge_canonical_turn_rows`` inside it, so one logical row can be
        backed by separate physical user and assistant rows. Fact authorship
        needs the physical rows, because that is where the actor and the reply
        edge actually live.
        """
        # A store that cannot enumerate physical rows cannot prove provenance.
        # That is a fail-closed state, not an error: the segment mapping stays
        # incomplete and every fact author stays empty, which is exactly the
        # behaviour required of an unprovable mapping.
        getter = getattr(self._store, "get_all_canonical_turns", None)
        if not callable(getter):
            return {}
        grouped: dict[int, list["CanonicalTurnRow"]] = {}
        for row in getter(self._config.conversation_id) or ():
            raw_group = getattr(row, "turn_group_number", -1)
            group = int(raw_group if raw_group is not None else -1)
            grouped.setdefault(group, []).append(row)
        return grouped

    @staticmethod
    def _segment_source_ids(segment) -> tuple[list[str], bool]:
        """First-seen deduplicated source ids of a segment, and completeness.

        Completeness requires every non-empty message to carry at least one
        source id. It is deliberately not derived from ``turn_count``: topic
        grouping is noncontiguous and the session splitter can turn one source
        message into two, so a positional slice is not a row mapping.
        """
        from ..types import SOURCE_CANONICAL_TURN_IDS_KEY

        ordered: list[str] = []
        seen: set[str] = set()
        complete = True
        for message in segment.messages:
            if not (getattr(message, "content", "") or "").strip():
                continue
            ids = (getattr(message, "metadata", None) or {}).get(
                SOURCE_CANONICAL_TURN_IDS_KEY
            ) or []
            if not ids:
                complete = False
                continue
            for cid in ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    ordered.append(cid)
        return ordered, complete

    def _build_actor_roster(self, segment, physical_by_id: dict) -> "ActorRoster":
        """Build one segment's actor roster and fact lanes from physical rows.

        Everything here comes from stored rows, never from model text or a
        positional cursor. A segment whose mapping is incomplete, or that spans
        more than one human, will attribute no fact author at all.
        """
        from ..types import (
            AUTHOR_ROLE_ASSISTANT,
            AUTHOR_ROLE_REQUESTER,
            AUTHOR_ROLE_SUBJECT,
            ActorRoster,
            FactLane,
        )

        ids, complete = self._segment_source_ids(segment)
        roster = ActorRoster(complete=complete)
        if not ids:
            roster.complete = False
            return roster

        for cid in ids:
            row = physical_by_id.get(cid)
            if row is None:
                # A source id that no longer resolves to a physical row makes
                # the mapping incomplete; it must not silently narrow a roster.
                roster.complete = False
                continue

            user_text = (row.user_content or "").strip()
            assistant_text = (row.assistant_content or "").strip()
            actor = (getattr(row, "sender_actor_id", "") or "").strip()
            label = (getattr(row, "sender", "") or "").strip()

            if user_text:
                if actor:
                    roster.actor_ids.add(actor)
                    if label:
                        roster.labels.setdefault(label.casefold(), set()).add(actor)
                else:
                    roster.has_unidentified_user_row = True

                # One requester lane per physical user row. It carries that
                # row's own words and NEVER its quote block.
                roster.lanes.append(FactLane(
                    role=AUTHOR_ROLE_REQUESTER,
                    text=user_text,
                    actor_id=actor,
                    source_message_id=(getattr(row, "source_message_id", "") or ""),
                    canonical_turn_id=row.canonical_turn_id,
                    speaker_label=label,
                ))

                quote = (getattr(row, "reply_target_body", "") or "").strip()
                target_id = (getattr(row, "reply_target_message_id", "") or "").strip()
                if int(getattr(row, "reply_attribution_version", 0) or 0) > 0:
                    roster.reply_bearing = True
                if quote:
                    # When the reply target resolves to a row we already hold,
                    # create NO subject lane: that row's own requester lane is
                    # the source of truth and will produce (or already produced)
                    # its facts. The quote is current-request context, not a
                    # second disclosure.
                    audience = (
                        getattr(row, "audience_conversation_id", "") or ""
                    ).strip()
                    channel = (getattr(row, "origin_channel_id", "") or "").strip()
                    target_candidates = [
                        candidate for candidate in physical_by_id.values()
                        if target_id
                        and (getattr(candidate, "source_message_id", "") or "").strip()
                        == target_id
                        and audience
                        and (
                            getattr(candidate, "audience_conversation_id", "") or ""
                        ).strip() == audience
                        and (
                            not channel
                            or (getattr(candidate, "origin_channel_id", "") or "").strip()
                            == channel
                        )
                        and (getattr(candidate, "user_content", "") or "").strip()
                    ]
                    target_present = len(target_candidates) == 1
                    if not target_present:
                        roster.lanes.append(FactLane(
                            role=AUTHOR_ROLE_SUBJECT,
                            text=quote,
                            # ONLY the resolved subject. Never the requester's
                            # id: that is the reply-chain contamination path.
                            actor_id=(
                                getattr(row, "reply_subject_actor_id", "") or ""
                            ).strip(),
                            source_message_id=target_id,
                            canonical_turn_id=row.canonical_turn_id,
                            speaker_label=(
                                getattr(row, "reply_subject_label", "") or ""
                            ).strip(),
                        ))

            if assistant_text:
                roster.lanes.append(FactLane(
                    role=AUTHOR_ROLE_ASSISTANT,
                    text=assistant_text,
                    actor_id="",
                    canonical_turn_id=row.canonical_turn_id,
                ))
        return roster

    def _load_compactable_rows(self) -> tuple[list["CanonicalTurnRow"], list["Message"]]:
        from ..types import SOURCE_CANONICAL_TURN_IDS_KEY, Message

        rows = list(
            self._store.get_uncompacted_canonical_turns(
                self._config.conversation_id,
                protected_recent_turns=self._config.monitor.protected_recent_turns,
            )
        )
        by_group = self._physical_rows_by_group()
        messages: list[Message] = []
        for row in rows:
            raw_group = getattr(row, "turn_group_number", -1)
            group = int(raw_group if raw_group is not None else -1)
            backing = by_group.get(group, [])
            # Split the backing physical rows by which side actually carries
            # content. A legacy combined row carries both, and may therefore
            # legitimately supply the same id to both messages.
            user_ids = [
                r.canonical_turn_id for r in backing if (r.user_content or "").strip()
            ]
            assistant_ids = [
                r.canonical_turn_id
                for r in backing
                if (r.assistant_content or "").strip()
            ]

            # ``_format_conversation`` labels a message with
            # ``get_sender_name(metadata) or role.capitalize()``. Carry the
            # stored sender in metadata, never in content: content feeds
            # hashes, excerpts, and the summary text itself. Only the user
            # half is attributed; a legacy row may carry the logical-turn
            # sender on both halves, and the assistant is not that speaker.
            #
            # The source ids ride alongside under a reserved key so each
            # segment's roster can be rebuilt from real rows instead of a
            # positional cursor. The session splitter copies Message.metadata
            # into both halves of a split, so the ids survive splits and
            # noncontiguous topic grouping.
            user_metadata: dict | None = None
            if (row.sender or "").strip() and (row.user_content or "").strip():
                user_metadata = {"sender": {"name": row.sender}}
            if user_ids:
                user_metadata = dict(user_metadata or {})
                user_metadata[SOURCE_CANONICAL_TURN_IDS_KEY] = list(user_ids)
            assistant_metadata: dict | None = None
            if assistant_ids:
                assistant_metadata = {
                    SOURCE_CANONICAL_TURN_IDS_KEY: list(assistant_ids)
                }
            timestamp = None
            for raw_timestamp in (
                row.first_seen_at,
                row.last_seen_at,
                row.created_at,
                row.updated_at,
            ):
                if not raw_timestamp:
                    continue
                try:
                    timestamp = datetime.fromisoformat(
                        str(raw_timestamp).replace("Z", "+00:00")
                    )
                    break
                except (TypeError, ValueError):
                    continue
            messages.append(Message(
                role="user",
                content=row.user_content,
                timestamp=timestamp,
                metadata=user_metadata,
            ))
            messages.append(Message(
                role="assistant",
                content=row.assistant_content,
                timestamp=timestamp,
                metadata=assistant_metadata,
            ))
        return rows, messages

    def _refresh_compaction_watermark(self) -> None:
        rows = list(self._store.get_all_canonical_turns(self._config.conversation_id))
        if not rows:
            self._engine_state.compacted_prefix_messages = 0
            self._engine_state.last_compacted_turn = -1
            return
        explicit_groups = [
            int(getattr(row, "turn_group_number"))
            if getattr(row, "turn_group_number", None) is not None
            else -1
            for row in rows
        ]
        if rows and all(group >= 0 for group in explicit_groups):
            grouped_rows: list[tuple[int, list["CanonicalTurnRow"]]] = []
            grouped_by_turn: dict[int, list["CanonicalTurnRow"]] = {}
            for row, turn_group_number in zip(rows, explicit_groups, strict=False):
                grouped_by_turn.setdefault(turn_group_number, []).append(row)
            grouped_rows = sorted(grouped_by_turn.items(), key=lambda item: item[0])
        else:
            grouped_rows = []
            pending: list["CanonicalTurnRow"] = []

            def _flush_pending() -> None:
                nonlocal pending
                if not pending:
                    return
                grouped_rows.append((len(grouped_rows), list(pending)))
                pending = []

            for row in rows:
                has_user = bool(getattr(row, "user_content", ""))
                has_assistant = bool(getattr(row, "assistant_content", ""))
                if has_user and has_assistant:
                    _flush_pending()
                    grouped_rows.append((len(grouped_rows), [row]))
                    continue
                if has_user:
                    _flush_pending()
                    pending = [row]
                    continue
                if has_assistant:
                    if pending:
                        pending.append(row)
                        _flush_pending()
                    else:
                        grouped_rows.append((len(grouped_rows), [row]))
                    continue
                _flush_pending()
            _flush_pending()

        last_prefix_turn = -1
        for turn_number, group_rows in grouped_rows:
            if group_rows and all(getattr(row, "compacted_at", None) for row in group_rows):
                last_prefix_turn = turn_number
                continue
            break
        if last_prefix_turn < 0:
            self._engine_state.compacted_prefix_messages = 0
            self._engine_state.last_compacted_turn = -1
            return
        self._engine_state.compacted_prefix_messages = (last_prefix_turn + 1) * 2
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
        results = self._compact_and_store(
            segments,
            len(compact_messages),
            compact_rows=compact_rows,
            progress_callback=progress_callback,
            generated_by_turn_id=generated_by_turn_id,
            operation_id=operation_id,
            disable_replacement_passes=disable_replacement_passes,
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

        physical_rows_by_group = self._physical_rows_by_group()
        physical_by_id = {
            row.canonical_turn_id: row
            for rows_in_group in physical_rows_by_group.values()
            for row in rows_in_group
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
                candidates = self._store.get_segments_by_tags(
                    tags=seg.tags, min_overlap=1, limit=merge_lookback,
                    conversation_id=self._config.conversation_id,
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

                for candidate in candidates:
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

                if best_candidate is not None:
                    # Combine turns: prepend existing segment's messages
                    old_meta = best_candidate.metadata
                    old_ids = list(
                        getattr(old_meta, "canonical_turn_ids", []) or []
                    )
                    old_complete = bool(
                        old_meta
                        and getattr(old_meta, "source_mapping_complete", False)
                        and old_ids
                        and all(cid in physical_by_id for cid in old_ids)
                    )
                    if old_complete:
                        candidate_messages = []
                        for cid in old_ids:
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
                                ))
                            if (old_row.assistant_content or "").strip():
                                candidate_messages.append(Message(
                                    role="assistant",
                                    content=old_row.assistant_content,
                                    metadata={
                                        SOURCE_CANONICAL_TURN_IDS_KEY: [cid],
                                    },
                                ))
                    else:
                        candidate_messages = [
                            Message(
                                role=m.get("role", "user"),
                                content=m.get("content", ""),
                                metadata=(m.get("metadata") or None),
                            )
                            for m in best_candidate.messages
                        ]
                    merged_mapping_prereqs[seg.id] = old_complete
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
        actor_rosters_by_segment = {
            seg.id: self._build_actor_roster(seg, physical_by_id)
            for seg in compactable
        }
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
                        logger.info("  Replaced %d old facts with %d new for segment %s",
                                    _deleted, _inserted, result.primary_tag)
                    else:
                        logger.info("  Stored %d facts for segment %s", _inserted, result.primary_tag)
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

        if not disable_replacement_passes:
            # A previous transient model/provider failure must not leave a
            # since-silent actor's card unreadable forever. Any ordinary
            # compaction services a bounded tenant-local retry queue after its
            # stored backoff expires. Terminal semantic disagreements remain
            # operator-visible and are not retried blindly.
            card_actors_to_rebuild.update(
                self._due_actor_card_rebuilds(limit=25)
            )
            for actor_id in sorted(card_actors_to_rebuild):
                try:
                    self._rebuild_actor_card(actor_id)
                except Exception:
                    # Canonical/fact mutation dirties any prior card before
                    # this call. A failed curation therefore leaves old
                    # entries auditable but unreadable.
                    logger.warning(
                        "actor card rebuild failed actor=%s",
                        actor_id[:24],
                        exc_info=True,
                    )

        return all_results
