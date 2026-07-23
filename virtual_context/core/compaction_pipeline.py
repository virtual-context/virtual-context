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
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from .engine_utils import extract_turn_pairs
from .store import ContextStore
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
        SegmentMetadata,
        StoredSegment,
        VirtualContextConfig,
    )

logger = logging.getLogger(__name__)


class _ActorCardAdmissionError(RuntimeError):
    """Validation failure that preserves a hashable, non-logged response."""

    def __init__(self, message: str, response_text: str = "") -> None:
        super().__init__(message)
        self.response_text = response_text


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

    def _rebuild_actor_card(self, actor_id: str, *, force: bool = False) -> int:
        """Curate and atomically replace one actor's rebuildable card cache."""
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
            CARD_SENSITIVITIES,
            CARD_SENSITIVITY_NORMAL,
            ActorCardEntry,
            ActorCardEntrySource,
        )

        tenant_id = self._config.tenant_id
        sources = list(self._store.list_actor_facts(
            tenant_id,
            actor_id,
            limit=int(self._config.assembler.actor_card_fact_limit),
        ))
        hash_payload = [
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
                "author_version": source.fact.author_attribution_version,
                "author_role": source.fact.author_source_role,
            }
            for source in sources
        ]
        input_hash = hashlib.sha256(json.dumps(
            {
                "policy": 5,
                "admission_model": getattr(
                    self._config.assembler,
                    "actor_card_admission_model",
                    "",
                ),
                "facts": hash_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        profile = self._store.get_actor_profile(tenant_id, actor_id)
        if profile is None:
            return 0
        if profile.card_input_hash == input_hash and not profile.card_dirty:
            return 0
        if not self._store.mark_actor_card_dirty(tenant_id, actor_id):
            return 0

        source_by_id = {source.fact.id: source for source in sources}
        raw_entries: list = []
        response_text = ""
        admission_response_text = ""
        parsed_entries = True
        model_exception: Exception | None = None
        admission_exception: Exception | None = None
        if sources:
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
                for source in sources
            ]
            system = (
                "Curate a compact person card from facts authored by one actor. "
                "The card is for durable interaction continuity, not a fact "
                "scrapbook or a transcript. Return JSON only with exactly one "
                "top-level key, entries, whose value is an array. Each entry "
                "must contain exactly: kind, body, confidence, sensitivity, "
                "and fact_ids. kind must be exactly one of "
                "\"communication_pref\", \"active_goal\", "
                "\"relevant_history\", or \"interaction_style\". "
                "sensitivity must be exactly the string \"normal\" or \"high\"; "
                "never use null, \"none\", booleans, or numbers. confidence "
                "must be a number from 0 through 1. fact_ids must cite only "
                "provided ids. Use a neutral concise body and do not invent "
                "identity or intent. Every body must be self-contained and "
                "unambiguous when read without the surrounding transcript; "
                "include essential referents such as the specific medication, "
                "goal, or preference rather than a generic placeholder. Write "
                "natural person-facing language, not a serialization of "
                "subject/verb/object fields, ontology names, or tag labels. "
                "Do not promote temporary, test-only, one-turn, session-only, "
                "or channel-only instructions into communication_pref or "
                "interaction_style. Do not retain a preference or goal that a "
                "later fact stopped, replaced, completed, or contradicted. "
                "Use mentioned_at and status to resolve conflicts, with the "
                "newest applicable fact winning. A communication preference or "
                "interaction style should be durable only when it is explicitly "
                "stated as lasting or consistently supported by repeated facts. "
                "Medical, sexual, financial, precise-location, credential, or "
                "similarly private personal material must be sensitivity "
                "\"high\". Ordinary non-private communication and interaction "
                "style may be \"normal\". If no durable entry is justified, "
                "return {\"entries\":[]}."
            )
            user = json.dumps({
                "facts": prompt_facts,
                "limits": {
                    "entries_per_kind": int(
                        self._config.assembler.actor_card_entries_per_kind
                    ),
                    "body_chars": CARD_ENTRY_BODY_MAX_CHARS,
                },
            }, separators=(",", ":"))
            try:
                response_text, _usage = self._compactor.llm.complete(
                    system=system,
                    user=user,
                    max_tokens=self._config.compactor.max_summary_tokens,
                )
                parsed = self._compactor._parse_response(response_text)
                if (
                    isinstance(parsed, dict)
                    and set(parsed) == {"entries"}
                    and isinstance(parsed["entries"], list)
                ):
                    raw_entries = parsed["entries"]
                else:
                    parsed_entries = False
            except Exception as exc:
                parsed_entries = False
                model_exception = exc

        now = datetime.now(timezone.utc).isoformat()
        per_kind: dict[str, int] = {}
        normalized: list[tuple[ActorCardEntry, list[ActorCardEntrySource]]] = []
        rejected: Counter[str] = Counter()
        for item in raw_entries:
            if not isinstance(item, dict):
                rejected["entry_not_object"] += 1
                continue
            kind = item.get("kind")
            body = item.get("body")
            sensitivity = item.get("sensitivity", CARD_SENSITIVITY_NORMAL)
            confidence = item.get("confidence")
            fact_ids = item.get("fact_ids")
            if kind not in CARD_KINDS:
                rejected["invalid_kind"] += 1
                continue
            if sensitivity not in CARD_SENSITIVITIES:
                rejected["invalid_sensitivity"] += 1
                continue
            if per_kind.get(kind, 0) >= int(
                self._config.assembler.actor_card_entries_per_kind
            ):
                rejected["per_kind_limit"] += 1
                continue
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
            if not isinstance(fact_ids, list) or not fact_ids:
                rejected["invalid_fact_ids"] += 1
                continue
            if any(
                not isinstance(fid, str) or fid not in source_by_id
                for fid in fact_ids
            ):
                rejected["unknown_fact_id"] += 1
                continue
            fact_ids = list(dict.fromkeys(fact_ids))

            scope = (
                CARD_SCOPE_CROSS_CONTEXT
                if sensitivity == CARD_SENSITIVITY_NORMAL
                and kind in CARD_CROSS_CONTEXT_KINDS
                else CARD_SCOPE_SAME_CONVERSATION
            )
            digest = hashlib.sha256(json.dumps(
                [actor_id, kind, body, fact_ids], separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:32]
            entry = ActorCardEntry(
                id=f"card-{digest}", tenant_id=tenant_id, actor_id=actor_id,
                kind=kind, body=body, confidence=confidence,
                sensitivity=sensitivity, audience_scope=scope,
                created_at=now, updated_at=now,
            )
            entry_sources = [
                ActorCardEntrySource(
                    entry_id=entry.id,
                    tenant_id=tenant_id,
                    owner_conversation_id=source_by_id[fid].owner_conversation_id,
                    audience_conversation_id=(
                        source_by_id[fid].audience_conversation_id
                    ),
                    audience_channel_id=source_by_id[fid].audience_channel_id,
                    fact_id=fid,
                )
                for fid in fact_ids
            ]
            normalized.append((entry, entry_sources))
            per_kind[kind] = per_kind.get(kind, 0) + 1

        basic_accepted_count = len(normalized)
        if parsed_entries and normalized:
            try:
                (
                    normalized,
                    admission_response_text,
                    admission_rejections,
                ) = self._admit_actor_card_entries(
                    actor_id,
                    sources,
                    normalized,
                )
                rejected.update(admission_rejections)
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
                        source_count=len(sources),
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
                len(sources),
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
                len(sources),
                len(raw_entries),
                json.dumps(dict(sorted(rejected.items())), separators=(",", ":")),
                response_hash[:16],
            )
            raise RuntimeError("actor card curation rejected every model entry")

        if admission_exception is not None:
            _record_status("admission_error")
            logger.warning(
                "ACTOR_CARD_REBUILD actor=%s sources=%d raw=%d "
                "basic_accepted=%d outcome=admission_error error_type=%s "
                "response_hash=%s",
                actor_id[:24],
                len(sources),
                len(raw_entries),
                basic_accepted_count,
                type(admission_exception).__name__,
                response_hash[:16],
            )
            raise RuntimeError("actor card semantic admission failed") from (
                admission_exception
            )

        expected_epochs: dict[str, int] = {}
        for source in sources:
            expected_epochs[source.owner_conversation_id] = (
                source.owner_lifecycle_epoch
            )
            expected_epochs[source.audience_conversation_id] = (
                source.audience_lifecycle_epoch
            )
        written = self._store.replace_actor_card(
            tenant_id,
            actor_id,
            normalized,
            input_hash=input_hash,
            expected_source_epochs=expected_epochs,
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
            len(sources),
            len(raw_entries),
            len(normalized),
            written,
            outcome,
            json.dumps(dict(sorted(rejected.items())), separators=(",", ":")),
            response_hash[:16],
        )
        return written

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
        base = self._compactor.llm
        from ..providers.anthropic import AnthropicProvider
        from ..providers.generic_openai import GenericOpenAIProvider

        if isinstance(base, GenericOpenAIProvider):
            return GenericOpenAIProvider(
                base_url=base.base_url,
                model=model,
                temperature=0.0,
                api_key=base.api_key,
                reasoning_effort="low",
            )
        if isinstance(base, AnthropicProvider):
            return AnthropicProvider(
                api_key=base.api_key,
                model=model,
                temperature=0.0,
            )
        raise RuntimeError(
            "actor-card admission model override is unsupported by "
            f"{type(base).__name__}"
        )

    def _actor_card_evidence_segments(
        self,
        actor_id: str,
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
        source_by_id = {source.fact.id: source for source in sources}
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
        sources: list,
        normalized: list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
    ) -> tuple[
        list[tuple["ActorCardEntry", list["ActorCardEntrySource"]]],
        str,
        Counter[str],
    ]:
        """Semantically admit immutable candidates against source evidence."""
        from ..types import (
            CARD_CROSS_CONTEXT_KINDS,
            CARD_SCOPE_CROSS_CONTEXT,
            CARD_SCOPE_SAME_CONVERSATION,
            CARD_SENSITIVITIES,
            CARD_SENSITIVITY_NORMAL,
        )

        provider = self._actor_card_admission_provider()
        if provider is None:
            raise RuntimeError(
                "actor-card semantic admission model is not configured"
            )

        candidate_fact_ids = {
            source.fact_id
            for _entry, entry_sources in normalized
            for source in entry_sources
        }
        evidence_segments, evidence_refs = (
            self._actor_card_evidence_segments(
                actor_id,
                sources,
                candidate_fact_ids,
            )
        )
        source_by_id = {source.fact.id: source for source in sources}
        candidates: list[dict] = []
        eligible: dict[
            str, tuple["ActorCardEntry", list["ActorCardEntrySource"]]
        ] = {}
        rejection_counts: Counter[str] = Counter()
        for entry, entry_sources in normalized:
            fact_ids = [source.fact_id for source in entry_sources]
            refs = {
                (
                    source_by_id[fact_id].owner_conversation_id,
                    source_by_id[fact_id].fact.segment_ref,
                )
                for fact_id in fact_ids
                if fact_id in source_by_id
            }
            if not refs or not refs.issubset(evidence_refs):
                rejection_counts["evidence_unavailable"] += 1
                continue
            eligible[entry.id] = (entry, entry_sources)
            candidates.append({
                "candidate_id": entry.id,
                "kind": entry.kind,
                "body": entry.body,
                "proposed_confidence": entry.confidence,
                "proposed_sensitivity": entry.sensitivity,
                "fact_ids": fact_ids,
                "source_segments": [
                    {
                        "owner_conversation_id": owner,
                        "segment_ref": ref,
                    }
                    for owner, ref in sorted(refs)
                ],
            })
        if not candidates:
            return [], "", rejection_counts

        compact_facts = [{
            "id": source.fact.id,
            "owner_conversation_id": source.owner_conversation_id,
            "segment_ref": source.fact.segment_ref,
            "fact": source.fact.format_for_prompt(),
            "status": source.fact.status,
            "mentioned_at": source.fact.mentioned_at.isoformat(),
        } for source in sources]
        system = (
            "You are the conservative semantic admission gate for a person "
            "card. Candidate bodies are immutable: you may admit or reject "
            "them and correct sensitivity, but you may not invent, rewrite, "
            "or merge candidates. Use only actor-authored facts and source "
            "messages. All available compact facts are supplied so later "
            "facts can revoke or replace a candidate; raw source messages are "
            "limited to the segments the candidate cites. Return JSON only "
            "with exactly one top-level key, "
            "decisions. Return exactly one decision for every candidate, with "
            "exactly candidate_id, admit, sensitivity, and reason. admit must "
            "be a boolean and sensitivity must be exactly \"normal\" or "
            "\"high\". You may raise sensitivity from normal to high, but "
            "must never lower high to normal. reason must be exactly one of "
            "\"durable\", \"temporary\", \"test_probe\", "
            "\"stopped_or_replaced\", \"completed\", \"contradicted\", "
            "\"insufficient_evidence\", \"not_durable\", or "
            "\"not_person_card\". Use reason \"durable\" if and only if admit "
            "is true. "
            "Reject temporary, test/probe, one-turn, session-only, "
            "channel-only, stopped, replaced, completed, or contradicted "
            "material. Later source messages revoke or replace earlier "
            "material. Requested answer prefixes and memory-system tests are "
            "not durable identity preferences. A communication preference or "
            "interaction style is admissible only when a source message "
            "explicitly establishes durability beyond the current test, "
            "session, and channel, or when consistent natural evidence spans "
            "at least two distinct source segments. Repeated test instructions "
            "do not establish a pattern. The immutable candidate body itself "
            "must be self-contained and unambiguous without relying on the "
            "surrounding segment; reject with insufficient_evidence when an "
            "essential referent (such as which medication, goal, or "
            "preference) is omitted. The body must be fully entailed by the "
            "cited actor-authored messages. Compact fact fields and tags help "
            "locate evidence but cannot independently justify body text. "
            "Reject with not_person_card when a body exposes internal "
            "ontology/tag language or serializes a machine fact triple rather "
            "than stating a natural person fact. Medical, sexual, financial, "
            "precise-location, credential, or similarly private material must "
            "be high sensitivity. When uncertain, reject."
        )
        user = json.dumps({
            "candidates": candidates,
            "facts": compact_facts,
            "evidence_segments": evidence_segments,
        }, separators=(",", ":"))
        response_text, _usage = provider.complete(
            system=system,
            user=user,
            max_tokens=max(
                800,
                min(
                    4000,
                    300 + 250 * len(candidates),
                ),
            ),
        )
        parsed = self._compactor._parse_response(response_text)
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"decisions"}
            or not isinstance(parsed["decisions"], list)
        ):
            raise _ActorCardAdmissionError(
                "actor-card admission response has no decisions array",
                response_text,
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
        }
        for decision in parsed["decisions"]:
            if (
                not isinstance(decision, dict)
                or set(decision)
                != {"candidate_id", "admit", "sensitivity", "reason"}
                or not isinstance(decision.get("candidate_id"), str)
                or not isinstance(decision.get("admit"), bool)
                or decision.get("sensitivity") not in CARD_SENSITIVITIES
                or decision.get("reason") not in valid_reasons
                or (
                    bool(decision.get("admit"))
                    != (decision.get("reason") == "durable")
                )
                or decision["candidate_id"] in decisions
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission response contains an invalid decision",
                    response_text,
                )
            decisions[decision["candidate_id"]] = decision
        if set(decisions) != set(eligible):
            raise _ActorCardAdmissionError(
                "actor-card admission response does not cover every candidate",
                response_text,
            )

        admitted: list[
            tuple["ActorCardEntry", list["ActorCardEntrySource"]]
        ] = []
        for candidate_id, (entry, entry_sources) in eligible.items():
            decision = decisions[candidate_id]
            if not decision["admit"]:
                rejection_counts[
                    f"semantic_{decision['reason']}"
                ] += 1
                continue
            sensitivity = decision["sensitivity"]
            if (
                entry.sensitivity != CARD_SENSITIVITY_NORMAL
                and sensitivity == CARD_SENSITIVITY_NORMAL
            ):
                raise _ActorCardAdmissionError(
                    "actor-card admission attempted to lower sensitivity",
                    response_text,
                )
            scope = (
                CARD_SCOPE_CROSS_CONTEXT
                if sensitivity == CARD_SENSITIVITY_NORMAL
                and entry.kind in CARD_CROSS_CONTEXT_KINDS
                else CARD_SCOPE_SAME_CONVERSATION
            )
            admitted.append((
                replace(
                    entry,
                    sensitivity=sensitivity,
                    audience_scope=scope,
                ),
                entry_sources,
            ))
        return admitted, response_text, rejection_counts

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
                _actors_to_rebuild = {
                    (fact.author_actor_id or "").strip()
                    for fact in [*_existing_facts_before, *result.facts]
                    if (fact.author_actor_id or "").strip()
                }
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
                    for actor_id in sorted(_actors_to_rebuild):
                        try:
                            self._rebuild_actor_card(actor_id)
                        except Exception:
                            # replace_facts_for_segment already dirtied the
                            # profile in the same transaction. A failed
                            # curation therefore leaves the old entries
                            # auditable but unreadable.
                            logger.warning(
                                "actor card rebuild failed actor=%s",
                                actor_id[:24], exc_info=True,
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

        return all_results
