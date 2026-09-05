"""ActorCardRebuildService: explicit dependencies for community memory work."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

from .evidence_manifest import evidence_digest

from .actor_card_policy import (
    _ACTOR_CARD_CITATION_LIMIT,
    _ACTOR_CARD_SINGLE_SOURCE_CONFIDENCE_CAP,
    _ACTOR_CARD_SINGLE_MESSAGE_STYLE_CONFIDENCE_CAP,
    _ActorCardAdmissionError,
    _ActorCardCoverageError,
    _format_rejection_counts,
)

if TYPE_CHECKING:
    from ...types import ActorCardEntry, ActorCardEntrySource

# Keep the existing operator log channel stable across the extraction.
logger = logging.getLogger("virtual_context.core.compaction_pipeline")


def _actor_card_message_keys(fact_sources, turn_sources, segment_source_ids):
    """Resolve citation identities using evidence already loaded for this build.

    Facts never inherit every message in their segment. Only a version-two
    requester fact with one exact native source match contributes an identity;
    legacy, quoted-subject and unavailable provenance cannot prove repetition.
    """
    keys = {}
    native_turns = {}
    for source in turn_sources:
        turn = source.turn
        owner = source.owner_conversation_id
        audience = source.audience_conversation_id
        native_id = turn.source_message_id or ""
        key = (
            ("message", owner, audience, native_id)
            if native_id else ("canonical", owner, audience, turn.canonical_turn_id)
        )
        keys[(owner, audience, "turn", turn.canonical_turn_id)] = key
        if native_id:
            native_turns.setdefault((owner, native_id), []).append(source)
    for source in fact_sources:
        fact = source.fact
        if (
            fact.author_attribution_version != 2
            or fact.author_source_role != "requester"
            or not fact.author_source_message_id
        ):
            continue
        owner = source.owner_conversation_id
        audience = source.audience_conversation_id
        matches = native_turns.get((owner, fact.author_source_message_id), [])
        if len(matches) != 1:
            continue
        turn_source = matches[0]
        turn = turn_source.turn
        if (
            turn_source.audience_conversation_id != audience
            or turn.sender_actor_id != fact.author_actor_id
            or turn.canonical_turn_id not in segment_source_ids.get(
                (owner, fact.segment_ref), (),
            )
        ):
            continue
        keys[(owner, audience, "fact", fact.id)] = keys[
            (owner, audience, "turn", turn.canonical_turn_id)
        ]
    return keys


def _actor_card_confidence(kind, confidence, citations, message_keys):
    from ...types import CARD_KIND_COMMUNICATION_PREF, CARD_KIND_INTERACTION_STYLE

    if kind in (CARD_KIND_COMMUNICATION_PREF, CARD_KIND_INTERACTION_STYLE):
        proved_messages = {message_keys[key] for key in citations if key in message_keys}
        if len(proved_messages) < 2:
            return min(confidence, _ACTOR_CARD_SINGLE_MESSAGE_STYLE_CONFIDENCE_CAP)
    elif len(citations) == 1:
        return min(confidence, _ACTOR_CARD_SINGLE_SOURCE_CONFIDENCE_CAP)
    return confidence


def _calibrate_actor_card_entries(entries, message_keys):
    for entry, entry_sources in entries:
        citations = [
            (source.owner_conversation_id, source.audience_conversation_id,
             "fact" if source.fact_id else "turn",
             source.fact_id or source.canonical_turn_id)
            for source in entry_sources
        ]
        entry.confidence = _actor_card_confidence(
            entry.kind, float(entry.confidence or 0.0), citations, message_keys,
        )


class ActorCardRebuildService:
    def __init__(
        self,
        *,
        store,
        config,
        compactor,
        curate_partition: Callable,
        admit_entries: Callable,
        policy_version: int,
        evidence_records: Callable,
    ) -> None:
        self._store = store
        self._config = config
        self._compactor = compactor
        self._curate_actor_card_partition = curate_partition
        self._admit_actor_card_entries = admit_entries
        self._policy_version = policy_version
        self._evidence_records = evidence_records

    def due_rebuilds(self, *, limit: int = 25) -> list[str]:
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
            return list(
                getter(
                    self._config.tenant_id,
                    due_at=datetime.now(timezone.utc).isoformat(),
                    limit=max(0, int(limit)),
                )
            )
        except Exception:
            logger.warning(
                "actor card retry queue read failed",
                exc_info=True,
            )
            return []

    def rebuild(self, actor_id: str, *, force: bool = False) -> int:
        """Curate and atomically replace one actor's rebuildable card cache.

        Facts are useful compact evidence, but they are not the membership
        criterion for a person card. Exact actor-authored canonical turns are
        also supplied so a substantive contributor can receive a meaningful
        card even when the fact extractor emitted nothing. A separate model
        independently judges substantive coverage and semantically admits each
        immutable candidate before the atomic replacement.
        """
        if not force and not getattr(self._config.assembler, "actor_card_enabled", False):
            return 0
        if self._compactor is None or not actor_id:
            return 0
        from datetime import datetime, timezone

        from ...types import (
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

        def _read_inputs() -> tuple[list, list, list, str, dict]:
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
            facts = list(
                self._store.list_actor_facts(
                    tenant_id,
                    actor_id,
                    limit=int(self._config.assembler.actor_card_fact_limit),
                )
            )
            turns = list(
                self._store.list_actor_turn_sources(
                    tenant_id,
                    actor_id,
                    limit=int(
                        getattr(
                            self._config.assembler,
                            "actor_card_turn_limit",
                            500,
                        )
                    ),
                )
            )
            carryover_getter = getattr(
                self._store,
                "list_actor_card_carryovers",
                None,
            )
            carryovers = (
                list(carryover_getter(tenant_id, actor_id)) if callable(carryover_getter) else []
            )
            required_fact_ids = sorted(
                {
                    source.fact_id
                    for _entry, sources in carryovers
                    for source in sources
                    if source.fact_id
                }
            )
            required_turn_ids = sorted(
                {
                    source.canonical_turn_id
                    for _entry, sources in carryovers
                    for source in sources
                    if source.canonical_turn_id
                }
            )
            fact_by_id = {source.fact.id: source for source in facts}
            turn_by_id = {source.turn.canonical_turn_id: source for source in turns}
            missing_fact_ids = [
                source_id for source_id in required_fact_ids if source_id not in fact_by_id
            ]
            missing_turn_ids = [
                source_id for source_id in required_turn_ids if source_id not in turn_by_id
            ]
            resolver = getattr(
                self._store,
                "resolve_actor_card_carryover_evidence",
                None,
            )
            if callable(resolver) and (missing_fact_ids or missing_turn_ids):
                resolved_facts, resolved_turns = resolver(
                    tenant_id,
                    actor_id,
                    fact_ids=missing_fact_ids,
                    turn_ids=missing_turn_ids,
                )
                for source in resolved_facts:
                    source_id = source.fact.id
                    if (
                        source_id in missing_fact_ids
                        and source.tenant_id == tenant_id
                        and source.fact.author_actor_id == actor_id
                    ):
                        fact_by_id[source_id] = source
                for source in resolved_turns:
                    source_id = source.turn.canonical_turn_id
                    if (
                        source_id in missing_turn_ids
                        and source.tenant_id == tenant_id
                        and source.turn.sender_actor_id == actor_id
                    ):
                        turn_by_id[source_id] = source
            # Put carryover citations first so the aggregate prompt bound cannot
            # hide the exact evidence needed to reconsider a live entry. Any
            # unresolved or truncated citation still fails closed downstream.
            facts = [
                fact_by_id[source_id] for source_id in required_fact_ids if source_id in fact_by_id
            ] + [source for source in facts if source.fact.id not in set(required_fact_ids)]
            turns = [
                turn_by_id[source_id] for source_id in required_turn_ids if source_id in turn_by_id
            ] + [
                source
                for source in turns
                if source.turn.canonical_turn_id not in set(required_turn_ids)
            ]
            fact_payload = (
                {
                    "id": source.fact.id,
                    "owner": source.owner_conversation_id,
                    "audience": source.audience_conversation_id,
                    "segment_ref": source.fact.segment_ref,
                    "author_actor_id": source.fact.author_actor_id,
                    "subject": source.fact.subject,
                    "verb": source.fact.verb,
                    "object": source.fact.object,
                    "what": source.fact.what,
                    "status": source.fact.status,
                    "superseded_by": source.fact.superseded_by,
                    "fact_type": source.fact.fact_type,
                    "mentioned_at": source.fact.mentioned_at.isoformat(),
                    "session_date": source.fact.session_date,
                    "author_version": (source.fact.author_attribution_version),
                    "author_role": source.fact.author_source_role,
                    "author_source_message_id": source.fact.author_source_message_id,
                }
                for source in facts
            )
            turn_payload = (
                {
                    "id": source.turn.canonical_turn_id,
                    "owner": source.owner_conversation_id,
                    "audience": source.audience_conversation_id,
                    "channel": source.audience_channel_id,
                    "content": source.turn.user_content,
                    "source_message_id": source.turn.source_message_id,
                    "created_at": (source.turn.created_at or source.turn.first_seen_at or ""),
                    "owner_epoch": source.owner_lifecycle_epoch,
                    "audience_epoch": source.audience_lifecycle_epoch,
                }
                for source in turns
            )
            carryover_payload = (
                {
                    "entry": {
                        "id": entry.id,
                        "kind": entry.kind,
                        "body": entry.body,
                        "confidence": entry.confidence,
                        "scope": entry.audience_scope,
                    },
                    "sources": sorted(
                        [
                            {
                                "owner": source.owner_conversation_id,
                                "audience": source.audience_conversation_id,
                                "channel": source.audience_channel_id,
                                "fact_id": source.fact_id,
                                "turn_id": source.canonical_turn_id,
                            }
                            for source in sources
                        ],
                        key=lambda item: json.dumps(
                            item,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                }
                for entry, sources in carryovers
            )
            curation_model = configured_curation_model or (
                getattr(self._compactor, "model_name", "")
                or getattr(
                    getattr(self._compactor, "llm", None),
                    "model",
                    "",
                )
                or type(getattr(self._compactor, "llm", None)).__name__
            )
            segment_source_ids = {}
            visible_ids_by_owner = {}
            for source in turns:
                visible_ids_by_owner.setdefault(source.owner_conversation_id, set()).add(
                    source.turn.canonical_turn_id,
                )

            def source_records():
                # Retain only source-id metadata from the existing streamed
                # manifest. Confidence calibration performs no extra DB reads.
                for record in self._evidence_records(facts, turns):
                    if record.get("kind") == "fact_segment":
                        segment_source_ids[(record["owner"], record["ref"])] = (
                            visible_ids_by_owner.get(record["owner"], set()).intersection(
                                record["source_ids"],
                            )
                        )
                    yield record

            digest = evidence_digest(
                {
                    "policy": self._policy_version,
                    "curation_model": curation_model,
                    "curation_fallback_model": curation_fallback_model,
                    "admission_model": admission_model,
                    "admission_fallback_model": admission_fallback_model,
                    "prompt_max_chars": int(
                        getattr(
                            self._config.assembler,
                            "actor_card_prompt_max_chars",
                            192_000,
                        )
                    ),
                },
                records={
                    "facts": fact_payload,
                    "turns": turn_payload,
                    "carryovers": carryover_payload,
                    "source_evidence": source_records(),
                },
            )
            return facts, turns, carryovers, digest, segment_source_ids

        (
            fact_sources,
            turn_sources,
            carryover_entries,
            input_hash,
            _segment_source_ids,
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
            status = status_getter(tenant_id, actor_id) if callable(status_getter) else None
            failed_outcomes = {
                "model_error",
                "invalid_response",
                "rejected_all",
                "admission_error",
                "coverage_disagreement",
                "coverage_gap",
                "stale_or_rejected_write",
            }
            # Coverage outcomes keep the timed retry backoff but are never
            # TERMINAL: a permanently cardless active member is not an
            # acceptable endpoint of a backoff policy.
            terminal_outcomes = failed_outcomes - {
                "coverage_disagreement",
                "coverage_gap",
            }
            if (
                status
                and (status.get("input_hash") or "") == input_hash
                and status.get("outcome") in failed_outcomes
            ):
                failures = int(status.get("failure_count") or 0)
                if failures >= 3 and status.get("outcome") in terminal_outcomes:
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
                    retry_at = datetime.fromisoformat(str(retry_raw).replace("Z", "+00:00"))
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
        build_marker = f"building:{input_hash}:{time.time_ns()}:{id(self)}"
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
            segment_source_ids,
        ) = _read_inputs()

        message_keys = _actor_card_message_keys(fact_sources, turn_sources, segment_source_ids)

        fact_source_by_audience_id = {
            (source.audience_conversation_id, source.fact.id): source for source in fact_sources
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
                curator_substantive_by_audience[audience_id] = partition_substantive
                raw_entries.extend(
                    (audience_id, item, visible_turn_ids) for item in partition_entries
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
            if len(body) > CARD_ENTRY_BODY_MAX_CHARS or any(
                ord(ch) < 32 or ord(ch) == 127 for ch in body
            ):
                rejected["invalid_body"] += 1
                continue
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
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
                    )
                    in turn_source_by_audience_id
                )

            unknown_fact_id: str | None = None
            for source_id in dict.fromkeys(fact_ids):
                if _valid_fact_id(source_id):
                    normalized_fact_ids.append(source_id)
                elif _valid_turn_id(source_id):
                    normalized_turn_ids.append(source_id)
                else:
                    unknown_fact_id = source_id
                    break
            if unknown_fact_id is not None:
                rejected["unknown_or_cross_audience_fact_id"] += 1
                # IDs only, never entry bodies: the guard exists to keep
                # audiences isolated, so its own diagnostics must not leak
                # content across them. The offending id is what separates a
                # hallucinated citation from a mangled real one from a
                # correctly refused cross-audience reference.
                logger.debug(
                    "ACTOR_CARD_CITATION_REJECTED "
                    "reason=unknown_or_cross_audience_fact_id "
                    "audience_id=%s source_id=%s",
                    audience_id,
                    unknown_fact_id,
                )
                continue

            unknown_turn_id: str | None = None
            for source_id in dict.fromkeys(turn_ids):
                if _valid_turn_id(source_id):
                    normalized_turn_ids.append(source_id)
                elif _valid_fact_id(source_id):
                    normalized_fact_ids.append(source_id)
                else:
                    unknown_turn_id = source_id
                    break
            if unknown_turn_id is not None:
                rejected["unknown_or_cross_audience_turn_id"] += 1
                logger.debug(
                    "ACTOR_CARD_CITATION_REJECTED "
                    "reason=unknown_or_cross_audience_turn_id "
                    "audience_id=%s source_id=%s",
                    audience_id,
                    unknown_turn_id,
                )
                continue

            fact_ids = list(dict.fromkeys(normalized_fact_ids))
            turn_ids = list(dict.fromkeys(normalized_turn_ids))
            # Citation order has no semantics. Canonicalize it before both
            # bounding and identity so even over-limit reordered copies retain
            # the same exact source subset and cannot evade deduplication.
            fact_ids = sorted(fact_ids)
            turn_ids = sorted(turn_ids)
            if len(fact_ids) + len(turn_ids) > _ACTOR_CARD_CITATION_LIMIT:
                remaining = _ACTOR_CARD_CITATION_LIMIT
                fact_ids = fact_ids[:remaining]
                remaining -= len(fact_ids)
                turn_ids = turn_ids[:remaining]
                rejected["citations_trimmed"] += 1
            if not fact_ids and not turn_ids:
                rejected["missing_citations"] += 1
                continue

            citations = [
                (fact_source_by_audience_id[(audience_id, fid)].owner_conversation_id,
                 audience_id, "fact", fid)
                for fid in fact_ids
            ] + [
                (turn_source_by_audience_id[(audience_id, tid)].owner_conversation_id,
                 audience_id, "turn", tid)
                for tid in turn_ids
            ]
            confidence = _actor_card_confidence(kind, confidence, citations, message_keys)

            scope = (
                CARD_SCOPE_CROSS_CONTEXT
                if kind in CARD_CROSS_CONTEXT_KINDS
                else CARD_SCOPE_SAME_CONVERSATION
            )
            digest = hashlib.sha256(
                json.dumps(
                    [actor_id, kind, body, fact_ids, turn_ids],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:32]
            entry = ActorCardEntry(
                id=f"card-{digest}",
                tenant_id=tenant_id,
                actor_id=actor_id,
                kind=kind,
                body=body,
                confidence=confidence,
                # Retained only for schema compatibility. Sensitivity is not
                # part of curation, admission, scoping, or serving policy.
                sensitivity=CARD_SENSITIVITY_NORMAL,
                audience_scope=scope,
                created_at=now,
                updated_at=now,
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
                        fact_source_by_audience_id[(audience_id, fid)].owner_conversation_id
                    ),
                    audience_conversation_id=(
                        fact_source_by_audience_id[(audience_id, fid)].audience_conversation_id
                    ),
                    audience_channel_id=(
                        fact_source_by_audience_id[(audience_id, fid)].audience_channel_id
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
                        turn_source_by_audience_id[(audience_id, turn_id)].owner_conversation_id
                    ),
                    audience_conversation_id=(
                        turn_source_by_audience_id[(audience_id, turn_id)].audience_conversation_id
                    ),
                    audience_channel_id=(
                        turn_source_by_audience_id[(audience_id, turn_id)].audience_channel_id
                    ),
                    canonical_turn_id=turn_id,
                )
                for turn_id in turn_ids
            )
            normalized_entries_by_key[semantic_key] = entry
            normalized.append((entry, entry_sources))
            normalized_by_audience.setdefault(audience_id, []).append((entry, entry_sources))
            per_audience_kind[quota_key] = per_audience_kind.get(quota_key, 0) + 1

        # A fresh curator is allowed to propose better identity/style entries,
        # but omission is not a deletion decision.  Re-submit every currently
        # active cross-context entry to semantic admission with its immutable
        # body and exact sources.  Same-conversation goals/history deliberately
        # retain replacement semantics: they are the rotating working set.
        existing_entry_ids_by_audience: dict[str, set[str]] = {}
        fresh_entry_ids = {entry.id for entry, _sources in normalized}
        for entry, entry_sources in carryover_entries:
            if (
                entry.kind not in CARD_CROSS_CONTEXT_KINDS
                or entry.audience_scope != CARD_SCOPE_CROSS_CONTEXT
                or not entry_sources
            ):
                logger.error(
                    "ACTOR_CARD_CARRYOVER_INVALID actor=%s entry=%s kind=%s scope=%s sources=%d",
                    actor_id[:24],
                    entry.id,
                    entry.kind,
                    entry.audience_scope,
                    len(entry_sources),
                )
                raise RuntimeError("actor card carryover violated the cross-context boundary")
            audiences = {
                (source.audience_conversation_id or "").strip() for source in entry_sources
            }
            if "" in audiences or len(audiences) != 1:
                # Never put evidence from two privacy audiences in one model
                # prompt.  Failing the refresh leaves the last-good card
                # served; silently dropping it would recreate the bug this
                # path exists to prevent.
                logger.error(
                    "ACTOR_CARD_CARRYOVER_AUDIENCE_INVALID actor=%s entry=%s audience_count=%d",
                    actor_id[:24],
                    entry.id,
                    len(audiences),
                )
                raise RuntimeError("actor card carryover has ambiguous source audience")
            audience_id = next(iter(audiences))
            existing_entry_ids_by_audience.setdefault(
                audience_id,
                set(),
            ).add(entry.id)
            if entry.id in fresh_entry_ids:
                continue
            normalized.append((entry, entry_sources))
            normalized_by_audience.setdefault(audience_id, []).append((entry, entry_sources))

        # Admission sees the same calibrated proposal for fresh and carried
        # entries. A stale stored confidence must not cause an otherwise valid
        # carryover to fail the current semantic confidence contract.
        _calibrate_actor_card_entries(normalized, message_keys)
        basic_accepted_count = len(normalized)
        independently_substantive = False
        coverage_gap = False
        if parsed_entries and (fact_sources or turn_sources or carryover_entries):
            try:
                admitted_entries: list[tuple[ActorCardEntry, list[ActorCardEntrySource]]] = []
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
                        curator_substantive=(curator_substantive_by_audience[audience_id]),
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
                    cap = int(self._config.assembler.actor_card_entries_per_kind)
                    limited: list[tuple[ActorCardEntry, list[ActorCardEntrySource]]] = []
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
                        limited.extend(ranked[: max(0, cap)])
                        if len(ranked) > max(0, cap):
                            rejected["post_admission_per_kind_limit"] += len(ranked) - max(0, cap)
                    partition_admitted = limited
                    admitted_entries.extend(partition_admitted)
                    admission_responses.append(partition_response)
                    rejected.update(admission_rejections)
                    independently_substantive = independently_substantive or partition_substantive
                    privacy_only_rejection = bool(normalized_by_audience.get(audience_id)) and (
                        admission_rejections.get(
                            "semantic_explicit_privacy_request",
                            0,
                        )
                        == len(normalized_by_audience[audience_id])
                        == sum(admission_rejections.values())
                    )
                    coverage_gap = coverage_gap or (
                        partition_substantive
                        and not partition_admitted
                        and not privacy_only_rejection
                    )
                normalized = admitted_entries
                admission_response_text = json.dumps(
                    admission_responses,
                    separators=(",", ":"),
                )
            except Exception as exc:
                admission_exception = exc
                admission_response_text = getattr(
                    exc,
                    "response_text",
                    "",
                )

        response_hash = (
            hashlib.sha256(
                json.dumps(
                    {
                        "curation": response_text,
                        "admission": admission_response_text,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if response_text or admission_response_text
            else ""
        )

        def _record_status(outcome: str, *, written_count: int = 0) -> None:
            recorder = getattr(
                self._store,
                "record_actor_card_rebuild_status",
                None,
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
                "ACTOR_CARD_REBUILD actor=%s sources=%d outcome=%s response_hash=%s error_type=%s",
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
                _format_rejection_counts(rejected),
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
            raise RuntimeError("actor card semantic admission failed") from (admission_exception)
        # A substantive actor whose offered entries all failed the
        # durability gate is a legitimate steady state, not a failure:
        # substantive describes the ACTOR's interaction, durable describes
        # ENTRIES. The (possibly empty) replacement below clears the dirty
        # and invalid flags and records the honest outcome instead of
        # wedging the card behind a gate the hardened admission rules can
        # never satisfy for banter- or request-heavy actors.

        # Carryovers must meet the same current evidence calibration as fresh
        # candidates; citation rows are not independent message observations.
        # Confidence is not part of the entry's immutable body or identity
        # digest, so the clamp changes no id and rewrites no body.
        _calibrate_actor_card_entries(normalized, message_keys)

        expected_epochs: dict[str, int] = {}
        for source in [*fact_sources, *turn_sources]:
            expected_epochs[source.owner_conversation_id] = source.owner_lifecycle_epoch
            expected_epochs[source.audience_conversation_id] = source.audience_lifecycle_epoch
        epoch_getter = getattr(self._store, "get_lifecycle_epoch", None)
        if carryover_entries and not callable(epoch_getter):
            raise RuntimeError("actor card carryover cannot prove lifecycle epochs")
        for _entry, entry_sources in carryover_entries:
            for source in entry_sources:
                for conversation_id in (
                    source.owner_conversation_id,
                    source.audience_conversation_id,
                ):
                    if conversation_id not in expected_epochs:
                        expected_epochs[conversation_id] = int(epoch_getter(conversation_id))
        written = self._store.replace_actor_card(
            tenant_id,
            actor_id,
            normalized,
            input_hash=input_hash,
            expected_source_epochs=expected_epochs,
            expected_build_marker=build_marker,
        )
        refreshed = self._store.get_actor_profile(tenant_id, actor_id)
        if refreshed is None or refreshed.card_dirty or (refreshed.card_input_hash != input_hash):
            # Distinguish losing the build marker from a bad write. Any
            # mutation to a conversation this actor speaks in re-dirties the
            # profile and clears the marker, so a rebuild whose model call
            # spans a live message reaches the commit holding a marker that
            # is no longer installed, and the replacement declines it. The
            # card is untouched and still dirty, and the next consolidation
            # rebuilds it against the newer evidence — the same protection,
            # reported for what it is. On a busy conversation this is the
            # expected outcome rather than the exceptional one, so it is
            # neither raised nor counted as a failed attempt.
            #
            # Proving it takes both halves: our input hash did not land
            # (nothing of ours was written) AND the marker is no longer
            # ours (someone else moved first). ``written == 0`` alone
            # cannot say this, because a clean-empty card also writes zero
            # rows.
            if (
                refreshed is not None
                and refreshed.card_input_hash != input_hash
                and (refreshed.card_build_marker or "") != build_marker
            ):
                _record_status("superseded", written_count=written)
                logger.info(
                    "ACTOR_CARD_REBUILD actor=%s sources=%d raw=%d "
                    "accepted=%d written=%d outcome=superseded "
                    "response_hash=%s",
                    actor_id[:24],
                    len(fact_sources) + len(turn_sources),
                    len(raw_entries),
                    len(normalized),
                    written,
                    response_hash[:16],
                )
                return 0
            _record_status("stale_or_rejected_write", written_count=written)
            raise RuntimeError("actor card replacement did not commit cleanly")
        outcome = (
            (
                "no_durable_entries"
                if coverage_gap
                else "clean_empty_filtered"
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
            _format_rejection_counts(rejected),
            response_hash[:16],
        )
        return written
