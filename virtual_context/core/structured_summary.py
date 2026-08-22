"""Shared helpers for source-backed, model-visible summaries.

The free-form ``segments.summary`` column remains a useful lossy retrieval
index.  A :class:`~virtual_context.types.StructuredSummary` is the separately
validated presentation artifact: every claim cites an exact physical lane and
an exact excerpt from that lane.  This module contains the quote-local semantic
checks shared by generation, migration, and rendering.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass


_UNSAFE_STATUS_CONTEXT_RE = re.compile(
    r"\b(?:almost|nearly|consider(?:ed|ing)?|thinking\s+about|wish|"
    r"remember|may\s+be\s+mistaken|might\s+be\s+mistaken|falsely|"
    r"which\s+(?:is|was)\s+false|(?:is|was)\s+false\s+that|"
    r"deny|denied|refute(?:d|s|ing)?|mistaken|"
    r"(?:said|claimed|reported)\s+(?:that\s+)?)\b",
    re.IGNORECASE,
)
_SAFETY_CRITICAL_PERSONAL_RE = re.compile(
    r"\b(?:"
    r"I\s+(?:(?:actually|currently|presently|definitely|certainly|absolutely)\s+)?"
    r"(?:do\s+not|don['’]t|did\s+not|didn['’]t|have\s+not|"
    r"haven['’]t|never|no\s+longer)\s+"
    r"(?:(?:actually|currently|presently|definitely|certainly|absolutely)\s+)?"
    r"(?:take|use|run|do|have|start|stop|discontinue|"
    r"complete|finish|cease|quit|pause|halt|suspend|restart|resume)\b|"
    r"I\s+(?:did\s+not|didn['’]t|have\s+not|haven['’]t|never)\s+"
    r"(?:started|begun|stopped|discontinued|completed|finished|ceased|quit|"
    r"paused|halted|suspended|restarted|resumed)\b|"
    r"I(?:\s+have\s+(?:not|never)|\s+haven['’]t|['’]ve\s+(?:not|never))\s+"
    r"(?:(?:actually|currently|presently|even|ever)\s+)?"
    r"(?:taken|used|run|done|had|been\s+(?:taking|using|running|doing|on))\b|"
    r"I(?:\s+have|['’]ve)\s+"
    r"(?:actually|currently|presently|definitely|certainly|absolutely)\s+"
    r"(?:not|never)\s+"
    r"(?:(?:actually|currently|presently|even|ever)\s+)?"
    r"(?:taken|used|run|done|had|been\s+(?:taking|using|running|doing|on))\b|"
    r"I(?:\s+(?:have|had)|['’](?:ve|d))?\s+never\s+"
    r"(?:(?:actually|even|ever)\s+)?"
    r"(?:take|took|taken|use|used|run|ran|done|had|start|started|begun)\b|"
    r"I\s+(?:was\s+not|wasn['’]t|was\s+never)\s+"
    r"(?:(?:currently|actually)\s+)?(?:taking|using|running|doing|on)\b|"
    r"I\s+(?:was\s+prescribed|got\s+(?:a\s+)?prescription\s+for)\b"
    r"[^.!?\n]{0,160}?(?:\b(?:but|and)\b|[,;])\s+"
    r"(?:(?:I\s+)?(?:never|did\s+not|didn['’]t|have\s+not|haven['’]t)|"
    r"I['’]ve\s+(?:not|never))\s+"
    r"(?:(?:actually|even|ever)\s+)?"
    r"(?:take|took|taken|use|used|start|started|fill|filled)\b|"
    r"I(?:['’]m|\s+am)\s+"
    r"(?:(?:actually|currently|presently|definitely|certainly|absolutely)\s+)?"
    r"not\s+"
    r"(?:(?:actually|currently|presently|definitely|certainly|absolutely)\s+)?"
    r"(?:taking|using|running|doing|on)\b|"
    r"I\s+(?:currently\s+)?(?:take|use)\s+no\b|"
    r"I(?:['’]m|\s+am)\s+(?:currently\s+)?"
    r"(?:taking|using|on)\s+(?:no\b|a\s+break\b)|"
    r"I\s+(?:have\s+)?(?:stopped|discontinued|quit|ceased|paused|halted|suspended|"
    r"restarted|resumed)\b|"
    r"I(?:['’]m|\s+am)\s+(?:currently\s+)?off\b|"
    r"I(?:['’]m|\s+am)\s+no\s+longer\b|"
    r"I(?:['’]m|\s+am)\s+back\s+on\b|"
    r"I\s+no\s+longer\b|"
    r"(?:glad|happy|relieved)\s+to\s+be\s+off\b|"
    r"where\s+did\s+that\s+come\s+from"
    r")",
    re.IGNORECASE,
)
_INCIDENTAL_STOP_RE = re.compile(
    r"\bI\s+(?:have\s+)?stopped\s+(?:by\b|at\b|the\s+(?:car|vehicle)\b|"
    r"a\s+(?:car|vehicle)\b|an\s+(?:automobile|vehicle)\b)",
    re.IGNORECASE,
)
_EXPLICIT_CORRECTION_RE = re.compile(
    r"\b(?:falsely|false|deny|denied|refute(?:d|s|ing)?|mistaken|"
    r"where\s+did\s+that\s+come\s+from)\b",
    re.IGNORECASE,
)

_CONDITIONAL_RE = re.compile(r"\b(?:if|unless|provided\s+that|in\s+case)\b", re.I)
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:hypothetical(?:ly)?|imagine|suppose|would|could|might)\b",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend(?:ed|s|ing)?|suggest(?:ed|s|ing)?|should|ought\s+to|advis(?:e|ed|es|ing))\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|I\s+think|I\s+guess|unsure|uncertain)\b",
    re.IGNORECASE,
)


def infer_temporal_status(text: str) -> str:
    """V1 never emits a free-standing temporal-status assertion.

    The exact requester lane can contain multiple entities, clauses, quotes,
    negations, and reversals. A regex can find words such as ``stopped`` or
    ``currently`` but cannot prove which entity they govern. The lane itself
    therefore carries the temporal meaning; the structured status stays empty
    until a later schema has deterministic clause/entity binding.
    """
    _ = text
    return ""


def is_safety_critical_personal_evidence(text: str) -> bool:
    """Whether an exact requester lane must survive summary selection.

    The summarizer may choose which ordinary details deserve compression, but
    it must not be able to omit a direct correction or state transition and
    thereby leave an older, contradictory state as the only visible memory.
    This predicate only controls inclusion of the complete exact lane; it does
    not turn its keywords into a factual status.
    """
    value = (text or "").strip()
    if not value:
        return False
    if _EXPLICIT_CORRECTION_RE.search(value) and re.search(r"\bI\b", value, re.I):
        return True
    if _INCIDENTAL_STOP_RE.search(value):
        return False
    return _SAFETY_CRITICAL_PERSONAL_RE.search(value) is not None


def infer_modality(text: str) -> str:
    """Return a conservative speech modality evidenced by one exact quote."""
    value = text or ""
    if not value:
        return "uncertain"
    if "?" in value:
        return "question"
    if _CONDITIONAL_RE.search(value):
        return "conditional"
    if _HYPOTHETICAL_RE.search(value):
        return "hypothetical"
    if _RECOMMENDATION_RE.search(value):
        return "recommendation"
    if _UNSAFE_STATUS_CONTEXT_RE.search(value):
        return "uncertain"
    if _UNCERTAIN_RE.search(value):
        return "uncertain"
    return "asserted"


def statuses_conflict(model_status: str, evidenced_status: str) -> bool:
    """Whether two non-empty statuses make incompatible state assertions."""
    model = (model_status or "").strip().lower()
    evidenced = (evidenced_status or "").strip().lower()
    if not model or model == "unspecified" or not evidenced:
        return False
    return model != evidenced


def event_time_is_supported(event_time: str, excerpts: Iterable[str], session_date: str) -> bool:
    """V1 never promotes an unbound time expression into claim metadata.

    Literal containment is insufficient: ``August 22`` may describe an
    appointment in a different clause from ``I stopped last month``.  Until a
    deterministic clause/entity binding exists, the exact evidence carries the
    time expression and the structured ``event_time`` must remain empty.
    """
    value = (event_time or "").strip()
    if value:
        return False
    # Keep the parameters explicit because writers and readers share this
    # policy boundary and future schema versions may bind them differently.
    _ = excerpts
    _ = session_date
    return True


def structured_claim_fingerprint(claim: object) -> str:
    """Stable content/provenance key used only for deterministic deduplication."""
    sources = []
    for source in tuple(getattr(claim, "sources", ()) or ()):
        sources.append({
            "canonical_turn_id": str(getattr(source, "canonical_turn_id", "") or ""),
            "source_role": str(getattr(source, "source_role", "") or ""),
            "speaker_label": str(getattr(source, "speaker_label", "") or ""),
            "evidence_excerpt": str(getattr(source, "evidence_excerpt", "") or ""),
            "session_date": str(getattr(source, "session_date", "") or ""),
            "source_provenance_digest": str(
                getattr(source, "source_provenance_digest", "") or "",
            ),
        })
    payload = {
        "text": str(getattr(claim, "text", "") or ""),
        "claim_type": str(getattr(claim, "claim_type", "") or ""),
        "temporal_status": str(getattr(claim, "temporal_status", "") or ""),
        "modality": str(getattr(claim, "modality", "") or ""),
        "event_time": str(getattr(claim, "event_time", "") or ""),
        "sources": sources,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"vc-structured-summary-claim-v1\0" + encoded).hexdigest()


def structured_source_digest(
    records: Iterable[Mapping[str, object]],
    *,
    namespace: str = "segment",
) -> str:
    """Hash the exact source snapshot used to construct a structured summary.

    Callers provide control-plane values, including actor ids, only as digest
    input. The returned SHA-256 is an opaque staleness/idempotency proof; none
    of the source values are serialized into a model-facing envelope.
    """
    normalized: list[dict[str, object]] = []
    for record in records:
        normalized.append({
            "canonical_turn_id": str(record.get("canonical_turn_id", "") or ""),
            "source_role": str(record.get("source_role", "") or ""),
            "actor_id": str(record.get("actor_id", "") or ""),
            "speaker_label": str(record.get("speaker_label", "") or ""),
            "content_sha256": hashlib.sha256(
                str(record.get("content", "") or "").encode("utf-8"),
            ).hexdigest(),
            "session_date": str(record.get("session_date", "") or ""),
            "audience_conversation_id": str(
                record.get("audience_conversation_id", "") or "",
            ),
            "origin_channel_id": str(
                record.get("origin_channel_id", "") or "",
            ),
            "audience_attribution_version": (
                record.get("audience_attribution_version", 0)
                if type(record.get("audience_attribution_version", 0)) is int
                else 0
            ),
        })
    # Preserve caller order. Segment ``canonical_turn_ids`` are a physical
    # chronology used to place a newer correction ahead of older state; if the
    # digest canonicalized this list as a set, mutating only that order would
    # keep the digest valid while changing which claim reaches a bounded
    # SUMMARY prefix. All v1 callers construct this sequence deterministically.
    payload = json.dumps(
        {
            "namespace": namespace,
            "schema_version": 1,
            "records": normalized,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"vc-structured-summary-source-v1\0" + payload).hexdigest()


def structured_tag_claim_digest(
    claims: Iterable[object],
    source_canonical_turn_ids: Iterable[object],
) -> str:
    """Bind a tag envelope to claims and complete ordered source coverage.

    SUMMARY rendering is prefix-bounded, so order is security-relevant: a
    digest over an unordered set would let an older active claim move ahead of
    a newer correction without invalidating the envelope. The complete ordered
    ``source_canonical_turn_ids`` list is equally load-bearing: deleting or
    reordering a parent source must invalidate the compact envelope even when
    none of its selected claims changed. Position-prefixed records make both
    orders and cardinalities explicit in the authenticated payload.
    """
    fingerprints = [structured_claim_fingerprint(claim) for claim in claims]
    source_ids = list(source_canonical_turn_ids)
    if (
        not source_ids
        or any(type(source_id) is not str or not source_id for source_id in source_ids)
        or any(source_id != source_id.strip() for source_id in source_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        return ""
    return structured_source_digest(
        [
            {
                "canonical_turn_id": f"claim:{index:06d}:{fingerprint}",
                "source_role": "selected-claim",
                "content": fingerprint,
            }
            for index, fingerprint in enumerate(fingerprints)
        ] + [
            {
                "canonical_turn_id": f"source:{index:06d}:{source_id}",
                "source_role": "source-coverage",
                "content": source_id,
            }
            for index, source_id in enumerate(source_ids)
        ],
        namespace="tag-selection-and-sources",
    )


@dataclass(frozen=True)
class _ValidatedTagRollupSegment:
    """Immutable signature of one physically revalidated segment envelope."""

    ref: str
    schema_version: int
    source_mapping_complete: bool
    source_digest: str
    generation_model: str
    canonical_turn_ids: tuple[str, ...]
    claim_fingerprints: tuple[str, ...]


_TAG_ROLLUP_PROOF_TOKEN = object()


@dataclass(frozen=True, init=False)
class ValidatedTagRollupInputs:
    """Opaque admission proof for lower-layer structured tag inputs.

    A :class:`~virtual_context.core.compactor.DomainCompactor` has no store and
    therefore cannot authenticate a segment envelope by itself. Callers obtain
    this proof from :func:`validate_tag_rollup_inputs`, which rehydrates the
    exact physical rows and binds each admitted segment's source ids, digest,
    model, and ordered claim fingerprints. Reusing a proof after any
    structured field changes fails :meth:`admits`.
    """

    _segments: tuple[_ValidatedTagRollupSegment, ...]

    def __init__(
        self,
        segments: Iterable[_ValidatedTagRollupSegment] = (),
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _TAG_ROLLUP_PROOF_TOKEN:
            raise TypeError(
                "ValidatedTagRollupInputs must come from "
                "validate_tag_rollup_inputs()",
            )
        object.__setattr__(self, "_segments", tuple(segments))

    @property
    def segment_refs(self) -> frozenset[str]:
        """Refs admitted by the physical validation pass."""
        return frozenset(segment.ref for segment in self._segments)

    def admits(self, summary: object) -> bool:
        """Whether *summary* is byte-for-byte the envelope that was proved."""
        signature = _tag_rollup_segment_signature(summary)
        return signature is not None and signature in self._segments


def _value(item: object, key: str, default: object = None) -> object:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _row_for_canonical_id(
    rows_by_id: Mapping[object, object],
    *,
    conversation_id: str,
    canonical_turn_id: str,
) -> object | None:
    row = rows_by_id.get(canonical_turn_id)
    if row is None:
        row = rows_by_id.get((conversation_id, canonical_turn_id))
    return row


def _tag_rollup_segment_signature(
    summary: object,
) -> _ValidatedTagRollupSegment | None:
    ref = _value(summary, "ref", "")
    metadata = _value(summary, "metadata")
    structured = _value(metadata, "structured_summary")
    canonical_ids = _value(metadata, "canonical_turn_ids")
    schema_version = _value(structured, "schema_version", 0)
    if (
        type(ref) is not str
        or not ref
        or not isinstance(canonical_ids, list)
        or not canonical_ids
        or any(
            type(canonical_id) is not str
            or not canonical_id
            or canonical_id != canonical_id.strip()
            for canonical_id in canonical_ids
        )
        or len(set(canonical_ids)) != len(canonical_ids)
        or type(schema_version) is not int
        or type(_value(structured, "source_digest", "")) is not str
        or type(_value(structured, "generation_model", "")) is not str
    ):
        return None
    claims = _value(structured, "claims")
    if not isinstance(claims, tuple) or not claims:
        return None
    return _ValidatedTagRollupSegment(
        ref=ref,
        schema_version=schema_version,
        source_mapping_complete=(
            _value(metadata, "source_mapping_complete", False) is True
        ),
        source_digest=str(_value(structured, "source_digest", "")),
        generation_model=str(_value(structured, "generation_model", "")),
        canonical_turn_ids=tuple(canonical_ids),
        claim_fingerprints=tuple(
            structured_claim_fingerprint(claim) for claim in claims
        ),
    )


def _validated_tag_rollup_segment(
    summary: object,
    rows_by_id: Mapping[object, object],
    *,
    conversation_id: str,
) -> _ValidatedTagRollupSegment | None:
    """Authenticate one stored segment against its exact canonical rows."""
    # Local imports keep this low-level digest module independent of the types
    # module during import initialization.
    from ..types import (
        AUDIENCE_ATTRIBUTION_VERSION,
        STRUCTURED_SUMMARY_SCHEMA_VERSION,
        strict_structured_summary,
        structured_summary_to_dict,
    )
    from .summary_identity import human_label_collision_key, is_safe_human_label

    signature = _tag_rollup_segment_signature(summary)
    metadata = _value(summary, "metadata")
    structured = _value(metadata, "structured_summary")
    if (
        signature is None
        or _value(metadata, "source_mapping_complete", False) is not True
        or _value(structured, "schema_version", 0)
        != STRUCTURED_SUMMARY_SCHEMA_VERSION
        or not signature.source_digest
        or not signature.generation_model.strip()
        or strict_structured_summary(structured_summary_to_dict(structured))
        != structured
    ):
        return None

    candidates: list[tuple[object, str, str, str]] = []
    actors_by_label: dict[str, set[str]] = {}
    for canonical_id in signature.canonical_turn_ids:
        row = _row_for_canonical_id(
            rows_by_id,
            conversation_id=conversation_id,
            canonical_turn_id=canonical_id,
        )
        if (
            row is None
            or _value(row, "conversation_id", "") != conversation_id
            or _value(row, "canonical_turn_id", "") != canonical_id
        ):
            return None
        user_content = str(_value(row, "user_content", "") or "")
        assistant_content = str(_value(row, "assistant_content", "") or "")
        if not user_content.strip() and not assistant_content.strip():
            return None
        if not user_content.strip():
            continue
        audience = str(
            _value(row, "audience_conversation_id", "") or "",
        ).strip()
        attribution_version = _value(
            row, "audience_attribution_version", 0,
        )
        if (
            not audience
            or type(attribution_version) is not int
            or attribution_version != AUDIENCE_ATTRIBUTION_VERSION
        ):
            return None
        actor = str(_value(row, "sender_actor_id", "") or "").strip()
        label = str(_value(row, "sender", "") or "").strip()
        if actor and is_safe_human_label(label, actor):
            actors_by_label.setdefault(
                human_label_collision_key(label), set(),
            ).add(actor)
            candidates.append((row, actor, label, user_content.strip()))

    colliding_labels = {
        key for key, actors in actors_by_label.items() if len(actors) > 1
    }
    records: list[dict[str, object]] = []
    admissible_by_id: dict[str, dict[str, object]] = {}
    for row, actor, label, content in candidates:
        if human_label_collision_key(label) in colliding_labels:
            continue
        canonical_id = str(_value(row, "canonical_turn_id", "") or "")
        record = {
            "canonical_turn_id": canonical_id,
            "source_role": "requester",
            "actor_id": actor,
            "speaker_label": label,
            "content": content,
            "session_date": str(
                _value(row, "session_date", "") or "",
            ).strip(),
            "audience_conversation_id": str(
                _value(row, "audience_conversation_id", "") or "",
            ),
            "origin_channel_id": str(
                _value(row, "origin_channel_id", "") or "",
            ),
            "audience_attribution_version": _value(
                row, "audience_attribution_version", 0,
            ),
        }
        records.append(record)
        admissible_by_id[canonical_id] = record
    if (
        not records
        or structured_source_digest(records, namespace="segment")
        != signature.source_digest
    ):
        return None

    claimed_source_ids: set[str] = set()
    for claim in tuple(_value(structured, "claims", ()) or ()):
        sources = _value(claim, "sources")
        if (
            _value(claim, "claim_type", "") != "conversation"
            or not isinstance(sources, tuple)
            or len(sources) != 1
        ):
            return None
        source = sources[0]
        canonical_id = _value(source, "canonical_turn_id", "")
        if (
            type(canonical_id) is not str
            or canonical_id not in admissible_by_id
            or canonical_id in claimed_source_ids
            or _value(source, "source_role", "") != "requester"
        ):
            return None
        claimed_source_ids.add(canonical_id)
        record = admissible_by_id[canonical_id]
        evidence = str(record["content"])
        if (
            _value(claim, "text", "") != evidence
            or _value(source, "evidence_excerpt", "") != evidence
            or _value(source, "speaker_label", "")
            != record["speaker_label"]
            or _value(source, "session_date", "")
            != record["session_date"]
            or _value(source, "source_provenance_digest", "")
            != structured_source_provenance_digest(record)
            or _value(claim, "temporal_status", "")
            != infer_temporal_status(evidence)
            or _value(claim, "modality", "") != infer_modality(evidence)
            or not event_time_is_supported(
                str(_value(claim, "event_time", "") or ""),
                [evidence],
                str(record["session_date"]),
            )
        ):
            return None

    required_critical_ids = {
        canonical_id
        for canonical_id, record in admissible_by_id.items()
        if is_safety_critical_personal_evidence(str(record["content"]))
    }
    if not required_critical_ids.issubset(claimed_source_ids):
        return None
    expected_critical_prefix = [
        str(record["canonical_turn_id"])
        for record in reversed(records)
        if str(record["canonical_turn_id"]) in required_critical_ids
    ]
    actual_claim_order = [
        str(_value(tuple(_value(claim, "sources", ()) or ())[0],
                   "canonical_turn_id", ""))
        for claim in tuple(_value(structured, "claims", ()) or ())
    ]
    if actual_claim_order[:len(expected_critical_prefix)] != expected_critical_prefix:
        return None
    return signature


def validate_tag_rollup_inputs(
    summaries: Iterable[object],
    rows_by_id: Mapping[object, object],
    *,
    conversation_id: str,
) -> ValidatedTagRollupInputs:
    """Return an opaque proof containing only physically valid segments.

    Duplicate refs with divergent envelopes are rejected as a unit. Invalid
    source prose may still participate in the retrieval-only tag synopsis; it
    simply receives no authority to contribute model-visible claims.
    """
    admitted_by_ref: dict[str, _ValidatedTagRollupSegment] = {}
    conflicted_refs: set[str] = set()
    for summary in summaries:
        signature = _validated_tag_rollup_segment(
            summary, rows_by_id, conversation_id=conversation_id,
        )
        ref = str(_value(summary, "ref", "") or "")
        if signature is None or not ref or ref in conflicted_refs:
            continue
        prior = admitted_by_ref.get(ref)
        if prior is not None and prior != signature:
            admitted_by_ref.pop(ref, None)
            conflicted_refs.add(ref)
            continue
        admitted_by_ref[ref] = signature
    return ValidatedTagRollupInputs(
        admitted_by_ref.values(), _token=_TAG_ROLLUP_PROOF_TOKEN,
    )


def apply_tag_claim_safety_floor(
    pool: Iterable[object],
    selected: Iterable[object],
    *,
    limit: int = 16,
) -> tuple[object, ...]:
    """Make exact personal corrections mandatory without semantic guessing.

    Candidate pool order comes from physically revalidated segment envelopes;
    each segment source digest binds its canonical source order and the writer
    places mandatory lanes in reverse physical order. Every exact requester
    correction/transition recognized by
    :func:`is_safety_critical_personal_evidence` survives selection. A
    mandatory claim is placed ahead of ordinary model-selected context. No
    claim is suppressed merely because the same speaker has another status or
    topic; if corrections alone exceed the bound, the tag layer fails empty so
    callers can fall back to exact segment evidence.
    """
    materialized_pool = tuple(pool)
    materialized_selected = tuple(selected)
    if limit < 1:
        return ()

    pool_index: dict[str, int] = {}
    mandatory: list[object] = []
    mandatory_fingerprints: set[str] = set()
    for index, claim in enumerate(materialized_pool):
        fingerprint = structured_claim_fingerprint(claim)
        pool_index.setdefault(fingerprint, index)
        if (
            fingerprint not in mandatory_fingerprints
            and is_safety_critical_personal_evidence(
                str(getattr(claim, "text", "") or ""),
            )
        ):
            mandatory_fingerprints.add(fingerprint)
            mandatory.append(claim)

    # Never truncate a set of corrections. An overfull correction set has no
    # defensible compact prefix;
    # return no tag claims so the caller falls back to lower-layer exact
    # segment evidence.
    if len(mandatory) > limit:
        return ()

    selected_unique: list[object] = []
    selected_seen: set[str] = set()
    for claim in materialized_selected:
        fingerprint = structured_claim_fingerprint(claim)
        if fingerprint in selected_seen or fingerprint not in pool_index:
            continue
        selected_seen.add(fingerprint)
        selected_unique.append(claim)

    # Corrections always precede ordinary model-selected context in their
    # authenticated pool order. All mandatory evidence is retained atomically.
    ordered: list[object] = list(mandatory)
    emitted = set(mandatory_fingerprints)
    for selected_claim in selected_unique:
        selected_fingerprint = structured_claim_fingerprint(selected_claim)
        if selected_fingerprint not in emitted:
            emitted.add(selected_fingerprint)
            ordered.append(selected_claim)

    return tuple(ordered[:limit])


def structured_source_provenance_digest(record: Mapping[str, object]) -> str:
    """Bind one claim source to its exact physical actor and audience row.

    The digest is safe to persist inside a claim source because it is a
    one-way control-plane proof.  Readers recompute it from the currently
    hydrated canonical row; actor ids and audience ids never enter the model
    rendering.
    """
    return structured_source_digest([record], namespace="claim-source")
