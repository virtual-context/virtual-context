"""Audience-scoped speaker labels and model-visible attribution projection.

This module owns two read-side jobs:

* **Batch label resolution.** ``resolve_speaker_labels`` maps internal actor
  ids to audience-safe display labels. A label comes only from the most
  recent audience-admissible physical canonical USER row for that actor,
  with a deterministic physical tiebreak. Tenant-global
  ``actor_profiles.display_name`` is never consulted: a DM may have most
  recently refreshed it with a private nickname, so the profile name is not
  audience-safe. A missing scoped label stays empty rather than falling back
  anywhere.

* **Source-class projection.** The ``project_*`` helpers build the
  model-visible attribution fields for each source class. They are
  allowlists, not dataclass dumps: every projected dict is built key by key
  and the internal actor id never appears in any output, so a serializer
  that merges a projection cannot leak identity.

Projection is annotation only. It adds no speaker input, does not touch
candidate generation or ranking, and fails open on retrieval: a label that
cannot be resolved is simply empty. Identity fails closed: an ineligible or
absent request context disables every audience-derived annotation instead of
substituting the resolved owner.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from ..types import AUDIENCE_ATTRIBUTION_VERSION, SpeakerRetrievalContext

if TYPE_CHECKING:
    from ..types import Fact, SourceProvenance

logger = logging.getLogger(__name__)

# Engine-owned reserved assistant identity. Assistant-only results present
# this label; they never carry a human actor. The matching handle is emitted
# only when a future roster snapshot explicitly contains the reserved
# assistant handle, so before rosters exist it is always empty.
ASSISTANT_SPEAKER_LABEL = "assistant"

SPEAKER_SCOPE_MIXED = "mixed"
SPEAKER_SCOPE_UNKNOWN = "unknown"

ATTRIBUTION_BASIS_MODEL_ASSISTED = "model_assisted"
ATTRIBUTION_BASIS_ROLE_LOCAL = "role_local"
ATTRIBUTION_BASIS_UNATTRIBUTED = "unattributed"

_ROLE_LOCAL_HUMAN_ROLES = frozenset({"requester", "subject"})

# Any of these keys on a result marks it as already speaker-annotated, so
# the aggregate marker must not stamp over it.
_SPEAKER_ANNOTATION_KEYS = frozenset({
    "speaker_label",
    "speaker_handle",
    "speaker_actor_known",
    "speaker_verified",
    "claimed_speaker_label",
    "speakers",
    "speaker_scope",
    "source_role",
    "attribution_basis",
})

# Stateless exposure: structural fields survive, human labels are blanked,
# claims and membership/verification flags are removed entirely.
_STATELESS_BLANKED_KEYS = ("speaker_label", "speaker_handle")
_STATELESS_DROPPED_KEYS = (
    "speaker_actor_known",
    "speaker_verified",
    "claimed_speaker_label",
    "speakers",
)

# How many most-recent physical rows one batch resolution may scan. Bounded
# so annotation cannot turn into a full-table read; an actor whose latest
# admissible row is older than the window simply keeps an empty label.
_DEFAULT_LABEL_SCAN_LIMIT = 400


def annotation_speaker_context(
    config: object,
    speaker_context: SpeakerRetrievalContext | None,
) -> SpeakerRetrievalContext | None:
    """Gate router for annotation at serialization boundaries.

    Returns the context unchanged only when ``speaker_annotations_enabled``
    is on AND the request proved a pre-alias audience. Anything else returns
    ``None``, which disables every newly speaker-aware annotation for the
    request — an unproved audience is never repaired to the resolved owner,
    and gate-off output stays byte-identical to the legacy serialization.
    """
    if speaker_context is None:
        return None
    search_config = getattr(config, "search", None)
    if not bool(getattr(search_config, "speaker_annotations_enabled", False)):
        return None
    if not getattr(speaker_context, "eligible", False):
        return None
    return speaker_context


def resolve_speaker_labels(
    store: object,
    actor_ids: Iterable[str],
    *,
    speaker_context: SpeakerRetrievalContext | None,
    scan_limit: int = _DEFAULT_LABEL_SCAN_LIMIT,
) -> dict[str, str]:
    """Batch audience-scoped display labels for internal actor ids.

    Each actor's label comes from its most recent audience-admissible
    physical canonical user row, decided by ``(sort_key,
    canonical_turn_id)`` so ties break on the physical row id. A row is
    admissible only when it is a human user lane with that
    ``sender_actor_id``, its validated pre-alias audience exactly equals the
    request audience, its audience attribution version is current, and —
    when the request carries a durable channel — its stored channel matches
    exactly (an empty stored channel fails closed).

    The scan reads recent rows under the alias-resolved owner and never
    consults tenant-global actor profiles. Failures and misses fail open to
    an empty label; they never widen scope or fall back to another source.
    The returned dict contains only actors that resolved a non-empty label,
    so ``labels.get(actor, "")`` is the read idiom.
    """
    wanted = {actor for actor in (actor_ids or ()) if actor}
    if not wanted:
        return {}
    if speaker_context is None or not getattr(speaker_context, "eligible", False):
        return {}
    owner = speaker_context.owner_conversation_id or ""
    if not owner:
        return {}

    try:
        rows = store.get_recent_canonical_turns(owner, limit=int(scan_limit))
    except Exception:
        logger.debug(
            "speaker label scan failed; labels stay empty", exc_info=True,
        )
        return {}

    audience = speaker_context.audience_conversation_id
    channel = speaker_context.audience_channel_id or ""
    # actor -> ((sort_key, canonical_turn_id), label-of-that-row)
    best: dict[str, tuple[tuple[float, str], str]] = {}
    try:
        for row in rows:
            actor = getattr(row, "sender_actor_id", "") or ""
            if actor not in wanted:
                continue
            if not (getattr(row, "user_content", "") or "").strip():
                # Not a human user lane; an assistant row never supplies a
                # human label even if a stored actor id survives on it.
                continue
            if (getattr(row, "audience_conversation_id", "") or "") != audience:
                continue
            version = int(getattr(row, "audience_attribution_version", 0) or 0)
            if version != AUDIENCE_ATTRIBUTION_VERSION:
                continue
            if channel:
                row_channel = getattr(row, "origin_channel_id", "") or ""
                if row_channel != channel:
                    # Includes the empty stored channel: fails closed.
                    continue
            key = (
                float(getattr(row, "sort_key", 0.0) or 0.0),
                getattr(row, "canonical_turn_id", "") or "",
            )
            current = best.get(actor)
            if current is None or key > current[0]:
                best[actor] = (key, (getattr(row, "sender", "") or "").strip())
    except Exception:
        logger.debug(
            "speaker label admission failed; labels stay empty", exc_info=True,
        )
        return {}

    return {actor: label for actor, (_, label) in best.items() if label}


def fact_attribution_basis(fact: "Fact") -> str:
    """Attribution class for one fact, from its persisted author fields only.

    Version 1 is model-assisted by definition. Version 2 is role-local only
    when the lane association is a human requester/subject lane AND the
    durable author actor is non-empty; an empty actor stays honestly
    unattributed. ``Fact.who`` and ``Fact.subject`` are content dimensions
    and never participate.
    """
    version = int(getattr(fact, "author_attribution_version", 0) or 0)
    if version <= 0:
        return ATTRIBUTION_BASIS_UNATTRIBUTED
    if version == 1:
        return ATTRIBUTION_BASIS_MODEL_ASSISTED
    role = getattr(fact, "author_source_role", "") or ""
    actor = getattr(fact, "author_actor_id", "") or ""
    if role in _ROLE_LOCAL_HUMAN_ROLES and actor:
        return ATTRIBUTION_BASIS_ROLE_LOCAL
    return ATTRIBUTION_BASIS_UNATTRIBUTED


def fact_author_actor_id(fact: "Fact") -> str:
    """Internal author actor, only when the fact's basis is role-local."""
    if fact_attribution_basis(fact) != ATTRIBUTION_BASIS_ROLE_LOCAL:
        return ""
    return getattr(fact, "author_actor_id", "") or ""


def collect_fact_author_actor_ids(facts: Iterable["Fact"]) -> set[str]:
    """Actor ids needing label resolution for a batch of fact hits."""
    return {
        actor
        for actor in (fact_author_actor_id(f) for f in (facts or ()))
        if actor
    }


def project_fact_speaker_fields(
    fact: "Fact",
    labels: dict[str, str] | None,
) -> dict[str, object]:
    """Model-visible attribution for one fact hit.

    Every projected fact discloses its numeric attribution version and its
    basis; the structural lane rides along as ``source_role`` when stamped.
    Singular speaker fields appear only when the basis is role-local: the
    audience-scoped label (possibly empty), an always-empty handle until a
    roster exists, and the known/verified flags. Model-assisted and
    unattributed facts never receive singular speaker fields, so a version-1
    authorship guess can never read as verified quote provenance.
    """
    basis = fact_attribution_basis(fact)
    fields: dict[str, object] = {
        "author_attribution_version": int(
            getattr(fact, "author_attribution_version", 0) or 0
        ),
        "attribution_basis": basis,
    }
    role = getattr(fact, "author_source_role", "") or ""
    if role:
        fields["source_role"] = role
    if basis == ATTRIBUTION_BASIS_ROLE_LOCAL:
        actor = getattr(fact, "author_actor_id", "") or ""
        fields["speaker_label"] = (labels or {}).get(actor, "")
        fields["speaker_handle"] = ""
        fields["speaker_actor_known"] = True
        fields["speaker_verified"] = True
    return fields


def project_quote_speaker_fields(
    provenance: "SourceProvenance | None",
    labels: dict[str, str] | None,
    handles: dict[str, str] | None = None,
) -> dict[str, object]:
    """Model-visible speaker annotation for one quote-like result.

    Uses only the physical role-local provenance projected at candidate
    construction. A requester lane speaks through ``sender_actor_id``, a
    subject lane through ``reply_subject_actor_id``, and an assistant lane
    through the reserved engine identity. An unresolved subject keeps empty
    singular fields; its raw stored reply label surfaces only as
    ``claimed_speaker_label`` with ``speaker_verified=false``. Mixed and
    unattributed text is never assigned one human speaker — it exposes a
    ``speaker_scope`` instead. A result with no provenance gets nothing.

    ``handles`` maps internal actor ids to this request's immutable roster
    snapshot handles. An actor absent from the map — outside the capped
    snapshot, no snapshot bound, or roster gate off — keeps its scoped
    label and known-actor flag but an empty handle; projection never
    mints, looks up, or reveals a hidden assignment.
    """
    if provenance is None:
        return {}
    role = getattr(provenance, "source_role", "") or "unattributed"
    fields: dict[str, object] = {"source_role": role}
    if role in _ROLE_LOCAL_HUMAN_ROLES:
        actor = getattr(provenance, "actor_id", "") or ""
        if actor:
            fields["speaker_label"] = (labels or {}).get(actor, "")
            fields["speaker_handle"] = (handles or {}).get(actor, "")
            fields["speaker_actor_known"] = True
            fields["speaker_verified"] = True
        else:
            fields["speaker_label"] = ""
            fields["speaker_handle"] = ""
            fields["speaker_actor_known"] = False
            fields["speaker_verified"] = False
            if role == "subject":
                claimed = getattr(provenance, "claimed_subject_label", "") or ""
                if claimed:
                    fields["claimed_speaker_label"] = claimed
    elif role == "assistant":
        fields["speaker_label"] = ASSISTANT_SPEAKER_LABEL
        fields["speaker_handle"] = ""
        fields["speaker_actor_known"] = False
        fields["speaker_verified"] = True
    elif role == "mixed":
        fields["speaker_scope"] = SPEAKER_SCOPE_MIXED
    else:
        fields["speaker_scope"] = SPEAKER_SCOPE_UNKNOWN
    return fields


def collect_quote_actor_ids(
    provenances: Iterable["SourceProvenance | None"],
) -> set[str]:
    """Actor ids needing label resolution for a batch of quote results."""
    wanted: set[str] = set()
    for provenance in provenances or ():
        if provenance is None:
            continue
        role = getattr(provenance, "source_role", "") or ""
        if role not in _ROLE_LOCAL_HUMAN_ROLES:
            continue
        actor = getattr(provenance, "actor_id", "") or ""
        if actor:
            wanted.add(actor)
    return wanted


def annotate_aggregate_entry(
    entry: object,
    scope: str = SPEAKER_SCOPE_UNKNOWN,
) -> object:
    """Mark one aggregate/tool-output result with an honest speaker scope.

    Aggregates that span sources never infer one speaker from the newest
    row, the session owner, or the requester; they disclose a scope
    instead. Idempotent and non-destructive: an entry that already carries
    any speaker annotation is left exactly as its projector produced it.
    """
    if not isinstance(entry, dict):
        return entry
    if any(key in entry for key in _SPEAKER_ANNOTATION_KEYS):
        return entry
    entry["speaker_scope"] = scope
    return entry


def strip_to_structural_speaker_fields(payload: object) -> object:
    """Reduce speaker annotation to the structural subset, in place.

    Stateless callers with no validated audience (for example MCP) may
    expose only structural attribution: ``source_role``, ``speaker_scope``,
    and the fact attribution version/basis. Human display labels are
    blanked, and claims, membership flags, and verification flags are
    removed, so no audience-derived identity can surface through a caller
    that never proved an audience.
    """
    if isinstance(payload, dict):
        for key in _STATELESS_BLANKED_KEYS:
            if key in payload:
                payload[key] = ""
        for key in _STATELESS_DROPPED_KEYS:
            payload.pop(key, None)
        for value in payload.values():
            strip_to_structural_speaker_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            strip_to_structural_speaker_fields(item)
    return payload
