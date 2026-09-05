"""Deterministic speaker-safety policy for model-visible summary prose.

Stored summaries are derived text, not quote provenance.  Older rows may call a
named participant ``the user`` even when a segment contains several people.
This module supplies a secondary lexical detector and the read-side provenance
gate used before that prose is shown to a model.  Absence of a suspicious word
is never treated as ownership proof: subjectless and passive summaries are
common, so an audience-scoped canonical-row mapping is the primary control.

The gate never guesses an owner and never treats a label as semantic proof.
Free-form generated prose is ranking-only. Versioned structured summaries may
select and classify exact excerpts, but presentation rebuilds their claim text
from the admitted excerpt and independently checks identity, role, modality,
time, and source integrity. Legacy rows fall back to exact human source text;
FULL depth reconstructs the exact role-separated canonical transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .rendered_memory import MemorySourceVersion
import json
import re
import unicodedata
import uuid
from typing import Iterable, TYPE_CHECKING

from ..types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS,
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
)
from .structured_summary import (
    event_time_is_supported,
    infer_modality,
    infer_temporal_status,
    is_safety_critical_personal_evidence,
    statuses_conflict,
    structured_claim_fingerprint,
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
)

if TYPE_CHECKING:
    from ..types import SpeakerRetrievalContext


# Possessives are always person referents. The non-possessive form excludes
# concrete software grammar, not broad words such as ``experience``,
# ``profile``, or ``data``: in a health summary those are precisely the words
# that carry a personal claim.
_GENERIC_HUMAN_NOUN = (
    r"(?:user|member|person|patient|client|participant|requester|speaker|"
    r"someone|individual|customer)"
)
_GENERIC_HUMAN_LABELS = frozenset({
    "user", "member", "person", "patient", "client", "participant",
    "requester", "speaker", "someone", "individual", "customer",
    "assistant",
})
_PERSONAL_PRONOUN_LABELS = frozenset({
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themself", "themselves",
})
_GENERIC_POSSESSIVE_RE = re.compile(
    rf"\b(?:(?:the|a|an|this|that)\s+)?{_GENERIC_HUMAN_NOUN}[\'\u2019]s\b",
    re.IGNORECASE,
)
_GENERIC_DETERMINED_RE = re.compile(
    # Permit bounded descriptors (``a Discord member`` / ``the historical
    # user``).  The post-noun exclusions keep ordinary technical compounds
    # such as ``a database user interface`` out of the policy.
    rf"\b(?:the|a|an|this|that)\s+(?:[a-z0-9_-]+\s+){{0,3}}?"
    rf"{_GENERIC_HUMAN_NOUN}\b"
    r"(?!-facing\b)"
    r"(?!\s+(?:interface|account|guide|manual|input|settings|request|message|"
    r"query|table|record|model|prompt|context|feedback|journey|flow|function|"
    r"method|class|library|sdk|object|type|field|column|property|schema|"
    r"endpoint|api)\b)",
    re.IGNORECASE,
)
_GENERIC_BARE_HUMAN_RE = re.compile(
    rf"\b{_GENERIC_HUMAN_NOUN}(?:[\'\u2019]s|\b(?=\s*(?:(?:\([^\n)]{{1,32}}\)\s*)?:|(?:is|was|has|had|"
    r"does|did|will|would|can|could|should|wants?|needs?|prefers?|plans?|"
    r"reports?|experiences?|takes?|uses?|stopped?|started?|said|says|asked|"
    r"asks|requested|shared|mentioned|discussed|noted|described|disclosed|"
    r"specified|stated|indicated|confirmed|selected|chose|switched|received|"
    r"implemented|configured|paid|bought|took|tried|felt|believes?|thinks?)\b)))",
    re.IGNORECASE,
)
_SUBJECT_PRONOUN_RE = re.compile(
    # Any-position on purpose: bullets and discourse prefixes (``Later, she``)
    # are normal summary grammar.  Contractions are detected separately and
    # fail closed because a mechanical rewrite cannot preserve their grammar.
    r"\b(?P<pronoun>i|we|you|he|she|they)\b(?=\s+[a-z])",
    re.IGNORECASE,
)
_POSSESSIVE_PRONOUN_RE = re.compile(
    r"\b(?P<pronoun>my|our|your|his|her|their)\b(?=\s+(?!"
    r"(?:api|interface|account|guide|manual|input|settings|request|message|"
    r"query|table|record|model|prompt|context|feedback|journey|flow|function|"
    r"method|class|library|sdk|object|type|field|column|property|schema|"
    r"endpoint)\b)[a-z0-9])",
    re.IGNORECASE,
)
_PERSONAL_PRONOUN_CONTRACTION_RE = re.compile(
    r"\b(?:i|we|you|he|she|they)[\'’](?:m|d|ll|re|ve|s)\b",
    re.IGNORECASE,
)
_TECHNICAL_BARE_HUMAN_RE = re.compile(
    # Dataclass/ORM prose often uses a capitalized type name as the grammatical
    # subject. These patterns are structural declarations, not people.
    r"\b(?:User|Member|Person|Client)\s+has\s+(?:(?:a|an|the)\s+)?"
    r"(?:(?:required|optional|nullable|indexed|string|integer|email)\s+)*"
    r"(?:field|column|property|attribute|schema)\b",
)
_INTERNAL_IDENTITY_LABEL_RE = re.compile(
    r"(?<![\w])(?:actor|sk|tenant|conversation|conv)\s*:", re.IGNORECASE,
)
_LABEL_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_COMMON_LABEL_EDGE_DECORATORS = " \t\r\n@!#<>()[]{}\"'`*_~.,;"
_DEFAULT_IGNORABLE_LABEL_RANGES = (
    (0x034F, 0x034F),  # combining grapheme joiner
    (0x115F, 0x1160),  # Hangul fillers
    (0x17B4, 0x17B5),  # Khmer inherent vowels
    (0x180B, 0x180F),  # Mongolian variation/free variation selectors
    (0x2060, 0x206F),  # word joiner and reserved format controls
    (0x3164, 0x3164),  # Hangul filler
    (0xFE00, 0xFE0F),  # variation selectors
    (0xFFA0, 0xFFA0),  # halfwidth Hangul filler
    (0xFFF0, 0xFFF8),  # unassigned default-ignorable code points
    (0x1BCA0, 0x1BCA3),  # shorthand format controls
    (0x1D173, 0x1D17A),  # musical-symbol format controls
    (0xE0000, 0xE0FFF),  # tags and variation-selector supplement
)


def _is_label_default_ignorable(character: str) -> bool:
    """Whether *character* may not create a display-label distinction."""
    if unicodedata.category(character) == "Cf":
        return True
    codepoint = ord(character)
    return any(
        start <= codepoint <= end
        for start, end in _DEFAULT_IGNORABLE_LABEL_RANGES
    )


def _normalized_label_policy_text(value: str) -> str:
    """Normalize display-label syntax without rewriting the rendered label."""
    compatible = unicodedata.normalize("NFKC", value or "")
    visible = "".join(
        character for character in compatible
        if not _is_label_default_ignorable(character)
    )
    return unicodedata.normalize("NFKC", visible).strip()


def human_label_collision_key(value: str) -> str:
    """Return the policy-normalized, caseless key for label ownership.

    Every boundary that decides whether two display labels identify distinct
    people must use this key rather than raw ``casefold()``. Compatibility
    forms and default-ignorable characters are presentation details, not
    identity distinctions.
    """
    return _normalized_label_policy_text(value).casefold()


def _contains_forbidden_human_label_token(value: str) -> bool:
    """Reject generic/pronoun label tokens despite cosmetic decoration.

    Source display labels are identifiers, not prose. A standalone ``He`` or
    ``User`` token therefore remains ambiguous when punctuation, mention
    syntax, or a parenthesized alias is added (for example ``He.`` or
    ``I (BigTex)``).
    """
    tokens = {
        match.group(0).casefold()
        for match in _LABEL_TOKEN_RE.finditer(
            _normalized_label_policy_text(value),
        )
    }
    return bool(tokens & (_GENERIC_HUMAN_LABELS | _PERSONAL_PRONOUN_LABELS))


def _looks_internal_identity_label(value: str, actor_id: str = "") -> bool:
    label = _normalized_label_policy_text(value)
    actor = _normalized_label_policy_text(actor_id)
    undecorated_label = label.strip(_COMMON_LABEL_EDGE_DECORATORS)
    return bool(
        not label
        or (actor and undecorated_label == actor)
        or _INTERNAL_IDENTITY_LABEL_RE.search(label)
    )


def is_safe_human_label(value: str, actor_id: str = "") -> bool:
    """Whether *value* is an explicit, non-internal human display label."""
    label = _normalized_label_policy_text(value)
    return bool(
        label
        and not _contains_forbidden_human_label_token(label)
        and not contains_ambiguous_human_referent(label)
        and not _looks_internal_identity_label(label, actor_id)
    )


SUMMARY_ATTRIBUTION_QUARANTINE = (
    "[summary withheld: speaker attribution is unresolved; retrieve exact "
    "source turns before making a person-specific claim]"
)

_SUMMARY_ATTRIBUTION_MARKER_RE = re.compile(
    r"</?(?:summary-attribution|historical-source-transcript|"
    r"structured-summary|canonical-source-transcript)\b",
    re.IGNORECASE,
)
_SOURCE_TRANSCRIPT_OPEN = "<historical-source-transcript>\n"
_SOURCE_TRANSCRIPT_CLOSE = "\n</historical-source-transcript>"
_STRUCTURED_SUMMARY_OPEN = "<structured-summary>\n"
_STRUCTURED_SUMMARY_CLOSE = "\n</structured-summary>"
_FULL_TRANSCRIPT_OPEN = "<canonical-source-transcript>\n"
_FULL_TRANSCRIPT_CLOSE = "\n</canonical-source-transcript>"
_SOURCE_SPEAKER_REF_RE = re.compile(r"^historical_[0-9a-f]{16}$")
# Bounds apply to the complete serialized JSON payload.  Oversized source
# groups are withheld atomically; no lane or speaker binding is ever sliced.
_SOURCE_TRANSCRIPT_MAX_LANES = 12
_SOURCE_TRANSCRIPT_MAX_UTF8_BYTES = 16 * 1024
_FULL_TRANSCRIPT_MAX_TURNS = 128
_FULL_TRANSCRIPT_MAX_LANES = _FULL_TRANSCRIPT_MAX_TURNS * 2
_FULL_TRANSCRIPT_MAX_UTF8_BYTES = 256 * 1024
_STRUCTURED_SUMMARY_MAX_CLAIMS = 8
_STRUCTURED_TAG_MAX_SELECTED_CLAIMS = 16
_STRUCTURED_SEGMENTS_MAX_CLAIMS = 64
_STRUCTURED_RENDER_MAX_UTF8_BYTES = 64 * 1024
_STRUCTURED_CLAIM_TYPES = frozenset({
    "personal", "world", "decision", "action", "technical", "conversation",
})
_STRUCTURED_MODALITIES = frozenset({
    "asserted", "hypothetical", "conditional", "question",
    "recommendation", "uncertain",
})
_STRUCTURED_TEMPORAL_STATUSES = frozenset({
    "", "active", "completed", "ceased", "planned", "abandoned", "recurring",
})

_DERIVED_SOURCE_PROSE_FIELDS = frozenset({
    "summary",
    "description",
    "excerpt",
    "summary_text",
    # remember_when derives these from summary excerpts before returning its
    # payload; guarding only the original excerpt leaves the copy exposed.
    "point",
    "supporting_point",
    "points",
    "evidence",
})
_DERIVED_INSTRUCTION_FIELDS = frozenset({"reader_hint"})


_REQUEST_LOCAL_RENDERING_TOKEN = object()


class _RequestLocalSummaryRendering(str):
    """A rendering that survived canonical hydration in this process.

    The serialized envelope remains deliberately self-describing, but its
    syntax is not provenance.  This ephemeral subtype is retained only until
    the model-boundary sanitizer runs; JSON serialization (or any ordinary
    string reconstruction) drops the in-process proof, as it should.
    """

    def __new__(
        cls,
        value: str,
        *,
        speaker_context: "SpeakerRetrievalContext",
        _token: object,
        source_versions: tuple = (),
        presented_source_ids: tuple[str, ...] = (),
        evidence_kind: str = "",
    ) -> "_RequestLocalSummaryRendering":
        if _token is not _REQUEST_LOCAL_RENDERING_TOKEN:
            raise TypeError("request-local summary renderings are internal")
        rendered = str.__new__(cls, value)
        rendered._speaker_context = speaker_context
        rendered._source_versions = source_versions
        rendered._presented_source_ids = presented_source_ids
        rendered._evidence_kind = evidence_kind
        return rendered

    def __copy__(self) -> "_RequestLocalSummaryRendering":
        return self

    def __deepcopy__(self, memo: dict) -> "_RequestLocalSummaryRendering":
        memo[id(self)] = self
        return self


def _mark_request_local_summary_rendering(
    text: str,
    *,
    speaker_context: "SpeakerRetrievalContext",
    source_versions: tuple = (),
    presented_source_ids: tuple[str, ...] = (),
    evidence_kind: str = "",
) -> str:
    """Carry successful exact-row proof to the next model boundary."""
    if (
        not getattr(speaker_context, "eligible", False)
        or not is_proved_summary_rendering(text)
    ):
        return SUMMARY_ATTRIBUTION_QUARANTINE
    return _RequestLocalSummaryRendering(
        text,
        speaker_context=speaker_context,
        _token=_REQUEST_LOCAL_RENDERING_TOKEN,
        source_versions=source_versions,
        presented_source_ids=presented_source_ids,
        evidence_kind=evidence_kind,
    )


def _is_request_local_summary_rendering(
    text: object,
    *,
    speaker_context: "SpeakerRetrievalContext | None",
) -> bool:
    """Require both the ephemeral proof carrier and a still-valid envelope."""
    return bool(
        type(text) is _RequestLocalSummaryRendering
        and speaker_context is not None
        and getattr(speaker_context, "eligible", False)
        and getattr(text, "_speaker_context", None) is speaker_context
        and is_proved_summary_rendering(text)
    )


def contains_ambiguous_human_referent(text: object) -> bool:
    """Whether *text* contains a generic singular human-speaker referent.

    The detector is intentionally narrow.  It targets the production failure
    shape without treating ordinary technical compounds (``user interface``,
    ``users table``) as identity claims.
    """
    if not isinstance(text, str) or not text:
        return False
    scrubbed = _TECHNICAL_BARE_HUMAN_RE.sub("", text)
    return bool(
        _GENERIC_POSSESSIVE_RE.search(scrubbed)
        or _GENERIC_DETERMINED_RE.search(scrubbed)
        or _GENERIC_BARE_HUMAN_RE.search(scrubbed)
        or _SUBJECT_PRONOUN_RE.search(scrubbed)
        or _POSSESSIVE_PRONOUN_RE.search(scrubbed)
        or _PERSONAL_PRONOUN_CONTRACTION_RE.search(scrubbed)
    )


def _decode_rendered_envelope(
    text: object,
    opening: str,
    closing: str,
    max_bytes: int,
) -> dict | None:
    """Decode one internally constructed, exactly bounded JSON envelope."""
    if not isinstance(text, str) or not text.startswith(opening):
        return None
    remainder = text[len(opening):]
    if remainder.count(closing) != 1:
        return None
    payload_text, trailing = remainder.split(closing, 1)
    if (
        trailing.strip()
        or len(payload_text.encode("utf-8")) > max_bytes
        or _SUMMARY_ATTRIBUTION_MARKER_RE.search(payload_text)
    ):
        return None
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _valid_rendered_source(
    source: object,
    *,
    allow_assistant: bool = False,
) -> bool:
    if not isinstance(source, dict) or set(source) != {
        "display_name", "role", "evidence_excerpt", "session_date",
        "current_requester_match",
    }:
        return False
    role = source.get("role")
    display_name = source.get("display_name")
    requester_match = source.get("current_requester_match")
    if role == "historical_human":
        if not isinstance(display_name, str) or not is_safe_human_label(display_name):
            return False
        if requester_match not in {
            "proved_same", "proved_different", "unproved",
        }:
            return False
    elif role == "historical_assistant":
        if (
            not allow_assistant
            or display_name != "Assistant"
            or requester_match != "not_applicable"
        ):
            return False
    else:
        return False
    excerpt = source.get("evidence_excerpt")
    session_date = source.get("session_date")
    return bool(
        isinstance(excerpt, str)
        and excerpt.strip()
        and len(excerpt) <= 800
        and isinstance(session_date, str)
        and len(session_date) <= 128
    )


def _is_proved_structured_rendering(text: object) -> bool:
    payload = _decode_rendered_envelope(
        text,
        _STRUCTURED_SUMMARY_OPEN,
        _STRUCTURED_SUMMARY_CLOSE,
        _STRUCTURED_RENDER_MAX_UTF8_BYTES,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "source", "depth", "claims",
    }:
        return False
    depth = payload.get("depth")
    if payload.get("source") != "structured_summary_v1" or depth not in {
        "summary", "segments",
    }:
        return False
    claims = payload.get("claims")
    if (
        not isinstance(claims, list)
        or not claims
        or len(claims) > (
            _STRUCTURED_SUMMARY_MAX_CLAIMS
            if depth == "summary"
            else _STRUCTURED_SEGMENTS_MAX_CLAIMS
        )
    ):
        return False
    for claim in claims:
        if not isinstance(claim, dict):
            return False
        common = {
            "text", "claim_type", "temporal_status", "modality",
        }
        expected = (
            common | {
                "display_name", "role", "evidence_excerpt", "session_date",
                "current_requester_match",
            }
            if depth == "summary"
            else common | {"event_time", "sources"}
        )
        if set(claim) != expected:
            return False
        claim_text = claim.get("text")
        if (
            not isinstance(claim_text, str)
            or not claim_text.strip()
            or len(claim_text) > 1000
            or _INTERNAL_IDENTITY_LABEL_RE.search(claim_text)
        ):
            return False
        if claim.get("claim_type") not in _STRUCTURED_CLAIM_TYPES:
            return False
        if claim.get("temporal_status") not in _STRUCTURED_TEMPORAL_STATUSES:
            return False
        if claim.get("modality") not in _STRUCTURED_MODALITIES:
            return False
        if depth == "summary":
            if not _valid_rendered_source({
                "display_name": claim.get("display_name"),
                "role": claim.get("role"),
                "evidence_excerpt": claim.get("evidence_excerpt"),
                "session_date": claim.get("session_date"),
                "current_requester_match": claim.get("current_requester_match"),
            }):
                return False
            if claim_text != claim.get("evidence_excerpt"):
                return False
        else:
            event_time = claim.get("event_time")
            sources = claim.get("sources")
            if (
                not isinstance(event_time, str)
                or len(event_time) > 128
                or not isinstance(sources, list)
                or len(sources) != 1
                or not all(_valid_rendered_source(source) for source in sources)
            ):
                return False
            if claim_text not in {
                source.get("evidence_excerpt") for source in sources
            }:
                return False
    return True


def _is_proved_full_rendering(text: object) -> bool:
    payload = _decode_rendered_envelope(
        text,
        _FULL_TRANSCRIPT_OPEN,
        _FULL_TRANSCRIPT_CLOSE,
        _FULL_TRANSCRIPT_MAX_UTF8_BYTES,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "source", "depth", "lanes",
    }:
        return False
    if payload.get("source") != "canonical_turns" or payload.get("depth") != "full":
        return False
    lanes = payload.get("lanes")
    if (
        not isinstance(lanes, list)
        or not lanes
        or len(lanes) > _FULL_TRANSCRIPT_MAX_LANES
    ):
        return False
    for lane in lanes:
        if not isinstance(lane, dict) or set(lane) != {
            "display_name", "role", "content", "session_date",
            "current_requester_match",
        }:
            return False
        source = {
            "display_name": lane.get("display_name"),
            "role": lane.get("role"),
            "evidence_excerpt": lane.get("content"),
            "session_date": lane.get("session_date"),
            "current_requester_match": lane.get("current_requester_match"),
        }
        # Full source lanes are exact and may exceed the summary-excerpt bound.
        content = source["evidence_excerpt"]
        source["evidence_excerpt"] = str(content)[:800] if isinstance(content, str) else content
        if (
            not isinstance(content, str)
            or not content.strip()
            or not _valid_rendered_source(source, allow_assistant=True)
        ):
            return False
    return True


def _is_proved_legacy_source_rendering(text: object) -> bool:
    payload = _decode_rendered_envelope(
        text,
        _SOURCE_TRANSCRIPT_OPEN,
        _SOURCE_TRANSCRIPT_CLOSE,
        _SOURCE_TRANSCRIPT_MAX_UTF8_BYTES,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "source", "generated_summary_prose_used", "lanes",
    }:
        return False
    if payload.get("source") != "canonical_turns":
        return False
    if payload.get("generated_summary_prose_used") is not False:
        return False
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        return False
    if len(lanes) > _SOURCE_TRANSCRIPT_MAX_LANES:
        return False
    ref_names: dict[str, str] = {}
    for lane in lanes:
        if not isinstance(lane, dict) or set(lane) != {
            "source_speaker_ref", "display_name", "role", "content", "session_date",
            "current_requester_match",
        }:
            return False
        role = lane.get("role")
        source_speaker_ref = lane.get("source_speaker_ref")
        display_name = lane.get("display_name")
        content = lane.get("content")
        session_date = lane.get("session_date")
        requester_match = lane.get("current_requester_match")
        if role != "historical_human":
            return False
        if (
            not isinstance(source_speaker_ref, str)
            or _SOURCE_SPEAKER_REF_RE.fullmatch(source_speaker_ref) is None
        ):
            return False
        if not isinstance(display_name, str) or not is_safe_human_label(display_name):
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        if not isinstance(session_date, str):
            return False
        if requester_match not in {
            "proved_same", "proved_different", "unproved",
        }:
            return False
        prior_name = ref_names.setdefault(source_speaker_ref, display_name)
        if prior_name != display_name:
            return False
    return True


def is_proved_summary_rendering(text: object) -> bool:
    """Validate one internally constructed model-facing memory envelope.

    Syntax is not source authority.  Callers may preserve a valid envelope only
    on the request-local path that just hydrated and checked its canonical rows.
    This parser is the final shape/escaping guard shared by all three depth
    renderers.
    """
    return bool(
        _is_proved_legacy_source_rendering(text)
        or _is_proved_structured_rendering(text)
        or _is_proved_full_rendering(text)
    )


@dataclass(frozen=True)
class HistoricalSourceLane:
    """One exact physical source lane safe to project to a model."""

    speaker: str
    role: str
    content: str
    session_date: str = ""
    requester_match: str = "unproved"
    actor_id: str = field(default="", repr=False)
    canonical_turn_id: str = field(default="", repr=False)


@dataclass(frozen=True)
class SummarySourceProjection:
    """Validated canonical replacement for one generated summary."""

    lanes: tuple[HistoricalSourceLane, ...] = ()
    complete: bool = False


def _projection_contains_internal_identity(
    projection: SummarySourceProjection,
) -> bool:
    """Whether exact projection text would disclose control-plane identity.

    The decision is projection-atomic.  Rewriting or slicing an exact lane
    would destroy its provenance semantics, so one internal identity marker or
    one admitted human actor id quarantines the complete projection.  Actor ids
    are collected before scanning so an assistant lane cannot echo a human
    lane's internal id into FULL output.
    """
    actor_ids = {
        lane.actor_id.strip()
        for lane in projection.lanes
        if lane.role == "historical_human" and lane.actor_id.strip()
    }
    return any(
        _content_contains_internal_identity(
            lane.content,
            admitted_actor_ids=actor_ids,
        )
        for lane in projection.lanes
    )


def _content_contains_internal_identity(
    content: object,
    *,
    admitted_actor_ids: Iterable[object],
) -> bool:
    """Detect control-plane identity after display-policy normalization."""
    if type(content) is not str:
        return False
    normalized_content = _normalized_label_policy_text(content)
    normalized_actor_ids = {
        _normalized_label_policy_text(actor_id)
        for actor_id in admitted_actor_ids
        if type(actor_id) is str and _normalized_label_policy_text(actor_id)
    }
    return bool(
        _INTERNAL_IDENTITY_LABEL_RE.search(normalized_content)
        or any(actor_id in normalized_content for actor_id in normalized_actor_ids)
    )


def structured_claims_contain_internal_identity(
    claims: Iterable[object],
    *,
    admitted_actor_ids: Iterable[object],
) -> bool:
    """Whether any structured evidence exposes an admitted internal identity.

    This predicate is artifact-atomic: callers must reject the complete
    structured envelope rather than deleting, rewriting, or slicing one claim.
    It is public so historical migration applies the same content rule as the
    runtime reader before classifying or persisting a v1 envelope.
    """
    actor_ids = tuple(admitted_actor_ids)
    for claim in claims:
        sources = getattr(claim, "sources", ())
        if not isinstance(sources, (tuple, list)):
            continue
        for source in sources:
            if _content_contains_internal_identity(
                getattr(source, "evidence_excerpt", None),
                admitted_actor_ids=actor_ids,
            ):
                return True
    return False


@dataclass(frozen=True)
class ValidatedClaimSource:
    """One claim-local source after exact-row validation."""

    display_name: str
    role: str
    evidence_excerpt: str
    session_date: str = ""
    requester_match: str = "unproved"
    actor_id: str = field(default="", repr=False)
    canonical_turn_id: str = field(default="", repr=False)


@dataclass(frozen=True)
class ValidatedSummaryClaim:
    """A generated claim whose identity, status, and evidence all survived."""

    text: str
    claim_type: str
    temporal_status: str
    modality: str
    event_time: str = ""
    sources: tuple[ValidatedClaimSource, ...] = ()


@dataclass(frozen=True)
class SummarySpeakerAttribution:
    """Internal proof used to present one stored summary safely.

    Actor ids are deliberately excluded from repr and never serialized by the
    renderer. ``label`` comes from an exact admitted canonical source row; it
    is never borrowed from a newer row or another channel.
    """

    actor_ids: frozenset[str] = field(default_factory=frozenset, repr=False)
    label: str = ""
    complete: bool = False
    requester_match: bool | None = field(default=None, repr=False)

    @property
    def is_proved_single_human(self) -> bool:
        label = self.label.strip()
        return bool(
            self.complete
            and len(self.actor_ids) == 1
            and is_safe_human_label(label)
        )


def _metadata_for(item: object) -> object | None:
    return getattr(item, "metadata", None)


def _context_owns_conversation(
    speaker_context: "SpeakerRetrievalContext | None",
    conversation_id: str,
) -> bool:
    expected_owner = str(
        getattr(speaker_context, "owner_conversation_id", "") or "",
    ).strip()
    return bool(expected_owner and expected_owner == str(conversation_id or "").strip())


def _structured_summary_for(item: object) -> object | None:
    direct = getattr(item, "structured_summary", None)
    if direct is not None:
        return direct
    return getattr(_metadata_for(item), "structured_summary", None)


def _canonical_ids_for(item: object) -> list[str]:
    """Return exact persisted source ids or fail empty on malformed shape."""
    metadata = _metadata_for(item)
    raw_ids = getattr(metadata, "canonical_turn_ids", None)
    if raw_ids is None:
        raw_ids = getattr(item, "source_canonical_turn_ids", None)
    if not isinstance(raw_ids, list) or not all(
        type(value) is str
        and bool(value)
        and value == value.strip()
        for value in raw_ids
    ):
        return []
    canonical_ids = list(raw_ids)
    # Source order and cardinality are authorization inputs. Silently
    # normalizing a duplicated persisted id would let a mutated tag/segment
    # validate against the digest for its former shorter source list.
    if len(set(canonical_ids)) != len(canonical_ids):
        return []
    return canonical_ids


def _structured_source_ids_for(item: object) -> list[str]:
    structured = _structured_summary_for(item)
    if (
        type(getattr(structured, "schema_version", None)) is not int
        or getattr(structured, "schema_version", 0)
        != STRUCTURED_SUMMARY_SCHEMA_VERSION
    ):
        return []
    ids: list[str] = []
    claims = getattr(structured, "claims", ())
    if not isinstance(claims, tuple):
        return []
    for claim in claims:
        sources = getattr(claim, "sources", ())
        if not isinstance(sources, tuple):
            continue
        for source in sources:
            canonical_id = getattr(source, "canonical_turn_id", None)
            if type(canonical_id) is str and canonical_id.strip():
                ids.append(canonical_id.strip())
    return list(dict.fromkeys(ids))


def _row_matches_exact_key(
    row: object,
    *,
    conversation_id: str,
    canonical_turn_id: str,
) -> bool:
    """Reject a store result mapped under a key its physical row does not own."""
    return bool(
        str(getattr(row, "conversation_id", "") or "").strip()
        == conversation_id
        and str(getattr(row, "canonical_turn_id", "") or "").strip()
        == canonical_turn_id
    )


def resolve_summary_speaker_attributions(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list[SummarySpeakerAttribution]:
    """Resolve exact source actors for *items* in one bounded batch.

    An absent/ineligible request context yields no proof.  The storage lookup
    itself enforces the request's resolved owner; audience-safe labels are then
    derived independently from audience-admissible physical rows.
    """
    materialized = list(items)
    empty = [SummarySpeakerAttribution() for _ in materialized]
    if (
        not materialized
        or store is None
        or not conversation_id
        or speaker_context is None
        or not getattr(speaker_context, "eligible", False)
        or not _context_owns_conversation(speaker_context, conversation_id)
    ):
        return empty

    getter = getattr(store, "get_canonical_turn_rows_by_id", None)
    if not callable(getter):
        return empty

    ids_by_item: list[list[str]] = []
    keys: list[tuple[str, str]] = []
    for item in materialized:
        metadata = _metadata_for(item)
        raw_ids = list(getattr(metadata, "canonical_turn_ids", []) or [])
        ids = list(dict.fromkeys(str(value) for value in raw_ids if value))
        ids_by_item.append(ids)
        keys.extend((conversation_id, canonical_id) for canonical_id in ids)

    if not keys:
        return empty
    try:
        rows = getter(list(dict.fromkeys(keys)), speaker_context=speaker_context)
    except Exception:
        return empty

    actor_sets: list[set[str]] = []
    label_candidates_by_item: list[list[tuple[float, str, str, str]]] = []
    complete_flags: list[bool] = []
    expected_audience = speaker_context.audience_conversation_id or ""
    expected_channel = speaker_context.audience_channel_id or ""
    channel_scope = str(
        getattr(speaker_context, "audience_channel_scope", "channel")
        or "channel"
    )
    request_origin_channel = str(
        getattr(speaker_context, "request_origin_channel_id", "") or ""
    ).strip()
    for item, canonical_ids in zip(materialized, ids_by_item, strict=True):
        metadata = _metadata_for(item)
        complete = bool(
            getattr(metadata, "source_mapping_complete", False)
            and canonical_ids
        )
        actors: set[str] = set()
        label_candidates: list[tuple[float, str, str, str]] = []
        for canonical_id in canonical_ids:
            row = rows.get((conversation_id, canonical_id))
            if row is None:
                complete = False
                continue
            # The exact-id hydration is owner-scoped.  A summary bridge is
            # stricter: every backing physical row must also prove the same
            # validated pre-alias audience (and channel when one is known).
            # A label for the same actor in the current audience cannot make
            # prose sourced from another DM/guild admissible.
            row_audience = (
                getattr(row, "audience_conversation_id", "") or ""
            )
            row_channel = (
                getattr(row, "origin_channel_id", "") or ""
            ).strip()
            current_attribution = int(
                getattr(row, "audience_attribution_version", 0) or 0
            ) == AUDIENCE_ATTRIBUTION_VERSION
            if channel_scope == "conversation":
                # Unified-guild mode relaxes only the origin-channel check.
                # The independently validated audience remains exact; a blank
                # channel is never permission to cross another guild or DM.
                audience_admitted = bool(
                    current_attribution
                    and row_audience == expected_audience
                    and request_origin_channel
                    and row_channel
                )
            else:
                audience_admitted = bool(
                    current_attribution
                    and row_audience == expected_audience
                    and row_channel == expected_channel
                )
            if not audience_admitted:
                complete = False
                continue
            if not (getattr(row, "user_content", "") or "").strip():
                actor = ""
            else:
                actor = (getattr(row, "sender_actor_id", "") or "").strip()
                if not actor:
                    complete = False
                else:
                    actors.add(actor)
                    label = (getattr(row, "sender", "") or "").strip()
                    if label:
                        label_candidates.append((
                            float(getattr(row, "sort_key", 0.0) or 0.0),
                            canonical_id,
                            actor,
                            label,
                        ))

            # A native reply carries another human's words in a structurally
            # separate lane. Even though that quote is excluded from requester
            # content, a legacy free-form summary may have absorbed it, so the
            # read bridge must count the subject before declaring one owner.
            if (getattr(row, "reply_target_body", "") or "").strip():
                subject_actor = (
                    getattr(row, "reply_subject_actor_id", "") or ""
                ).strip()
                if not subject_actor:
                    complete = False
                else:
                    actors.add(subject_actor)
                    subject_label = (
                        getattr(row, "reply_subject_label", "") or ""
                    ).strip()
                    if subject_label:
                        label_candidates.append((
                            float(getattr(row, "sort_key", 0.0) or 0.0),
                            canonical_id,
                            subject_actor,
                            subject_label,
                        ))
        actor_sets.append(actors)
        label_candidates_by_item.append(label_candidates)
        complete_flags.append(complete)

    resolved: list[SummarySpeakerAttribution] = []
    for actors, candidates, complete in zip(
        actor_sets,
        label_candidates_by_item,
        complete_flags,
        strict=True,
    ):
        label = ""
        if complete and len(actors) == 1:
            sole_actor = next(iter(actors))
            matching = [
                candidate for candidate in candidates
                if candidate[2] == sole_actor
            ]
            if matching:
                label = max(matching, key=lambda candidate: (
                    candidate[0], candidate[1], candidate[3],
                ))[3]
            if not is_safe_human_label(label, sole_actor):
                label = ""
        resolved.append(SummarySpeakerAttribution(
            actor_ids=frozenset(actors),
            label=label,
            complete=complete,
            requester_match=(
                next(iter(actors))
                == str(getattr(speaker_context, "requester_actor_id", "") or "").strip()
                if len(actors) == 1
                and bool(str(getattr(
                    speaker_context, "requester_actor_id", "",
                ) or "").strip())
                else None
            ),
        ))
    return resolved


def resolve_segment_ref_attributions(
    segment_refs: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> dict[str, SummarySpeakerAttribution]:
    """Resolve source proof for stored segment refs without exposing actors.

    Search and temporal code frequently discard the ``StoredSegment`` before
    creating derived bundles.  This helper reloads each unique ref, performs
    one canonical-row batch, and returns only an in-process proof map. Missing
    segments are intentionally absent rather than represented as safe.
    """
    if store is None:
        return {}
    getter = getattr(store, "get_segment", None)
    if not callable(getter):
        return {}

    refs = list(dict.fromkeys(
        str(value).strip()
        for value in segment_refs
        if value is not None and str(value).strip()
    ))
    segments: list[object] = []
    loaded_refs: list[str] = []
    for ref in refs:
        try:
            segment = getter(ref, conversation_id=conversation_id or None)
        except TypeError:
            # Small alternate stores predate the owner keyword. Their returned
            # segment still has to pass the exact canonical-row lookup below.
            try:
                segment = getter(ref)
            except Exception:
                segment = None
        except Exception:
            segment = None
        if segment is None:
            continue
        segments.append(segment)
        loaded_refs.append(ref)

    attributions = resolve_summary_speaker_attributions(
        segments,
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )
    return dict(zip(loaded_refs, attributions, strict=True))


def _source_row_is_admitted(
    row: object,
    speaker_context: "SpeakerRetrievalContext",
) -> bool:
    """Whether one physical source row is inside this exact read scope."""
    try:
        current = (
            type(getattr(row, "audience_attribution_version", None)) is int
            and getattr(row, "audience_attribution_version")
            == AUDIENCE_ATTRIBUTION_VERSION
        )
    except Exception:
        return False
    row_audience = str(
        getattr(row, "audience_conversation_id", "") or "",
    ).strip()
    row_channel = str(getattr(row, "origin_channel_id", "") or "").strip()
    expected_audience = str(
        getattr(speaker_context, "audience_conversation_id", "") or "",
    ).strip()
    if not current or not expected_audience or row_audience != expected_audience:
        return False
    scope = str(
        getattr(speaker_context, "audience_channel_scope", "channel")
        or "channel",
    )
    if scope == "conversation":
        # Conversation scope is for a proved group audience.  It permits a
        # sibling channel inside that same audience, never a DM or another
        # audience hidden behind the same storage owner.
        request_channel = str(
            getattr(speaker_context, "request_origin_channel_id", "") or "",
        ).strip()
        return bool(request_channel and row_channel)
    expected_channel = str(
        getattr(speaker_context, "audience_channel_id", "") or "",
    ).strip()
    return row_channel == expected_channel


def resolve_summary_source_projections(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
    _preloaded_rows: dict[tuple[str, str], object] | None = None,
    _known_colliding_labels: frozenset[str] = frozenset(),
) -> list[SummarySourceProjection]:
    """Reconstruct exact, role-separated canonical source for each item.

    The generated ``summary`` field is intentionally never read.  Every
    canonical id must resolve to one physical row inside the current request's
    audience boundary. Human lanes additionally require a durable actor and a
    non-generic exact row label. Copied reply-edge bodies and assistant output
    are deliberately not projected as human evidence.
    """
    materialized = list(items)
    empty = [SummarySourceProjection() for _ in materialized]
    if (
        not materialized
        or (store is None and _preloaded_rows is None)
        or not conversation_id
        or speaker_context is None
        or not getattr(speaker_context, "eligible", False)
        or not _context_owns_conversation(speaker_context, conversation_id)
    ):
        return empty

    ids_by_item: list[list[str]] = []
    keys: list[tuple[str, str]] = []
    for item in materialized:
        ids = _canonical_ids_for(item)
        if len(ids) > _SOURCE_TRANSCRIPT_MAX_LANES:
            ids = []
        ids_by_item.append(ids)
        keys.extend((conversation_id, canonical_id) for canonical_id in ids)
    if not keys:
        return empty
    if _preloaded_rows is None:
        getter = getattr(store, "get_canonical_turn_rows_by_id", None)
        if not callable(getter):
            return empty
        try:
            rows = getter(
                list(dict.fromkeys(keys)), speaker_context=speaker_context,
            )
        except Exception:
            return empty
    else:
        rows = _preloaded_rows

    requester_actor = str(
        getattr(speaker_context, "requester_actor_id", "") or "",
    ).strip()

    def requester_match(actor_id: str) -> str:
        if not requester_actor:
            return "unproved"
        return "proved_same" if actor_id == requester_actor else "proved_different"

    projections: list[SummarySourceProjection] = []
    global_label_actors: dict[str, set[str]] = {}
    for item, canonical_ids in zip(materialized, ids_by_item, strict=True):
        metadata = _metadata_for(item)
        complete = bool(
            (
                getattr(metadata, "source_mapping_complete", False) is True
                or (
                    metadata is None
                    and isinstance(
                        getattr(item, "source_canonical_turn_ids", None), list,
                    )
                )
            )
            and canonical_ids
        )
        lanes: list[HistoricalSourceLane] = []
        label_actors: dict[str, set[str]] = {}

        for canonical_id in canonical_ids:
            row = rows.get((conversation_id, canonical_id))
            if (
                row is None
                or not _row_matches_exact_key(
                    row,
                    conversation_id=conversation_id,
                    canonical_turn_id=canonical_id,
                )
                or not _source_row_is_admitted(row, speaker_context)
            ):
                complete = False
                continue
            user_content = str(getattr(row, "user_content", "") or "").strip()
            assistant_content = str(
                getattr(row, "assistant_content", "") or "",
            ).strip()
            reply_content = str(
                getattr(row, "reply_target_body", "") or "",
            ).strip()
            session_date = str(getattr(row, "session_date", "") or "").strip()
            if not user_content and not assistant_content and not reply_content:
                complete = False
                continue

            # ``reply_target_body`` and ``reply_subject_label`` are copied edge
            # claims on the replying row, not the exact target row. Until the
            # target message can be independently hydrated and byte/actor
            # checked, omit that lane completely. It must never be presented
            # as an exact quote merely because the copied subject actor exists.
            if reply_content:
                pass

            if user_content:
                actor = str(
                    getattr(row, "sender_actor_id", "") or "",
                ).strip()
                label = str(getattr(row, "sender", "") or "").strip()
                if (
                    not actor
                    or not is_safe_human_label(label, actor)
                ):
                    complete = False
                else:
                    label_key = human_label_collision_key(label)
                    label_actors.setdefault(label_key, set()).add(actor)
                    global_label_actors.setdefault(
                        label_key, set(),
                    ).add(actor)
                    lanes.append(HistoricalSourceLane(
                        speaker=label,
                        role="historical_human",
                        content=user_content,
                        session_date=session_date,
                        requester_match=requester_match(actor),
                        actor_id=actor,
                        canonical_turn_id=str(getattr(row, "canonical_turn_id", "")),
                    ))

            # Historical assistant output is itself model-generated and can
            # contain the very false personal-state claim being remediated.
            # It remains available through exact transcript/debug paths, but
            # summary-backed evidence projects only humans' own words.
            if assistant_content:
                pass

        if any(len(actors) != 1 for actors in label_actors.values()):
            complete = False
        projections.append(SummarySourceProjection(
            lanes=tuple(lanes) if complete else (),
            complete=bool(complete and lanes),
        ))

    # A label collision across two separately retrieved segments is just as
    # ambiguous as one inside a segment. Include the current requester in the
    # same collision set so an identically named different actor cannot be
    # mistaken for the person being addressed.
    if requester_actor:
        try:
            from .speaker_labels import resolve_speaker_labels

            requester_label = resolve_speaker_labels(
                store,
                [requester_actor],
                speaker_context=speaker_context,
            ).get(requester_actor, "").strip()
        except Exception:
            requester_label = ""
        if requester_label:
            global_label_actors.setdefault(
                human_label_collision_key(requester_label), set(),
            ).add(requester_actor)
    for roster_label, roster_actor in (
        getattr(speaker_context, "roster_label_actor_pairs", ()) or ()
    ):
        if (
            type(roster_label) is str
            and roster_label.strip()
            and type(roster_actor) is str
            and roster_actor.strip()
        ):
            global_label_actors.setdefault(
                human_label_collision_key(roster_label), set(),
            ).add(roster_actor.strip())
    colliding = set(_known_colliding_labels) | {
        label for label, actors in global_label_actors.items()
        if len(actors) > 1
    }
    if colliding:
        projections = [
            SummarySourceProjection()
            if any(
                lane.role != "historical_assistant"
                and human_label_collision_key(lane.speaker) in colliding
                for lane in projection.lanes
            )
            else projection
            for projection in projections
        ]
    return projections


def _requester_match_for_actor(
    actor_id: str,
    speaker_context: "SpeakerRetrievalContext",
) -> str:
    requester_actor = str(
        getattr(speaker_context, "requester_actor_id", "") or "",
    ).strip()
    if not requester_actor:
        return "unproved"
    return "proved_same" if actor_id == requester_actor else "proved_different"


def _known_label_collisions(
    rows: dict[tuple[str, str], object],
    *,
    store: object | None,
    speaker_context: "SpeakerRetrievalContext",
) -> frozenset[str]:
    """Find display names that cannot safely stand for one source actor."""
    actors_by_label: dict[str, set[str]] = {}
    seen_rows: set[int] = set()
    for row in rows.values():
        if id(row) in seen_rows or not _source_row_is_admitted(
            row, speaker_context,
        ):
            continue
        seen_rows.add(id(row))
        if not str(getattr(row, "user_content", "") or "").strip():
            continue
        actor = str(getattr(row, "sender_actor_id", "") or "").strip()
        label = str(getattr(row, "sender", "") or "").strip()
        if actor and is_safe_human_label(label, actor):
            actors_by_label.setdefault(
                human_label_collision_key(label), set(),
            ).add(actor)

    for roster_label, roster_actor in (
        getattr(speaker_context, "roster_label_actor_pairs", ()) or ()
    ):
        if (
            type(roster_label) is str
            and roster_label.strip()
            and type(roster_actor) is str
            and roster_actor.strip()
        ):
            actors_by_label.setdefault(
                human_label_collision_key(roster_label), set(),
            ).add(roster_actor.strip())

    requester_actor = str(
        getattr(speaker_context, "requester_actor_id", "") or "",
    ).strip()
    if requester_actor and store is not None:
        try:
            from .speaker_labels import resolve_speaker_labels

            requester_label = resolve_speaker_labels(
                store,
                [requester_actor],
                speaker_context=speaker_context,
            ).get(requester_actor, "").strip()
        except Exception:
            requester_label = ""
        if requester_label:
            actors_by_label.setdefault(
                human_label_collision_key(requester_label), set(),
            ).add(requester_actor)

    return frozenset(
        label for label, actors in actors_by_label.items()
        if len(actors) > 1
    )


def _validate_structured_claim(
    claim: object,
    *,
    rows: dict[tuple[str, str], object],
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext",
    colliding_labels: frozenset[str],
) -> ValidatedSummaryClaim | None:
    """Validate one v1 claim against its exact current canonical source."""
    text = getattr(claim, "text", None)
    claim_type = getattr(claim, "claim_type", None)
    temporal_status = getattr(claim, "temporal_status", None)
    modality = getattr(claim, "modality", None)
    event_time = getattr(claim, "event_time", None)
    raw_sources = getattr(claim, "sources", None)
    if (
        type(text) is not str
        or not text.strip()
        or len(text) > 1000
        or claim_type not in _STRUCTURED_CLAIM_TYPES
        or temporal_status not in _STRUCTURED_TEMPORAL_STATUSES
        or modality not in _STRUCTURED_MODALITIES
        or type(event_time) is not str
        or len(event_time) > 128
        or not isinstance(raw_sources, tuple)
        or len(raw_sources) != 1
    ):
        return None

    sources: list[ValidatedClaimSource] = []
    human_actors: set[str] = set()
    evidence_excerpts: list[str] = []
    source_session_dates: list[str] = []
    source_roles: set[str] = set()

    for source in raw_sources:
        canonical_id = getattr(source, "canonical_turn_id", None)
        source_role = getattr(source, "source_role", None)
        speaker_label = getattr(source, "speaker_label", None)
        evidence = getattr(source, "evidence_excerpt", None)
        source_session_date = getattr(source, "session_date", None)
        source_provenance_digest = getattr(
            source, "source_provenance_digest", None,
        )
        if (
            type(canonical_id) is not str
            or not canonical_id.strip()
            or len(canonical_id) > 256
            or source_role != "requester"
            or type(speaker_label) is not str
            or not speaker_label.strip()
            or len(speaker_label) > 256
            or type(evidence) is not str
            or not evidence
            or len(evidence) > 800
            or type(source_session_date) is not str
            or len(source_session_date) > 128
            or type(source_provenance_digest) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}", source_provenance_digest,
            ) is None
        ):
            return None
        canonical_id = canonical_id.strip()
        source_roles.add(source_role)
        row = rows.get((conversation_id, canonical_id))
        if (
            row is None
            or not _row_matches_exact_key(
                row,
                conversation_id=conversation_id,
                canonical_turn_id=canonical_id,
            )
            or not _source_row_is_admitted(row, speaker_context)
        ):
            return None

        row_session_date = str(
            getattr(row, "session_date", "") or "",
        ).strip()
        if source_session_date.strip() != row_session_date:
            return None

        lane_content = str(getattr(row, "user_content", "") or "")
        actor_id = str(
            getattr(row, "sender_actor_id", "") or "",
        ).strip()
        canonical_label = str(getattr(row, "sender", "") or "").strip()
        if (
            not actor_id
            or not is_safe_human_label(canonical_label, actor_id)
            or speaker_label.strip() != canonical_label
            or human_label_collision_key(canonical_label) in colliding_labels
        ):
            return None
        human_actors.add(actor_id)
        rendered_source = ValidatedClaimSource(
            display_name=canonical_label,
            role="historical_human",
            evidence_excerpt=evidence,
            session_date=row_session_date,
            requester_match=_requester_match_for_actor(
                actor_id, speaker_context,
            ),
            actor_id=actor_id,
            canonical_turn_id=str(getattr(row, "canonical_turn_id", "")),
        )

        # A selected substring can invert its containing sentence (for
        # example ``currently taking`` inside ``no longer currently taking``).
        # Structured evidence is therefore admitted only when it is the entire
        # trimmed requester lane, byte-for-byte.
        if not lane_content.strip() or evidence != lane_content.strip():
            return None
        if source_provenance_digest != structured_source_provenance_digest({
            "canonical_turn_id": canonical_id,
            "source_role": "requester",
            "actor_id": actor_id,
            "speaker_label": canonical_label,
            "content": lane_content.strip(),
            "session_date": row_session_date,
            "audience_conversation_id": str(
                getattr(row, "audience_conversation_id", "") or "",
            ),
            "origin_channel_id": str(
                getattr(row, "origin_channel_id", "") or "",
            ),
            "audience_attribution_version": getattr(
                row, "audience_attribution_version", 0,
            ),
        }):
            return None
        # Control-plane ids must never re-enter through exact evidence text.
        if _INTERNAL_IDENTITY_LABEL_RE.search(evidence) or (
            rendered_source.actor_id and rendered_source.actor_id in evidence
        ):
            return None

        evidence_excerpts.append(evidence)
        source_session_dates.append(row_session_date)
        sources.append(rendered_source)

    if len(source_roles) != 1 or len(human_actors) > 1:
        return None
    evidence_text = "\n".join(evidence_excerpts)
    evidenced_status = infer_temporal_status(evidence_text)
    if statuses_conflict(temporal_status, evidenced_status):
        return None
    if infer_modality(evidence_text) != modality:
        return None
    if event_time and not any(
        event_time_is_supported(
            event_time,
            evidence_excerpts,
            session_date,
        )
        for session_date in source_session_dates
    ):
        return None

    return ValidatedSummaryClaim(
        # ``claim.text`` is a generated selection/classification aid, never
        # factual evidence. Reconstruct the model-visible claim extractively
        # from the exact lane instead of trusting even a well-attributed
        # paraphrase that could invert stopped/planned/current semantics.
        text=evidence_excerpts[0],
        # A model-selected taxonomy cannot prove that a quoted/refuted
        # statement is the requester's personal state. V1 exposes every exact
        # lane conservatively as conversation evidence.
        claim_type="conversation",
        # A model classification is never positive proof of current state. If
        # the exact quote does not deterministically establish a status, render
        # it as unspecified instead of manufacturing ``active`` by fallback.
        temporal_status=evidenced_status,
        modality=infer_modality(evidence_text),
        event_time=event_time,
        sources=tuple(sources),
    )


def _structured_source_digest_matches(
    item: object,
    *,
    raw_claims: tuple[object, ...],
    expected_digest: str,
    rows: dict[tuple[str, str], object],
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext",
) -> bool:
    """Recompute the v1 source snapshot before trusting any of its claims."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        return False

    metadata = _metadata_for(item)
    if metadata is None and hasattr(item, "tag"):
        # Tag rollups copy lower-layer claims unchanged. Their digest is over
        # selected claim fingerprints in presentation order. SUMMARY is
        # prefix-bounded, so authenticating only an unordered set would allow
        # an old active state to move ahead of a newer cessation.
        fingerprints = [
            structured_claim_fingerprint(claim) for claim in raw_claims
        ]
        if (
            not fingerprints
            or len(fingerprints) > _STRUCTURED_TAG_MAX_SELECTED_CLAIMS
            or len(set(fingerprints)) != len(fingerprints)
        ):
            return False
        return structured_tag_claim_digest(
            raw_claims, _canonical_ids_for(item),
        ) == expected_digest

    canonical_ids = _canonical_ids_for(item)
    if (
        metadata is None
        or getattr(metadata, "source_mapping_complete", False) is not True
        or not canonical_ids
    ):
        return False
    candidates: list[tuple[object, str, str, str]] = []
    label_actors: dict[str, set[str]] = {}
    for canonical_id in canonical_ids:
        row = rows.get((conversation_id, canonical_id))
        # Compressed v1 evidence is requester-only. Legacy assistant-only
        # canonical rows can predate audience attribution and are therefore
        # absent from scoped hydration in production; they neither contribute
        # to nor invalidate the requester source digest. FULL independently
        # requires every row and remains fail-closed below this path.
        if row is None:
            continue
        requester_text = str(
            getattr(row, "user_content", "") or "",
        ).strip()
        if not requester_text:
            continue
        if (
            not _row_matches_exact_key(
                row,
                conversation_id=conversation_id,
                canonical_turn_id=canonical_id,
            )
            or not _source_row_is_admitted(row, speaker_context)
        ):
            return False
        actor_id = str(
            getattr(row, "sender_actor_id", "") or "",
        ).strip()
        label = str(getattr(row, "sender", "") or "").strip()
        if actor_id and is_safe_human_label(label, actor_id):
            label_actors.setdefault(
                human_label_collision_key(label), set(),
            ).add(actor_id)
            candidates.append((row, "requester", label, requester_text))

    colliding = {
        label for label, actors in label_actors.items() if len(actors) > 1
    }
    records: list[dict[str, object]] = []
    for row, source_role, label, content in candidates:
        if (
            source_role == "requester"
            and human_label_collision_key(label) in colliding
        ):
            continue
        records.append({
            "canonical_turn_id": str(
                getattr(row, "canonical_turn_id", "") or "",
            ).strip(),
            "source_role": source_role,
            "actor_id": (
                str(getattr(row, "sender_actor_id", "") or "").strip()
                if source_role == "requester"
                else ""
            ),
            "speaker_label": label,
            "content": content,
            "session_date": str(
                getattr(row, "session_date", "") or "",
            ).strip(),
            "audience_conversation_id": str(
                getattr(row, "audience_conversation_id", "") or "",
            ),
            "origin_channel_id": str(
                getattr(row, "origin_channel_id", "") or "",
            ),
            "audience_attribution_version": (
                getattr(row, "audience_attribution_version", 0)
                if type(getattr(row, "audience_attribution_version", 0)) is int
                else 0
            ),
        })
    return bool(
        records
        and structured_source_digest(records, namespace="segment")
        == expected_digest
    )


def _validated_structured_claims(
    item: object,
    *,
    rows: dict[tuple[str, str], object],
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext",
    colliding_labels: frozenset[str],
) -> tuple[ValidatedSummaryClaim, ...]:
    structured = _structured_summary_for(item)
    if (
        type(getattr(structured, "schema_version", None)) is not int
        or getattr(structured, "schema_version", 0)
        != STRUCTURED_SUMMARY_SCHEMA_VERSION
    ):
        return ()
    raw_claims = getattr(structured, "claims", None)
    if not isinstance(raw_claims, tuple):
        return ()
    expected_digest = str(
        getattr(structured, "source_digest", "") or "",
    ).strip()
    if not _structured_source_digest_matches(
        item,
        raw_claims=raw_claims,
        expected_digest=expected_digest,
        rows=rows,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    ):
        return ()
    metadata = _metadata_for(item)
    is_tag_summary = metadata is None and hasattr(item, "tag")
    required_source_ids: set[str] = set()
    admitted_actor_ids: set[str] = set()
    segment_source_order: dict[str, int] = {}
    if metadata is not None:
        # Segment source_digest authenticates the full source snapshot, but it
        # does not prove that the claim list covers that snapshot. Every exact
        # requester lane that fits in one atomic claim must survive validation;
        # otherwise an old/partial v1 payload could silently omit ordinary
        # context or a newer correction. A safety-critical lane that is too
        # large for the claim schema also makes the compressed layer unsafe:
        # reject it so the caller uses the complete canonical fallback.
        canonical_ids = _canonical_ids_for(item)
        segment_source_order = {
            canonical_id: index
            for index, canonical_id in enumerate(canonical_ids)
        }
        for canonical_id in canonical_ids:
            row = rows.get((conversation_id, canonical_id))
            if row is None:
                continue
            requester_text = str(
                getattr(row, "user_content", "") or "",
            ).strip()
            if not requester_text:
                continue
            if (
                not _row_matches_exact_key(
                    row,
                    conversation_id=conversation_id,
                    canonical_turn_id=canonical_id,
                )
                or not _source_row_is_admitted(row, speaker_context)
            ):
                return ()
            actor_id = str(
                getattr(row, "sender_actor_id", "") or "",
            ).strip()
            if actor_id:
                admitted_actor_ids.add(actor_id)
            if len(requester_text) <= STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS:
                required_source_ids.add(canonical_id)
            elif is_safety_critical_personal_evidence(requester_text):
                return ()
    elif is_tag_summary:
        # The selected-claim digest proves only the compact layer-two
        # envelope. It cannot prove that selection retained a newer stop,
        # denial, or restart from its lower-layer history. Recheck the tag's
        # complete canonical coverage and require every currently admitted
        # safety-critical requester lane to survive as a valid selected claim.
        # Otherwise an active-only selection could be internally authentic
        # while omitting the exact correction that makes it unsafe.
        canonical_ids = _canonical_ids_for(item)
        if not canonical_ids:
            return ()
        canonical_id_set = set(canonical_ids)
        for claim in raw_claims:
            sources = getattr(claim, "sources", None)
            if not isinstance(sources, tuple) or len(sources) != 1:
                return ()
            selected_id = str(
                getattr(sources[0], "canonical_turn_id", "") or "",
            ).strip()
            if not selected_id or selected_id not in canonical_id_set:
                return ()
        for canonical_id in canonical_ids:
            row = rows.get((conversation_id, canonical_id))
            if (
                row is None
                or not _row_matches_exact_key(
                    row,
                    conversation_id=conversation_id,
                    canonical_turn_id=canonical_id,
                )
            ):
                return ()
            if not _source_row_is_admitted(row, speaker_context):
                continue
            actor_id = str(
                getattr(row, "sender_actor_id", "") or "",
            ).strip()
            if actor_id:
                admitted_actor_ids.add(actor_id)
            requester_text = str(
                getattr(row, "user_content", "") or "",
            ).strip()
            if (
                requester_text
                and is_safety_critical_personal_evidence(requester_text)
            ):
                required_source_ids.add(canonical_id)
    if structured_claims_contain_internal_identity(
        raw_claims,
        admitted_actor_ids=admitted_actor_ids,
    ):
        return ()
    validated: list[tuple[str, ValidatedSummaryClaim]] = []
    validated_source_ids: set[str] = set()
    for claim in raw_claims:
        projection = _validate_structured_claim(
            claim,
            rows=rows,
            conversation_id=conversation_id,
            speaker_context=speaker_context,
            colliding_labels=colliding_labels,
        )
        # A layer-two selection is one authenticated envelope, not a bag of
        # independently salvageable claims. If a newer correction has been
        # mutated/deleted while an older active claim still validates, keeping
        # only that survivor reverses the state presented to the model. The
        # same atomic rule covers malformed source cardinality and duplicate
        # canonical selections: reject the whole tag so the assembler can use
        # its exact segment/canonical fallback.
        if is_tag_summary and projection is None:
            return ()
        if projection is not None:
            raw_sources = tuple(getattr(claim, "sources", ()) or ())
            canonical_id = ""
            if len(raw_sources) == 1:
                canonical_id = str(
                    getattr(raw_sources[0], "canonical_turn_id", "") or "",
                ).strip()
            if is_tag_summary and (
                not canonical_id or canonical_id in validated_source_ids
            ):
                return ()
            # A segment's source digest authenticates only its persisted
            # canonical id sequence. An otherwise valid claim from some other
            # hydrated row has no authenticated position in this segment and
            # cannot participate in its bounded model-facing prefix.
            if metadata is not None and canonical_id not in segment_source_order:
                continue
            if canonical_id not in validated_source_ids:
                validated.append((canonical_id, projection))
            if canonical_id:
                validated_source_ids.add(canonical_id)
    if is_tag_summary and len(validated) != len(raw_claims):
        return ()
    if required_source_ids - validated_source_ids:
        return ()
    if metadata is not None:
        # Validate and reconstruct every candidate before deciding which ones
        # are safety-critical. Persisted ``claim.text`` and taxonomy are model
        # output, so consulting either before validation lets a forged older
        # claim occupy the bounded SUMMARY prefix while a newer exact stop is
        # pushed past it. The projection text is the whole canonical requester
        # lane; order critical projections by the digest-authenticated physical
        # chronology. Ordinary claims also use physical chronology: an older
        # provider-built v1 envelope may contain every required lane while
        # retaining a model-selected order, and persisted order is not source
        # authority.
        critical = sorted(
            (
                entry for entry in validated
                if is_safety_critical_personal_evidence(entry[1].text)
            ),
            key=lambda entry: segment_source_order[entry[0]],
            reverse=True,
        )
        critical_source_ids = {canonical_id for canonical_id, _ in critical}
        ordinary = sorted(
            (
                entry for entry in validated
                if entry[0] not in critical_source_ids
            ),
            key=lambda entry: segment_source_order[entry[0]],
        )
        validated = critical + ordinary
    return tuple(
        projection for _, projection in validated[
            :_STRUCTURED_SEGMENTS_MAX_CLAIMS
        ]
    )


def _escaped_json(payload: dict) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_structured_claims_for_model(
    claims: Iterable[ValidatedSummaryClaim],
    *,
    depth: str,
) -> str:
    """Render compact tag claims or richer segment claims without source ids."""
    max_claims = (
        _STRUCTURED_SUMMARY_MAX_CLAIMS
        if depth == "summary"
        else _STRUCTURED_SEGMENTS_MAX_CLAIMS
    )
    materialized = [
        claim for claim in claims if len(claim.sources) == 1
    ][:max_claims]
    if not materialized or depth not in {"summary", "segments"}:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    rendered_claims: list[dict] = []
    for claim in materialized:
        common = {
            "text": claim.text,
            "claim_type": claim.claim_type,
            "temporal_status": claim.temporal_status,
            "modality": claim.modality,
        }
        if depth == "summary":
            source = claim.sources[0]
            common.update({
                "display_name": source.display_name,
                "role": source.role,
                "evidence_excerpt": source.evidence_excerpt,
                "session_date": source.session_date,
                "current_requester_match": source.requester_match,
            })
        else:
            common.update({
                "event_time": claim.event_time,
                "sources": [
                    {
                        "display_name": source.display_name,
                        "role": source.role,
                        "evidence_excerpt": source.evidence_excerpt,
                        "session_date": source.session_date,
                        "current_requester_match": source.requester_match,
                    }
                    for source in claim.sources
                ],
            })
        rendered_claims.append(common)
    encoded = _escaped_json({
        "source": "structured_summary_v1",
        "depth": depth,
        "claims": rendered_claims,
    })
    if len(encoded.encode("utf-8")) > _STRUCTURED_RENDER_MAX_UTF8_BYTES:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    rendered = _STRUCTURED_SUMMARY_OPEN + encoded + _STRUCTURED_SUMMARY_CLOSE
    return rendered if is_proved_summary_rendering(
        rendered,
    ) else SUMMARY_ATTRIBUTION_QUARANTINE


def _full_source_projection(
    item: object,
    *,
    rows: dict[tuple[str, str], object],
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext",
    colliding_labels: frozenset[str],
) -> SummarySourceProjection:
    canonical_ids = _canonical_ids_for(item)
    metadata = _metadata_for(item)
    mapping_complete = bool(
        (
            getattr(metadata, "source_mapping_complete", False) is True
            or (
                metadata is None
                and isinstance(
                    getattr(item, "source_canonical_turn_ids", None), list,
                )
            )
        )
        and canonical_ids
        and len(canonical_ids) <= _FULL_TRANSCRIPT_MAX_TURNS
    )
    lanes: list[HistoricalSourceLane] = []
    for canonical_id in canonical_ids:
        row = rows.get((conversation_id, canonical_id))
        if (
            row is None
            or not _row_matches_exact_key(
                row,
                conversation_id=conversation_id,
                canonical_turn_id=canonical_id,
            )
            or not _source_row_is_admitted(row, speaker_context)
        ):
            mapping_complete = False
            continue
        session_date = str(getattr(row, "session_date", "") or "").strip()
        user_content = str(getattr(row, "user_content", "") or "")
        assistant_content = str(
            getattr(row, "assistant_content", "") or "",
        )
        if not user_content.strip() and not assistant_content.strip():
            mapping_complete = False
            continue
        if user_content.strip():
            actor_id = str(
                getattr(row, "sender_actor_id", "") or "",
            ).strip()
            label = str(getattr(row, "sender", "") or "").strip()
            if (
                not actor_id
                or not is_safe_human_label(label, actor_id)
                or human_label_collision_key(label) in colliding_labels
            ):
                mapping_complete = False
            else:
                lanes.append(HistoricalSourceLane(
                    speaker=label,
                    role="historical_human",
                    content=user_content,
                    session_date=session_date,
                    requester_match=_requester_match_for_actor(
                        actor_id, speaker_context,
                    ),
                    actor_id=actor_id,
                    canonical_turn_id=str(getattr(row, "canonical_turn_id", "")),
                ))
        if assistant_content.strip():
            lanes.append(HistoricalSourceLane(
                speaker="Assistant",
                role="historical_assistant",
                content=assistant_content,
                session_date=session_date,
                requester_match="not_applicable",
                canonical_turn_id=str(getattr(row, "canonical_turn_id", "")),
            ))
    return SummarySourceProjection(
        lanes=tuple(lanes) if mapping_complete else (),
        complete=bool(mapping_complete and lanes),
    )


def render_full_source_projection_for_model(
    projection: SummarySourceProjection | None,
) -> str:
    """Render the exact role-separated canonical transcript for FULL depth."""
    if projection is None or not projection.complete or not projection.lanes:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    if _projection_contains_internal_identity(projection):
        return SUMMARY_ATTRIBUTION_QUARANTINE
    payload = {
        "source": "canonical_turns",
        "depth": "full",
        "lanes": [
            {
                "display_name": lane.speaker,
                "role": lane.role,
                "content": lane.content,
                "session_date": lane.session_date,
                "current_requester_match": lane.requester_match,
            }
            for lane in projection.lanes
        ],
    }
    encoded = _escaped_json(payload)
    if len(encoded.encode("utf-8")) > _FULL_TRANSCRIPT_MAX_UTF8_BYTES:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    rendered = _FULL_TRANSCRIPT_OPEN + encoded + _FULL_TRANSCRIPT_CLOSE
    return rendered if is_proved_summary_rendering(
        rendered,
    ) else SUMMARY_ATTRIBUTION_QUARANTINE


def render_source_projection_for_model(
    projection: SummarySourceProjection | None,
) -> str:
    """Serialize one canonical projection without generated prose or ids."""
    if projection is None or not projection.complete or not projection.lanes:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    if _projection_contains_internal_identity(projection):
        return SUMMARY_ATTRIBUTION_QUARANTINE
    if len(projection.lanes) > _SOURCE_TRANSCRIPT_MAX_LANES:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    source_refs: dict[str, str] = {}

    def source_ref(lane: HistoricalSourceLane, index: int) -> str:
        # Random request-local references avoid both raw-id disclosure and
        # cross-call display-name collisions.  Actor ids are used only as an
        # in-memory equality key and never enter the serialized value.
        key = lane.actor_id or f"lane:{index}"
        return source_refs.setdefault(
            key, f"historical_{uuid.uuid4().hex[:16]}",
        )

    payload = {
        "source": "canonical_turns",
        "generated_summary_prose_used": False,
        "lanes": [
            {
                "source_speaker_ref": source_ref(lane, index),
                "display_name": lane.speaker,
                "role": lane.role,
                "content": lane.content,
                "session_date": lane.session_date,
                "current_requester_match": lane.requester_match,
            }
            for index, lane in enumerate(projection.lanes)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > _SOURCE_TRANSCRIPT_MAX_UTF8_BYTES:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    # Reserved markup inside an exact source message remains data rather than
    # becoming a second envelope visible to the model.
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    if len(encoded.encode("utf-8")) > _SOURCE_TRANSCRIPT_MAX_UTF8_BYTES:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    rendered = _SOURCE_TRANSCRIPT_OPEN + encoded + _SOURCE_TRANSCRIPT_CLOSE
    return rendered if is_proved_summary_rendering(
        rendered,
    ) else SUMMARY_ATTRIBUTION_QUARANTINE


def render_segment_refs_for_model(
    segment_refs: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> dict[str, str]:
    """Render exact canonical transcripts for loaded segment references."""
    if store is None:
        return {}
    getter = getattr(store, "get_segment", None)
    if not callable(getter):
        return {}
    refs = list(dict.fromkeys(
        str(value).strip()
        for value in segment_refs
        if value is not None and str(value).strip()
    ))
    loaded_refs: list[str] = []
    segments: list[object] = []
    for ref in refs:
        try:
            segment = getter(ref, conversation_id=conversation_id or None)
        except TypeError:
            try:
                segment = getter(ref)
            except Exception:
                segment = None
        except Exception:
            segment = None
        if segment is not None:
            loaded_refs.append(ref)
            segments.append(segment)
    projections = resolve_summary_source_projections(
        segments,
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )
    return {
        ref: _mark_request_local_summary_rendering(
            render_source_projection_for_model(projection),
            speaker_context=speaker_context,
        )
        for ref, projection in zip(loaded_refs, projections, strict=True)
    }


def render_summary_for_model(
    text: object,
    attribution: SummarySpeakerAttribution | None = None,
    *,
    require_proved_scope: bool = False,
) -> str:
    """Return only non-evidentiary prose; never bless a generated summary.

    ``attribution`` proves which actors participated in source rows, not that
    a generated sentence preserved speaker, negation, intent, or time.  A
    model-visible summary therefore cannot become safe merely by attaching a
    label. Callers that require evidence must use
    :func:`render_source_projection_for_model` instead.
    """
    value = text if isinstance(text, str) else ""
    if _SUMMARY_ATTRIBUTION_MARKER_RE.search(value):
        return SUMMARY_ATTRIBUTION_QUARANTINE
    ambiguous = contains_ambiguous_human_referent(value)
    # Segment participation is not sentence-level coreference proof. Even a
    # generic noun can denote an addressee or third party (``BigTex advised
    # the user`` / ``described a patient``), so no ambiguous referent is ever
    # rebound mechanically. Current generation retries it; historical prose
    # is quarantined and its exact source turns remain retrievable.
    if ambiguous:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    if require_proved_scope:
        return SUMMARY_ATTRIBUTION_QUARANTINE
    return value


def render_summary_items_for_model(
    requests: Iterable[tuple[object, object]],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list[str]:
    """Render mixed SUMMARY/SEGMENTS/FULL requests with one row hydration.

    Version-one segment claims are validated independently. Version-one tag
    selections validate atomically so a failed newer correction cannot expose
    only an older surviving state. If validation fails (or the item is legacy),
    SUMMARY and SEGMENTS fall back to existing exact human-source projection;
    the assembler resolves invalid tags through their source segments. FULL
    never reads ``StoredSegment.full_text``; it reconstructs both roles from
    exact canonical lanes.
    """
    materialized: list[tuple[object, str]] = []
    for item, raw_depth in requests:
        depth = str(getattr(raw_depth, "value", raw_depth) or "").strip().lower()
        materialized.append((item, depth))
    if not materialized:
        return []
    empty = [SUMMARY_ATTRIBUTION_QUARANTINE for _ in materialized]
    if (
        store is None
        or not conversation_id
        or speaker_context is None
        or not getattr(speaker_context, "eligible", False)
        or not _context_owns_conversation(speaker_context, conversation_id)
    ):
        return empty
    getter = getattr(store, "get_canonical_turn_rows_by_id", None)
    if not callable(getter):
        return empty

    keys: list[tuple[str, str]] = []
    for item, _depth in materialized:
        # A layer-two TagSummary can cover thousands of canonical turns. Its
        # compact v1 presentation still has to hydrate the tag's full source
        # id set: selected-claim provenance alone cannot detect omission of a
        # newer safety-critical correction. Invalid tags remain bounded at
        # presentation time by the assembler's source-segment fallback.
        ids = _canonical_ids_for(item) + _structured_source_ids_for(item)
        keys.extend(
            (conversation_id, canonical_id)
            for canonical_id in dict.fromkeys(ids)
        )
    if not keys:
        return empty
    try:
        loaded = getter(
            list(dict.fromkeys(keys)), speaker_context=speaker_context,
        )
    except Exception:
        return empty
    if not hasattr(loaded, "get"):
        return empty
    rows = loaded
    colliding_labels = _known_label_collisions(
        rows,
        store=store,
        speaker_context=speaker_context,
    )
    items = [item for item, _depth in materialized]
    legacy_projections = resolve_summary_source_projections(
        items,
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
        _preloaded_rows=rows,
        _known_colliding_labels=colliding_labels,
    )

    rendered: list[str] = []
    for (item, depth), legacy_projection in zip(
        materialized, legacy_projections, strict=True,
    ):
        dependency_ids = tuple(dict.fromkeys(
            _canonical_ids_for(item) + _structured_source_ids_for(item)
        ))
        versions = tuple(
            MemorySourceVersion.from_row(rows[(conversation_id, source_id)])
            for source_id in dependency_ids if (conversation_id, source_id) in rows
        )

        def mark(text: str, source_ids: Iterable[str], kind: str) -> str:
            return _mark_request_local_summary_rendering(
                text, speaker_context=speaker_context, source_versions=versions,
                presented_source_ids=tuple(dict.fromkeys(source_ids)), evidence_kind=kind,
            )

        if depth == "full":
            projection = _full_source_projection(
                item, rows=rows, conversation_id=conversation_id,
                speaker_context=speaker_context, colliding_labels=colliding_labels,
            )
            rendered.append(mark(
                render_full_source_projection_for_model(projection),
                (lane.canonical_turn_id for lane in projection.lanes), "canonical_transcript",
            ))
            continue
        if depth not in {"summary", "segments"}:
            rendered.append(SUMMARY_ATTRIBUTION_QUARANTINE)
            continue
        claims = _validated_structured_claims(
            item,
            rows=rows,
            conversation_id=conversation_id,
            speaker_context=speaker_context,
            colliding_labels=colliding_labels,
        )
        candidate = render_structured_claims_for_model(claims, depth=depth)
        if candidate != SUMMARY_ATTRIBUTION_QUARANTINE:
            selected_claims = claims[:(
                _STRUCTURED_SUMMARY_MAX_CLAIMS if depth == "summary"
                else _STRUCTURED_SEGMENTS_MAX_CLAIMS
            )]
            rendered.append(mark(
                candidate,
                (source.canonical_turn_id for claim in selected_claims for source in claim.sources),
                "structured_summary_v1",
            ))
            continue

        structured = _structured_summary_for(item)
        is_v1 = (
            type(getattr(structured, "schema_version", None)) is int
            and getattr(structured, "schema_version", 0)
            == STRUCTURED_SUMMARY_SCHEMA_VERSION
        )
        metadata = _metadata_for(item)
        raw_claims = getattr(structured, "claims", ())
        # Legacy artifacts have no structured authority and retain the exact
        # canonical fallback. A present v1 artifact may use that fallback only
        # when its full segment source snapshot still matches. Otherwise a
        # detected source-id deletion/reorder would be converted into a stale
        # partial transcript (for example old active without the later stop).
        v1_segment_snapshot_proved = bool(
            is_v1
            and metadata is not None
            and isinstance(raw_claims, tuple)
            and _structured_source_digest_matches(
                item,
                raw_claims=raw_claims,
                expected_digest=str(
                    getattr(structured, "source_digest", "") or "",
                ).strip(),
                rows=rows,
                conversation_id=conversation_id,
                speaker_context=speaker_context,
            )
        )
        rendered.append(mark(
            (
                render_source_projection_for_model(legacy_projection)
                if not is_v1 or v1_segment_snapshot_proved
                else SUMMARY_ATTRIBUTION_QUARANTINE
            ),
            (lane.canonical_turn_id for lane in legacy_projection.lanes)
            if legacy_projection is not None else (),
            "canonical_human_evidence",
        ))
    return rendered


def render_summaries_for_model(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
    depth: object = "summary",
) -> list[str]:
    """Compatibility wrapper for rendering one depth across a batch."""
    return render_summary_items_for_model(
        ((item, depth) for item in items),
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )


def sanitize_summary_payload_for_model(
    payload: object,
    *,
    allow_proved_renderings: bool = False,
    speaker_context: "SpeakerRetrievalContext | None" = None,
) -> object:
    """Quarantine anonymous derived prose in a JSON-like model payload.

    This stateless boundary is for aggregate/search/time tools. A syntactically
    valid envelope is still untrusted input: ``allow_proved_renderings`` only
    preserves the ephemeral subtype emitted by a canonical-row renderer bound
    to the exact same immutable ``speaker_context`` object. It mutates
    dictionaries/lists in place, matching the existing speaker-field
    projection helpers.
    """
    if isinstance(payload, dict):
        for key, value in list(payload.items()):
            if key in _DERIVED_SOURCE_PROSE_FIELDS:
                if isinstance(value, str):
                    payload[key] = (
                        value
                        if (
                            allow_proved_renderings
                            and _is_request_local_summary_rendering(
                                value,
                                speaker_context=speaker_context,
                            )
                        )
                        else render_summary_for_model(
                            value, require_proved_scope=True,
                        )
                    )
                elif isinstance(value, list):
                    payload[key] = [
                        (
                            item
                            if (
                                allow_proved_renderings
                                and _is_request_local_summary_rendering(
                                    item,
                                    speaker_context=speaker_context,
                                )
                            )
                            else render_summary_for_model(
                                item, require_proved_scope=True,
                            )
                        )
                        if isinstance(item, str)
                        else sanitize_summary_payload_for_model(
                            item,
                            allow_proved_renderings=allow_proved_renderings,
                            speaker_context=speaker_context,
                        )
                        for item in value
                    ]
                continue
            if key in _DERIVED_INSTRUCTION_FIELDS and isinstance(value, str):
                payload[key] = render_summary_for_model(value)
                continue
            sanitize_summary_payload_for_model(
                value,
                allow_proved_renderings=allow_proved_renderings,
                speaker_context=speaker_context,
            )
    elif isinstance(payload, list):
        for value in payload:
            sanitize_summary_payload_for_model(
                value,
                allow_proved_renderings=allow_proved_renderings,
                speaker_context=speaker_context,
            )
    return payload
