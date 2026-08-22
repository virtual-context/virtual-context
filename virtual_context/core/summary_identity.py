"""Deterministic speaker-safety policy for model-visible summary prose.

Stored summaries are derived text, not quote provenance.  Older rows may call a
named participant ``the user`` even when a segment contains several people.
This module supplies a secondary lexical detector and the read-side provenance
gate used before that prose is shown to a model.  Absence of a suspicious word
is never treated as ownership proof: subjectless and passive summaries are
common, so an audience-scoped canonical-row mapping is the primary control.

The gate never guesses an owner and never treats a label as semantic proof.
Generated prose is ranking-only. Model-facing summary surfaces either rebuild
role-local human text from exact audience-scoped canonical rows or withhold the
item; assistant output and copied reply bodies are not summary evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import unicodedata
import uuid
from typing import Iterable, TYPE_CHECKING

from ..types import AUDIENCE_ATTRIBUTION_VERSION

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
    r"</?(?:summary-attribution|historical-source-transcript)\b",
    re.IGNORECASE,
)
_SOURCE_TRANSCRIPT_OPEN = "<historical-source-transcript>\n"
_SOURCE_TRANSCRIPT_CLOSE = "\n</historical-source-transcript>"
_SOURCE_SPEAKER_REF_RE = re.compile(r"^historical_[0-9a-f]{16}$")
# Bounds apply to the complete serialized JSON payload.  Oversized source
# groups are withheld atomically; no lane or speaker binding is ever sliced.
_SOURCE_TRANSCRIPT_MAX_LANES = 12
_SOURCE_TRANSCRIPT_MAX_UTF8_BYTES = 16 * 1024

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


def is_proved_summary_rendering(text: object) -> bool:
    """Validate one canonical-source transcript projection.

    Generated summary prose can neither prove its speaker nor preserve the
    source's plan/ceased/current modality.  Consequently the only rendering
    trusted by model-facing summary surfaces is a JSON envelope built from
    exact, audience-admitted canonical rows.  Syntax alone is never authority;
    callers may preserve a valid envelope only on an internal path that just
    constructed it from those rows.
    """
    if not isinstance(text, str) or not text.startswith(_SOURCE_TRANSCRIPT_OPEN):
        return False
    remainder = text[len(_SOURCE_TRANSCRIPT_OPEN):]
    if remainder.count(_SOURCE_TRANSCRIPT_CLOSE) != 1:
        return False
    payload_text, trailing = remainder.split(_SOURCE_TRANSCRIPT_CLOSE, 1)
    if (
        trailing.strip()
        or len(payload_text.encode("utf-8")) > _SOURCE_TRANSCRIPT_MAX_UTF8_BYTES
        or _SUMMARY_ATTRIBUTION_MARKER_RE.search(payload_text)
    ):
        return False
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        return False
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


@dataclass(frozen=True)
class HistoricalSourceLane:
    """One exact physical source lane safe to project to a model."""

    speaker: str
    role: str
    content: str
    session_date: str = ""
    requester_match: str = "unproved"
    actor_id: str = field(default="", repr=False)


@dataclass(frozen=True)
class SummarySourceProjection:
    """Validated canonical replacement for one generated summary."""

    lanes: tuple[HistoricalSourceLane, ...] = ()
    complete: bool = False


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
        or store is None
        or not conversation_id
        or speaker_context is None
        or not getattr(speaker_context, "eligible", False)
    ):
        return empty
    getter = getattr(store, "get_canonical_turn_rows_by_id", None)
    if not callable(getter):
        return empty

    ids_by_item: list[list[str]] = []
    keys: list[tuple[str, str]] = []
    for item in materialized:
        metadata = _metadata_for(item)
        raw_ids = getattr(metadata, "canonical_turn_ids", None)
        if not isinstance(raw_ids, list) or not all(
            type(value) is str and bool(value.strip()) for value in raw_ids
        ):
            ids: list[str] = []
        else:
            ids = list(dict.fromkeys(value.strip() for value in raw_ids))
            if len(ids) > _SOURCE_TRANSCRIPT_MAX_LANES:
                ids = []
        ids_by_item.append(ids)
        keys.extend((conversation_id, canonical_id) for canonical_id in ids)
    if not keys:
        return empty
    try:
        rows = getter(list(dict.fromkeys(keys)), speaker_context=speaker_context)
    except Exception:
        return empty

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
            getattr(metadata, "source_mapping_complete", False) is True
            and canonical_ids
        )
        lanes: list[HistoricalSourceLane] = []
        label_actors: dict[str, set[str]] = {}

        for canonical_id in canonical_ids:
            row = rows.get((conversation_id, canonical_id))
            if row is None or not _source_row_is_admitted(row, speaker_context):
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
    colliding = {
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


def render_source_projection_for_model(
    projection: SummarySourceProjection | None,
) -> str:
    """Serialize one canonical projection without generated prose or ids."""
    if projection is None or not projection.complete or not projection.lanes:
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
        ref: render_source_projection_for_model(projection)
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


def render_summaries_for_model(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list[str]:
    """Batch-resolve and render objects carrying ``summary`` + ``metadata``."""
    materialized = list(items)
    projections = resolve_summary_source_projections(
        materialized,
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )
    return [
        render_source_projection_for_model(projection)
        for projection in projections
    ]


def sanitize_summary_payload_for_model(
    payload: object,
    *,
    allow_proved_renderings: bool = False,
) -> object:
    """Quarantine anonymous derived prose in a JSON-like model payload.

    This stateless boundary is for aggregate/search/time tools whose result no
    longer carries enough exact row provenance to bridge a legacy referent.  It
    mutates dictionaries/lists in place, matching the existing speaker-field
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
                            and is_proved_summary_rendering(value)
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
                                and is_proved_summary_rendering(item)
                            )
                            else render_summary_for_model(
                                item, require_proved_scope=True,
                            )
                        )
                        if isinstance(item, str)
                        else sanitize_summary_payload_for_model(
                            item,
                            allow_proved_renderings=allow_proved_renderings,
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
            )
    elif isinstance(payload, list):
        for value in payload:
            sanitize_summary_payload_for_model(
                value,
                allow_proved_renderings=allow_proved_renderings,
            )
    return payload
