"""Model-visible presentation of stored summaries.

A stored summary is shown as the text the summarizer wrote; speakers are named
at write time. The one boundary enforced when reading is audience: a summary
whose source turns were written in a different audience (another guild or a
DM attached to the same owner) is withheld from this request. Rows that carry
no audience are the owner's own history.

The label helpers below keep display labels unambiguous where summaries and
speaker rosters are written.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, TYPE_CHECKING

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


SUMMARY_ATTRIBUTION_QUARANTINE = (
    "[summary withheld: its source turns belong to another conversation]"
)

_DEPTHS = frozenset({"summary", "segments", "full"})


def _request_audience(
    speaker_context: "SpeakerRetrievalContext | None",
    conversation_id: str,
) -> str:
    """The audience this request reads as: its proved route, else the owner."""
    audience = ""
    if speaker_context is not None and getattr(speaker_context, "eligible", False):
        audience = str(getattr(speaker_context, "audience_conversation_id", "") or "")
    owner = str(getattr(speaker_context, "owner_conversation_id", "") or "") if speaker_context else ""
    return (audience or owner or conversation_id or "").strip()


def _source_ids(item: object) -> list[str]:
    metadata = getattr(item, "metadata", None)
    raw = getattr(metadata, "canonical_turn_ids", None)
    if not raw:
        raw = getattr(item, "source_canonical_turn_ids", None)
    if not isinstance(raw, (list, tuple)):
        return []
    return [value.strip() for value in raw if isinstance(value, str) and value.strip()]


def _load_audiences(store: object | None, keys: list[tuple[str, str]]) -> dict | None:
    """Stored audience of each exact source key; ``None`` when the lookup failed."""
    if not keys:
        return {}
    getter = getattr(store, "get_canonical_turn_audiences", None)
    rows_getter = getattr(store, "get_canonical_turn_rows_by_id", None)
    try:
        if callable(getter):
            audiences = getter(keys)
        elif callable(rows_getter):
            try:
                rows = rows_getter(keys, speaker_context=None, internal_validation=True)
            except TypeError:
                rows = rows_getter(keys, speaker_context=None)
            audiences = {
                key: (getattr(row, "audience_conversation_id", ""),
                      getattr(row, "audience_attribution_version", 0))
                for key, row in rows.items()
            }
        else:
            return {}
    except Exception:
        return None
    return audiences if hasattr(audiences, "get") else None


def summaries_admitted_for_audience(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list[bool]:
    """Whether each item may be shown to a request in its audience.

    An item is withheld when one of its source turns carries a proved
    audience other than the request's, or when its source turns could not be
    read.
    """
    materialized = list(items)
    audience = _request_audience(speaker_context, conversation_id)
    ids_by_item = [_source_ids(item) for item in materialized]
    keys = list(dict.fromkeys(
        (conversation_id, source_id)
        for ids in ids_by_item for source_id in ids
    ))
    audiences = _load_audiences(store, keys)
    admitted: list[bool] = []
    for ids in ids_by_item:
        if audiences is None:
            # The audience could not be checked, so nothing sourced is shown.
            admitted.append(not ids)
            continue
        foreign = False
        for source_id in ids:
            stored = audiences.get((conversation_id, source_id))
            if not isinstance(stored, tuple) or len(stored) != 2:
                continue
            row_audience, version = stored
            if not isinstance(row_audience, str) or not isinstance(version, int):
                continue
            if row_audience.strip() and version > 0 and row_audience.strip() != audience:
                foreign = True
                break
        admitted.append(not foreign)
    return admitted


def _text_for_depth(item: object, depth: str) -> str:
    if depth == "full":
        full = getattr(item, "full_text", None)
        if isinstance(full, str) and full.strip():
            return full
    summary = getattr(item, "summary", None)
    return summary if isinstance(summary, str) else ""


def render_summary_items_for_model(
    requests: Iterable[tuple[object, object]],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
    judgment_runtime: object | None = None,
) -> list[str]:
    """Stored text for each ``(item, depth)``, or the withheld marker."""
    materialized = [
        (item, str(getattr(depth, "value", depth) or "summary").strip().lower())
        for item, depth in requests
    ]
    admitted = summaries_admitted_for_audience(
        (item for item, _depth in materialized),
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )
    rendered: list[str] = []
    for (item, depth), ok in zip(materialized, admitted, strict=True):
        text = _text_for_depth(item, depth if depth in _DEPTHS else "summary")
        rendered.append(text if ok and text.strip() else SUMMARY_ATTRIBUTION_QUARANTINE)
    return rendered


def render_summaries_for_model(
    items: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
    depth: object = "summary",
    judgment_runtime: object | None = None,
) -> list[str]:
    """Render one depth across a batch."""
    return render_summary_items_for_model(
        ((item, depth) for item in items),
        store=store,
        conversation_id=conversation_id,
        speaker_context=speaker_context,
    )


def render_summary_for_model(text: object, *_args, **_kwargs) -> str:
    """Stored summary text as the model sees it."""
    return text if isinstance(text, str) else ""


def is_proved_summary_rendering(text: object) -> bool:
    """Whether *text* is a summary the renderer admitted."""
    return isinstance(text, str) and bool(text.strip()) and text != SUMMARY_ATTRIBUTION_QUARANTINE


def sanitize_summary_payload_for_model(payload: object, **_kwargs) -> object:
    """Tool payloads carry stored summary text unchanged."""
    return payload


def facts_admitted_for_audience(
    facts: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list[bool]:
    """Whether each fact may be shown to a request in its audience.

    A fact's source turns are those of the segment it was extracted from, so
    a fact is admitted exactly when its segment would be. A fact with no
    segment carries no source turns and is admitted, as an unsourced summary
    is; a segment that could not be read withholds its facts. A store with no
    segment reader has no sources to check, as with summaries.
    """
    materialized = list(facts)
    refs = list(dict.fromkeys(
        ref for ref in (str(getattr(f, "segment_ref", "") or "").strip() for f in materialized)
        if ref
    ))
    segment_ok: dict[str, bool] = {}
    loaded: list[tuple[str, object]] = []
    getter = getattr(store, "get_segment", None)
    for ref in refs:
        if not callable(getter):
            segment_ok[ref] = True
            continue
        try:
            segment = getter(ref, conversation_id=conversation_id or None)
        except Exception:
            segment_ok[ref] = False
            continue
        if segment is None:
            segment_ok[ref] = True
        else:
            loaded.append((ref, segment))
    if loaded:
        admitted = summaries_admitted_for_audience(
            (segment for _ref, segment in loaded),
            store=store,
            conversation_id=conversation_id,
            speaker_context=speaker_context,
        )
        for (ref, _segment), ok in zip(loaded, admitted, strict=True):
            segment_ok[ref] = ok
    return [
        segment_ok.get(str(getattr(f, "segment_ref", "") or "").strip(), True)
        for f in materialized
    ]


def facts_for_audience(
    facts: Iterable[object],
    *,
    store: object | None,
    conversation_id: str,
    speaker_context: "SpeakerRetrievalContext | None",
) -> list:
    """``facts`` without those whose source turns belong to another audience."""
    materialized = list(facts)
    admitted = facts_admitted_for_audience(
        materialized, store=store, conversation_id=conversation_id,
        speaker_context=speaker_context,
    )
    return [fact for fact, ok in zip(materialized, admitted, strict=True) if ok]
