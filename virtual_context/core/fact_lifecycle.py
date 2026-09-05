"""Immutable proposals and deterministic admission for derived fact changes.

Extraction and comparison models propose changes; source identity, current
audience proof and chronology decide whether a proposal may change live state.
The same pure policy runs before model selection and inside the store transaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from collections.abc import Mapping, Sequence
import hashlib
import json

from ..types import Fact, REPLY_ATTRIBUTION_VERSION

POLICY_VERSION = "fact-admission-v1"


def fact_version(fact: Fact) -> str:
    """Stable persisted-fact fingerprint for optimistic proposal admission."""
    value = asdict(fact)
    value.pop("session_date", None)  # Query-derived presentation, not source state.
    value["tags"] = sorted(set(value.get("tags") or []))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class FactProposal:
    action: str
    old_fact_id: str
    new_fact_id: str
    proposed_fields: tuple[tuple[str, str], ...] = ()
    observed_at: str = ""
    event_date: str = ""
    source_versions: tuple[tuple[str, str], ...] = ()
    policy_version: str = POLICY_VERSION
    expected_old_version: str = ""
    expected_new_version: str = ""

    def __post_init__(self):
        for name in ("proposed_fields", "source_versions"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(pair) is not tuple or len(pair) != 2 or any(type(part) is not str for part in pair) for pair in value):
                raise TypeError(f"{name} must be an immutable tuple of string pairs")


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    policy_version: str = POLICY_VERSION


def parse_fact_date(value: str) -> date | None:
    if not value or value == "(unknown)":
        return None
    for pattern in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], pattern).date()
        except (ValueError, TypeError):
            pass
    return None


def is_attributed(fact: Fact) -> bool:
    return bool(fact.author_actor_id or fact.author_attribution_version or fact.author_source_role or fact.author_source_message_id)


def source_author_matches(fact: Fact, rows: Sequence[Mapping]) -> bool:
    """Re-prove the stored author against the current physical source rows.

Version two's requester ID identifies its own physical message; a subject ID
identifies the reply target, as stamped by community attribution. An ambiguous
external ID cannot stand in for physical identity. Version one requires a
complete one-human, non-reply roster, matching its original admission policy.
"""
    if not is_attributed(fact):
        return True  # Explicit legacy lane; audience policy remains separate.
    if not fact.author_actor_id or type(fact.author_attribution_version) is not int:
        return False
    values = [dict(row) for row in rows]
    if not values or any(row.get("conversation_id") != fact.conversation_id for row in values):
        return False
    humans = [row for row in values if (row.get("user_content") or "").strip()]
    if fact.author_attribution_version == 1:
        if fact.author_source_role not in ("", "requester") or not humans:
            return False
        if any(row.get("reply_attribution_version") or row.get("reply_target_body") or row.get("reply_target_message_id") or row.get("reply_subject_actor_id") for row in humans):
            return False
        return {row.get("sender_actor_id") or "" for row in humans} == {fact.author_actor_id}
    if fact.author_attribution_version != 2 or not fact.author_source_message_id:
        return False
    if fact.author_source_role == "requester":
        matches = [row for row in humans if row.get("source_message_id") == fact.author_source_message_id]
        return len(matches) == 1 and matches[0].get("sender_actor_id") == fact.author_actor_id
    if fact.author_source_role == "subject":
        matches = [row for row in humans if row.get("reply_target_message_id") == fact.author_source_message_id]
        if len(matches) != 1:
            return False
        row = matches[0]
        # A present physical target has its own requester lane. Attribution
        # intentionally suppresses a duplicate quoted subject lane in this case.
        if any(candidate.get("source_message_id") == fact.author_source_message_id
               and candidate.get("audience_conversation_id") == row.get("audience_conversation_id")
               and (candidate.get("origin_channel_id") or "") == (row.get("origin_channel_id") or "") for candidate in humans):
            return False
        version = row.get("reply_attribution_version")
        return (type(version) is int and version == REPLY_ATTRIBUTION_VERSION
                and bool((row.get("reply_target_body") or "").strip())
                and row.get("reply_subject_actor_id") == fact.author_actor_id)
    return False  # Assistant text is never proof of a human author's claim.


def decide_supersession(
    new: Fact, old: Fact, *,
    new_audience: tuple[str, str] | None = None,
    old_audience: tuple[str, str] | None = None,
) -> AdmissionDecision:
    """Admit only an active replacement in the same proved source scope.

Legacy facts with no attribution fields retain explicitly labelled local
behavior. An attributed endpoint never falls back to this compatibility lane.
An unknown date cannot manufacture chronology; equal/unknown dates preserve
legacy comparison behavior but are visible in the decision reason.
"""
    def reject(reason):
        return AdmissionDecision(False, reason)

    if not new.id or not old.id or new.id == old.id:
        return reject("invalid_fact_identity")
    if not new.conversation_id or new.conversation_id != old.conversation_id:
        return reject("conversation_mismatch")
    if new.superseded_by or old.superseded_by:
        return reject("inactive_fact")
    if not new.subject.strip() or new.subject.strip().casefold() != old.subject.strip().casefold():
        return reject("subject_mismatch")
    attributed = is_attributed(new) or is_attributed(old)
    if attributed:
        if not is_attributed(new) or not is_attributed(old):
            return reject("attribution_mismatch")
        if type(new.author_attribution_version) is not int or type(old.author_attribution_version) is not int or new.author_attribution_version not in (1, 2) or old.author_attribution_version not in (1, 2):
            return reject("unproved_author_version")
        if not new.author_actor_id or new.author_actor_id != old.author_actor_id:
            return reject("author_mismatch")
        if new.author_source_role != old.author_source_role:
            return reject("source_role_mismatch")
        if new.author_source_role not in ("", "requester", "subject", "assistant"):
            return reject("unproved_source_role")
        if not new_audience or not old_audience or not new_audience[0] or not old_audience[0]:
            return reject("unproved_audience")
        if new_audience != old_audience:
            return reject("audience_mismatch")
    elif new_audience or old_audience:
        # Legacy author fields never erase an audience that canonical rows can
        # already prove (for example, two channels merged into one owner).
        if not new_audience or not old_audience:
            return reject("unproved_audience")
        if new_audience != old_audience:
            return reject("audience_mismatch")
    if new.status == "planned" and old.status != "planned":
        return reject("plan_is_not_observed_outcome")
    new_date = parse_fact_date(new.when_date or new.session_date)
    old_date = parse_fact_date(old.when_date or old.session_date)
    if new_date and old_date and old_date > new_date:
        return reject("older_evidence")
    reason = "proved_source_scope" if attributed else "legacy_unattributed"
    if new_date is None or old_date is None:
        reason += ":unknown_chronology"
    return AdmissionDecision(True, reason)
