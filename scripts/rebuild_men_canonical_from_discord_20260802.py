#!/usr/bin/env python3
"""Build an externally attested Men-guild canonical-history replacement.

This script is deliberately read-only with respect to the database.  It turns
the frozen Discord guild transcript into a deterministic, reviewable staging
artifact.  Raw Discord defines identity, chronology, channel, reply edges, and
the delivered Vast response chunks.  OpenClaw user events may corroborate a
raw trigger; they are never allowed to change a raw Discord author.  The sole
deleted trigger can be recovered only when its Discord snowflake timestamp,
channel, and one unique OpenClaw event agree exactly.

The artifact is consumed by a separate guarded database swap only after a
restored-snapshot rehearsal and independent review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from virtual_context.core.canonical_turns import (
    HASH_VERSION,
    compute_turn_hash_from_raw,
    normalize_turn_text,
)
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    REPLY_ATTRIBUTION_VERSION,
)


MEN_GUILD_ID = "1524917037191925871"
VAST_USER_ID = "1485681229608259666"
OWNER_CONVERSATION_ID = f"sk:agent:vast:discord:guild:{MEN_GUILD_ID}"
AGENT_SCOPE_ID = "vast"
ACCOUNT_ID = "vast"
PLATFORM = "discord"
PROJECTION_VERSION = "discord-history-raw-v1"
DELETED_PROJECTION_VERSION = "discord-snowflake+archived-session-chain-v1"
DISPOSITION_SHA256 = (
    "bff6c25c08b66b1254412c7c2b47aa82efa830730c00d24cbbce648972ac12b8"
)
TRANSCRIPT_SHA256 = (
    "2f90cfd63159ba2428cb4c0ec7de067b6d7e2785bcef4a1461a7b6ae402d4eb6"
)
RECOVERY_SESSION_BUNDLE_SHA256 = (
    "ccb0ffd142caf902afe5dc1793fe3b8c1d0a711b84d9f76cca9a506b56a95cdb"
)
EXPECTED_RAW_DISCORD_MESSAGES = 11667
EXPECTED_RAW_VAST_DELIVERIES = 2204
EXPECTED_CANONICAL_DELIVERIES = 1961
EXPECTED_EXCLUDED_DELIVERIES = 237
EXPECTED_QUARANTINED_DELIVERIES = 6
EXPECTED_CANONICAL_GROUPS = 1594
EXPECTED_MULTI_CHUNK_GROUPS = 215
EXPECTED_MAX_CHUNKS = 18
EXPECTED_RECOVERY_EVIDENCE_RECORDS = 533
EXPECTED_RECOVERY_SESSION_FILES = 14
EXPECTED_CANONICAL_ROWS = 3188
EXPECTED_SOURCE_MEMBERSHIPS = 1594
EXPECTED_ACTORS = 18
DELETED_TRIGGER_ID = "1533227206493605888"
CANONICAL_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://virtual-context.local/canonical/discord-history/v1",
)
MENTION_RE = re.compile(r"<@!?(\d{15,24})>")


class EvidenceError(RuntimeError):
    """Raised when the frozen external evidence is ambiguous or inconsistent."""


@dataclass(frozen=True)
class OpenClawUserEvent:
    channel_id: str
    timestamp_ms: int
    sender_id: str
    sender_name: str
    content: str
    session_id: str
    event_id: str
    mirror_identity: str
    idempotency_key: str


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvidenceError(f"non-object JSONL {path}:{line_number}")
            yield value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso_to_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def _snowflake_ms(message_id: str) -> int:
    if not re.fullmatch(r"\d{15,24}", message_id or ""):
        raise EvidenceError(f"invalid Discord snowflake: {message_id!r}")
    return (int(message_id) >> 22) + 1420070400000


def _event_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part).strip()


def _preferred_name(message: dict[str, Any]) -> str:
    names = message.get("author_names")
    if isinstance(names, list):
        for value in names:
            if isinstance(value, str) and value.strip():
                return value.strip()
    author_id = str(message.get("author_id") or "").strip()
    return f"Discord user {author_id}" if author_id else "Unknown speaker"


def _build_name_index(messages: Iterable[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for message in messages:
        author_id = str(message.get("author_id") or "").strip()
        if not author_id:
            continue
        names = message.get("author_names")
        if isinstance(names, list):
            for rank, value in enumerate(names):
                if isinstance(value, str) and value.strip():
                    # Prefer the first (display) name while retaining enough
                    # observations for a deterministic fallback.
                    counts[author_id][value.strip()] += 2 if rank == 0 else 1
    counts[VAST_USER_ID]["Vast"] += 1_000_000
    return {
        author_id: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for author_id, counter in counts.items()
        if counter
    }


def _attachment_marker(attachment: dict[str, Any]) -> str:
    filename = str(attachment.get("filename") or attachment.get("id") or "attachment")
    media_type = str(attachment.get("content_type") or "").strip()
    if media_type:
        return f"[attachment: {filename} ({media_type})]"
    return f"[attachment: {filename}]"


def _project_discord_body(
    message: dict[str, Any],
    names_by_id: dict[str, str],
) -> str:
    content = str(message.get("content") or "").strip()

    def replace_mention(match: re.Match[str]) -> str:
        member_id = match.group(1)
        return "@" + names_by_id.get(member_id, f"discord:{member_id}")

    content = MENTION_RE.sub(replace_mention, content)
    supplements: list[str] = []
    for attachment in message.get("attachments") or []:
        if isinstance(attachment, dict):
            supplements.append(_attachment_marker(attachment))
    for sticker in message.get("sticker_items") or []:
        if isinstance(sticker, dict):
            label = str(sticker.get("name") or sticker.get("id") or "sticker")
            supplements.append(f"[sticker: {label}]")
    if not content:
        for embed in message.get("embeds") or []:
            if not isinstance(embed, dict):
                continue
            title = str(embed.get("title") or "").strip()
            description = str(embed.get("description") or "").strip()
            url = str(embed.get("url") or "").strip()
            rendered = " — ".join(value for value in (title, description, url) if value)
            if rendered:
                supplements.append(f"[embed: {rendered}]")
    projected = "\n".join(value for value in (content, *supplements) if value).strip()
    return projected or "[empty Discord message]"


def _project_assistant_group(
    responses: list[dict[str, Any]],
    names_by_id: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    chunks: list[str] = []
    raw_blocks: list[dict[str, str]] = []
    for response in responses:
        text = _project_discord_body(response, names_by_id)
        chunks.append(text)
        raw_blocks.append({
            "type": "text",
            "text": text,
            "source_message_id": str(response.get("id") or ""),
        })
    return "\n\n".join(chunks).strip(), raw_blocks


def _load_openclaw_users(
    sessions_dir: Path,
) -> tuple[dict[tuple[str, str, int], list[OpenClawUserEvent]], dict[str, int]]:
    sessions_path = sessions_dir / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    if not isinstance(sessions, dict):
        raise EvidenceError("sessions.json is not an object")
    session_to_channel: dict[str, str] = {}
    for key, value in sessions.items():
        match = re.fullmatch(r"agent:vast:discord:channel:(\d{15,24})", str(key))
        if not match or not isinstance(value, dict):
            continue
        session_id = str(value.get("sessionId") or "").strip()
        if session_id and (sessions_dir / f"{session_id}.jsonl").exists():
            session_to_channel[session_id] = match.group(1)

    indexed: dict[tuple[str, str, int], list[OpenClawUserEvent]] = defaultdict(list)
    dedupe: set[str] = set()
    stats = Counter()
    for session_id, channel_id in sorted(session_to_channel.items()):
        for entry in _read_jsonl(sessions_dir / f"{session_id}.jsonl"):
            message = entry.get("message")
            if entry.get("type") != "message" or not isinstance(message, dict):
                continue
            if message.get("role") != "user" or message.get("sourceChannel") != "discord":
                continue
            sender_id = str(message.get("senderId") or "").strip()
            timestamp = message.get("timestamp")
            if not sender_id or not isinstance(timestamp, (int, float)):
                stats["skipped_missing_identity"] += 1
                continue
            timestamp_ms = int(timestamp)
            openclaw = message.get("__openclaw")
            mirror = (
                str(openclaw.get("mirrorIdentity") or "").strip()
                if isinstance(openclaw, dict)
                else ""
            )
            idempotency_key = str(message.get("idempotencyKey") or "").strip()
            dedupe_key = mirror or idempotency_key or (
                f"{session_id}:{entry.get('id')}:{sender_id}:{timestamp_ms}"
            )
            if dedupe_key in dedupe:
                stats["duplicates"] += 1
                continue
            dedupe.add(dedupe_key)
            event = OpenClawUserEvent(
                channel_id=channel_id,
                timestamp_ms=timestamp_ms,
                sender_id=sender_id,
                sender_name=str(message.get("senderName") or "").strip(),
                content=_event_text(message.get("content")),
                session_id=session_id,
                event_id=str(entry.get("id") or ""),
                mirror_identity=mirror,
                idempotency_key=idempotency_key,
            )
            indexed[(channel_id, sender_id, timestamp_ms)].append(event)
            stats["unique_user_events"] += 1
    stats["channel_sessions"] = len(session_to_channel)
    return indexed, dict(stats)


def _exact_openclaw_match(
    index: dict[tuple[str, str, int], list[OpenClawUserEvent]],
    *,
    channel_id: str,
    sender_id: str,
    timestamp_ms: int,
) -> OpenClawUserEvent | None:
    candidates = index.get((channel_id, sender_id, timestamp_ms), [])
    unique = {
        (event.sender_id, event.sender_name, event.content): event
        for event in candidates
    }
    if len(unique) > 1:
        raise EvidenceError(
            "ambiguous OpenClaw match for "
            f"channel={channel_id} sender={sender_id} timestamp={timestamp_ms}"
        )
    return next(iter(unique.values()), None)


def _canonical_id(kind: str, guild_id: str, source: str) -> str:
    return str(uuid.uuid5(CANONICAL_NAMESPACE, f"{kind}:{guild_id}:{source}"))


def _source_raw_content(text: str) -> str:
    return json.dumps(
        [{"type": "text", "text": text}],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_row(
    *,
    canonical_turn_id: str,
    turn_group_number: int,
    sort_key: float,
    role: str,
    content: str,
    raw_content: str,
    session_date: str,
    sender: str,
    channel_id: str,
    channel_label: str,
    sender_actor_id: str,
    source_message_id: str,
    reply_target_message_id: str,
    reply_subject_actor_id: str,
    reply_subject_label: str,
    reply_target_body: str,
    timestamp: str,
    batch_id: str,
    covered_entries: int,
) -> dict[str, Any]:
    user_content = content if role == "user" else ""
    assistant_content = content if role == "assistant" else ""
    turn_hash, norm_user, norm_assistant = compute_turn_hash_from_raw(
        user_content,
        assistant_content,
        version=HASH_VERSION,
    )
    return {
        "canonical_turn_id": canonical_turn_id,
        "conversation_id": OWNER_CONVERSATION_ID,
        "turn_group_number": turn_group_number,
        "sort_key": sort_key,
        "turn_hash": turn_hash,
        "hash_version": HASH_VERSION,
        "normalized_user_text": norm_user,
        "normalized_assistant_text": norm_assistant,
        "user_content": user_content,
        "assistant_content": assistant_content,
        "user_raw_content": raw_content if role == "user" else None,
        "assistant_raw_content": raw_content if role == "assistant" else None,
        "primary_tag": "_general",
        "tags_json": "[]",
        "session_date": session_date,
        "sender": sender if role == "user" else "",
        "fact_signals_json": "[]",
        "code_refs_json": "[]",
        "tagged_at": None,
        "compacted_at": None,
        "first_seen_at": timestamp,
        "last_seen_at": timestamp,
        "source_batch_id": batch_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "covered_ingestible_entries": covered_entries,
        "compaction_operation_id": None,
        "origin_conversation_id": f"sk:agent:vast:discord:channel:{channel_id}",
        "origin_channel_id": channel_id,
        "origin_channel_label": f"#{channel_label}" if channel_label else "",
        "sender_actor_id": sender_actor_id if role == "user" else "",
        "source_message_id": source_message_id if role == "user" else "",
        "reply_target_message_id": reply_target_message_id if role == "user" else "",
        "reply_subject_actor_id": reply_subject_actor_id if role == "user" else "",
        "reply_subject_label": reply_subject_label if role == "user" else "",
        "reply_target_body": reply_target_body if role == "user" else "",
        "reply_attribution_version": (
            REPLY_ATTRIBUTION_VERSION
            if role == "user" and reply_target_message_id
            else 0
        ),
        "audience_conversation_id": OWNER_CONVERSATION_ID,
        "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
    }


def _raw_body_equivalent(raw: str, openclaw: str) -> bool:
    normalized_raw = MENTION_RE.sub(
        lambda match: "@Vast" if match.group(1) == VAST_USER_ID else match.group(0),
        raw,
    )
    return normalize_turn_text(normalized_raw) == normalize_turn_text(openclaw)


def _load_disposition_groups(
    *,
    disposition_path: Path,
    all_messages: list[dict[str, Any]],
    by_message_id: dict[str, dict[str, Any]],
    sessions_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the all-delivery disposition and return canonical response groups.

    The frozen disposition is the authority for whether each raw Vast delivery
    is a conversational response.  Raw Discord remains the authority for every
    delivered byte, timestamp, channel, attachment, and available trigger
    sender.  Nothing is inferred from adjacency.
    """
    disposition_sha = _sha256_file(disposition_path)
    if disposition_sha != DISPOSITION_SHA256:
        raise EvidenceError(
            "unexpected all-delivery disposition checksum: "
            f"{disposition_sha} != {DISPOSITION_SHA256}"
        )
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    if not isinstance(disposition, dict) or disposition.get("schema_version") != 1:
        raise EvidenceError("unsupported all-delivery disposition schema")
    records = disposition.get("records")
    if not isinstance(records, list):
        raise EvidenceError("all-delivery disposition records are missing")

    expected_decisions = {
        "canonical_response_member": EXPECTED_CANONICAL_DELIVERIES,
        "exclude_from_canonical_response_membership": EXPECTED_EXCLUDED_DELIVERIES,
        "manual_review_rejected_by_exact_authority": EXPECTED_QUARANTINED_DELIVERIES,
    }
    if (
        disposition.get("raw_vast_delivery_count") != EXPECTED_RAW_VAST_DELIVERIES
        or disposition.get("unique_response_id_count") != EXPECTED_RAW_VAST_DELIVERIES
        or disposition.get("decision_counts") != expected_decisions
        or len(records) != EXPECTED_RAW_VAST_DELIVERIES
    ):
        raise EvidenceError("all-delivery disposition anti-vacuity counts changed")
    actual_decisions = Counter(str(record.get("decision") or "") for record in records)
    if dict(actual_decisions) != expected_decisions:
        raise EvidenceError("all-delivery disposition record decisions do not match header")

    raw_vast_messages = [
        message
        for message in all_messages
        if str(message.get("author_id") or "") == VAST_USER_ID
        and bool(message.get("author_bot"))
    ]
    raw_vast_ids = [str(message.get("id") or "") for message in raw_vast_messages]
    if (
        len(raw_vast_ids) != EXPECTED_RAW_VAST_DELIVERIES
        or len(set(raw_vast_ids)) != EXPECTED_RAW_VAST_DELIVERIES
    ):
        raise EvidenceError("raw Discord Vast-delivery coverage changed")

    record_ids = [str(record.get("response_id") or "") for record in records]
    if len(set(record_ids)) != EXPECTED_RAW_VAST_DELIVERIES:
        raise EvidenceError("disposition response ids are missing or duplicated")
    if set(record_ids) != set(raw_vast_ids):
        missing = sorted(set(raw_vast_ids) - set(record_ids))
        extra = sorted(set(record_ids) - set(raw_vast_ids))
        raise EvidenceError(
            f"disposition/raw Discord response coverage differs: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    special_deleted_trigger: dict[str, Any] | None = None
    canonical_categories = Counter()
    for record in records:
        response_id = str(record.get("response_id") or "")
        raw_response = by_message_id.get(response_id)
        if raw_response is None:
            raise EvidenceError(f"disposition response {response_id} is absent from Discord")
        raw_attachment_filenames = [
            str(attachment.get("filename") or "")
            for attachment in raw_response.get("attachments") or []
            if isinstance(attachment, dict)
        ]
        exact_fields = {
            "channel_id": str(raw_response.get("channel_id") or ""),
            "channel_name": str(raw_response.get("channel_name") or ""),
            "content": str(raw_response.get("content") or ""),
            "timestamp": str(raw_response.get("timestamp") or ""),
            "discord_type": raw_response.get("type"),
            "attachment_filenames": raw_attachment_filenames,
        }
        for field, expected in exact_fields.items():
            if record.get(field) != expected:
                raise EvidenceError(
                    f"disposition response {response_id} disagrees with Discord field {field}"
                )
        if record.get("decision") != "canonical_response_member":
            continue

        category = str(record.get("category") or "")
        canonical_categories[category] += 1
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            raise EvidenceError(f"canonical response {response_id} lacks evidence")
        selectors = [
            str(evidence.get(field) or "").strip()
            for field in ("trigger_id", "trigger_message_id", "raw_trigger_id")
            if str(evidence.get(field) or "").strip()
        ]
        if not selectors or len(set(selectors)) != 1:
            raise EvidenceError(
                f"canonical response {response_id} has ambiguous trigger selectors"
            )
        trigger_id = selectors[0]
        if str(raw_response.get("guild_id") or "") != MEN_GUILD_ID:
            raise EvidenceError(f"response {response_id} is outside the Men guild")
        if category == "native_referenced_response" and (
            str(raw_response.get("reference_message_id") or "") != trigger_id
        ):
            raise EvidenceError(f"native response {response_id} lost its Discord reference")
        if category == "native_reference_plus_session_chain_recovery":
            if special_deleted_trigger is not None or trigger_id != DELETED_TRIGGER_ID:
                raise EvidenceError("deleted-trigger recovery is not singular and exact")
            if trigger_id in by_message_id or evidence.get("raw_trigger_available") is not False:
                raise EvidenceError("deleted-trigger recovery unexpectedly has a raw trigger")
            trigger_ms = _snowflake_ms(trigger_id)
            required_special = {
                "session_channel_id": str(raw_response.get("channel_id") or ""),
                "trigger_snowflake_timestamp_ms": trigger_ms,
                "user_native_timestamp_ms": trigger_ms,
                "user_source_channel": "discord",
                "assistant_body_literal_equals_raw_response": True,
            }
            if any(evidence.get(key) != value for key, value in required_special.items()):
                raise EvidenceError("deleted-trigger embedded proof is inconsistent")
            session_file = Path(str(evidence.get("session_file") or "")).name
            if not session_file or not (sessions_dir / session_file).exists():
                raise EvidenceError("deleted-trigger archived session proof is unavailable")
            sender_id = str(evidence.get("user_sender_id") or "").strip()
            sender_name = str(evidence.get("user_sender_name") or "").strip()
            user_body = str(evidence.get("user_body") or "").strip()
            if not sender_id or not sender_name or not user_body:
                raise EvidenceError("deleted-trigger embedded sender proof is incomplete")
            special_deleted_trigger = {
                "id": trigger_id,
                "guild_id": MEN_GUILD_ID,
                "channel_id": str(raw_response.get("channel_id") or ""),
                "channel_name": str(evidence.get("session_channel_name") or "").lstrip("#"),
                "author_id": sender_id,
                "author_names": [sender_name],
                "author_bot": False,
                "content": user_body,
                "attachments": [],
                "embeds": [],
                "sticker_items": [],
                "reference_message_id": "",
                "timestamp": datetime.fromtimestamp(
                    trigger_ms / 1000,
                    timezone.utc,
                ).isoformat(),
            }
        grouped[trigger_id].append((record, raw_response))

    expected_categories = {
        "native_referenced_response": 1428,
        "recovered_literal": 445,
        "recovered_fence": 78,
        "recovered_whitespace": 8,
        "recovered_both": 1,
        "native_reference_plus_session_chain_recovery": 1,
    }
    if dict(canonical_categories) != expected_categories:
        raise EvidenceError("canonical disposition categories changed")
    if len(grouped) != EXPECTED_CANONICAL_GROUPS:
        raise EvidenceError(
            f"canonical response group count changed: {len(grouped)}"
        )

    groups: list[dict[str, Any]] = []
    multi_chunk_groups = 0
    max_chunks = 0
    available_trigger_count = 0
    for trigger_id, record_responses in grouped.items():
        record_responses.sort(
            key=lambda item: (_snowflake_ms(str(item[1].get("id") or "")), str(item[1].get("id") or ""))
        )
        records_for_group = [item[0] for item in record_responses]
        responses = [item[1] for item in record_responses]
        channels = {str(response.get("channel_id") or "") for response in responses}
        if len(channels) != 1:
            raise EvidenceError(f"trigger {trigger_id} crosses Discord channels")
        trigger = by_message_id.get(trigger_id)
        trigger_available = trigger is not None
        if trigger_available:
            available_trigger_count += 1
            if (
                str(trigger.get("guild_id") or "") != MEN_GUILD_ID
                or bool(trigger.get("author_bot"))
                or str(trigger.get("channel_id") or "") not in channels
            ):
                raise EvidenceError(f"raw trigger {trigger_id} has invalid authority")
        elif trigger_id == DELETED_TRIGGER_ID:
            trigger = special_deleted_trigger
        else:
            raise EvidenceError(f"canonical trigger {trigger_id} is absent from raw Discord")
        if not isinstance(trigger, dict):
            raise EvidenceError(f"canonical trigger {trigger_id} was not reconstructed")
        if _snowflake_ms(trigger_id) > min(
            _snowflake_ms(str(response.get("id") or "")) for response in responses
        ):
            raise EvidenceError(f"trigger {trigger_id} occurs after its response")
        chunk_count = len(responses)
        if chunk_count > 1:
            multi_chunk_groups += 1
        max_chunks = max(max_chunks, chunk_count)
        groups.append({
            "trigger_message_id": trigger_id,
            "trigger_available": trigger_available,
            "trigger": trigger,
            "response_message_ids": [str(response.get("id") or "") for response in responses],
            "responses": responses,
            "disposition_categories": [str(record.get("category") or "") for record in records_for_group],
            "disposition_evidence": [record.get("evidence") for record in records_for_group],
        })
    if (
        available_trigger_count != EXPECTED_CANONICAL_GROUPS - 1
        or multi_chunk_groups != EXPECTED_MULTI_CHUNK_GROUPS
        or max_chunks != EXPECTED_MAX_CHUNKS
        or sum(len(group["responses"]) for group in groups) != EXPECTED_CANONICAL_DELIVERIES
    ):
        raise EvidenceError("canonical response-group structure changed")

    return groups, {
        "sha256": disposition_sha,
        "raw_vast_deliveries": EXPECTED_RAW_VAST_DELIVERIES,
        "canonical_deliveries": EXPECTED_CANONICAL_DELIVERIES,
        "excluded_deliveries": EXPECTED_EXCLUDED_DELIVERIES,
        "quarantined_deliveries": EXPECTED_QUARANTINED_DELIVERIES,
        "canonical_groups": EXPECTED_CANONICAL_GROUPS,
        "multi_chunk_groups": EXPECTED_MULTI_CHUNK_GROUPS,
        "max_chunks": EXPECTED_MAX_CHUNKS,
    }


def _recovery_session_bundle(
    disposition_path: Path,
    sessions_dir: Path,
) -> dict[str, Any]:
    """Pin every archived session file used to recover response membership."""
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    records = disposition.get("records") if isinstance(disposition, dict) else None
    if not isinstance(records, list):
        raise EvidenceError("disposition records unavailable for session bundle")
    recovery_categories = {
        "recovered_literal",
        "recovered_fence",
        "recovered_whitespace",
        "recovered_both",
        "native_reference_plus_session_chain_recovery",
    }
    recovery_records = [
        record for record in records
        if record.get("decision") == "canonical_response_member"
        and str(record.get("category") or "") in recovery_categories
    ]
    if len(recovery_records) != EXPECTED_RECOVERY_EVIDENCE_RECORDS:
        raise EvidenceError("recovery-session evidence record count changed")
    filenames: set[str] = set()
    for record in recovery_records:
        evidence = record.get("evidence")
        filename = Path(str(
            evidence.get("session_file") if isinstance(evidence, dict) else ""
        )).name
        if not filename or not (sessions_dir / filename).is_file():
            raise EvidenceError("recovery evidence references a missing session file")
        filenames.add(filename)
    if len(filenames) != EXPECTED_RECOVERY_SESSION_FILES:
        raise EvidenceError("recovery-session file coverage changed")
    files = [
        {
            "filename": filename,
            "sha256": _sha256_file(sessions_dir / filename),
            "bytes": (sessions_dir / filename).stat().st_size,
        }
        for filename in sorted(filenames)
    ]
    checksum_text = "".join(
        f"{item['sha256']}  {item['filename']}\n" for item in files
    )
    bundle_sha = _sha256_text(checksum_text)
    if bundle_sha != RECOVERY_SESSION_BUNDLE_SHA256:
        raise EvidenceError(
            "recovery-session evidence bundle changed: "
            f"{bundle_sha} != {RECOVERY_SESSION_BUNDLE_SHA256}"
        )
    return {
        "sha256": bundle_sha,
        "evidence_records": len(recovery_records),
        "files": files,
    }


def build_artifact(
    *,
    transcript_dir: Path,
    sessions_dir: Path,
    old_canonical_path: Path,
    disposition_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    messages_path = transcript_dir / "messages.jsonl"
    transcript_sha = _sha256_file(messages_path)
    if transcript_sha != TRANSCRIPT_SHA256:
        raise EvidenceError(
            f"unexpected Discord transcript checksum: {transcript_sha}"
        )
    all_messages = list(_read_jsonl(messages_path))
    if len(all_messages) != EXPECTED_RAW_DISCORD_MESSAGES:
        raise EvidenceError("raw Discord transcript message count changed")
    by_message_id = {
        str(message.get("id") or ""): message
        for message in all_messages
        if str(message.get("id") or "")
    }
    names_by_id = _build_name_index(all_messages)
    openclaw_index, openclaw_stats = _load_openclaw_users(sessions_dir)
    recovery_session_bundle = _recovery_session_bundle(
        disposition_path,
        sessions_dir,
    )
    old_canonical_sha = _sha256_file(old_canonical_path)
    old_input_rows = list(_read_jsonl(old_canonical_path))

    if any(str(message.get("guild_id") or "") != MEN_GUILD_ID for message in all_messages):
        raise EvidenceError("transcript contains a message outside the Men guild")
    groups, disposition_stats = _load_disposition_groups(
        disposition_path=disposition_path,
        all_messages=all_messages,
        by_message_id=by_message_id,
        sessions_dir=sessions_dir,
    )

    def group_order(group: dict[str, Any]) -> tuple[int, str]:
        trigger_id = str(group.get("trigger_message_id") or "")
        return _snowflake_ms(trigger_id), trigger_id

    groups.sort(key=group_order)
    canonical_rows: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    group_evidence: list[dict[str, Any]] = []
    transport_projection_by_source: dict[str, str] = {}
    seen_trigger_ids: set[str] = set()
    seen_response_ids: set[str] = set()
    repair_batch_id = _canonical_id(
        "batch",
        MEN_GUILD_ID,
        ":".join((
            disposition_stats["sha256"],
            transcript_sha,
            recovery_session_bundle["sha256"],
            old_canonical_sha,
        )),
    )
    stats = Counter()

    for group_number, group in enumerate(groups):
        trigger_id = str(group.get("trigger_message_id") or "").strip()
        if not trigger_id or trigger_id in seen_trigger_ids:
            raise EvidenceError(f"missing or duplicated trigger id: {trigger_id!r}")
        seen_trigger_ids.add(trigger_id)
        responses = group.get("responses")
        if not isinstance(responses, list) or not responses:
            raise EvidenceError(f"trigger {trigger_id} has no responses")
        response_ids = [str(response.get("id") or "") for response in responses]
        declared_response_ids = [str(value) for value in group.get("response_message_ids") or []]
        if response_ids != declared_response_ids:
            raise EvidenceError(f"response id order mismatch for trigger {trigger_id}")
        if any(not value or value in seen_response_ids for value in response_ids):
            raise EvidenceError(f"missing or duplicated response id for trigger {trigger_id}")
        seen_response_ids.update(response_ids)

        trigger = group.get("trigger")
        trigger_available = bool(group.get("trigger_available"))
        projection_version = PROJECTION_VERSION
        deleted_recovery: OpenClawUserEvent | None = None
        if trigger_available:
            if not isinstance(trigger, dict) or str(trigger.get("id") or "") != trigger_id:
                raise EvidenceError(f"invalid available trigger {trigger_id}")
        else:
            # Exactly one deleted trigger has already been reconstructed from
            # the checksummed disposition's archived-session proof. Corroborate
            # its explicit sender/channel/timestamp against the frozen OpenClaw
            # archive; never search a timestamp and choose a nearby speaker.
            first_response = responses[0]
            channel_id = str(first_response.get("channel_id") or "")
            timestamp_ms = _snowflake_ms(trigger_id)
            if (
                trigger_id != DELETED_TRIGGER_ID
                or not isinstance(trigger, dict)
                or str(trigger.get("id") or "") != trigger_id
                or str(trigger.get("channel_id") or "") != channel_id
            ):
                raise EvidenceError(f"unapproved deleted trigger {trigger_id}")
            deleted_recovery = _exact_openclaw_match(
                openclaw_index,
                channel_id=channel_id,
                sender_id=str(trigger.get("author_id") or ""),
                timestamp_ms=timestamp_ms,
            )
            if (
                deleted_recovery is None
                or normalize_turn_text(deleted_recovery.content)
                != normalize_turn_text(str(trigger.get("content") or ""))
            ):
                raise EvidenceError(
                    f"deleted trigger {trigger_id} lacks one exact archived-session event"
                )
            projection_version = DELETED_PROJECTION_VERSION
            stats["deleted_triggers_recovered"] += 1

        assert isinstance(trigger, dict)
        guild_id = str(trigger.get("guild_id") or "")
        channel_id = str(trigger.get("channel_id") or "")
        sender_id = str(trigger.get("author_id") or "")
        timestamp = str(trigger.get("timestamp") or "")
        if (
            guild_id != MEN_GUILD_ID
            or not channel_id
            or not sender_id
            or bool(trigger.get("author_bot"))
            or not timestamp
        ):
            raise EvidenceError(f"invalid trigger authority for {trigger_id}")
        disposition_categories = list(group.get("disposition_categories") or [])
        if len(disposition_categories) != len(responses):
            raise EvidenceError(f"disposition evidence count mismatch for {trigger_id}")
        for response, response_category in zip(
            responses,
            disposition_categories,
            strict=True,
        ):
            if (
                str(response.get("guild_id") or "") != MEN_GUILD_ID
                or str(response.get("channel_id") or "") != channel_id
                or str(response.get("author_id") or "") != VAST_USER_ID
                or not bool(response.get("author_bot"))
            ):
                raise EvidenceError(f"invalid response authority for trigger {trigger_id}")
            raw_reference = str(response.get("reference_message_id") or "")
            if response_category in {
                "native_referenced_response",
                "native_reference_plus_session_chain_recovery",
            }:
                if raw_reference != trigger_id:
                    raise EvidenceError(
                        f"native response lost trigger reference for {trigger_id}"
                    )
            elif response_category.startswith("recovered_"):
                if raw_reference or int(response.get("type") or 0) != 0:
                    raise EvidenceError(
                        f"recovered response has unexpected native reference for {trigger_id}"
                    )
            else:
                raise EvidenceError(
                    f"unsupported canonical response category {response_category!r}"
                )

        trigger_ms = _iso_to_ms(timestamp)
        openclaw_match = deleted_recovery or _exact_openclaw_match(
            openclaw_index,
            channel_id=channel_id,
            sender_id=sender_id,
            timestamp_ms=trigger_ms,
        )
        if openclaw_match:
            stats["openclaw_exact_matches"] += 1
        else:
            stats["openclaw_unmatched_raw_triggers"] += 1

        sender_name = _preferred_name(trigger)
        transport_projection = _project_discord_body(trigger, names_by_id)
        transport_projection_by_source[trigger_id] = transport_projection
        user_content = f"{sender_name}: {transport_projection}"
        assistant_content, assistant_raw_blocks = _project_assistant_group(
            responses,
            names_by_id,
        )
        if not assistant_content:
            raise EvidenceError(f"empty assistant projection for trigger {trigger_id}")

        reply_target_id = str(trigger.get("reference_message_id") or "").strip()
        reply_subject_actor_id = ""
        reply_subject_label = ""
        reply_target_body = ""
        reply_status = "none"
        if reply_target_id:
            target = by_message_id.get(reply_target_id)
            if target is None:
                reply_status = "unavailable"
            else:
                target_author_id = str(target.get("author_id") or "")
                reply_subject_actor_id = (
                    f"actor:discord:{target_author_id}" if target_author_id else ""
                )
                reply_subject_label = _preferred_name(target)
                reply_target_body = _project_discord_body(target, names_by_id)
                reply_status = "resolved"
                stats["reply_edges_resolved"] += 1
        user_id = _canonical_id("user", guild_id, trigger_id)
        assistant_id = _canonical_id(
            "assistant",
            guild_id,
            ",".join(response_ids),
        )
        user_row = _canonical_row(
            canonical_turn_id=user_id,
            turn_group_number=group_number,
            sort_key=float(group_number * 2000 + 1000),
            role="user",
            content=user_content,
            raw_content=_source_raw_content(transport_projection),
            session_date=timestamp[:10],
            sender=sender_name,
            channel_id=channel_id,
            channel_label=str(trigger.get("channel_name") or ""),
            sender_actor_id=f"actor:discord:{sender_id}",
            source_message_id=trigger_id,
            reply_target_message_id=reply_target_id,
            reply_subject_actor_id=reply_subject_actor_id,
            reply_subject_label=reply_subject_label,
            reply_target_body=reply_target_body,
            timestamp=timestamp,
            batch_id=repair_batch_id,
            covered_entries=1,
        )
        assistant_timestamp = str(responses[-1].get("timestamp") or timestamp)
        assistant_row = _canonical_row(
            canonical_turn_id=assistant_id,
            turn_group_number=group_number,
            sort_key=float(group_number * 2000 + 2000),
            role="assistant",
            content=assistant_content,
            raw_content=json.dumps(
                assistant_raw_blocks,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            session_date=timestamp[:10],
            sender="",
            channel_id=channel_id,
            channel_label=str(trigger.get("channel_name") or ""),
            sender_actor_id="",
            source_message_id="",
            reply_target_message_id="",
            reply_subject_actor_id="",
            reply_subject_label="",
            reply_target_body="",
            timestamp=assistant_timestamp,
            batch_id=repair_batch_id,
            covered_entries=len(responses),
        )
        canonical_rows.extend((user_row, assistant_row))
        memberships.append({
            "tenant_id": "__TENANT_ID_REQUIRED_AT_APPLY__",
            "agent_scope_id": AGENT_SCOPE_ID,
            "platform": PLATFORM,
            "account_id": ACCOUNT_ID,
            "message_id": trigger_id,
            "canonical_turn_id": user_id,
            "assistant_canonical_turn_id": assistant_id,
            "assistant_turn_hash": assistant_row["turn_hash"],
            # Deprecated compatibility field only. The immutable ledger binds
            # row ids/hashes; pair placement is derived from canonical rows so
            # VCMERGE/resequencing can move both halves safely.
            "turn_group_number": -1,
            "pair_version": 1,
            "audience_conversation_id": OWNER_CONVERSATION_ID,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "author_id": sender_id,
            "source_actor_id": f"actor:discord:{sender_id}",
            "transport_body_sha256": _sha256_text(
                str(trigger.get("content") or "")
            ),
            "canonical_body_sha256": _sha256_text(user_content),
            "projection_version": projection_version,
            "canonical_turn_hash": user_row["turn_hash"],
            "reply_target_message_id": reply_target_id,
            "observed_at": timestamp,
            # Deterministic evidence artifact; the apply report records the
            # operational insertion time separately.
            "created_at": timestamp,
        })
        group_evidence.append({
            "trigger_message_id": trigger_id,
            "response_message_ids": response_ids,
            "user_canonical_turn_id": user_id,
            "assistant_canonical_turn_id": assistant_id,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "trigger_available_in_discord_fetch": trigger_available,
            "projection_version": projection_version,
            "openclaw_exact_match": bool(openclaw_match),
            "openclaw_session_id": openclaw_match.session_id if openclaw_match else "",
            "openclaw_event_id": openclaw_match.event_id if openclaw_match else "",
            "openclaw_body_equivalent": (
                _raw_body_equivalent(
                    str(trigger.get("content") or ""),
                    openclaw_match.content,
                )
                if openclaw_match
                else False
            ),
            "reply_target_message_id": reply_target_id,
            "reply_status": reply_status,
            "assistant_chunk_count": len(responses),
            "disposition_categories": disposition_categories,
        })
        stats["groups"] += 1
        stats["canonical_rows"] += 2
        stats["source_memberships"] += 1
        stats["assistant_chunks"] += len(responses)

    actor_ids = {
        str(row["sender_actor_id"])
        for row in canonical_rows
        if row["user_content"]
    }
    if (
        len(canonical_rows) != EXPECTED_CANONICAL_ROWS
        or len(canonical_rows) != len(groups) * 2
        or len(memberships) != EXPECTED_SOURCE_MEMBERSHIPS
        or len(memberships) != len(groups)
        or len(actor_ids) != EXPECTED_ACTORS
    ):
        raise EvidenceError("anti-vacuity count check failed")
    if len({row["canonical_turn_id"] for row in canonical_rows}) != len(canonical_rows):
        raise EvidenceError("canonical id collision")
    if len({row["sort_key"] for row in canonical_rows}) != len(canonical_rows):
        raise EvidenceError("canonical sort-key collision")
    if len({row["source_message_id"] for row in canonical_rows if row["user_content"]}) != len(groups):
        raise EvidenceError("source-message coverage mismatch")

    new_by_source = {
        row["source_message_id"]: row
        for row in canonical_rows
        if row["user_content"]
    }
    new_assistant_by_normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        if row["assistant_content"]:
            new_assistant_by_normalized[row["normalized_assistant_text"]].append(row)
    old_map: list[dict[str, Any]] = []
    old_rows: list[dict[str, Any]] = []
    old_stats = Counter()
    for old in old_input_rows:
        if str(old.get("conversation_id") or "") != OWNER_CONVERSATION_ID:
            continue
        old_rows.append(old)
        old_id = str(old.get("canonical_turn_id") or "")
        source_id = str(old.get("source_message_id") or "")
        mapped_ids: list[str] = []
        classification = "unmapped_no_external_identity"
        if source_id and source_id in new_by_source:
            replacement = new_by_source[source_id]
            old_body = str(old.get("user_content") or "")
            canonical_body_match = old_body == replacement["user_content"]
            transport_body_match = (
                old_body == transport_projection_by_source[source_id]
            )
            actor_match = (
                str(old.get("sender_actor_id") or "")
                == replacement["sender_actor_id"]
            )
            if canonical_body_match and actor_match:
                classification = "source_actor_canonical_body_exact"
                mapped_ids = [replacement["canonical_turn_id"]]
            elif transport_body_match and actor_match:
                classification = "source_actor_transport_body_needs_label"
                mapped_ids = [replacement["canonical_turn_id"]]
            elif canonical_body_match or transport_body_match:
                classification = "source_body_exact_actor_wrong_or_missing"
            elif actor_match:
                classification = "source_actor_exact_body_corrupt"
            else:
                classification = "source_body_and_actor_corrupt"
        elif str(old.get("assistant_content") or "").strip():
            normalized = normalize_turn_text(str(old.get("assistant_content") or ""))
            candidates = new_assistant_by_normalized.get(normalized, [])
            if len(candidates) == 1:
                mapped_ids = [candidates[0]["canonical_turn_id"]]
                classification = "unique_external_assistant_body"
        old_stats[classification] += 1
        old_map.append({
            "old_canonical_turn_id": old_id,
            "old_source_message_id": source_id,
            "classification": classification,
            "new_canonical_turn_ids": mapped_ids,
        })

    new_group_by_id = {
        str(row["canonical_turn_id"]): int(row["turn_group_number"])
        for row in canonical_rows
    }
    mapped_new_ids_by_old = {
        str(row["old_canonical_turn_id"]): list(row["new_canonical_turn_ids"])
        for row in old_map
    }
    old_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        raw_group_number = row.get("turn_group_number", -1)
        old_groups[int(raw_group_number if raw_group_number is not None else -1)].append(row)
    old_group_map: list[dict[str, Any]] = []
    old_group_stats = Counter()
    for old_group_number, old_group_rows in sorted(old_groups.items()):
        candidates = sorted({
            new_group_by_id[new_id]
            for old_row in old_group_rows
            for new_id in mapped_new_ids_by_old.get(
                str(old_row.get("canonical_turn_id") or ""),
                [],
            )
            if new_id in new_group_by_id
        })
        if old_group_number < 0:
            classification = "quarantined_missing_old_group_identity"
            mapped_group = None
        elif len(candidates) == 1:
            classification = "exactly_one_external_group"
            mapped_group = candidates[0]
        elif not candidates:
            classification = "quarantined_no_external_group"
            mapped_group = None
        else:
            classification = "quarantined_conflicting_external_groups"
            mapped_group = None
        old_group_stats[classification] += 1
        old_group_map.append({
            "old_turn_group_number": old_group_number,
            "old_canonical_turn_ids": [
                str(row.get("canonical_turn_id") or "")
                for row in old_group_rows
            ],
            "candidate_new_turn_group_numbers": candidates,
            "new_turn_group_number": mapped_group,
            "classification": classification,
        })

    output_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    for filename, rows in (
        ("canonical_rows.jsonl", canonical_rows),
        ("source_memberships.jsonl", memberships),
        ("group_evidence.jsonl", group_evidence),
        ("old_to_new.jsonl", old_map),
        ("old_group_to_new_group.jsonl", old_group_map),
    ):
        count, digest = _write_jsonl(output_dir / filename, rows)
        files[filename] = {"rows": count, "sha256": digest}

    manifest = {
        "version": 4,
        "authority": "raw Discord; exact OpenClaw corroboration only",
        "guild_id": MEN_GUILD_ID,
        "owner_conversation_id": OWNER_CONVERSATION_ID,
        "agent_scope_id": AGENT_SCOPE_ID,
        "account_id": ACCOUNT_ID,
        "vast_user_id": VAST_USER_ID,
        "repair_batch_id": repair_batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": dict(sorted(stats.items())),
        "openclaw_stats": openclaw_stats,
        "all_delivery_disposition": {
            "filename": disposition_path.name,
            **disposition_stats,
        },
        "inputs": {
            "discord_transcript": {
                "filename": messages_path.name,
                "sha256": transcript_sha,
                "rows": len(all_messages),
            },
            "recovery_session_bundle": recovery_session_bundle,
            "old_canonical": {
                "filename": old_canonical_path.name,
                "sha256": old_canonical_sha,
                "rows": len(old_input_rows),
                "owner_rows": len(old_rows),
            },
        },
        "old_row_classifications": dict(sorted(old_stats.items())),
        "old_group_classifications": dict(sorted(old_group_stats.items())),
        "files": files,
        "invariants": {
            "one_user_and_one_assistant_row_per_response_group": True,
            "one_source_membership_per_group": True,
            "source_membership_binds_assistant_row_and_hash": True,
            "source_membership_group_is_non_authoritative": True,
            "all_user_rows_have_actor_and_source_id": True,
            "all_response_chunks_have_decision_specific_exact_evidence": True,
            "no_adjacency_inference": True,
            "database_mutated": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = [
        f"{meta['sha256']}  {filename}"
        for filename, meta in sorted(files.items())
    ]
    checksums.append(
        f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  manifest.json"
    )
    (output_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--old-canonical", type=Path, required=True)
    parser.add_argument("--disposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        manifest = build_artifact(
            transcript_dir=args.transcript_dir.resolve(),
            sessions_dir=args.sessions_dir.resolve(),
            old_canonical_path=args.old_canonical.resolve(),
            disposition_path=args.disposition.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except (EvidenceError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "ok",
        "output_dir": str(args.output_dir.resolve()),
        "stats": manifest["stats"],
        "old_row_classifications": manifest["old_row_classifications"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
