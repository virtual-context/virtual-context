#!/usr/bin/env python3
"""Extend the reviewed Men-guild Vast-delivery disposition to final high-water.

The previously reviewed 2,204-delivery partition is immutable input. Every one
of its records is revalidated against the final raw Discord export. New Vast
deliveries are admitted automatically only when Discord itself supplies an
exact reply edge to an available, earlier, human message in the same channel.
Any new direct/tool/operational delivery fails closed for explicit review; the
script never infers membership from adjacency, body similarity, or OpenClaw
history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GUILD_ID = "1524917037191925871"
VAST_USER_ID = "1485681229608259666"
PREVIOUS_DISPOSITION_SHA256 = (
    "bff6c25c08b66b1254412c7c2b47aa82efa830730c00d24cbbce648972ac12b8"
)
PREVIOUS_DELIVERIES = 2204


class DispositionError(RuntimeError):
    """Raised when final evidence does not extend the reviewed partition."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DispositionError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, value


def _exact_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "channel_id": str(raw.get("channel_id") or ""),
        "channel_name": str(raw.get("channel_name") or ""),
        "content": str(raw.get("content") or ""),
        "timestamp": str(raw.get("timestamp") or ""),
        "discord_type": raw.get("type"),
        "attachment_filenames": [
            str(attachment.get("filename") or "")
            for attachment in raw.get("attachments") or []
            if isinstance(attachment, dict)
        ],
    }


def _base_record(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_id": str(raw.get("id") or ""),
        **_exact_fields(raw),
    }


def extend(
    *,
    transcript_dir: Path,
    previous_disposition_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    previous_sha = _sha256(previous_disposition_path)
    if previous_sha != PREVIOUS_DISPOSITION_SHA256:
        raise DispositionError("previous disposition checksum is not the reviewed input")
    previous = json.loads(previous_disposition_path.read_text(encoding="utf-8"))
    previous_records = previous.get("records") if isinstance(previous, dict) else None
    if (
        previous.get("schema_version") != 1
        or previous.get("raw_vast_delivery_count") != PREVIOUS_DELIVERIES
        or not isinstance(previous_records, list)
        or len(previous_records) != PREVIOUS_DELIVERIES
    ):
        raise DispositionError("previous disposition anti-vacuity counts changed")
    previous_by_id = {
        str(record.get("response_id") or ""): record
        for record in previous_records
        if isinstance(record, dict)
    }
    if "" in previous_by_id or len(previous_by_id) != PREVIOUS_DELIVERIES:
        raise DispositionError("previous disposition response ids are incomplete")

    manifest_path = transcript_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "virtual-context.discord-guild-transcript.v2"
        or manifest.get("guild_id") != GUILD_ID
        or manifest.get("bot_user_id") != VAST_USER_ID
        or manifest.get("maintenance_freeze_confirmed") is not True
        or (manifest.get("counts") or {}).get(
            "post_high_water_vast_messages_observed"
        ) != 0
    ):
        raise DispositionError("transcript is not a frozen final Men-guild export")
    messages_path = transcript_dir / "messages.jsonl"
    groups_path = transcript_dir / "vast-response-groups.jsonl"
    expected_files = manifest.get("files") or {}
    if (
        _sha256(messages_path) != expected_files.get("messages.jsonl")
        or _sha256(groups_path) != expected_files.get("vast-response-groups.jsonl")
    ):
        raise DispositionError("transcript file checksums do not match its manifest")

    messages = [row for _line, row in _read_jsonl(messages_path)]
    by_id = {str(message.get("id") or ""): message for message in messages}
    if "" in by_id or len(by_id) != len(messages):
        raise DispositionError("final transcript message ids are missing or duplicated")
    raw_vast = [
        message for message in messages
        if str(message.get("author_id") or "") == VAST_USER_ID
        and bool(message.get("author_bot"))
    ]
    raw_vast_by_id = {str(message["id"]): message for message in raw_vast}
    if len(raw_vast_by_id) != len(raw_vast):
        raise DispositionError("final Vast delivery ids are duplicated")
    if not set(previous_by_id).issubset(raw_vast_by_id):
        missing = sorted(set(previous_by_id) - set(raw_vast_by_id), key=int)
        raise DispositionError(
            f"final transcript lost reviewed Vast deliveries: {missing[:3]}"
        )

    native: dict[str, dict[str, Any]] = {}
    for line_number, group in _read_jsonl(groups_path):
        trigger_id = str(group.get("trigger_message_id") or "")
        for response_id_value in group.get("response_message_ids") or []:
            response_id = str(response_id_value or "")
            if not response_id or response_id in native:
                raise DispositionError("native response group membership is duplicated")
            native[response_id] = {
                "trigger_id": trigger_id,
                "trigger_available": bool(group.get("trigger_available")),
                "group_line": line_number,
                "final_high_water_extension": True,
            }

    records: list[dict[str, Any]] = []
    unclassified: list[str] = []
    for response_id, raw in sorted(raw_vast_by_id.items(), key=lambda item: int(item[0])):
        prior = previous_by_id.get(response_id)
        if prior is not None:
            for field, value in _exact_fields(raw).items():
                if prior.get(field) != value:
                    raise DispositionError(
                        f"reviewed response {response_id} changed Discord field {field}"
                    )
            records.append(prior)
            continue

        evidence = native.get(response_id)
        trigger_id = str((evidence or {}).get("trigger_id") or "")
        trigger = by_id.get(trigger_id)
        if (
            evidence is None
            or evidence.get("trigger_available") is not True
            or str(raw.get("reference_message_id") or "") != trigger_id
            or not isinstance(trigger, dict)
            or bool(trigger.get("author_bot"))
            or str(trigger.get("guild_id") or "") != GUILD_ID
            or str(trigger.get("channel_id") or "")
               != str(raw.get("channel_id") or "")
            or int(trigger_id or 0) >= int(response_id)
        ):
            unclassified.append(response_id)
            continue
        records.append({
            **_base_record(raw),
            "decision": "canonical_response_member",
            "category": "native_referenced_response",
            "evidence": evidence,
        })
    if unclassified:
        raise DispositionError(
            "new Vast deliveries require explicit evidence review: "
            + ",".join(unclassified[:20])
        )
    if len(records) != len(raw_vast) or len({row["response_id"] for row in records}) != len(records):
        raise DispositionError("final disposition is not an exhaustive disjoint partition")

    categories = Counter(str(record.get("category") or "") for record in records)
    decisions = Counter(str(record.get("decision") or "") for record in records)
    document = {
        "schema_version": 1,
        "raw_vast_delivery_count": len(records),
        "unique_response_id_count": len(records),
        "category_counts": dict(sorted(categories.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "inputs": {
            "final_transcript_manifest": str(manifest_path.resolve()),
            "final_transcript_manifest_sha256": _sha256(manifest_path),
            "previous_disposition": str(previous_disposition_path.resolve()),
            "previous_disposition_sha256": previous_sha,
            "previous_delivery_count": PREVIOUS_DELIVERIES,
            "new_delivery_count": len(records) - PREVIOUS_DELIVERIES,
        },
        "records": records,
    }
    resolved_output = output_path.expanduser().resolve(strict=False)
    if resolved_output.exists():
        raise DispositionError(f"refusing to overwrite output: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with resolved_output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    resolved_output.chmod(0o600)
    return {
        "output": str(resolved_output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "raw_vast_deliveries": len(records),
        "new_deliveries": len(records) - PREVIOUS_DELIVERIES,
        "decision_counts": document["decision_counts"],
        "category_counts": document["category_counts"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--previous-disposition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = _parse_args()
    try:
        result = extend(
            transcript_dir=args.transcript_dir.resolve(),
            previous_disposition_path=args.previous_disposition.resolve(),
            output_path=args.output,
        )
    except (DispositionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps({"status": "complete", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
