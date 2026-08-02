#!/usr/bin/env python3
"""Export an immutable, checksummed Discord guild transcript.

The exporter is intentionally independent of OpenClaw conversation history.
Discord is the authority for message id, author, channel, reply edge, body, and
delivery order.  A per-channel high-water mark is captured before pagination,
so human traffic that arrives while the gateway is frozen cannot create a
mixed-time corpus.  A second pass fails if the configured bot delivered any
message beyond those marks.

The bot token is read from the local OpenClaw config (or an environment
variable) and is never written or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE = "https://discord.com/api/v10"
DIRECT_MESSAGE_CHANNEL_TYPES = {0, 2, 5, 13}
THREAD_PARENT_CHANNEL_TYPES = {0, 5, 15, 16}
THREAD_CHANNEL_TYPES = {10, 11, 12}
KNOWN_GUILD_CHANNEL_TYPES = {0, 2, 4, 5, 13, 14, 15, 16}


class ExportError(RuntimeError):
    """Raised when Discord cannot provide a complete, stable export."""


class DiscordClient:
    def __init__(self, token: str) -> None:
        self._headers = {
            "Authorization": f"Bot {token}",
            "User-Agent": "virtual-context-history-audit/1",
        }

    def get(self, path: str, **params: Any) -> Any:
        query = urllib.parse.urlencode(
            {key: value for key, value in params.items() if value not in (None, "")}
        )
        url = f"{API_BASE}{path}" + (f"?{query}" if query else "")
        for _attempt in range(12):
            request = urllib.request.Request(url, headers=self._headers)
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    body = json.load(response)
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    reset_after = float(
                        response.headers.get("X-RateLimit-Reset-After") or 0
                    )
                    if remaining == "0" and reset_after > 0:
                        time.sleep(reset_after + 0.05)
                    return body
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                if exc.code != 429:
                    raise ExportError(
                        f"Discord GET {path} failed with HTTP {exc.code}"
                    ) from exc
                try:
                    retry_after = float(json.loads(payload).get("retry_after", 1))
                except (ValueError, TypeError, json.JSONDecodeError):
                    retry_after = 1.0
                time.sleep(max(retry_after, 0.05) + 0.1)
            except urllib.error.URLError as exc:
                raise ExportError(f"Discord GET {path} failed: {exc.reason}") from exc
        raise ExportError(f"Discord rate-limit retries exhausted for {path}")


def _token(config_path: Path, account_name: str, token_env: str) -> str:
    env_token = os.environ.get(token_env, "").strip()
    if env_token:
        return env_token
    document = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        value = document["channels"]["discord"]["accounts"][account_name]["token"]
    except (KeyError, TypeError) as exc:
        raise ExportError(
            f"Discord token for account {account_name!r} is absent from config"
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"Discord token for account {account_name!r} is empty")
    return value.strip()


def _iso(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _names(message: dict[str, Any]) -> list[str]:
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    member = message.get("member") if isinstance(message.get("member"), dict) else {}
    values = [member.get("nick"), author.get("global_name"), author.get("username")]
    output: list[str] = []
    for value in values:
        name = str(value or "").strip()
        if name and name not in output:
            output.append(name)
    return output


def _normalize_message(
    message: dict[str, Any],
    *,
    guild_id: str,
    channel: dict[str, Any],
    page_count: int,
) -> dict[str, Any]:
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    reference = (
        message.get("message_reference")
        if isinstance(message.get("message_reference"), dict)
        else {}
    )
    return {
        "id": str(message.get("id") or ""),
        "guild_id": guild_id,
        "channel_id": str(channel.get("id") or ""),
        "channel_name": str(channel.get("name") or ""),
        "channel_page_count": page_count,
        "author_id": str(author.get("id") or ""),
        "author_names": _names(message),
        "author_bot": bool(author.get("bot")),
        "content": str(message.get("content") or ""),
        "timestamp": _iso(message.get("timestamp")),
        "edited_timestamp": _iso(message.get("edited_timestamp")),
        "attachments": message.get("attachments") or [],
        "embeds": message.get("embeds") or [],
        "sticker_items": message.get("sticker_items") or [],
        "mentioned_user_ids": [
            str(item.get("id") or "")
            for item in message.get("mentions") or []
            if isinstance(item, dict) and item.get("id")
        ],
        "mentioned_role_ids": [str(value) for value in message.get("mention_roles") or []],
        "reference_message_id": str(reference.get("message_id") or ""),
        "reference_channel_id": str(reference.get("channel_id") or ""),
        "reference_guild_id": str(reference.get("guild_id") or ""),
        "type": int(message.get("type") or 0),
        "flags": int(message.get("flags") or 0),
        "pinned": bool(message.get("pinned")),
        "tts": bool(message.get("tts")),
    }


def _list_archived_threads(
    client: DiscordClient,
    parent_id: str,
    *,
    private: bool,
) -> list[dict[str, Any]]:
    # The joined-private endpoint is only a subset of guild history.  A
    # repair-grade export requires the full private archive and fails loudly
    # if the bot lacks MANAGE_THREADS rather than certifying a partial view.
    suffix = "threads/archived/private" if private else "threads/archived/public"
    threads: list[dict[str, Any]] = []
    before = ""
    seen_anchors: set[str] = set()
    while True:
        document = client.get(
            f"/channels/{parent_id}/{suffix}",
            limit=100,
            before=before,
        )
        page = document.get("threads") if isinstance(document, dict) else None
        if not isinstance(page, list):
            raise ExportError(f"Discord returned malformed archived threads for {parent_id}")
        threads.extend(item for item in page if isinstance(item, dict))
        if not document.get("has_more"):
            return threads
        if not page:
            raise ExportError(f"Discord archived-thread pagination stalled for {parent_id}")
        metadata = page[-1].get("thread_metadata") or {}
        anchor = str(metadata.get("archive_timestamp") or "").strip()
        if not anchor or anchor in seen_anchors:
            raise ExportError(f"Discord archived-thread cursor did not advance for {parent_id}")
        seen_anchors.add(anchor)
        before = anchor


def _channels(client: DiscordClient, guild_id: str) -> list[dict[str, Any]]:
    base = client.get(f"/guilds/{guild_id}/channels")
    if not isinstance(base, list):
        raise ExportError("Discord returned a malformed guild channel list")
    unknown_base = [
        item for item in base
        if isinstance(item, dict)
        and int(item.get("type", -1)) not in KNOWN_GUILD_CHANNEL_TYPES
    ]
    if unknown_base:
        raise ExportError(
            "Discord returned an unknown guild channel type; refusing a "
            "potentially incomplete export"
        )
    direct = [
        item for item in base
        if isinstance(item, dict)
        and int(item.get("type", -1)) in DIRECT_MESSAGE_CHANNEL_TYPES
    ]
    parents = [
        item for item in base
        if isinstance(item, dict)
        and int(item.get("type", -1)) in THREAD_PARENT_CHANNEL_TYPES
    ]
    active = client.get(f"/guilds/{guild_id}/threads/active")
    active_threads = active.get("threads") if isinstance(active, dict) else None
    if not isinstance(active_threads, list):
        raise ExportError("Discord returned malformed active threads")
    discovered: dict[str, dict[str, Any]] = {
        str(item.get("id") or ""): dict(item) for item in direct
    }
    for parent in parents:
        parent_id = str(parent.get("id") or "")
        parent_type = int(parent.get("type", -1))
        private_modes = (False, True) if parent_type == 0 else (False,)
        for private in private_modes:
            for thread in _list_archived_threads(client, parent_id, private=private):
                thread_type = int(thread.get("type", -1))
                if thread_type not in THREAD_CHANNEL_TYPES:
                    raise ExportError(
                        "Discord returned an unsupported archived thread type"
                    )
                discovered[str(thread.get("id") or "")] = dict(thread)
    for thread in active_threads:
        if isinstance(thread, dict):
            if int(thread.get("type", -1)) not in THREAD_CHANNEL_TYPES:
                raise ExportError(
                    "Discord returned an unsupported active thread type"
                )
            discovered[str(thread.get("id") or "")] = dict(thread)
    if "" in discovered:
        raise ExportError("Discord returned a channel without an id")
    for item in discovered.values():
        item.setdefault("guild_id", guild_id)
    return sorted(
        discovered.values(),
        key=lambda item: (int(item.get("position") or 0), int(item["id"])),
    )


def _channel_high_water(client: DiscordClient, channel_id: str) -> str:
    page = client.get(f"/channels/{channel_id}/messages", limit=1)
    if not isinstance(page, list):
        raise ExportError(f"Discord returned malformed messages for {channel_id}")
    return str(page[0].get("id") or "") if page else ""


def _channel_messages(
    client: DiscordClient,
    channel: dict[str, Any],
    high_water: str,
) -> tuple[list[dict[str, Any]], int]:
    if not high_water:
        return [], 0
    before = str(int(high_water) + 1)
    raw_rows: list[dict[str, Any]] = []
    pages = 0
    seen: set[str] = set()
    while True:
        page = client.get(
            f"/channels/{channel['id']}/messages",
            limit=100,
            before=before,
        )
        if not isinstance(page, list):
            raise ExportError(f"Discord returned malformed messages for {channel['id']}")
        if not page:
            break
        pages += 1
        ids = [str(item.get("id") or "") for item in page if isinstance(item, dict)]
        if len(ids) != len(page) or any(not value.isdigit() for value in ids):
            raise ExportError(f"Discord returned invalid message ids for {channel['id']}")
        if seen.intersection(ids):
            raise ExportError(f"Discord repeated a message page for {channel['id']}")
        seen.update(ids)
        raw_rows.extend(page)
        next_before = min(ids, key=int)
        if int(next_before) >= int(before):
            raise ExportError(f"Discord message cursor did not advance for {channel['id']}")
        before = next_before
        if len(page) < 100:
            break
    return [
        _normalize_message(
            item,
            guild_id=str(channel.get("guild_id") or ""),
            channel=channel,
            page_count=pages,
        )
        for item in raw_rows
    ], pages


def _messages_after(
    client: DiscordClient,
    channel_id: str,
    high_water: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Snowflake zero is older than every real Discord message. This also
    # exhaustively covers a channel that was empty at the initial high-water
    # read but received messages while the export was running.
    after = high_water or "0"
    while True:
        page = client.get(
            f"/channels/{channel_id}/messages",
            limit=100,
            after=after,
        )
        if not isinstance(page, list):
            raise ExportError(f"Discord returned malformed drift messages for {channel_id}")
        if not page:
            return output
        new_rows = [item for item in page if int(str(item.get("id") or "0")) > int(after)]
        if not new_rows:
            raise ExportError(f"Discord drift cursor did not advance for {channel_id}")
        output.extend(new_rows)
        after = str(max(int(str(item.get("id") or "0")) for item in new_rows))
        if len(page) < 100:
            return output


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_durable(path: Path, payload: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)
    return hashlib.sha256(payload).hexdigest()


def export(args: argparse.Namespace) -> dict[str, Any]:
    if not args.maintenance_freeze_confirmed:
        raise ExportError("a confirmed gateway/ingest freeze is required")
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise ExportError(f"refusing to overwrite output directory: {output_dir}")
    client = DiscordClient(_token(args.openclaw_config, args.account, args.token_env))
    bot = client.get("/users/@me")
    observed_bot_id = str(bot.get("id") or "") if isinstance(bot, dict) else ""
    if observed_bot_id != args.bot_user_id:
        raise ExportError(
            "authenticated Discord bot identity does not match --bot-user-id"
        )
    channels = _channels(client, args.guild_id)
    for channel in channels:
        channel["guild_id"] = args.guild_id
    initial_channel_ids = [str(channel["id"]) for channel in channels]
    high_water = {
        str(channel["id"]): _channel_high_water(client, str(channel["id"]))
        for channel in channels
    }
    messages: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    for channel in channels:
        rows, pages = _channel_messages(
            client,
            channel,
            high_water[str(channel["id"])],
        )
        messages.extend(rows)
        page_counts[str(channel["id"])] = pages
    ids = [str(message.get("id") or "") for message in messages]
    if len(ids) != len(set(ids)):
        raise ExportError("the guild export contains duplicate message ids")
    for message in messages:
        channel_id = str(message["channel_id"])
        if (
            message["guild_id"] != args.guild_id
            or not high_water.get(channel_id)
            or int(message["id"]) > int(high_water[channel_id])
        ):
            raise ExportError("a message falls outside the captured guild high-water")
    messages.sort(key=lambda message: int(message["id"]))

    final_channels = _channels(client, args.guild_id)
    final_channel_ids = [str(channel["id"]) for channel in final_channels]
    if final_channel_ids != initial_channel_ids:
        raise ExportError("Discord channel/thread membership changed during export")
    drift_messages: list[dict[str, Any]] = []
    for channel in channels:
        drift_messages.extend(
            _messages_after(
                client,
                str(channel["id"]),
                high_water[str(channel["id"])],
            )
        )
    post_cutoff_vast = [
        message for message in drift_messages
        if str((message.get("author") or {}).get("id") or "") == args.bot_user_id
    ]
    if post_cutoff_vast:
        raise ExportError(
            "the configured bot delivered messages after the captured high-water marks"
        )

    by_id = {str(message["id"]): message for message in messages}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if (
            message["author_id"] == args.bot_user_id
            and message["author_bot"]
            and message["reference_message_id"]
        ):
            grouped[message["reference_message_id"]].append(message)
    response_groups: list[dict[str, Any]] = []
    for trigger_id, responses in grouped.items():
        responses.sort(key=lambda message: int(message["id"]))
        trigger = by_id.get(trigger_id)
        response_groups.append({
            "trigger_message_id": trigger_id,
            "trigger_available": trigger is not None,
            "trigger": trigger,
            "response_message_ids": [message["id"] for message in responses],
            "responses": responses,
        })
    response_groups.sort(key=lambda group: int(group["response_message_ids"][0]))

    output_dir.mkdir(mode=0o700, parents=True)
    output_dir.chmod(0o700)
    messages_sha = _write_durable(output_dir / "messages.jsonl", _jsonl_bytes(messages))
    groups_sha = _write_durable(
        output_dir / "vast-response-groups.jsonl",
        _jsonl_bytes(response_groups),
    )
    channel_manifest = [
        {
            "id": str(channel["id"]),
            "name": str(channel.get("name") or ""),
            "position": int(channel.get("position") or 0),
            "type": int(channel.get("type") or 0),
            "messages": sum(
                message["channel_id"] == str(channel["id"])
                for message in messages
            ),
            "pages": page_counts[str(channel["id"])],
            "high_water_message_id": high_water[str(channel["id"])],
        }
        for channel in channels
    ]
    vast_messages = [
        message for message in messages
        if message["author_id"] == args.bot_user_id and message["author_bot"]
    ]
    manifest = {
        "schema": "virtual-context.discord-guild-transcript.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "maintenance_freeze_confirmed": True,
        "guild_id": args.guild_id,
        "account": args.account,
        "bot_user_id": args.bot_user_id,
        "channels": channel_manifest,
        "counts": {
            "channels": len(channels),
            "messages": len(messages),
            "human_messages": sum(not message["author_bot"] for message in messages),
            "bot_messages": sum(message["author_bot"] for message in messages),
            "vast_messages": len(vast_messages),
            "vast_response_groups": len(response_groups),
            "response_groups_missing_trigger": sum(
                not group["trigger_available"] for group in response_groups
            ),
            "post_high_water_messages_observed": len(drift_messages),
            "post_high_water_human_messages_observed": sum(
                not bool((message.get("author") or {}).get("bot"))
                for message in drift_messages
            ),
            "post_high_water_vast_messages_observed": 0,
        },
        "high_water": {
            "max_message_id": max(ids, key=int) if ids else "",
            "max_vast_message_id": (
                max((message["id"] for message in vast_messages), key=int)
                if vast_messages else ""
            ),
            "channel_message_ids": high_water,
        },
        "files": {
            "messages.jsonl": messages_sha,
            "vast-response-groups.jsonl": groups_sha,
        },
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_sha = _write_durable(output_dir / "manifest.json", manifest_payload)
    sums = (
        f"{messages_sha}  messages.jsonl\n"
        f"{groups_sha}  vast-response-groups.jsonl\n"
        f"{manifest_sha}  manifest.json\n"
    ).encode("utf-8")
    _write_durable(output_dir / "SHA256SUMS", sums)
    directory_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guild-id", required=True)
    parser.add_argument("--bot-user-id", required=True)
    parser.add_argument("--account", default="vast")
    parser.add_argument(
        "--openclaw-config",
        type=Path,
        default=Path.home() / ".openclaw" / "openclaw.json",
    )
    parser.add_argument("--token-env", default="DISCORD_BOT_TOKEN")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maintenance-freeze-confirmed",
        action="store_true",
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    try:
        manifest = export(_parse_args())
    except (ExportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps({"status": "exported", "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
