"""Turn a host's embedded history replay back into real conversation turns.

Some hosts (the OpenClaw Codex harness among them) cannot hand prior turns to
the model as separate messages, so they paste recent history into the current
user message as a ``<conversation_context>`` block. Left in place, that block
is part of the protected current turn and nothing in it can be trimmed. This
module splits it into ordinary user/assistant items placed before the current
message, so the history is managed like any other history: protected recent
turns stay verbatim and older compacted turns can be dropped in favour of
summaries.

Only the outbound payload is rewritten. Ingestion reads the client's own shape
before this runs, so expanded turns are never stored a second time.
"""

from __future__ import annotations

import copy
import re

import json

from ._envelope import (
    _HOST_ASSEMBLED_LABEL,
    _HOST_ASSEMBLED_LINE_RE,
    _HOST_CTX_LABEL_LINE_RE,
    _HOST_REQUEST_LINE_RE,
)

_BLOCK_RE = re.compile(r"<conversation_context>([\s\S]*?)</conversation_context>")
# A section header is the role alone, or the role followed by attributes the
# host attaches to it, e.g. ``[user sender={"id":"1","name":"optics"}]``.
_SECTION_RE = re.compile(
    r"(?m)^[ \t]*\[(user|assistant|toolResult|toolCall|tool|system|compactionSummary)"
    r"(?:[ \t]+([^\n]*?))?\][ \t]*$"
)
_PREAMBLE_RE = re.compile(
    r"(?m)^[ \t]*" + re.escape(_HOST_ASSEMBLED_LABEL) + r"[ \t]*\n"
    r"(?:[ \t]*Treat the conversation context below[^\n]*\n)?\s*$"
)
_USER_ROLES = {"user", "compactionSummary", "system"}
# A group host may also list recent room messages as their own section, one
# message per line: "#session:<id> <date> <time> UTC <speaker>: <text>". Lines
# quoting the message a reply answers carry "#<message id>" and "[reply target]".
_SELECTED_HEADER_RE = re.compile(
    r"(?m)^[ \t]*Conversation context \(chronological, selected for current message\):[ \t]*⟦openclaw:ctx⟧[ \t]*\n"
)
_BLANK_LINE_RE = re.compile(r"(?m)^[ \t]*$")
_SELECTED_LINE_RE = re.compile(r"^#(session:\S+|\d+) \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC (.*)$")
_HOST_SPEAKERS = {"OpenClaw"}


def _attributed_speaker(attrs: str | None) -> str:
    """Display name from a header's ``sender={json}`` attribute, or ``""``."""
    if not attrs or "sender=" not in attrs:
        return ""
    raw = attrs[attrs.index("sender=") + len("sender="):]
    try:
        sender, _end = json.JSONDecoder().raw_decode(raw)
    except ValueError:
        return ""
    if not isinstance(sender, dict):
        return ""
    return str(sender.get("name") or sender.get("username") or sender.get("id") or "")


def parse_replay_block(block: str) -> list[tuple[str, str]]:
    """Ordered ``(role, text)`` entries; non-user sections between users fold into one assistant entry.

    A user header may carry the speaker (``[user sender={...}]``); the entry
    then starts with that speaker's name, as selected room history does.
    """
    heads = list(_SECTION_RE.finditer(block))
    entries: list[tuple[str, str]] = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(block)
        text = block[head.end():end].strip()
        label = head.group(1)
        if not text:
            continue
        if label in _USER_ROLES:
            if label != "user":
                text = f"[{label}]\n{text}"
            else:
                speaker = _attributed_speaker(head.group(2))
                if speaker:
                    text = f"{speaker}: {text}"
            entries.append(("user", text))
        elif entries and entries[-1][0] == "assistant":
            entries[-1] = ("assistant", entries[-1][1] + "\n\n" + text)
        else:
            entries.append(("assistant", text))
    return entries


def _selected_speaker(rest: str) -> tuple[str, str, bool]:
    """``(speaker, text, is_reply_target)`` from what follows the line's timestamp."""
    reply = rest.startswith("[reply target] ")
    if reply:
        rest = rest[len("[reply target] "):]
        rest = re.sub(r"^->#\d+ ", "", rest)
    if rest.startswith("{"):
        try:
            sender, end = json.JSONDecoder().raw_decode(rest)
        except ValueError:
            sender, end = None, 0
        if isinstance(sender, dict) and rest[end:end + 2] == ": ":
            name = str(sender.get("name") or sender.get("username") or sender.get("id") or "user")
            return name, rest[end + 2:], reply
    speaker, sep, text = rest.partition(": ")
    return (speaker, text, reply) if sep else ("", rest, reply)


def parse_selected_history(section: str) -> list[tuple[str, str]]:
    """Ordered ``(role, text)`` entries from a selected room-history section.

    The host's own lines become assistant entries and fold together when
    adjacent; every other speaker's line is a user entry prefixed with the
    speaker. A reply-target line and unprefixed continuation lines extend the
    entry before them.
    """
    entries: list[tuple[str, str]] = []
    for line in section.split("\n"):
        match = _SELECTED_LINE_RE.match(line)
        if not match:
            if entries and line.strip():
                role, text = entries[-1]
                entries[-1] = (role, text + "\n" + line)
            continue
        speaker, text, reply = _selected_speaker(match.group(2))
        if reply:
            label = f"[reply target] {speaker}: {text}" if speaker else f"[reply target] {text}"
            if entries:
                role, prior = entries[-1]
                entries[-1] = (role, prior + "\n\n" + label)
            else:
                entries.append(("user", label))
            continue
        if speaker in _HOST_SPEAKERS or speaker.endswith("(you)"):
            if entries and entries[-1][0] == "assistant":
                entries[-1] = ("assistant", entries[-1][1] + "\n\n" + text)
            else:
                entries.append(("assistant", text))
        else:
            entries.append(("user", f"{speaker}: {text}" if speaker else text))
    return entries


def _take_selected_history(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove a selected room-history section from *text*; return what is left and its entries."""
    header = _SELECTED_HEADER_RE.search(text)
    if not header:
        return text, []
    start = header.end()
    ends = [len(text)]
    for pattern in (_HOST_CTX_LABEL_LINE_RE, _HOST_ASSEMBLED_LINE_RE, _HOST_REQUEST_LINE_RE):
        found = pattern.search(text, start)
        if found:
            ends.append(found.start())
    block = text.find("<conversation_context>", start)
    if block >= 0:
        ends.append(block)
    # The section is one line per message; a blank line ends it. A reused
    # thread appends the request after that blank line without a label.
    blank = _BLANK_LINE_RE.search(text, start)
    if blank:
        ends.append(blank.start())
    end = min(ends)
    entries = parse_selected_history(text[start:end])
    if not entries:
        return text, []
    return text[:header.start()] + text[end:].lstrip("\n"), entries


def _item_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _message(role: str, text: str) -> dict:
    kind = "input_text" if role == "user" else "output_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def find_replay_block(body: dict) -> str | None:
    """The newest user message's ``<conversation_context>`` block text, if any."""
    items = body.get("input") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return None
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user":
            match = _BLOCK_RE.search(_item_text(item))
            return match.group(1) if match else None
    return None


def _carries_history(text: str) -> bool:
    return "<conversation_context>" in text or bool(_SELECTED_HEADER_RE.search(text))


def _newest_user_with(items: list, carries) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user":
            if carries(_item_text(item)):
                return index
    return None


def _is_text_part(part) -> bool:
    return isinstance(part, dict) and isinstance(part.get("text"), str)


def _history_span(item: dict) -> tuple[list, tuple[int, int, str] | None]:
    """Content parts and the ``(first, last, joined_text)`` span holding the newest history.

    The history is normally one text part. A host that places an attachment
    from the replayed history inside the message splits the block across the
    text parts around it; the span then runs to the part that closes the block
    and the joined text is what gets parsed.
    """
    content = item.get("content")
    parts = content if isinstance(content, list) else [{"type": "input_text", "text": content}]
    start = next((i for i, part in enumerate(parts)
                  if _is_text_part(part) and _carries_history(part["text"])), None)
    if start is None:
        return parts, None
    end = start
    text = parts[start]["text"]
    if "<conversation_context>" in text and "</conversation_context>" not in text:
        for j in range(start + 1, len(parts)):
            if not _is_text_part(parts[j]):
                continue
            text += parts[j]["text"]
            end = j
            if "</conversation_context>" in parts[j]["text"]:
                break
    return parts, (start, end, text)


def _has_block(text: str) -> bool:
    return "<conversation_context>" in text


def _has_selected(text: str) -> bool:
    return bool(_SELECTED_HEADER_RE.search(text))


def expand_host_replay(body: dict) -> tuple[dict, int]:
    """Split the newest host history in the payload into turns before its message.

    The history is normally in the current message. A host that projects
    history once into a thread's first message and then appends later turns
    leaves it in an older user message. The newest replay block becomes turns
    before the message holding it, and the newest selected room-history
    section becomes turns before its message (ahead of the block's turns when
    both are in one message). Older replay blocks are left as they are; older
    selected sections are removed, since the newest one supersedes them.

    Returns ``(body, inserted_item_count)``. The input body is never mutated;
    when there is nothing to expand the same object is returned with 0.
    """
    items = body.get("input") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return body, 0
    block_at = _newest_user_with(items, _has_block)
    selected_at = _newest_user_with(items, _has_selected)
    if block_at is None and selected_at is None:
        return body, 0
    texts: dict[int, tuple[list, tuple[int, int, str], str]] = {}
    inserted: dict[int, list[tuple[str, str]]] = {}
    # Only the text holding the history is rewritten; every other part of the
    # message (images, files, audio, further text) is kept as-is. Text parts an
    # inline attachment split the block across are merged back into one.
    for index, item in enumerate(items):
        if index > (selected_at if selected_at is not None else -1) and index != block_at:
            continue
        if not (isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user"):
            continue
        parts, span = _history_span(item)
        if span is None:
            continue
        text, entries = _take_selected_history(span[2])
        if index != selected_at:
            entries = []
        replayed = []
        if index == block_at:
            match = _BLOCK_RE.search(text)
            replayed = parse_replay_block(match.group(1)) if match else []
            if match and replayed:
                before = _PREAMBLE_RE.sub("", text[:match.start()]).rstrip()
                after = text[match.end():].lstrip()
                text = (before + "\n\n" + after) if before else after
        if text == span[2] and not entries and not replayed:
            continue
        texts[index] = (parts, span, text.strip())
        inserted[index] = entries + replayed
    if not inserted:
        return body, 0
    out_items: list = []
    count = 0
    for index, item in enumerate(items):
        if index in texts:
            parts, (start, end, _joined), text = texts[index]
            expanded = [_message(role, entry) for role, entry in inserted[index]]
            out_items.extend(expanded)
            count += len(expanded)
            current = copy.deepcopy(item)
            new_parts = copy.deepcopy(parts)
            merged = {**new_parts[start], "text": text}
            kept_between = [part for part in new_parts[start + 1:end + 1] if not _is_text_part(part)]
            current["content"] = new_parts[:start] + [merged] + kept_between + new_parts[end + 1:]
            out_items.append(current)
        else:
            out_items.append(item)
    if not count:
        return body, 0
    out = dict(body)
    out["input"] = out_items
    return out, count


def without_host_replayed_groups(replay_messages: list, body: dict, fmt) -> list:
    """Stored turn groups minus those the host already replayed into ``body``.

    A replayed user message names its platform message id in the host speaker
    tag. A stored group whose user message has one of those ids is already in
    the payload, so it is left out whole; other groups are kept whole.
    """
    if not replay_messages:
        return replay_messages
    from ..core.history_catchup import read_host_speaker

    replayed = set()
    for item in fmt.get_messages(body) or []:
        if isinstance(item, dict) and item.get("role") == "user":
            speaker = read_host_speaker(_item_text(item))
            if speaker is not None:
                replayed.add(speaker.message_id)
    if not replayed:
        return replay_messages

    def _meta(message) -> dict:
        metadata = getattr(message, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    dropped = {
        _meta(m).get("db_recent_group_key")
        for m in replay_messages
        if m.role == "user" and _meta(m).get("source_message_id") in replayed
    }
    dropped.discard(None)
    return [m for m in replay_messages if _meta(m).get("db_recent_group_key") not in dropped]
