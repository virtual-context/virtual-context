"""Message filtering: removes irrelevant history turns from API request bodies.

Pure functions — no ProxyState dependency. Extracted from proxy/server.py.
"""

from __future__ import annotations

from ..core.turn_tag_index import TurnTagIndex
from .formats import PayloadFormat, detect_format


def filter_body_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    matched_tags: list[str],
    *,
    recent_turns: int = 3,
    broad: bool = False,
    temporal: bool = False,
    compacted_turn: int = 0,
    fmt: PayloadFormat | None = None,
) -> tuple[dict, int]:
    """Filter request body messages to remove irrelevant history turns.

    Operates on the raw API body, preserving original message format
    (content blocks, metadata, etc.).  Uses the TurnTagIndex to decide
    which user+assistant pairs to keep based on tag overlap.

    When *compacted_turn* > 0 (paging active), pairs whose turn index is
    below the watermark are unconditionally dropped — their content lives
    in VC summaries and is retrievable via ``vc_expand_topic``.

    Returns (filtered_body, turns_dropped).
    """
    if fmt is None:
        fmt = detect_format(body)

    # Determine message key and assistant role based on format
    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    messages = body.get(_msg_key, [])
    if not messages:
        return body, 0

    # Separate system messages (OpenAI format) and chat messages
    prefix: list[dict] = []
    chat_msgs: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not chat_msgs:
            prefix.append(msg)
        else:
            chat_msgs.append(msg)

    if not chat_msgs:
        return body, 0

    # Split trailing user message (current turn) from history pairs
    current_user = None
    if chat_msgs and chat_msgs[-1].get("role") == "user":
        current_user = chat_msgs[-1]
        chat_msgs = chat_msgs[:-1]

    # Group into user+assistant pairs, tracking which message indices are paired.
    # Unpaired messages (tool_results between consecutive users, batched messages,
    # etc.) are always kept — they're structural and may be required by the API.
    pairs: list[tuple[int, int]] = []  # (msg_idx_user, msg_idx_assistant)
    paired_indices: set[int] = set()
    i = 0
    while i + 1 < len(chat_msgs):
        if (chat_msgs[i].get("role") == "user"
                and chat_msgs[i + 1].get("role") == _asst_role):
            pairs.append((i, i + 1))
            paired_indices.add(i)
            paired_indices.add(i + 1)
            i += 2
        else:
            i += 1

    total_pairs = len(pairs)
    protected = min(recent_turns, total_pairs)

    if total_pairs <= protected or not turn_tag_index.entries:
        return body, 0

    # Broad/temporal: keep everything
    if broad or temporal:
        return body, 0

    tag_set = set(matched_tags)

    # First pass: mark each pair as keep/drop based on tags
    keep_pair = [False] * total_pairs
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if pair_idx >= total_pairs - protected:
            keep_pair[pair_idx] = True
        elif compacted_turn > 0 and pair_idx < compacted_turn:
            # PROXY-023: paging active — compacted turns are dropped
            # unconditionally. Their content is in VC summaries and
            # retrievable via vc_expand_topic.
            keep_pair[pair_idx] = False
        else:
            entry = turn_tag_index.get_tags_for_turn(pair_idx)
            if entry is None:
                keep_pair[pair_idx] = True
            elif "rule" in entry.tags or set(entry.tags) & tag_set:
                keep_pair[pair_idx] = True

    # Second pass: fix tool_use/tool_result dependencies.
    # If assistant has tool_use, the next pair (with tool_result) must also be kept.
    # If user has tool_result, the previous pair (with tool_use) must also be kept.
    # Iterate until stable (handles multi-step tool chains).
    changed = True
    while changed:
        changed = False
        for pair_idx in range(total_pairs):
            if not keep_pair[pair_idx]:
                continue
            u_idx, a_idx = pairs[pair_idx]
            if _has_tool_use(chat_msgs[a_idx]) and pair_idx + 1 < total_pairs and not keep_pair[pair_idx + 1]:
                keep_pair[pair_idx + 1] = True
                changed = True
            if _has_tool_result(chat_msgs[u_idx]) and pair_idx > 0 and not keep_pair[pair_idx - 1]:
                keep_pair[pair_idx - 1] = True
                changed = True

    # Build per-message keep set: unpaired messages always kept, pairs based on filter
    keep_msg: set[int] = set()
    for msg_idx in range(len(chat_msgs)):
        if msg_idx not in paired_indices:
            keep_msg.add(msg_idx)  # always keep unpaired messages
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if keep_pair[pair_idx]:
            keep_msg.add(u_idx)
            keep_msg.add(a_idx)

    # Final tool chain safety: any kept assistant with tool_use must have its
    # tool_result in the immediately following message(s) also kept, and vice versa.
    changed = True
    while changed:
        changed = False
        for msg_idx in range(len(chat_msgs)):
            if msg_idx not in keep_msg:
                continue
            msg = chat_msgs[msg_idx]
            if msg.get("role") == _asst_role and _has_tool_use(msg):
                # Keep all following messages until we find the tool_result
                for j in range(msg_idx + 1, len(chat_msgs)):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if _has_tool_result(chat_msgs[j]):
                        break
            if _has_tool_result(msg):
                # Keep all preceding messages back to the tool_use
                for j in range(msg_idx - 1, -1, -1):
                    if j not in keep_msg:
                        keep_msg.add(j)
                        changed = True
                    if chat_msgs[j].get("role") == _asst_role and _has_tool_use(chat_msgs[j]):
                        break

    # Build filtered message list preserving original order
    kept: list[dict] = list(prefix)
    for msg_idx in range(len(chat_msgs)):
        if msg_idx in keep_msg:
            kept.append(chat_msgs[msg_idx])

    if current_user:
        kept.append(current_user)

    # PROXY-022: Enforce strict role alternation.
    # OpenClaw can send consecutive same-role messages (batched Telegram messages,
    # tool_result followed by new user text without intervening assistant). When
    # we drop pairs around these "unpaired" messages the result can have
    # consecutive same-role entries, which the Anthropic API rejects.
    # Fix: walk the kept list and drop any message that repeats the previous role.
    alternating: list[dict] = []
    for msg in kept:
        if alternating and msg.get("role") == alternating[-1].get("role"):
            continue  # skip — would create consecutive same-role
        alternating.append(msg)
    kept = alternating

    # Compute drops from final kept list (not incremental tracking, which
    # undercounts when alternation enforcement removes additional messages).
    kept_user_count = sum(1 for m in kept if m.get("role") == "user")
    if current_user:
        kept_user_count -= 1  # exclude current turn from pair count
    dropped = max(0, total_pairs - kept_user_count)
    if dropped == 0:
        return body, 0

    body = dict(body)
    body[_msg_key] = kept
    return body, dropped


def _has_tool_use(msg: dict) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _has_tool_result(msg: dict) -> bool:
    """Check if a user message contains tool_result blocks."""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False
