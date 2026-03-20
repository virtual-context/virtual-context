"""Message filtering: removes irrelevant history turns from API request bodies.

Pure functions — no ProxyState dependency. Extracted from proxy/server.py.
"""

from __future__ import annotations

import hashlib
import logging

from ..core.turn_tag_index import TurnTagIndex
from ._envelope import _strip_envelope
from .formats import PayloadFormat, detect_format

logger = logging.getLogger(__name__)


def filter_body_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    matched_tags: list[str],
    *,
    recent_turns: int = 3,
    compacted_turn: int = 0,
    fmt: PayloadFormat | None = None,
    pre_compaction_mode: str = "aggressive",
) -> tuple[dict, int]:
    """Filter request body messages to remove irrelevant history turns.

    Operates on the raw API body, preserving original message format
    (content blocks, metadata, etc.).  Uses the TurnTagIndex to decide
    which user+assistant pairs to keep based on tag overlap.

    When *compacted_turn* > 0 (paging active), pairs whose turn index is
    below the watermark are unconditionally dropped — their content lives
    in VC summaries and is retrievable via ``vc_expand_topic``.

    Pre-compaction mode (when *compacted_turn* == 0):
    - ``"off"``: skip all tag-based filtering, return body unchanged
    - ``"conservative"``: double the *recent_turns* protection window
    - ``"aggressive"``: use *recent_turns* as-is (default, backward compatible)

    Returns (filtered_body, turns_dropped).
    """
    if fmt is None:
        fmt = detect_format(body)

    # Determine message key and assistant role based on format
    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    elif fmt.name == "openai_responses":
        _msg_key = "input"
        _asst_role = "assistant"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    messages = body.get(_msg_key, [])
    if not messages:
        return body, 0

    # Pre-compaction mode check
    if compacted_turn == 0:
        if pre_compaction_mode == "off":
            return body, 0
        elif pre_compaction_mode == "conservative":
            recent_turns = recent_turns * 2

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

    # Second pass: build tool_use_id → message index maps for precise
    # referential integrity.  Every tool_use block has an "id" field;
    # every tool_result block has a "tool_use_id" field that must match
    # a tool_use in an earlier message.
    #
    # Map: tool_use_id → msg_idx that contains the tool_use
    # Map: tool_use_id → msg_idx that contains the tool_result
    tooluse_to_msg: dict[str, int] = {}   # id → assistant msg index
    toolresult_to_msg: dict[str, int] = {}  # tool_use_id → user msg index
    for msg_idx, msg in enumerate(chat_msgs):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and "id" in block:
                tooluse_to_msg[block["id"]] = msg_idx
            elif block.get("type") == "tool_result" and "tool_use_id" in block:
                toolresult_to_msg[block["tool_use_id"]] = msg_idx

    # Build per-message keep set: unpaired messages always kept, pairs based on filter
    keep_msg: set[int] = set()
    for msg_idx in range(len(chat_msgs)):
        if msg_idx not in paired_indices:
            keep_msg.add(msg_idx)  # always keep unpaired messages
    for pair_idx, (u_idx, a_idx) in enumerate(pairs):
        if keep_pair[pair_idx]:
            keep_msg.add(u_idx)
            keep_msg.add(a_idx)

    # Build reverse map: msg_idx → its pair partner (if any)
    msg_to_partner: dict[int, int] = {}
    for u_idx, a_idx in pairs:
        msg_to_partner[u_idx] = a_idx
        msg_to_partner[a_idx] = u_idx

    # Enforce tool_use_id referential integrity: if a message is kept,
    # the message containing the matching tool_use or tool_result must
    # also be kept.  When a force-kept message belongs to a pair, its
    # pair partner is also kept (otherwise we'd have a half-pair that
    # breaks role alternation).  Iterate until stable (chains can be long).
    changed = True
    while changed:
        changed = False
        for tid, use_idx in tooluse_to_msg.items():
            result_idx = toolresult_to_msg.get(tid)
            if result_idx is None:
                continue
            if use_idx in keep_msg and result_idx not in keep_msg:
                keep_msg.add(result_idx)
                changed = True
            if result_idx in keep_msg and use_idx not in keep_msg:
                keep_msg.add(use_idx)
                changed = True
        # Keep pair partners of any force-added messages
        for msg_idx in list(keep_msg):
            partner = msg_to_partner.get(msg_idx)
            if partner is not None and partner not in keep_msg:
                keep_msg.add(partner)
                changed = True

    # Ensure the first kept chat message is role=user.  When pair 0 is
    # dropped but an unpaired assistant at index 2 is force-kept (via
    # tool_use_id integrity), the filtered output starts with assistant —
    # which the Anthropic API rejects.  Fix: walk kept messages in order
    # and force-keep all earlier messages (and their pair partners) up to
    # the first user message.
    first_kept_indices = sorted(keep_msg)
    for idx in first_kept_indices:
        if chat_msgs[idx].get("role") == "user":
            break  # first kept message is already user — no fix needed
        # Force-keep this message and all messages before it
        for backfill in range(idx + 1):
            if backfill not in keep_msg:
                keep_msg.add(backfill)
                changed = True
        # Also keep pair partners of anything we just added
        for backfill in range(idx + 1):
            partner = msg_to_partner.get(backfill)
            if partner is not None and partner not in keep_msg:
                keep_msg.add(partner)
    # Re-run referential integrity if we added messages
    if changed:
        changed = True
        while changed:
            changed = False
            for tid, use_idx in tooluse_to_msg.items():
                result_idx = toolresult_to_msg.get(tid)
                if result_idx is None:
                    continue
                if use_idx in keep_msg and result_idx not in keep_msg:
                    keep_msg.add(result_idx)
                    changed = True
                if result_idx in keep_msg and use_idx not in keep_msg:
                    keep_msg.add(use_idx)
                    changed = True
            for msg_idx in list(keep_msg):
                partner = msg_to_partner.get(msg_idx)
                if partner is not None and partner not in keep_msg:
                    keep_msg.add(partner)
                    changed = True

    # Adjacent gap fill: when we keep messages A and C but not B,
    # the dropped B may break role alternation or context flow.
    # If all three are part of tool chains, keep B too.
    # This is handled by the alternation enforcement below.

    # Build filtered message list preserving original order
    kept: list[dict] = list(prefix)
    any_dropped = len(keep_msg) < len(chat_msgs)
    for msg_idx in range(len(chat_msgs)):
        if msg_idx in keep_msg:
            kept.append(chat_msgs[msg_idx])

    if current_user:
        kept.append(current_user)

    # Strip thinking blocks when messages were dropped.  Thinking blocks
    # carry cryptographic signatures that chain across turns; dropping a
    # message mid-chain breaks the chain and the Anthropic API returns 500.
    # Thinking blocks are ephemeral (LLM scratchpad) — the actual answer
    # lives in the text/tool_use blocks, so stripping is safe.
    if any_dropped:
        kept = _strip_thinking_blocks(kept)

    # Tag messages that carry tool_use/tool_result links — these must
    # survive role alternation enforcement.  Uses a sentinel key that
    # will be stripped before returning.
    #
    # PROXY-004c: We tag messages in `kept`, NOT in `chat_msgs`.
    #
    # Why: _strip_thinking_blocks (above) creates shallow copies of any
    # assistant message that contains thinking blocks.  If we set
    # _vc_critical on the original dict in chat_msgs, the COPY in `kept`
    # won't have the sentinel.  The alternation enforcement step then
    # can't see it and drops the copy — orphaning the tool_result.
    #
    # Evidence from A/B run 2026-03-01 (request_log/000038):
    #   chat_msgs[49] = assistant [thinking, text]        (pair 24)
    #   chat_msgs[50] = assistant [thinking, text, tool_use(X)]  (UNPAIRED)
    #   chat_msgs[51] = user [tool_result(X)]             (pair 25)
    # After _strip_thinking_blocks, msg[50] is a new dict.  Sentinel on
    # original chat_msgs[50] is invisible to alternation → msg[50] dropped
    # → tool_result(X) orphaned → Anthropic API 400.
    #
    # Fix: walk `kept` in parallel with keep_msg to find which kept-list
    # positions correspond to critical chat_msgs indices, then tag the
    # actual objects in `kept`.
    _critical_indices: set[int] = set()
    for tid, use_idx in tooluse_to_msg.items():
        result_idx = toolresult_to_msg.get(tid)
        if result_idx is not None and use_idx in keep_msg and result_idx in keep_msg:
            _critical_indices.add(use_idx)
            _critical_indices.add(result_idx)
    # Build a set of kept-list positions that are critical (have tool_use/result
    # partners).  Uses an index set instead of mutating message dicts with sentinel
    # keys — avoids leaking temporary keys if an exception interrupts cleanup.
    _critical_kept: set[int] = set()
    _prefix_len = len(prefix)
    _chat_kept_i = 0
    for kept_i in range(_prefix_len, len(kept)):
        if current_user and kept_i == len(kept) - 1:
            break  # last entry is current_user, not from chat_msgs
        while _chat_kept_i < len(chat_msgs) and _chat_kept_i not in keep_msg:
            _chat_kept_i += 1
        if _chat_kept_i < len(chat_msgs) and _chat_kept_i in _critical_indices:
            _critical_kept.add(kept_i)
        _chat_kept_i += 1

    # PROXY-022: Enforce strict role alternation.
    # OpenClaw can send consecutive same-role messages (batched Telegram messages,
    # tool_result followed by new user text without intervening assistant). When
    # we drop pairs around these "unpaired" messages the result can have
    # consecutive same-role entries, which the Anthropic API rejects.
    # Fix: walk the kept list and drop any message that repeats the previous role,
    # UNLESS the message contains a tool_use/tool_result that is part of a kept
    # referential pair — dropping it would orphan the partner.
    # Note: bare items (no role, e.g. function_call in Responses API) are always kept.
    alternating: list[dict] = []
    _alt_critical: list[bool] = []  # parallel list tracking criticality
    for kept_i, msg in enumerate(kept):
        role = msg.get("role")
        if role is None:
            alternating.append(msg)
            _alt_critical.append(kept_i in _critical_kept)
            continue
        if alternating and role == alternating[-1].get("role"):
            if kept_i in _critical_kept:
                # This message has a tool_use/tool_result partner that's also
                # kept — dropping it would create an orphan.  Instead, drop
                # the PREVIOUS same-role message if it's not critical.
                if not _alt_critical[-1]:
                    alternating[-1] = msg  # replace previous with current
                    _alt_critical[-1] = True
                else:
                    alternating.append(msg)  # both critical — keep both
                    _alt_critical.append(True)
            else:
                continue  # skip — would create consecutive same-role
        else:
            alternating.append(msg)
            _alt_critical.append(kept_i in _critical_kept)
    kept = alternating

    # PROXY-004c: Final safety net — verify no orphaned tool_result blocks.
    # The tagging fix above should prevent orphans, but if any slip through
    # (e.g. a message layout we haven't seen yet), returning the unfiltered
    # body is always better than sending a guaranteed 400 to the API.
    _final_tu: set[str] = set()
    _final_tr: set[str] = set()
    for msg in kept:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and "id" in block:
                _final_tu.add(block["id"])
            elif block.get("type") == "tool_result" and "tool_use_id" in block:
                _final_tr.add(block["tool_use_id"])
    _orphaned = _final_tr - _final_tu
    if _orphaned:
        logger.info(
            "MSG-FILTER PROXY-004c safety: %d orphaned tool_result(s) after filtering "
            "-- returning unfiltered body to avoid 400 (ids: %s)",
            len(_orphaned), list(_orphaned)[:3],
        )
        return body, 0

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


def _strip_thinking_blocks(messages: list[dict]) -> list[dict]:
    """Remove thinking blocks from assistant messages.

    Thinking blocks carry chained cryptographic signatures.  When the
    message filter drops any message, the chain is broken and the
    Anthropic API returns 500.  Thinking blocks are ephemeral (LLM
    scratchpad) — the actual answer lives in text/tool_use blocks.
    Stripping also eliminates any prompt-caching benefit, but caching
    is already invalidated by message filtering.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "assistant" or not isinstance(content, list):
            out.append(msg)
            continue
        filtered = [b for b in content
                    if not (isinstance(b, dict) and b.get("type") == "thinking")]
        if len(filtered) == len(content):
            out.append(msg)  # no thinking blocks — keep original
        else:
            out.append({**msg, "content": filtered})
    return out


def _extract_text_for_stub_hash(msg: dict) -> str:
    """Extract the last user-visible text from a message for hashing.

    Handles both plain-string content and content-block arrays.
    Skips tool_use / tool_result blocks.  Strips OpenClaw envelope.

    Uses reversed iteration over content blocks so that multi-text-block
    messages (common in Anthropic tool_use responses) return the **last**
    text block — matching the engine's ``_last_text_block`` behaviour.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return _strip_envelope(content).strip()
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                return _strip_envelope(text).strip()
    return ""


def stub_compacted_messages(
    body: dict,
    turn_tag_index: TurnTagIndex,
    compacted_through: int,
    *,
    fmt: PayloadFormat | None = None,
) -> tuple[dict, int]:
    """Replace compacted turns with lightweight stubs using hash-based identification.

    Walks user-text messages in the client payload, computes SHA-256 hash
    (matching engine's combined_text), looks up in TurnTagIndex.  If the
    matched entry's turn < compacted_through // 2, stubs the entire
    message group (user + assistant + any tool chain messages).

    Returns (modified_body, stub_count).
    """
    if compacted_through <= 0:
        return body, 0
    if not turn_tag_index.entries:
        return body, 0

    if fmt is None:
        fmt = detect_format(body)

    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    elif fmt.name == "openai_responses":
        _msg_key = "input"
        _asst_role = "assistant"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    watermark_turn = compacted_through // 2

    messages = body.get(_msg_key, [])
    if not messages:
        return body, 0

    # Phase 1: identify user-text message indices (messages with extractable text
    # that are NOT tool_result-only).
    user_text_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        # Skip tool_result-only user messages (they're part of a tool chain,
        # not the start of a new turn).
        content = msg.get("content", "")
        if isinstance(content, list):
            has_text = any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in content
            )
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
            if has_tool_result and not has_text:
                continue
        text = _extract_text_for_stub_hash(msg)
        if text:
            user_text_indices.append(i)

    if not user_text_indices:
        return body, 0

    # Phase 2: group messages into turns.
    # Each turn spans from one user-text message to the next.
    turn_groups: list[tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
    for g, uti in enumerate(user_text_indices):
        if g + 1 < len(user_text_indices):
            end = user_text_indices[g + 1]
        else:
            end = len(messages)
        turn_groups.append((uti, end))

    # Phase 3: hash each turn group and match against the index.
    stubs: list[tuple[int, int, object]] = []  # (start, end, TurnTagEntry)
    for start, end in turn_groups:
        msg = messages[start]
        user_text = _extract_text_for_stub_hash(msg)
        if not user_text:
            continue

        # Find first assistant message in group for combined hash
        asst_text = ""
        for j in range(start + 1, end):
            if messages[j].get("role") == _asst_role:
                asst_text = _extract_text_for_stub_hash(messages[j])
                break

        combined = f"{user_text} {asst_text}"
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]

        entry = turn_tag_index.get_entry_by_hash(h)
        if entry is not None and entry.turn_number < watermark_turn:
            stubs.append((start, end, entry))

    if not stubs:
        return body, 0

    # Phase 4: build new message list, replacing stub ranges with lightweight markers.
    stub_starts: dict[int, tuple[int, int, object]] = {
        s[0]: s for s in sorted(stubs, key=lambda s: s[0])
    }
    new_messages: list[dict] = []
    i = 0
    while i < len(messages):
        stub = stub_starts.get(i)
        if stub:
            start, end, entry = stub
            tags_str = ", ".join(entry.tags[:5])
            new_messages.append({
                "role": "user",
                "content": f"[Compacted turn {entry.turn_number}]",
            })
            new_messages.append({
                "role": _asst_role,
                "content": [{"type": "text", "text":
                    f"[Compacted turn {entry.turn_number}: "
                    f"topics={tags_str}. "
                    f"Content stored in virtual-context.]"
                }],
            })
            i = end
        else:
            new_messages.append(messages[i])
            i += 1

    body = dict(body)
    body[_msg_key] = new_messages
    return body, len(stubs)
