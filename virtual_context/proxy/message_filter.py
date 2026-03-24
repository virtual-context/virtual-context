"""Message filtering: removes irrelevant history turns from API request bodies.

Pure functions — no ProxyState dependency. Extracted from proxy/server.py.
"""

from __future__ import annotations

import hashlib
import logging
import math

from ..core.turn_tag_index import TurnTagIndex
from ._envelope import _strip_envelope
from .formats import PayloadFormat, detect_format

logger = logging.getLogger(__name__)


def _is_tool_result_only_user(msg: dict) -> bool:
    """Whether a user message is tool-result scaffolding, not a real turn start."""
    if msg.get("role") not in ("user", "human"):
        return False
    content = msg.get("content", "")
    if not isinstance(content, list):
        return False
    ctypes = {block.get("type") for block in content if isinstance(block, dict)}
    return bool(ctypes and ctypes <= {"tool_result"})


def _consume_responses_tool_round(
    messages: list[dict],
    start: int,
    assistant_role: str,
) -> tuple[list[int], int] | None:
    """Consume a bare Responses tool round plus its closing assistant."""
    bare_indices: list[int] = []
    idx = start
    while idx < len(messages):
        item = messages[idx]
        if (
            not isinstance(item, dict)
            or item.get("role") is not None
            or item.get("type") not in ("function_call", "function_call_output")
        ):
            break
        bare_indices.append(idx)
        idx += 1
    if not bare_indices:
        return None
    if idx < len(messages) and messages[idx].get("role") == assistant_role:
        bare_indices.append(idx)
        return bare_indices, idx + 1
    return None


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
    - ``"aggressive"``: use *recent_turns* as-is (default)

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

    # Group into atomic turn chains, tracking which message indices are paired.
    # A chain is: user + assistant + zero or more tool rounds.
    # Tool round shapes:
    #   - OpenAI Chat: one or more role="tool" messages → assistant
    #   - Anthropic: user with tool_result-only content → assistant
    #   - Responses API: bare function_call + function_call_output items
    # Unpaired messages (those not captured by any chain) are always kept —
    # they're structural and may be required by the API.
    pairs: list[tuple[int, ...]] = []  # tuple of ALL msg indices in the chain
    paired_indices: set[int] = set()
    i = 0
    while i + 1 < len(chat_msgs):
        if (chat_msgs[i].get("role") == "user"
                and chat_msgs[i + 1].get("role") == _asst_role):
            chain_indices: list[int] = [i, i + 1]
            j = i + 2
            # Extend chain through tool rounds (same logic as trim_to_upstream_limit)
            while j < len(chat_msgs) - 1:
                next_msg = chat_msgs[j]
                next_role = next_msg.get("role", "")
                # OpenAI tool response (role="tool")
                if next_role == "tool":
                    tool_batch = [j]
                    k = j + 1
                    while k < len(chat_msgs) and chat_msgs[k].get("role") == "tool":
                        tool_batch.append(k)
                        k += 1
                    if k < len(chat_msgs) and chat_msgs[k].get("role") == _asst_role:
                        chain_indices.extend(tool_batch)
                        chain_indices.append(k)
                        j = k + 1
                        continue
                    else:
                        break  # tool messages without following assistant — stop
                # OpenAI Responses bare function_call/function_call_output round
                responses_round = _consume_responses_tool_round(chat_msgs, j, _asst_role)
                if responses_round:
                    round_indices, next_index = responses_round
                    chain_indices.extend(round_indices)
                    j = next_index
                    continue
                # Anthropic tool result (user message with tool_result-only content)
                if next_role not in ("user", "human"):
                    break
                if not _is_tool_result_only_user(next_msg):
                    break  # real user message, not part of tool chain
                if j + 1 >= len(chat_msgs) or chat_msgs[j + 1].get("role") != _asst_role:
                    break  # tool_result without following assistant — stop
                chain_indices.extend([j, j + 1])
                j += 2
            pairs.append(tuple(chain_indices))
            for ci in chain_indices:
                paired_indices.add(ci)
            i = chain_indices[-1] + 1
        else:
            i += 1

    total_pairs = len(pairs)
    protected = min(recent_turns, total_pairs)

    if total_pairs <= protected or not turn_tag_index.entries:
        return body, 0

    tag_set = set(matched_tags)

    # First pass: mark each chain as keep/drop based on tags
    keep_pair = [False] * total_pairs
    for pair_idx, chain in enumerate(pairs):
        if pair_idx >= total_pairs - protected:
            keep_pair[pair_idx] = True
        elif compacted_turn > 0 and pair_idx < compacted_turn:
            # PROXY-023: paging active — compacted turns are dropped
            # unconditionally. Their content is in VC summaries and
            # retrievable via vc_expand_topic.
            keep_pair[pair_idx] = False
            logger.debug("T%d DROP (paging: below compacted watermark %d)", pair_idx, compacted_turn)
        else:
            entry = turn_tag_index.get_tags_for_turn(pair_idx)
            if entry is None:
                keep_pair[pair_idx] = True
            elif "rule" in entry.tags or set(entry.tags) & tag_set:
                keep_pair[pair_idx] = True
            else:
                logger.debug("T%d DROP (no tag match: turn_tags=%s query_tags=%s)", pair_idx, entry.tags, matched_tags)

    _kept = sum(keep_pair)
    _dropped = total_pairs - _kept
    if _dropped:
        logger.info("Turn filter: %d/%d kept, %d dropped (protected=%d, watermark=%d)",
                     _kept, total_pairs, _dropped, protected, compacted_turn)

    # Second pass: build tool_use_id → message index maps for precise
    # referential integrity.  Every tool_use block has an "id" field;
    # every tool_result block has a "tool_use_id" field that must match
    # a tool_use in an earlier message.
    #
    # Map: tool_use_id → msg_idx that contains the tool_use
    # Map: tool_use_id → msg_idx that contains the tool_result
    tooluse_to_msg: dict[str, int] = {}   # id → assistant msg index
    toolresult_to_msg: dict[str, int] = {}  # tool_use_id → user/tool msg index
    for msg_idx, msg in enumerate(chat_msgs):
        # Anthropic: content blocks with type=tool_use / type=tool_result
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and "id" in block:
                    tooluse_to_msg[block["id"]] = msg_idx
                elif block.get("type") == "tool_result" and "tool_use_id" in block:
                    toolresult_to_msg[block["tool_use_id"]] = msg_idx
        # OpenAI: assistant message with tool_calls array
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict) and "id" in tc:
                tooluse_to_msg[tc["id"]] = msg_idx
        # OpenAI: tool role message with tool_call_id
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            toolresult_to_msg[msg["tool_call_id"]] = msg_idx

    # Build per-message keep set: unpaired messages always kept, chains based on filter
    keep_msg: set[int] = set()
    for msg_idx in range(len(chat_msgs)):
        if msg_idx not in paired_indices:
            keep_msg.add(msg_idx)  # always keep unpaired messages
    for pair_idx, chain in enumerate(pairs):
        if keep_pair[pair_idx]:
            for ci in chain:
                keep_msg.add(ci)

    # Build reverse map: msg_idx → all OTHER indices in its chain.
    # When any index in a chain is force-kept, all chain members must be kept
    # to preserve tool-call referential integrity and role alternation.
    msg_to_chain_peers: dict[int, set[int]] = {}
    for chain in pairs:
        chain_set = set(chain)
        for ci in chain:
            msg_to_chain_peers[ci] = chain_set - {ci}

    # Enforce tool_use_id referential integrity: if a message is kept,
    # the message containing the matching tool_use or tool_result must
    # also be kept.  When a force-kept message belongs to a chain, all
    # chain members are also kept (otherwise we'd have a partial chain
    # that breaks role alternation or orphans tool results).
    # Iterate until stable (chains can be long).
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
        # Keep all chain peers of any force-added messages
        for msg_idx in list(keep_msg):
            peers = msg_to_chain_peers.get(msg_idx)
            if peers:
                for peer in peers:
                    if peer not in keep_msg:
                        keep_msg.add(peer)
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
        # Also keep chain peers of anything we just added
        for backfill in range(idx + 1):
            peers = msg_to_chain_peers.get(backfill)
            if peers:
                for peer in peers:
                    if peer not in keep_msg:
                        keep_msg.add(peer)
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
                peers = msg_to_chain_peers.get(msg_idx)
                if peers:
                    for peer in peers:
                        if peer not in keep_msg:
                            keep_msg.add(peer)
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
    # Checks both Anthropic (tool_use/tool_result in content) and OpenAI
    # (tool_calls on assistant / role="tool" with tool_call_id) formats.
    _final_tu: set[str] = set()
    _final_tr: set[str] = set()
    for msg in kept:
        # Anthropic content blocks
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and "id" in block:
                    _final_tu.add(block["id"])
                elif block.get("type") == "tool_result" and "tool_use_id" in block:
                    _final_tr.add(block["tool_use_id"])
        # OpenAI tool_calls
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict) and "id" in tc:
                _final_tu.add(tc["id"])
        # OpenAI tool role
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            _final_tr.add(msg["tool_call_id"])
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

    # Identify user-text message indices (messages with extractable text
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

    # Group messages into turns.
    # Each turn spans from one user-text message to the next.
    turn_groups: list[tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
    for g, uti in enumerate(user_text_indices):
        if g + 1 < len(user_text_indices):
            end = user_text_indices[g + 1]
        else:
            end = len(messages)
        turn_groups.append((uti, end))

    # Hash each turn group and match against the index.
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

    # Build new message list, replacing stub ranges with lightweight markers.
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
            # Detect tool activity in the compacted range
            has_tools = any(
                messages[j].get("role") == "tool"
                or messages[j].get("tool_calls")
                or messages[j].get("type") in ("function_call", "function_call_output")
                or (isinstance(messages[j].get("content"), list) and any(
                    isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                    for b in messages[j]["content"]
                ))
                for j in range(start, end)
            )
            new_messages.append({
                "role": "user",
                "content": f"[Compacted turn {entry.turn_number}]",
            })
            if has_tools:
                stub_text = (
                    f"[Compacted turn {entry.turn_number}: "
                    f"tool activity compacted. "
                    f"Full tool output stored in virtual-context; "
                    f"use vc_find_quote(...) to search it or "
                    f"vc_expand_topic(...) for linked topic context.]"
                )
            else:
                stub_text = (
                    f"[Compacted turn {entry.turn_number}: "
                    f"topics={tags_str}. "
                    f"Content stored in virtual-context.]"
                )
            new_messages.append({
                "role": _asst_role,
                "content": [{"type": "text", "text": stub_text}],
            })
            i = end
        else:
            new_messages.append(messages[i])
            i += 1

    body = dict(body)
    body[_msg_key] = new_messages
    return body, len(stubs)


# ---------------------------------------------------------------------------
# Upstream context-window trimming
# ---------------------------------------------------------------------------


def _get_message_key(fmt) -> str:
    """Return the body key holding the message array for this format."""
    name = fmt.name
    if name == "gemini":
        return "contents"
    if name == "openai_responses":
        return "input"
    return "messages"


def trim_to_upstream_limit(
    body: dict,
    upstream_limit: int,
    fmt,
) -> tuple[dict, int]:
    """Trim oldest message pairs so payload fits within upstream context window.

    Uses adaptive batch removal: estimate excess fraction, drop proportional
    batch of oldest pairs, re-estimate with empirical tokens-per-pair.

    Always protects the last 2 user/assistant pairs (current + previous turn).
    System prompt and tools are never trimmed.

    Returns (trimmed_body, pairs_removed). Returns (body, 0) if no trim needed.
    """
    total = fmt.estimate_payload_tokens(body)
    # Output budget key varies by provider
    output_budget = body.get("max_tokens", 0)
    if not output_budget:
        gen_cfg = body.get("generationConfig", {})
        if isinstance(gen_cfg, dict):
            output_budget = gen_cfg.get("maxOutputTokens", 0)
    if not output_budget:
        output_budget = 4096
    input_limit = upstream_limit - output_budget

    if total <= input_limit:
        return body, 0

    msg_key = _get_message_key(fmt)
    original_messages = body.get(msg_key, [])
    if not original_messages or not isinstance(original_messages, list):
        return body, 0

    # Identify system prefix (not trimmable)
    system_prefix = 0
    if original_messages and original_messages[0].get("role") in ("system",):
        system_prefix = 1

    # Identify atomic units: regular pairs or tool chains.
    # A tool chain is an assistant[tool_use] → user[tool_result] → ... sequence
    # that must be dropped or kept together.
    # Each "pair" is a tuple of ALL message indices in the atomic unit.
    pairs: list[tuple[int, ...]] = []
    i = system_prefix
    while i < len(original_messages) - 1:
        msg = original_messages[i]
        if msg.get("role") not in ("user", "human"):
            i += 1
            continue
        # Verify next message is assistant/model — skip consecutive users
        if original_messages[i + 1].get("role") not in ("assistant", "model"):
            i += 1
            continue
        # Start of a turn: user message + assistant response
        chain_indices = [i, i + 1]  # user + assistant
        j = i + 2
        # Extend chain through tool rounds.
        # Anthropic: user[tool_result] → assistant → ...
        # OpenAI: tool (role="tool") → assistant → ...  (assistant has "tool_calls")
        while j < len(original_messages) - 1:
            next_msg = original_messages[j]
            next_role = next_msg.get("role", "")
            # OpenAI tool response (role="tool")
            if next_role == "tool":
                # Consume all consecutive tool messages + following assistant
                tool_batch = [j]
                k = j + 1
                while k < len(original_messages) and original_messages[k].get("role") == "tool":
                    tool_batch.append(k)
                    k += 1
                if k < len(original_messages) and original_messages[k].get("role") == "assistant":
                    chain_indices.extend(tool_batch)
                    chain_indices.append(k)
                    j = k + 1
                    continue
                else:
                    break  # tool messages without following assistant — stop
            # OpenAI Responses bare function_call/function_call_output round
            responses_round = _consume_responses_tool_round(original_messages, j, "assistant")
            if responses_round:
                round_indices, next_index = responses_round
                chain_indices.extend(round_indices)
                j = next_index
                continue
            # Anthropic tool result (user message with tool_result content blocks)
            if next_role not in ("user", "human"):
                break
            if not _is_tool_result_only_user(next_msg):
                break  # real user message, not part of tool chain
            # Verify next message is assistant before extending
            if j + 1 >= len(original_messages) or original_messages[j + 1].get("role") != "assistant":
                break  # tool_result without following assistant — malformed, stop
            # tool_result user + next assistant = part of chain
            chain_indices.extend([j, j + 1])
            j += 2
        pairs.append(tuple(chain_indices))
        i = chain_indices[-1] + 1

    if len(pairs) <= 2:
        return body, 0

    fixed = fmt._estimate_system_tokens(body) + fmt.estimate_tools_tokens(body)
    msg_tokens = total - fixed
    available = input_limit - fixed

    if msg_tokens <= 0:
        return body, 0

    total_pairs_removed = 0
    trimmable_count = len(pairs) - 2
    tokens_before = total

    # When available <= 0, system+tools+max_tokens already exceed the limit.
    # Best effort: drop all trimmable pairs.
    if available <= 0:
        total_pairs_removed = trimmable_count
        drop_indices: set[int] = set()
        for pair_idx in range(total_pairs_removed):
            for idx in pairs[pair_idx]:
                drop_indices.add(idx)
        new_messages = [m for idx, m in enumerate(original_messages) if idx not in drop_indices]
        trimmed_body = dict(body)
        trimmed_body[msg_key] = new_messages
        return trimmed_body, total_pairs_removed

    for _iteration in range(3):
        if msg_tokens <= available:
            break

        if total_pairs_removed > 0:
            tokens_per_pair = (tokens_before - total) / total_pairs_removed
            pairs_needed = math.ceil((msg_tokens - available) / max(tokens_per_pair, 1))
        else:
            excess_ratio = (msg_tokens - available) / msg_tokens
            pairs_needed = math.ceil(excess_ratio * trimmable_count)

        pairs_needed = max(1, min(pairs_needed, trimmable_count - total_pairs_removed))
        if pairs_needed <= 0:
            break

        total_pairs_removed += pairs_needed

        drop_indices: set[int] = set()
        for pair_idx in range(total_pairs_removed):
            for idx in pairs[pair_idx]:
                drop_indices.add(idx)

        new_messages = [m for idx, m in enumerate(original_messages) if idx not in drop_indices]
        trimmed_body = dict(body)
        trimmed_body[msg_key] = new_messages

        total = fmt.estimate_payload_tokens(trimmed_body)
        msg_tokens = total - fixed

    if total_pairs_removed == 0:
        return body, 0

    # Post-trim cleanup: remove any orphaned tool_use/tool_result pairs
    # that survived trimming without their counterpart.
    trimmed_body = _cleanup_orphaned_tools(trimmed_body, msg_key)

    return trimmed_body, total_pairs_removed


def _cleanup_orphaned_tools(body: dict, msg_key: str) -> dict:
    """Remove messages with orphaned tool_use/tool_result (Anthropic) or
    tool_calls/tool (OpenAI) blocks."""
    messages = body.get(msg_key, [])
    if not messages:
        return body

    for _pass in range(3):  # max 3 passes (removing one orphan may expose another)
        # Collect all tool IDs from both Anthropic and OpenAI formats
        tool_use_ids: set[str] = set()    # IDs offered (assistant side)
        tool_result_ids: set[str] = set()  # IDs answered (user/tool side)
        for m in messages:
            # Anthropic: content blocks with type=tool_use / type=tool_result
            content = m.get("content", [])
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use" and "id" in b:
                        tool_use_ids.add(b["id"])
                    elif b.get("type") == "tool_result" and "tool_use_id" in b:
                        tool_result_ids.add(b["tool_use_id"])
            # OpenAI: assistant message with tool_calls array
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict) and "id" in tc:
                    tool_use_ids.add(tc["id"])
            # OpenAI: tool role message with tool_call_id
            if m.get("role") == "tool" and "tool_call_id" in m:
                tool_result_ids.add(m["tool_call_id"])

        orphan_use_ids = tool_use_ids - tool_result_ids
        orphan_result_ids = tool_result_ids - tool_use_ids
        if not orphan_use_ids and not orphan_result_ids:
            break

        # Remove messages containing orphaned blocks
        cleaned: list[dict] = []
        for m in messages:
            has_orphan = False
            # Check Anthropic content blocks
            content = m.get("content", [])
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use" and b.get("id") in orphan_use_ids:
                        has_orphan = True
                        break
                    if b.get("type") == "tool_result" and b.get("tool_use_id") in orphan_result_ids:
                        has_orphan = True
                        break
            # Check OpenAI tool_calls
            if not has_orphan:
                for tc in m.get("tool_calls", []):
                    if isinstance(tc, dict) and tc.get("id") in orphan_use_ids:
                        has_orphan = True
                        break
            # Check OpenAI tool role
            if not has_orphan and m.get("role") == "tool" and m.get("tool_call_id") in orphan_result_ids:
                has_orphan = True
            if not has_orphan:
                cleaned.append(m)
        messages = cleaned

    body = dict(body)
    body[msg_key] = messages
    return body
