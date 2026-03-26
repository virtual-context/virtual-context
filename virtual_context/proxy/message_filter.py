"""Message filtering: removes irrelevant history turns from API request bodies.

Pure functions — no ProxyState dependency. Extracted from proxy/server.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import TYPE_CHECKING

from ..core.turn_tag_index import TurnTagIndex
from ._envelope import _strip_envelope
from .formats import PayloadFormat, detect_format

if TYPE_CHECKING:
    from ..core.store import ContextStore

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


def sanitize_vc_tool_errors(body: dict, fmt: PayloadFormat) -> dict:
    """Replace stale vc_restore_tool error results with benign acknowledgements.

    When the client SDK (e.g. Claude Code) didn't recognise vc_restore_tool,
    it returned ``<tool_use_error>`` XML.  These errors poison the conversation
    history — the model sees them and refuses to call the tool again.  Replace
    them with a neutral message so future calls aren't inhibited.
    """
    # Determine message key by format name
    _fname = fmt.name
    if _fname == "gemini":
        msg_key = "contents"
    elif _fname == "openai_responses":
        msg_key = "input"
    else:
        msg_key = "messages"
    messages = body.get(msg_key) if msg_key else None
    if not messages or not isinstance(messages, list):
        return body

    _ERROR_NEEDLE = "No such tool available: vc_restore_tool"
    _REPLACEMENT = (
        "The restore was handled internally. The original content is "
        "available in your conversation history above."
    )
    changed = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _ERROR_NEEDLE in content:
            msg["content"] = _REPLACEMENT
            if "is_error" in msg:
                del msg["is_error"]
            changed = True
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            # Anthropic tool_result blocks
            if block.get("type") == "tool_result":
                bc = block.get("content", "")
                if isinstance(bc, str) and _ERROR_NEEDLE in bc:
                    block["content"] = _REPLACEMENT
                    if "is_error" in block:
                        del block["is_error"]
                    changed = True
            # OpenAI function_call_output
            if block.get("type") == "function_call_output":
                out = block.get("output", "")
                if isinstance(out, str) and _ERROR_NEEDLE in out:
                    block["output"] = _REPLACEMENT
                    changed = True

    if changed:
        logger.info("SANITIZE: replaced stale vc_restore_tool error(s) in history")
    return body


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


def _chain_has_tool_activity(messages: list[dict], indices: tuple[int, ...] | range) -> bool:
    """Return True if any message in *indices* contains tool-call or tool-output signals.

    Checks all supported provider formats:
    - OpenAI Chat: ``role: "tool"`` or ``tool_calls`` array on assistant
    - OpenAI Responses: bare items with ``type`` of ``function_call`` / ``function_call_output``
    - Anthropic: content blocks with ``type`` of ``tool_use`` / ``tool_result``
    """
    for ci in indices:
        msg = messages[ci]
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
        if msg.get("type") in ("function_call", "function_call_output"):
            return True
        content = msg.get("content", [])
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for b in content
        ):
            return True
    return False


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

    # Detect which chains contain tool activity — these are drop-exempt.
    chain_has_tools = [False] * total_pairs
    for pair_idx, chain in enumerate(pairs):
        if _chain_has_tool_activity(chat_msgs, chain):
            chain_has_tools[pair_idx] = True

    # First pass: mark each chain as keep/drop based on tags
    keep_pair = [False] * total_pairs
    for pair_idx, chain in enumerate(pairs):
        if pair_idx >= total_pairs - protected:
            keep_pair[pair_idx] = True
        elif chain_has_tools[pair_idx]:
            # Tool-bearing chains are drop-exempt — always kept regardless
            # of tag match. Tool history is handled by position-based
            # stubbing, not semantic filtering.
            keep_pair[pair_idx] = True
            logger.debug("T%d KEEP (tool activity: drop-exempt)", pair_idx)
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
            # Skip tool-bearing turn groups — position-based stubbing
            # handles their tool outputs separately.
            if _chain_has_tool_activity(messages, range(start, end)):
                continue
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
            # Tool-bearing turns are skipped above (position-based
            # stubbing handles them), so only non-tool turns reach here.
            new_messages.append({
                "role": "user",
                "content": f"[Compacted turn {entry.turn_number}]",
            })
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
    input_limit = upstream_limit

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

    # Collect trailing messages not in any pair (e.g. the user's current
    # message with no assistant response yet). These are always kept.
    paired_indices: set[int] = set()
    for pair in pairs:
        for idx in pair:
            paired_indices.add(idx)
    trailing = tuple(
        idx for idx in range(system_prefix, len(original_messages))
        if idx not in paired_indices
    )

    if len(pairs) <= 2 and not trailing:
        return body, 0

    fixed = fmt._estimate_system_tokens(body) + fmt.estimate_tools_tokens(body)
    available = input_limit - fixed

    logger.info(
        "TRIM_BUDGET: total=%d fixed(system+tools)=%d "
        "input_limit=%d available_for_msgs=%d pairs=%d",
        total, fixed, input_limit, available, len(pairs),
    )

    if available <= 0:
        # System+tools alone exceed the limit — drop all trimmable pairs
        keep_indices: set[int] = set()
        for idx in pairs[-2]:
            keep_indices.add(idx)
        if len(pairs) > 2:
            for idx in pairs[-1]:
                keep_indices.add(idx)
        new_messages = [m for idx, m in enumerate(original_messages) if idx in keep_indices]
        trimmed_body = dict(body)
        trimmed_body[msg_key] = new_messages
        return trimmed_body, len(pairs) - 2

    # Build from newest to oldest. Start with trailing messages + last 2 pairs
    # (protected), then add older pairs one at a time until the budget is full.
    keep_pairs: list[tuple[int, ...]] = list(pairs[-2:])  # always keep last 2
    budget_used = 0
    # Always count trailing messages (user's current message)
    for idx in trailing:
        budget_used += fmt._count(json.dumps(original_messages[idx], default=str))
    for pair in keep_pairs:
        for idx in pair:
            budget_used += fmt._count(json.dumps(original_messages[idx], default=str))

    # Walk backwards from the third-to-last pair
    added = 0
    for pair_idx in range(len(pairs) - 3, -1, -1):
        pair = pairs[pair_idx]
        pair_tokens = 0
        for idx in pair:
            pair_tokens += fmt._count(json.dumps(original_messages[idx], default=str))
        if budget_used + pair_tokens <= available:
            keep_pairs.insert(0, pair)
            budget_used += pair_tokens
            added += 1
        # else: skip this pair, too big — but keep trying older smaller ones

    total_pairs_removed = len(pairs) - 2 - added
    if total_pairs_removed == 0:
        return body, 0

    keep_indices: set[int] = set()
    # Always keep system prefix message if present
    for idx in range(system_prefix):
        keep_indices.add(idx)
    for pair in keep_pairs:
        for idx in pair:
            keep_indices.add(idx)
    for idx in trailing:
        keep_indices.add(idx)

    new_messages = [m for idx, m in enumerate(original_messages) if idx in keep_indices]
    trimmed_body = dict(body)
    trimmed_body[msg_key] = new_messages

    logger.info(
        "TRIM_RESULT: kept=%d/%d pairs (%d msgs), budget_used=%d/%d",
        len(keep_pairs), len(pairs), len(new_messages), budget_used, available,
    )

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


# ---------------------------------------------------------------------------
# Position-based tool output stubbing
# ---------------------------------------------------------------------------


def _summarise_arguments(args: object, max_len: int = 200) -> str:
    """Produce a compact single-line summary of tool call arguments."""
    if args is None:
        return ""
    if isinstance(args, str):
        text = args
    elif isinstance(args, dict):
        # For dict args, format as key=value pairs
        parts: list[str] = []
        for k, v in args.items():
            v_str = str(v) if not isinstance(v, str) else v
            parts.append(f"{k}={v_str}")
        text = " ".join(parts)
    else:
        text = str(args)
    # Collapse whitespace and cap length
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def _replace_anthropic_content(block: dict, new_text: str) -> None:
    """Replace entire content of an Anthropic tool_result block.

    Replaces ALL content (including non-text blocks like images) with a
    single text block. When stubbing, the stub is the only thing that
    should remain. When restoring, the full text replaces everything.
    """
    block["content"] = new_text


def stub_tool_outputs_by_position(
    body: dict,
    fmt: PayloadFormat,
    protected_recent_turns: int,
    turn_tag_index: TurnTagIndex,
    store: ContextStore,
    conversation_id: str,
    **kwargs,
) -> tuple[dict, int, list[str]]:
    """Replace tool outputs outside the protected window with lightweight stubs.

    Scans all tool outputs in the payload.  Outputs in the last
    *protected_recent_turns* turn chains pass through unmodified.  Older
    outputs are stored with a unique ref and replaced in-place with a stub
    containing the ref, tool name, argument summary, and restore call.

    Returns ``(body, stub_count, list_of_refs)``.
    """
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
        return body, 0, []

    # ------------------------------------------------------------------
    # 1. Parse turn chains (same pattern as filter_body_messages)
    # ------------------------------------------------------------------
    prefix_count = 0
    chat_msgs: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not chat_msgs:
            prefix_count += 1
        else:
            chat_msgs.append(msg)

    if not chat_msgs:
        return body, 0, []

    # Separate trailing user message (current turn)
    has_trailing_user = bool(chat_msgs and chat_msgs[-1].get("role") == "user")
    history_msgs = chat_msgs[:-1] if has_trailing_user else chat_msgs

    pairs: list[tuple[int, ...]] = []  # tuples of GLOBAL message indices
    paired_indices: set[int] = set()
    i = 0
    while i + 1 < len(history_msgs):
        gi = i + prefix_count  # global index into messages list
        if (history_msgs[i].get("role") == "user"
                and history_msgs[i + 1].get("role") == _asst_role):
            chain_indices: list[int] = [gi, gi + 1]
            j = i + 2
            while j < len(history_msgs) - 1:
                next_msg = history_msgs[j]
                next_role = next_msg.get("role", "")
                gj = j + prefix_count
                # OpenAI tool response (role="tool")
                if next_role == "tool":
                    tool_batch = [gj]
                    k = j + 1
                    while k < len(history_msgs) and history_msgs[k].get("role") == "tool":
                        tool_batch.append(k + prefix_count)
                        k += 1
                    if k < len(history_msgs) and history_msgs[k].get("role") == _asst_role:
                        chain_indices.extend(tool_batch)
                        chain_indices.append(k + prefix_count)
                        j = k + 1
                        continue
                    else:
                        break
                # OpenAI Responses bare function_call/function_call_output round
                responses_round = _consume_responses_tool_round(history_msgs, j, _asst_role)
                if responses_round:
                    round_indices, next_index = responses_round
                    chain_indices.extend([ri + prefix_count for ri in round_indices])
                    j = next_index
                    continue
                # Anthropic tool result (user message with tool_result-only content)
                if next_role not in ("user", "human"):
                    break
                if not _is_tool_result_only_user(next_msg):
                    break
                if j + 1 >= len(history_msgs) or history_msgs[j + 1].get("role") != _asst_role:
                    break
                chain_indices.extend([gj, gj + 1])
                j += 2
            pairs.append(tuple(chain_indices))
            for ci in chain_indices:
                paired_indices.add(ci)
            i = chain_indices[-1] - prefix_count + 1
        else:
            i += 1

    if not pairs:
        return body, 0, []

    # ------------------------------------------------------------------
    # 2. Identify protected window (last N chains)
    # ------------------------------------------------------------------
    total_chains = len(pairs)
    protected = min(protected_recent_turns, total_chains)
    protected_start = total_chains - protected  # chains >= this index are protected

    # Conditional intrusion: if protected zone exceeds a percentage of the
    # context budget, allow stubbing into the protected zone except for
    # the last 2 chains (the current exchange).
    _intrusion_threshold = kwargs.get("protected_intrusion_threshold", 0.0)
    _context_budget = kwargs.get("context_budget", 0)
    _stub_protected_start = total_chains  # default: no intrusion

    if _intrusion_threshold > 0 and _context_budget > 0 and protected > 2:
        # Estimate protected zone token size
        _prot_bytes = 0
        for chain_idx in range(protected_start, total_chains):
            for gi in pairs[chain_idx]:
                _prot_bytes += len(json.dumps(messages[gi], default=str))
        _prot_tokens = _prot_bytes // 4
        _prot_ratio = _prot_tokens / _context_budget if _context_budget else 0

        if _prot_ratio > _intrusion_threshold:
            # Allow stubbing inside protected zone except last 2 chains
            _stub_protected_start = total_chains - 2
            logger.info(
                "PROTECTED_INTRUSION: protected zone %dt is %.0f%% of budget %dt "
                "(threshold %.0f%%) — stubbing turns 3+ in protected window",
                _prot_tokens, _prot_ratio * 100, _context_budget,
                _intrusion_threshold * 100,
            )
        else:
            _stub_protected_start = total_chains  # no intrusion needed

    # Build global-index → chain-index map
    msg_to_chain: dict[int, int] = {}
    for chain_idx, chain in enumerate(pairs):
        for gi in chain:
            msg_to_chain[gi] = chain_idx

    # ------------------------------------------------------------------
    # 3. Build tool call map: call_id → {name, arguments}
    # ------------------------------------------------------------------
    tool_call_map: dict[str, dict] = {}
    for tc in fmt.iter_tool_calls(body):
        if tc.call_id:
            tool_call_map[tc.call_id] = {
                "name": tc.name,
                "arguments": tc.arguments,
            }

    # ------------------------------------------------------------------
    # 4. Resolve canonical turn numbers per chain via hash lookup
    # ------------------------------------------------------------------
    chain_canonical_turn: dict[int, int] = {}
    for chain_idx, chain in enumerate(pairs):
        # Find user-text message and first assistant message in chain
        user_text = ""
        asst_text = ""
        for gi in chain:
            msg = messages[gi]
            if msg.get("role") == "user" and not _is_tool_result_only_user(msg):
                user_text = _extract_text_for_stub_hash(msg)
            elif msg.get("role") == _asst_role and not asst_text:
                asst_text = _extract_text_for_stub_hash(msg)
        if user_text:
            combined = f"{user_text} {asst_text}"
            h = hashlib.sha256(combined.encode()).hexdigest()[:16]
            entry = turn_tag_index.get_entry_by_hash(h)
            if entry is not None:
                chain_canonical_turn[chain_idx] = entry.turn_number
                continue
        # No canonical turn resolved — mark as unknown.
        # Do NOT use body-local chain index: after filtering drops turns,
        # chain_idx no longer corresponds to the real turn number.
        chain_canonical_turn[chain_idx] = -1

    # ------------------------------------------------------------------
    # 5. Stub tool outputs outside protected window
    # ------------------------------------------------------------------
    # Import VC_TOOL_NAMES to skip VC tool results
    from ..core.tool_loop import VC_TOOL_NAMES

    stub_count = 0
    stub_refs: list[str] = []

    # Hard-protected message indices: last 4 messages (last 2 turns) are never stubbed
    _hard_protected_msg_start = max(0, len(messages) - 4)

    for output in fmt.iter_tool_outputs(body):
        # Hard-protect last 2 turns by message index
        if output.msg_index >= _hard_protected_msg_start:
            continue

        # If in a recognized chain, use chain-level protection
        chain_idx = msg_to_chain.get(output.msg_index)
        if chain_idx is not None:
            if chain_idx >= _stub_protected_start:
                continue
            if chain_idx >= protected_start and _stub_protected_start >= total_chains:
                continue
        else:
            # Not in any chain — use message-index-based protection
            # Only protect if within the last protected_recent_turns * 2 messages
            _soft_protected_msg_start = max(0, len(messages) - protected_recent_turns * 2)
            if output.msg_index >= _soft_protected_msg_start and _stub_protected_start >= total_chains:
                continue

        # Skip VC tool outputs to prevent feedback loops
        call_info = tool_call_map.get(output.call_id, {})
        tool_name = call_info.get("name", "")
        if tool_name in VC_TOOL_NAMES:
            continue

        content_text = output.content
        if not content_text:
            continue

        # Content-addressed ref: same content always produces the same ref.
        # Prevents duplicate storage when the client resends the full history.
        ref = f"tool_{hashlib.sha256(content_text.encode()).hexdigest()[:12]}"

        # Resolve canonical turn
        canonical_turn = chain_canonical_turn.get(chain_idx, chain_idx)

        # Summarise arguments for the stub
        args_summary = _summarise_arguments(call_info.get("arguments"))

        # Store the full content
        try:
            store.store_tool_output(
                ref=ref,
                conversation_id=conversation_id,
                tool_name=tool_name,
                command=args_summary,
                turn=canonical_turn,
                content=content_text,
                original_bytes=len(content_text.encode("utf-8")),
            )
        except Exception:
            logger.warning("TOOL-STUB: failed to store ref=%s", ref, exc_info=True)
            continue

        # Write turn link (only if canonical turn was resolved)
        if canonical_turn < 0:
            logger.debug("TOOL-STUB: skipping turn link for ref=%s (no canonical turn)", ref)
        else:
            try:
                store.link_turn_tool_output(conversation_id, canonical_turn, ref)
            except Exception:
                pass  # non-critical

        # Build stub text
        stub_text = (
            f"[tool output ref={ref}"
            f" | tool={tool_name or 'unknown'}"
            f' args="{args_summary}"'
            f' | call vc_restore_tool(ref="{ref}")]'
        )

        # Replace content in-place
        carrier = output.carrier
        carrier_type = output.carrier_type
        if carrier_type == "anthropic":
            _replace_anthropic_content(carrier, stub_text)
        elif carrier_type == "openai_chat":
            carrier["content"] = stub_text
        elif carrier_type == "openai_responses":
            carrier["output"] = stub_text

        stub_count += 1
        stub_refs.append(ref)

    return body, stub_count, stub_refs


# ---------------------------------------------------------------------------
# Stage 2: Full turn chain collapse (post-compaction)
# ---------------------------------------------------------------------------


def _extract_tool_metadata_from_chain(
    messages: list[dict],
    chain_global_indices: tuple[int, ...],
    tool_call_map: dict[str, dict],
) -> list[str]:
    """Extract compact tool call descriptions from a turn chain.

    Returns a list of strings like ``"Read(file.py)"``, ``"Exec(cmd)"``.
    """
    from ..core.tool_loop import VC_TOOL_NAMES

    seen: list[str] = []
    seen_ids: set[str] = set()
    for gi in chain_global_indices:
        msg = messages[gi]
        # Anthropic tool_use blocks
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and "id" in block:
                    call_id = block["id"]
                    if call_id in seen_ids:
                        continue
                    seen_ids.add(call_id)
                    name = block.get("name", "")
                    if name in VC_TOOL_NAMES:
                        continue
                    args = _summarise_arguments(block.get("input"), max_len=60)
                    seen.append(f"{name}({args})" if args else name)
        # OpenAI tool_calls
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict) and "id" in tc:
                call_id = tc["id"]
                if call_id in seen_ids:
                    continue
                seen_ids.add(call_id)
                name = tc.get("function", {}).get("name", "")
                if name in VC_TOOL_NAMES:
                    continue
                args = _summarise_arguments(tc.get("function", {}).get("arguments"), max_len=60)
                seen.append(f"{name}({args})" if args else name)
    return seen


def _find_pre_filter_chain(
    pre_filter_messages: list[dict],
    user_text_needle: str,
    _asst_role: str,
    used_indices: set[int] | None = None,
) -> list[int] | None:
    """Find the turn chain in pre_filter_messages that starts with a user
    message whose extracted text matches *user_text_needle*.

    *used_indices* tracks starting positions already claimed by previous
    calls, preventing duplicate user text (e.g. "continue", "yes") from
    always matching the same first occurrence.

    Returns list of global indices into pre_filter_messages, or None.
    """
    if used_indices is None:
        used_indices = set()
    for i, msg in enumerate(pre_filter_messages):
        if i in used_indices:
            continue
        if msg.get("role") != "user":
            continue
        text = _extract_text_for_stub_hash(msg)
        if text != user_text_needle:
            continue
        # Found the matching user message — build the chain
        chain: list[int] = [i]
        if i + 1 >= len(pre_filter_messages):
            continue
        if pre_filter_messages[i + 1].get("role") != _asst_role:
            continue
        chain.append(i + 1)
        j = i + 2
        while j < len(pre_filter_messages):
            next_msg = pre_filter_messages[j]
            next_role = next_msg.get("role", "")
            # OpenAI tool response
            if next_role == "tool":
                tool_batch = [j]
                k = j + 1
                while k < len(pre_filter_messages) and pre_filter_messages[k].get("role") == "tool":
                    tool_batch.append(k)
                    k += 1
                if k < len(pre_filter_messages) and pre_filter_messages[k].get("role") == _asst_role:
                    chain.extend(tool_batch)
                    chain.append(k)
                    j = k + 1
                    continue
                else:
                    break
            # OpenAI Responses bare round
            responses_round = _consume_responses_tool_round(pre_filter_messages, j, _asst_role)
            if responses_round:
                round_indices, next_index = responses_round
                chain.extend(round_indices)
                j = next_index
                continue
            # Anthropic tool result
            if next_role not in ("user", "human"):
                break
            if not _is_tool_result_only_user(next_msg):
                break
            if j + 1 >= len(pre_filter_messages) or pre_filter_messages[j + 1].get("role") != _asst_role:
                break
            chain.extend([j, j + 1])
            j += 2
        return chain
    return None


def collapse_turn_chains(
    body: dict,
    fmt: PayloadFormat,
    pre_filter_body: dict,
    protected_recent_turns: int,
    turn_tag_index: TurnTagIndex,
    store: ContextStore,
    conversation_id: str,
) -> tuple[dict, int, list[str]]:
    """Collapse turn chains outside protected window to stub pairs (stage 2).

    Each turn chain outside the protected window is replaced with a compact
    stub pair (user + assistant) that includes tool names and a restore ref.
    The original messages (including thinking blocks) are captured from
    *pre_filter_body* and stored as a chain snapshot for later restore.
    Individual tool outputs are also stored for FTS search.

    Returns ``(modified_body, collapse_count, list_of_chain_refs)``.
    """
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
        return body, 0, []

    # ------------------------------------------------------------------
    # 1. Parse turn chains (same pattern as stub_tool_outputs_by_position)
    # ------------------------------------------------------------------
    prefix_count = 0
    chat_msgs: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not chat_msgs:
            prefix_count += 1
        else:
            chat_msgs.append(msg)

    if not chat_msgs:
        return body, 0, []

    has_trailing_user = bool(chat_msgs and chat_msgs[-1].get("role") == "user")
    history_msgs = chat_msgs[:-1] if has_trailing_user else chat_msgs

    pairs: list[tuple[int, ...]] = []  # tuples of GLOBAL message indices
    i = 0
    while i + 1 < len(history_msgs):
        gi = i + prefix_count
        if (history_msgs[i].get("role") == "user"
                and history_msgs[i + 1].get("role") == _asst_role):
            chain_indices: list[int] = [gi, gi + 1]
            j = i + 2
            while j < len(history_msgs) - 1:
                next_msg = history_msgs[j]
                next_role = next_msg.get("role", "")
                gj = j + prefix_count
                if next_role == "tool":
                    tool_batch = [gj]
                    k = j + 1
                    while k < len(history_msgs) and history_msgs[k].get("role") == "tool":
                        tool_batch.append(k + prefix_count)
                        k += 1
                    if k < len(history_msgs) and history_msgs[k].get("role") == _asst_role:
                        chain_indices.extend(tool_batch)
                        chain_indices.append(k + prefix_count)
                        j = k + 1
                        continue
                    else:
                        break
                responses_round = _consume_responses_tool_round(history_msgs, j, _asst_role)
                if responses_round:
                    round_indices, next_index = responses_round
                    chain_indices.extend([ri + prefix_count for ri in round_indices])
                    j = next_index
                    continue
                if next_role not in ("user", "human"):
                    break
                if not _is_tool_result_only_user(next_msg):
                    break
                if j + 1 >= len(history_msgs) or history_msgs[j + 1].get("role") != _asst_role:
                    break
                chain_indices.extend([gj, gj + 1])
                j += 2
            pairs.append(tuple(chain_indices))
            i = chain_indices[-1] - prefix_count + 1
        else:
            i += 1

    if not pairs:
        return body, 0, []

    # ------------------------------------------------------------------
    # 2. Identify protected window (last N chains)
    # ------------------------------------------------------------------
    total_chains = len(pairs)
    protected = min(protected_recent_turns, total_chains)
    protected_start = total_chains - protected

    # ------------------------------------------------------------------
    # 3. Build tool call map
    # ------------------------------------------------------------------
    tool_call_map: dict[str, dict] = {}
    for tc in fmt.iter_tool_calls(body):
        if tc.call_id:
            tool_call_map[tc.call_id] = {
                "name": tc.name,
                "arguments": tc.arguments,
            }

    # ------------------------------------------------------------------
    # 4. Resolve canonical turn numbers via hash lookup
    # ------------------------------------------------------------------
    chain_canonical_turn: dict[int, int] = {}
    chain_user_text: dict[int, str] = {}  # for pre_filter matching
    for chain_idx, chain in enumerate(pairs):
        user_text = ""
        asst_text = ""
        for gi in chain:
            msg = messages[gi]
            if msg.get("role") == "user" and not _is_tool_result_only_user(msg):
                user_text = _extract_text_for_stub_hash(msg)
            elif msg.get("role") == _asst_role and not asst_text:
                asst_text = _extract_text_for_stub_hash(msg)
        chain_user_text[chain_idx] = user_text
        if user_text:
            combined = f"{user_text} {asst_text}"
            h = hashlib.sha256(combined.encode()).hexdigest()[:16]
            entry = turn_tag_index.get_entry_by_hash(h)
            if entry is not None:
                chain_canonical_turn[chain_idx] = entry.turn_number
                continue
        chain_canonical_turn[chain_idx] = -1

    # ------------------------------------------------------------------
    # 5. Prepare pre_filter messages for snapshot extraction
    # ------------------------------------------------------------------
    pre_filter_messages = pre_filter_body.get(_msg_key, [])

    # ------------------------------------------------------------------
    # 6. Collapse chains outside the protected window
    # ------------------------------------------------------------------
    from ..core.tool_loop import VC_TOOL_NAMES

    collapse_count = 0
    _pf_used: set[int] = set()  # track claimed pre-filter indices
    chain_refs: list[str] = []
    # Map: chain_idx → (ref, stub_user, stub_asst) for chains to collapse
    collapse_map: dict[int, tuple[str, dict, dict]] = {}

    for chain_idx in range(protected_start):
        chain = pairs[chain_idx]
        canonical_turn = chain_canonical_turn.get(chain_idx, -1)

        # Skip chains without tool activity — they're handled by
        # stub_compacted_messages
        if not _chain_has_tool_activity(messages, chain):
            continue

        user_text = chain_user_text.get(chain_idx, "")
        if not user_text:
            continue

        # a. Find corresponding messages in pre_filter_body
        pf_chain = _find_pre_filter_chain(pre_filter_messages, user_text, _asst_role, used_indices=_pf_used)
        if pf_chain is None:
            logger.debug("CHAIN-COLLAPSE: no pre-filter match for chain %d (turn %d)", chain_idx, canonical_turn)
            continue

        # Claim these pre-filter indices so duplicate user text doesn't re-match
        _pf_used.update(pf_chain)

        # b. Serialize chain snapshot from pre-filter body
        pf_msgs = [pre_filter_messages[pi] for pi in pf_chain]
        chain_json = json.dumps(pf_msgs, default=str)

        # c. Generate composite ref
        ref = f"chain_{canonical_turn}_{hashlib.sha256(chain_json.encode()).hexdigest()[:12]}"

        # d. Store individual tool outputs
        tool_output_refs_for_chain: list[str] = []
        for output in fmt.iter_tool_outputs(body):
            if output.msg_index not in chain:
                continue
            call_info = tool_call_map.get(output.call_id, {})
            tool_name = call_info.get("name", "")
            if tool_name in VC_TOOL_NAMES:
                continue
            content_text = output.content
            if not content_text:
                continue
            tool_ref = f"tool_{hashlib.sha256(content_text.encode()).hexdigest()[:12]}"
            args_summary = _summarise_arguments(call_info.get("arguments"))
            try:
                store.store_tool_output(
                    ref=tool_ref,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    command=args_summary,
                    turn=canonical_turn,
                    content=content_text,
                    original_bytes=len(content_text.encode("utf-8")),
                )
            except Exception:
                logger.warning("CHAIN-COLLAPSE: failed to store tool output ref=%s", tool_ref, exc_info=True)
                continue
            if canonical_turn >= 0:
                try:
                    store.link_turn_tool_output(conversation_id, canonical_turn, tool_ref)
                except Exception:
                    pass
            tool_output_refs_for_chain.append(tool_ref)

        # e. Store chain snapshot
        try:
            store.store_chain_snapshot(
                ref=ref,
                conversation_id=conversation_id,
                turn_number=canonical_turn,
                chain_json=chain_json,
                message_count=len(pf_msgs),
                tool_output_refs=",".join(tool_output_refs_for_chain),
            )
        except Exception:
            logger.warning("CHAIN-COLLAPSE: failed to store chain snapshot ref=%s", ref, exc_info=True)
            continue

        # f. Build tool metadata for stub
        tool_descs = _extract_tool_metadata_from_chain(messages, chain, tool_call_map)
        tool_str = ", ".join(tool_descs) if tool_descs else ""

        # g. Build stub pair
        turn_label = canonical_turn if canonical_turn >= 0 else chain_idx
        stub_user = {
            "role": "user",
            "content": f"[Compacted turn {turn_label}]",
        }
        # Build descriptive line
        desc_parts = [f"Compacted turn {turn_label}"]
        if canonical_turn >= 0:
            entry = turn_tag_index.get_tags_for_turn(canonical_turn)
            if entry and entry.tags:
                desc_parts.append(f"topics={', '.join(entry.tags[:5])}")
        if tool_str:
            desc_parts.append(tool_str)
        desc_line = " | ".join(desc_parts)
        # Build explicit restore instruction
        stub_text = (
            f"[{desc_line}.\n"
            f'To restore and uncompact full tool call results in place: '
            f'{{"type": "tool_use", "name": "vc_restore_tool", '
            f'"input": {{"ref": "{ref}"}}}}]'
        )
        stub_asst = {
            "role": _asst_role,
            "content": [{"type": "text", "text": stub_text}],
        }

        collapse_map[chain_idx] = (ref, stub_user, stub_asst)
        chain_refs.append(ref)
        collapse_count += 1

    if not collapse_map:
        return body, 0, []

    # ------------------------------------------------------------------
    # 7. Build new message list with collapsed chains
    # ------------------------------------------------------------------
    # Collect all global indices that belong to collapsed chains
    # and all tool_use IDs within those chains (for orphan cleanup).
    collapsed_indices: set[int] = set()
    collapsed_tool_use_ids: set[str] = set()
    for chain_idx, (ref, stub_user, stub_asst) in collapse_map.items():
        chain = pairs[chain_idx]
        for gi in chain:
            collapsed_indices.add(gi)
            msg = messages[gi]
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                    ):
                        _tid = block.get("id")
                        if _tid:
                            collapsed_tool_use_ids.add(_tid)

    # Build index → insertion stubs map
    insert_at: dict[int, tuple[dict, dict]] = {}
    for chain_idx, (ref, stub_user, stub_asst) in collapse_map.items():
        insert_at[pairs[chain_idx][0]] = (stub_user, stub_asst)

    new_messages: list[dict] = []
    for mi in range(len(messages)):
        if mi in collapsed_indices:
            if mi in insert_at:
                stub_user, stub_asst = insert_at[mi]
                new_messages.append(stub_user)
                new_messages.append(stub_asst)
            # else: skip (interior of a collapsed chain)
        else:
            msg = messages[mi]
            # Strip orphaned tool_result blocks whose tool_use was collapsed
            if collapsed_tool_use_ids and isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list):
                    cleaned = [
                        block for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and block.get("tool_use_id") in collapsed_tool_use_ids
                        )
                    ]
                    if len(cleaned) != len(content):
                        msg = dict(msg)
                        msg["content"] = cleaned if cleaned else [{"type": "text", "text": "[tool results removed — parent tool call was compacted]"}]
            new_messages.append(msg)

    body = dict(body)
    body[_msg_key] = new_messages
    return body, collapse_count, chain_refs
