"""Message filtering: removes irrelevant history turns from API request bodies.

Pure functions — no ProxyState dependency. Extracted from proxy/server.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.turn_tag_index import TurnTagIndex
from ._envelope import _strip_envelope
from .formats import PayloadFormat, detect_format

if TYPE_CHECKING:
    from ..core.store import ContextStore
    from ..types import AssembledContext

logger = logging.getLogger(__name__)


def _is_tool_result_only_user(msg: dict) -> bool:
    """Whether a user message is tool-result scaffolding, not a real turn start.

    Handles Anthropic (tool_result content blocks) and Gemini (functionResponse
    parts).
    """
    if msg.get("role") not in ("user", "human"):
        return False
    # Anthropic: content blocks with type="tool_result"
    content = msg.get("content", "")
    if isinstance(content, list):
        ctypes = {block.get("type") for block in content if isinstance(block, dict)}
        if ctypes and ctypes <= {"tool_result"}:
            return True
    # Gemini: parts with functionResponse
    parts = msg.get("parts", [])
    if isinstance(parts, list) and parts:
        has_func_response = any(
            isinstance(p, dict) and "functionResponse" in p for p in parts
        )
        has_text = any(
            isinstance(p, dict) and "text" in p and "functionResponse" not in p
            for p in parts
        )
        if has_func_response and not has_text:
            return True
    return False


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
    - Gemini: parts with ``functionCall`` or ``functionResponse``
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
        # Gemini: functionCall/functionResponse in parts
        parts = msg.get("parts", [])
        if isinstance(parts, list) and any(
            isinstance(p, dict) and ("functionCall" in p or "functionResponse" in p)
            for p in parts
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

    Handles all four provider formats:
    - Anthropic: content blocks with type="text"
    - OpenAI Chat: string content or content blocks with type="text"
    - OpenAI Responses: content blocks with type="input_text" or "output_text"
    - Gemini: parts array with {"text": "..."} dicts (no type field)

    Uses reversed iteration over content blocks so that multi-text-block
    messages (common in Anthropic tool_use responses) return the **last**
    text block — matching the engine's ``_last_text_block`` behaviour.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return _strip_envelope(content).strip()
    if isinstance(content, list):
        # Accept "text" (Anthropic/OpenAI Chat), "input_text" (Responses user),
        # and "output_text" (Responses assistant).
        _text_types = {"text", "input_text", "output_text"}
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") in _text_types:
                text = block.get("text", "")
                return _strip_envelope(text).strip()
    # Gemini: messages use "parts" instead of "content"
    parts = msg.get("parts", [])
    if isinstance(parts, list):
        for part in reversed(parts):
            if isinstance(part, dict) and "text" in part:
                # Skip functionCall/functionResponse parts
                if "functionCall" in part or "functionResponse" in part:
                    continue
                text = part.get("text", "")
                return _strip_envelope(text).strip()
    return ""


def drop_compacted_turns(
    body: dict,
    turn_tag_index: TurnTagIndex,
    compacted_through: int,
    *,
    fmt: PayloadFormat | None = None,
    protected_recent_turns: int = 6,
) -> tuple[dict, int]:
    """Drop non-tool turns outside the protected window.

    Uses ``fmt.group_into_turns()`` to identify turn boundaries and drops
    all non-tool turns outside the last *protected_recent_turns* groups.
    These turns are already represented by compacted segments in the VC
    context injection.  Tool-bearing turns are left for chain collapse.

    No hash matching — all non-tool history turns are dropped regardless
    of whether they match the turn tag index.

    Returns (modified_body, drop_count).
    """
    if compacted_through <= 0:
        return body, 0

    if fmt is None:
        fmt = detect_format(body)

    turn_groups = fmt.group_into_turns(body)
    if not turn_groups:
        return body, 0

    # Separate trailing user-only group (current question) from history.
    messages = fmt.get_messages(body)
    last_group = turn_groups[-1]
    last_idx = last_group.indices[-1] if last_group.indices else -1
    last_msg = messages[last_idx] if 0 <= last_idx < len(messages) else {}
    if last_msg.get("role") in ("user", "human"):
        all_user = all(
            messages[i].get("role") in ("user", "human")
            for i in last_group.indices
            if 0 <= i < len(messages)
        )
        if all_user:
            history_turns = turn_groups[:-1]
        else:
            history_turns = turn_groups
    else:
        history_turns = turn_groups

    if not history_turns:
        return body, 0

    total = len(history_turns)
    protected = min(protected_recent_turns, total)
    protected_start = total - protected

    # Collect indices of non-tool turns outside the protected window
    drop_indices: set[int] = set()
    drop_count = 0
    for tidx in range(protected_start):
        turn = history_turns[tidx]
        if turn.has_tool_activity:
            continue  # chain collapse handles these
        for gi in turn.indices:
            drop_indices.add(gi)
        drop_count += 1

    if not drop_indices:
        return body, 0

    # Remove dropped indices
    fmt.remove_items(body, sorted(drop_indices))

    return body, drop_count


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
    """Trim oldest turn groups so payload fits within upstream context window.

    Uses ``fmt.group_into_turns(body)`` for atomic turn grouping across all
    provider formats.  Builds from newest turn groups to oldest, filling the
    budget greedily.  Always protects the last 2 turn groups (current +
    previous turn).  System prompt and tools are never trimmed.

    Returns (trimmed_body, turns_removed). Returns (body, 0) if no trim needed.
    """
    total = fmt.estimate_payload_tokens(body)
    input_limit = upstream_limit

    if total <= input_limit:
        return body, 0

    msg_key = _get_message_key(fmt)
    original_messages = body.get(msg_key, [])
    if not original_messages or not isinstance(original_messages, list):
        return body, 0

    # Use format abstraction for atomic turn grouping
    all_groups = fmt.group_into_turns(body)

    # Separate system-prefix groups (always kept, not counted against budget)
    # from real turn groups.  A "system prefix" group is one at the start of
    # the array whose items all have role="system".
    system_groups: list = []
    turn_groups: list = []
    for g in all_groups:
        if not turn_groups:
            # Still scanning for system-only prefix groups
            all_system = all(
                original_messages[idx].get("role") == "system"
                for idx in g.indices
            )
            if all_system:
                system_groups.append(g)
                continue
        turn_groups.append(g)

    # Collect trailing items: indices not covered by any turn group.
    # group_into_turns covers all items, so this is normally empty, but
    # handle the edge case defensively.
    grouped_indices: set[int] = set()
    for g in all_groups:
        grouped_indices.update(g.indices)
    trailing = tuple(
        idx for idx in range(len(original_messages))
        if idx not in grouped_indices
    )

    if len(turn_groups) <= 2 and not trailing:
        return body, 0

    fixed = fmt._estimate_system_tokens(body) + fmt.estimate_tools_tokens(body)
    available = input_limit - fixed

    logger.info(
        "TRIM_BUDGET: total=%d fixed(system+tools)=%d "
        "input_limit=%d available_for_msgs=%d turns=%d",
        total, fixed, input_limit, available, len(turn_groups),
    )

    # Protect the last 2 turn groups (or all if fewer than 2)
    protected_count = min(2, len(turn_groups))

    if available <= 0:
        # System+tools alone exceed the limit — keep only protected turns
        keep_indices: set[int] = set()
        for g in system_groups:
            keep_indices.update(g.indices)
        for g in turn_groups[-protected_count:]:
            keep_indices.update(g.indices)
        for idx in trailing:
            keep_indices.add(idx)
        new_messages = [m for idx, m in enumerate(original_messages) if idx in keep_indices]
        trimmed_body = dict(body)
        trimmed_body[msg_key] = new_messages
        return trimmed_body, len(turn_groups) - protected_count

    # Build from newest to oldest.  Start with protected turn groups +
    # trailing items, then add older groups until the budget is full.
    keep_groups = list(turn_groups[-protected_count:])
    budget_used = 0

    # Always count trailing items
    for idx in trailing:
        budget_used += fmt._count(json.dumps(original_messages[idx], default=str))
    # Count system-prefix groups (always kept)
    for g in system_groups:
        for idx in g.indices:
            budget_used += fmt._count(json.dumps(original_messages[idx], default=str))
    # Count protected turn groups
    for g in keep_groups:
        for idx in g.indices:
            budget_used += fmt._count(json.dumps(original_messages[idx], default=str))

    # Walk backwards from the oldest unprotected turn group
    added = 0
    trimmable = turn_groups[:-protected_count] if protected_count else turn_groups
    for g_idx in range(len(trimmable) - 1, -1, -1):
        g = trimmable[g_idx]
        g_tokens = 0
        for idx in g.indices:
            g_tokens += fmt._count(json.dumps(original_messages[idx], default=str))
        if budget_used + g_tokens <= available:
            keep_groups.insert(0, g)
            budget_used += g_tokens
            added += 1
        # else: skip — too big, but keep trying older smaller ones

    total_removed = len(trimmable) - added
    if total_removed == 0:
        return body, 0

    keep_indices: set[int] = set()
    for g in system_groups:
        keep_indices.update(g.indices)
    for g in keep_groups:
        keep_indices.update(g.indices)
    for idx in trailing:
        keep_indices.add(idx)

    new_messages = [m for idx, m in enumerate(original_messages) if idx in keep_indices]
    trimmed_body = dict(body)
    trimmed_body[msg_key] = new_messages

    logger.info(
        "TRIM_RESULT: kept=%d/%d turns (%d msgs), budget_used=%d/%d",
        len(keep_groups), len(turn_groups), len(new_messages), budget_used, available,
    )

    # Post-trim cleanup: remove any orphaned tool_use/tool_result pairs
    # that survived trimming without their counterpart.
    trimmed_body = _cleanup_orphaned_tools(trimmed_body, msg_key)

    return trimmed_body, total_removed


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


def drop_topic_only_stubs(
    body: dict,
    fmt: PayloadFormat,
) -> tuple[dict, int]:
    """Remove VC stub messages that have no restore ref.

    Stubs are identified by the _vc_stub marker (via fmt.is_vc_stub).
    A stub is topic-only if its text does not contain 'vc_restore_tool'.
    These are dead weight — summaries already cover their content.

    Stubs are grouped into user+assistant pairs. If either message in a
    pair contains a restore ref, the entire pair is kept.

    Uses fmt.extract_text_from_item for format-aware text extraction.

    Returns (body, stubs_dropped).
    """
    messages = fmt.get_messages(body)

    # Group consecutive stubs into pairs (user + assistant).
    # Walk through stubs in the contiguous run and pair them up.
    stub_pairs: list[list[int]] = []
    current_pair: list[int] = []
    for i in range(len(messages)):
        if not fmt.is_vc_stub(body, i):
            if current_pair:
                stub_pairs.append(current_pair)
                current_pair = []
            continue
        current_pair.append(i)
        if len(current_pair) == 2:
            stub_pairs.append(current_pair)
            current_pair = []
    if current_pair:
        stub_pairs.append(current_pair)

    # For each pair, check if ANY message in the pair has a restore ref
    drop_indices = []
    for pair in stub_pairs:
        has_restore = any(
            "vc_restore_tool" in fmt.extract_text_from_item(body, i)
            for i in pair
        )
        if not has_restore:
            drop_indices.extend(pair)

    if drop_indices:
        fmt.remove_items(body, drop_indices)

    return body, len(drop_indices)


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

    Uses ``fmt.group_into_turns(body)`` for turn detection and protection
    window calculation, ``fmt.iter_tool_outputs(body)`` for finding outputs,
    and ``fmt.replace_tool_output_content()`` for replacement.

    Returns ``(body, stub_count, list_of_refs)``.
    """
    if fmt.name == "gemini":
        _asst_role = "model"
    else:
        _asst_role = "assistant"

    messages = fmt.get_messages(body)
    if not messages:
        return body, 0, []

    # ------------------------------------------------------------------
    # 1. Group into turns using format-aware turn detection
    # ------------------------------------------------------------------
    turns = fmt.group_into_turns(body)
    if not turns:
        return body, 0, []

    total_turns = len(turns)

    # ------------------------------------------------------------------
    # 2. Identify protected window (last N turns)
    # ------------------------------------------------------------------
    protected_start = max(0, total_turns - protected_recent_turns)
    hard_protected_start = max(0, total_turns - 2)

    # Conditional intrusion: if protected zone exceeds a percentage of the
    # context budget, allow stubbing into the protected zone except for
    # the last 2 turns (the current exchange).
    _intrusion_threshold = kwargs.get("protected_intrusion_threshold", 0.0)
    _context_budget = kwargs.get("context_budget", 0)
    intrusion_active = False

    if _intrusion_threshold > 0 and _context_budget > 0 and (total_turns - protected_start) > 2:
        # Estimate protected zone token size
        _prot_bytes = 0
        for ti in range(protected_start, total_turns):
            for idx in turns[ti].indices:
                _prot_bytes += len(json.dumps(messages[idx], default=str))
        _prot_tokens = _prot_bytes // 4
        _prot_ratio = _prot_tokens / _context_budget if _context_budget else 0

        if _prot_ratio > _intrusion_threshold:
            intrusion_active = True
            logger.info(
                "PROTECTED_INTRUSION: protected zone %dt is %.0f%% of budget %dt "
                "(threshold %.0f%%) — stubbing turns 3+ in protected window",
                _prot_tokens, _prot_ratio * 100, _context_budget,
                _intrusion_threshold * 100,
            )

    # Build message-index → turn-index map
    idx_to_turn: dict[int, int] = {}
    for ti, turn in enumerate(turns):
        for idx in turn.indices:
            idx_to_turn[idx] = ti

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
    # 4. Resolve canonical turn numbers per turn via hash lookup
    # ------------------------------------------------------------------
    turn_canonical: dict[int, int] = {}
    for ti, turn in enumerate(turns):
        user_text = ""
        asst_text = ""
        for idx in turn.indices:
            msg = messages[idx]
            if msg.get("role") in ("user", "human") and not _is_tool_result_only_user(msg):
                user_text = _extract_text_for_stub_hash(msg)
            elif msg.get("role") == _asst_role and not asst_text:
                asst_text = _extract_text_for_stub_hash(msg)
        if user_text:
            combined = f"{user_text} {asst_text}"
            h = hashlib.sha256(combined.encode()).hexdigest()[:16]
            entry = turn_tag_index.get_entry_by_hash(h)
            if entry is not None:
                turn_canonical[ti] = entry.turn_number
                continue
        turn_canonical[ti] = -1

    # ------------------------------------------------------------------
    # 5. Stub tool outputs outside protected window
    # ------------------------------------------------------------------
    from ..core.tool_loop import VC_TOOL_NAMES

    stub_count = 0
    stub_refs: list[str] = []

    for output in fmt.iter_tool_outputs(body):
        turn_idx = idx_to_turn.get(output.msg_index)

        # Messages not in any turn group are structural — skip
        if turn_idx is None:
            continue

        # Last 2 turns — never stub (hard-protected)
        if turn_idx >= hard_protected_start:
            continue

        # Turns in protected window — only stub if intrusion is active
        if turn_idx >= protected_start and not intrusion_active:
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
        canonical_turn = turn_canonical.get(turn_idx, -1)

        # Summarise arguments for the stub
        args_summary = _summarise_arguments(call_info.get("arguments"))

        # Store the full content (skip if store is None — e.g. in tests)
        if store is not None:
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
            if canonical_turn >= 0:
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

        # Replace content in-place using format method
        fmt.replace_tool_output_content(body, output, stub_text)

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
    """Extract deduplicated tool call descriptions from a turn chain.

    Collects all tool calls, deduplicates by ``name(args)`` string, and
    returns compact descriptions with repeat counts.  e.g.::

        ["Read(/root/.openclaw/memory/2026-03-25.md)",
         "Read(/root/.openclaw/memory/2026-03-26.md)",
         "session_status (x20)",
         "message (x3)"]
    """
    from ..core.tool_loop import VC_TOOL_NAMES

    # Collect raw descriptions (with duplicates)
    raw: list[str] = []
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
                    raw.append(f"{name}({args})" if args else name)
        # OpenAI Chat tool_calls
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
                raw.append(f"{name}({args})" if args else name)
        # OpenAI Responses bare function_call items
        if msg.get("type") == "function_call" and "call_id" in msg:
            call_id = msg["call_id"]
            if call_id not in seen_ids:
                seen_ids.add(call_id)
                name = msg.get("name", "")
                if name not in VC_TOOL_NAMES:
                    args = _summarise_arguments(msg.get("arguments"), max_len=60)
                    raw.append(f"{name}({args})" if args else name)
        # Gemini functionCall parts
        parts = msg.get("parts", [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and "functionCall" in part:
                    fc = part["functionCall"]
                    name = fc.get("name", "")
                    if name in VC_TOOL_NAMES:
                        continue
                    args = _summarise_arguments(fc.get("args"), max_len=60)
                    raw.append(f"{name}({args})" if args else name)

    # Deduplicate: preserve first-seen order, count repeats
    from collections import Counter
    counts = Counter(raw)
    seen_descs: set[str] = set()
    result: list[str] = []
    for desc in raw:
        if desc in seen_descs:
            continue
        seen_descs.add(desc)
        count = counts[desc]
        if count > 1:
            result.append(f"{desc} (x{count})")
        else:
            result.append(desc)
    return result


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
        # Found the matching user message — build the chain.
        # For OpenAI Responses, bare items (function_call, reasoning) can
        # appear between the user message and the first assistant message.
        # Scan forward past them to find the assistant.
        chain: list[int] = [i]
        j = i + 1
        while j < len(pre_filter_messages):
            _next = pre_filter_messages[j]
            _next_role = _next.get("role", "")
            _next_type = _next.get("type", "")
            if _next_role == _asst_role:
                chain.append(j)
                j += 1
                break
            # Accept bare Responses items (function_call, function_call_output,
            # reasoning) between user and first assistant
            if _next_type in ("function_call", "function_call_output", "reasoning"):
                chain.append(j)
                j += 1
                continue
            # Anything else (user, developer, system) means no assistant follows
            break
        else:
            continue  # no assistant found
        if len(chain) < 2:
            continue  # no assistant was appended
        # j is now past the first assistant — continue looking for tool rounds
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
    protected_recent_turns: int,
    turn_tag_index: TurnTagIndex,
    store: ContextStore | None,
    conversation_id: str,
    pre_filter_body: dict | None = None,
    deep_compaction_ratio: float = 0.5,
    client_truncated: bool = False,
) -> tuple[dict, int, list[str], int]:
    """Collapse turn chains outside protected window to stub pairs (stage 2).

    Each turn chain outside the protected window is replaced with a compact
    stub pair (user + assistant) that includes tool names and a restore ref.
    The original messages (including thinking blocks) are captured from
    *pre_filter_body* and stored as a chain snapshot for later restore.
    Individual tool outputs are also stored for FTS search.

    **Deep compaction:** Turns whose canonical turn number is below
    ``compacted_through * deep_compaction_ratio`` are dropped entirely
    (no stub pair inserted).  Their tool outputs are already linked to
    segments and recoverable via ``vc_find_quote`` / ``vc_expand_topic``.
    Set *deep_compaction_ratio* to 0 to disable (keep all stubs).

    Uses ``fmt.group_into_turns()`` for turn boundary detection and format
    mutation methods for list manipulation, making this work across all
    provider formats (Anthropic, OpenAI Chat, OpenAI Responses, Gemini).

    Returns ``(modified_body, collapse_count, list_of_chain_refs, recovered_count)``.
    """
    if pre_filter_body is None:
        pre_filter_body = body

    _recovered_stubs: list[tuple[int, dict, dict]] = []
    _recovered_count = 0

    if fmt.name == "gemini":
        _msg_key = "contents"
        _asst_role = "model"
    elif fmt.name == "openai_responses":
        _msg_key = "input"
        _asst_role = "assistant"
    else:
        _msg_key = "messages"
        _asst_role = "assistant"

    messages = fmt.get_messages(body)
    if not messages:
        return body, 0, [], 0

    # ------------------------------------------------------------------
    # 1. Identify turn groups via format abstraction
    # ------------------------------------------------------------------
    turn_groups = fmt.group_into_turns(body)
    if not turn_groups:
        return body, 0, [], 0

    # Separate trailing user-only group (current question) from history turns.
    # A trailing group is one whose last item is a user message (no assistant
    # response yet).
    last_group = turn_groups[-1]
    last_idx = last_group.indices[-1] if last_group.indices else -1
    last_msg = messages[last_idx] if 0 <= last_idx < len(messages) else {}
    if last_msg.get("role") in ("user", "human"):
        # Check if this group is ONLY user messages (trailing question)
        all_user = all(
            messages[i].get("role") in ("user", "human")
            for i in last_group.indices
            if 0 <= i < len(messages)
        )
        if all_user:
            history_turns = turn_groups[:-1]
        else:
            history_turns = turn_groups
    else:
        history_turns = turn_groups

    if not history_turns:
        return body, 0, [], 0

    # ------------------------------------------------------------------
    # 2. Identify protected window (last N turn groups)
    # ------------------------------------------------------------------
    total_turns = len(history_turns)
    protected = min(protected_recent_turns, total_turns)
    protected_start = total_turns - protected

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
    turn_canonical: dict[int, int] = {}
    turn_user_text: dict[int, str] = {}
    for tidx, turn in enumerate(history_turns):
        user_text = ""
        asst_text = ""
        for gi in turn.indices:
            msg = messages[gi]
            if msg.get("role") in ("user", "human") and not _is_tool_result_only_user(msg):
                user_text = _extract_text_for_stub_hash(msg)
            elif msg.get("role") == _asst_role and not asst_text:
                asst_text = _extract_text_for_stub_hash(msg)
        turn_user_text[tidx] = user_text
        if user_text:
            combined = f"{user_text} {asst_text}"
            h = hashlib.sha256(combined.encode()).hexdigest()[:16]
            entry = turn_tag_index.get_entry_by_hash(h)
            if entry is not None:
                turn_canonical[tidx] = entry.turn_number
                continue
        turn_canonical[tidx] = -1

    # ------------------------------------------------------------------
    # 5. Prepare pre_filter messages for snapshot extraction
    # ------------------------------------------------------------------
    pre_filter_messages = pre_filter_body.get(_msg_key, [])

    # ------------------------------------------------------------------
    # 6. Collapse turns outside the protected window
    # ------------------------------------------------------------------
    from ..core.tool_loop import VC_TOOL_NAMES

    collapse_count = 0
    _pf_used: set[int] = set()
    chain_refs: list[str] = []
    # Map: turn_idx -> (ref, stub_user, stub_asst) for turns to collapse
    collapse_map: dict[int, tuple[str, dict, dict]] = {}
    # Set of turn indices to deep-drop (remove without stub)
    deep_drop_set: set[int] = set()

    # Compute deep compaction threshold from canonical turn numbers.
    # Turns below this threshold are dropped entirely (no stub).
    _max_canonical = max((v for v in turn_canonical.values() if v >= 0), default=0)
    _deep_threshold = int(_max_canonical * deep_compaction_ratio) if deep_compaction_ratio > 0 else 0

    _skip_no_tool = 0
    _skip_no_user_text = 0
    _skip_no_pf_match = 0
    _skip_no_canonical = 0
    _deep_dropped = 0

    logger.info(
        "CHAIN-COLLAPSE: analyzing %d turns (%d protected, %d collapsible) format=%s deep_threshold=%d",
        total_turns, protected, protected_start, fmt.name, _deep_threshold,
    )

    for tidx in range(protected_start):
        turn = history_turns[tidx]
        canonical_turn = turn_canonical.get(tidx, -1)

        # Skip turns without tool activity -- handled by drop_compacted_turns
        if not turn.has_tool_activity:
            _skip_no_tool += 1
            continue

        user_text = turn_user_text.get(tidx, "")
        if not user_text:
            _skip_no_user_text += 1
            continue

        # a. Find corresponding messages in pre_filter_body
        pf_chain = _find_pre_filter_chain(pre_filter_messages, user_text, _asst_role, used_indices=_pf_used)
        if pf_chain is None:
            _skip_no_pf_match += 1
            if _skip_no_pf_match <= 3:
                logger.info(
                    "CHAIN-COLLAPSE: no pre-filter match turn %d (canonical %d) user_text=%s",
                    tidx, canonical_turn, user_text[:80],
                )
            continue

        _pf_used.update(pf_chain)

        # b. Serialize chain snapshot from pre-filter body
        pf_msgs = [pre_filter_messages[pi] for pi in pf_chain]
        chain_json = json.dumps(pf_msgs, default=str)

        # c. Generate composite ref
        ref = f"chain_{canonical_turn}_{hashlib.sha256(chain_json.encode()).hexdigest()[:12]}"

        # d. Store individual tool outputs (only when store is available)
        tool_output_refs_for_chain: list[str] = []
        if store is not None:
            for output in fmt.iter_tool_outputs(body):
                if output.msg_index not in turn.indices:
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

        # e. Store chain snapshot (only when store is available)
        if store is not None:
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
        tool_descs = _extract_tool_metadata_from_chain(messages, turn.indices, tool_call_map)
        tool_str = ", ".join(tool_descs) if tool_descs else ""

        # g. Build stub pair (format-aware content structure)
        turn_label = canonical_turn if canonical_turn >= 0 else tidx
        _fname = fmt.name

        desc_parts = [f"Compacted turn {turn_label}"]
        if canonical_turn >= 0:
            entry = turn_tag_index.get_tags_for_turn(canonical_turn)
            if entry and entry.tags:
                desc_parts.append(f"topics={', '.join(entry.tags[:5])}")
        if tool_str:
            desc_parts.append(tool_str)
        desc_line = " | ".join(desc_parts)
        stub_text = (
            f"[{desc_line}.\n"
            f'To restore and uncompact full tool call results in place: '
            f'{{"type": "tool_use", "name": "vc_restore_tool", '
            f'"input": {{"ref": "{ref}"}}}}]'
        )

        if _fname == "gemini":
            stub_user = {
                "role": "user",
                "parts": [{"text": f"[Compacted turn {turn_label}]"}],
            }
            stub_asst = {
                "role": "model",
                "parts": [{"text": stub_text}],
            }
        elif _fname == "openai_responses":
            stub_user = {
                "role": "user",
                "content": f"[Compacted turn {turn_label}]",
            }
            stub_asst = {
                "role": "assistant",
                "content": [{"type": "output_text", "text": stub_text}],
            }
        else:
            # Anthropic / OpenAI Chat
            stub_user = {
                "role": "user",
                "content": f"[Compacted turn {turn_label}]",
            }
            stub_asst = {
                "role": _asst_role,
                "content": [{"type": "text", "text": stub_text}],
            }

        # Deep compaction: if this turn is well below the compaction frontier,
        # drop it entirely instead of inserting a stub.  Segment linkage
        # already provides recovery via vc_find_quote / vc_expand_topic.
        if _deep_threshold > 0 and canonical_turn >= 0 and canonical_turn < _deep_threshold:
            deep_drop_set.add(tidx)
            chain_refs.append(ref)
            collapse_count += 1
            _deep_dropped += 1
            continue

        fmt.mark_as_vc_stub(stub_user)
        fmt.mark_as_vc_stub(stub_asst)

        collapse_map[tidx] = (ref, stub_user, stub_asst)
        chain_refs.append(ref)
        collapse_count += 1

    logger.info(
        "CHAIN-COLLAPSE: results — stubbed=%d deep_dropped=%d skip_no_tool=%d skip_no_user_text=%d "
        "skip_no_pf_match=%d skip_no_canonical=%d",
        len(collapse_map), _deep_dropped, _skip_no_tool, _skip_no_user_text,
        _skip_no_pf_match, _skip_no_canonical,
    )

    # ------------------------------------------------------------------
    # 6b. Store-backed chain recovery (when client truncated)
    # ------------------------------------------------------------------
    if client_truncated and store is not None:
        _tti_entries = turn_tag_index.entries
        _canonical_max = _tti_entries[-1].turn_number if _tti_entries else 0
        _recovery_deep = int(_canonical_max * deep_compaction_ratio) if deep_compaction_ratio > 0 else 0
        _protected_canonical = _canonical_max - protected_recent_turns + 1

        stored_snapshots = store.get_chain_snapshots_for_conversation(
            conversation_id, min_turn=_recovery_deep,
        )
        _existing_refs = set(chain_refs)

        for snap in stored_snapshots:
            snap_turn = snap["turn_number"]
            snap_ref = snap["ref"]
            if snap_ref in _existing_refs:
                continue
            if snap_turn >= _protected_canonical:
                continue

            tool_names_str = ""
            raw_refs = [r.strip() for r in snap.get("tool_output_refs", "").split(",") if r.strip()]
            if raw_refs:
                try:
                    names = store.get_tool_names_for_refs(raw_refs)
                    tool_names_str = ", ".join(names) if names else "tools used"
                except Exception:
                    tool_names_str = "tools used"

            desc_parts = [f"Compacted turn {snap_turn}"]
            if tool_names_str:
                desc_parts.append(tool_names_str)
            desc_line = " | ".join(desc_parts)
            stub_text = (
                f"[{desc_line}.\n"
                f'To restore and uncompact full tool call results in place: '
                f'{{"type": "tool_use", "name": "vc_restore_tool", '
                f'"input": {{"ref": "{snap_ref}"}}}}]'
            )

            _fname = fmt.name
            if _fname == "gemini":
                stub_user = {"role": "user", "parts": [{"text": f"[Compacted turn {snap_turn}]"}]}
                stub_asst = {"role": "model", "parts": [{"text": stub_text}]}
            elif _fname == "openai_responses":
                stub_user = {"role": "user", "content": f"[Compacted turn {snap_turn}]"}
                stub_asst = {"role": "assistant", "content": [{"type": "output_text", "text": stub_text}]}
            else:
                stub_user = {"role": "user", "content": f"[Compacted turn {snap_turn}]"}
                stub_asst = {"role": _asst_role, "content": [{"type": "text", "text": stub_text}]}

            fmt.mark_as_vc_stub(stub_user)
            fmt.mark_as_vc_stub(stub_asst)
            _recovered_stubs.append((snap_turn, stub_user, stub_asst))
            chain_refs.append(snap_ref)
            collapse_count += 1

        if _recovered_stubs:
            logger.info(
                "STORE-RECOVERY: recovered %d chain stubs from store (deep_threshold=%d, protected=%d)",
                len(_recovered_stubs), _recovery_deep, _protected_canonical,
            )
        _recovered_count = len(_recovered_stubs)

    if not collapse_map and not deep_drop_set and not _recovered_stubs:
        return body, 0, [], 0

    # ------------------------------------------------------------------
    # 7. Apply mutations: remove collapsed items, insert stubs, clean orphans
    # ------------------------------------------------------------------
    # Collect all indices to remove and tool_use IDs for orphan cleanup.
    # Includes both stubbed turns (collapse_map) and deep-dropped turns.
    collapsed_indices: set[int] = set()
    collapsed_tool_use_ids: set[str] = set()
    _all_collapsed_tidxs = set(collapse_map.keys()) | deep_drop_set
    for tidx in _all_collapsed_tidxs:
        turn = history_turns[tidx]
        for gi in turn.indices:
            collapsed_indices.add(gi)
            msg = messages[gi]
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        _tid = block.get("id")
                        if _tid:
                            collapsed_tool_use_ids.add(_tid)
            # Also collect call_ids from OpenAI Responses bare function_call items
            if msg.get("type") == "function_call":
                _cid = msg.get("call_id")
                if _cid:
                    collapsed_tool_use_ids.add(_cid)
            # And from OpenAI Chat tool_calls arrays
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    _tid2 = tc.get("id")
                    if _tid2:
                        collapsed_tool_use_ids.add(_tid2)

    # Build ordered list of (first_index_of_turn, stub_user, stub_asst)
    # sorted by position so we can apply removals and insertions correctly.
    insertions: list[tuple[int, dict, dict]] = []
    for tidx, (ref, stub_user, stub_asst) in collapse_map.items():
        turn = history_turns[tidx]
        insertions.append((turn.indices[0], stub_user, stub_asst))
    insertions.sort(key=lambda x: x[0])

    # Strip orphaned tool_result / function_call_output blocks from
    # non-collapsed messages before removing collapsed indices.
    for mi in range(len(messages)):
        if mi in collapsed_indices:
            continue
        msg = messages[mi]
        if not isinstance(msg, dict) or not collapsed_tool_use_ids:
            continue
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
                msg["content"] = cleaned if cleaned else [{"type": "text", "text": "[tool results removed — parent tool call was compacted]"}]
        # OpenAI Responses: function_call_output items outside collapsed
        # chains but referencing collapsed call_ids
        if msg.get("type") == "function_call_output" and msg.get("call_id") in collapsed_tool_use_ids:
            msg["output"] = "[tool results removed — parent tool call was compacted]"

    # Remove collapsed items (highest index first to preserve positions)
    fmt.remove_items(body, sorted(collapsed_indices))

    # Insert stub pairs. After removal, indices shift, so we track the offset.
    # For each insertion point, count how many collapsed indices were below it
    # and adjust accordingly.
    offset = 0
    for orig_pos, stub_user, stub_asst in insertions:
        # How many collapsed items were before (or at) this position?
        removed_before = sum(1 for ci in collapsed_indices if ci < orig_pos)
        # How many stub pairs were already inserted before this position?
        adjusted_pos = orig_pos - removed_before + offset
        fmt.insert_items(body, adjusted_pos, [stub_user, stub_asst])
        offset += 2  # each stub pair adds 2 items

    # Insert recovered stubs after system/developer prefix
    if _recovered_stubs:
        _recovered_stubs.sort(key=lambda x: x[0])
        messages = fmt.get_messages(body)
        insert_at = 0
        for i, m in enumerate(messages):
            if m.get("role") in ("system", "developer"):
                insert_at = i + 1
            else:
                break
        recovery_items = []
        for _, stub_user, stub_asst in _recovered_stubs:
            recovery_items.extend([stub_user, stub_asst])
        fmt.insert_items(body, insert_at, recovery_items)

    return body, collapse_count, chain_refs, _recovered_count


@dataclass
class ReducibleItem:
    """An item in the payload that can be reduced to save tokens."""
    msg_index: int          # index into the messages list
    block_index: int        # index into message content blocks (-1 for string content)
    category: str           # thinking_sig, tool_result, tool_result_last2, vc_context, conversation_text, image
    size_bytes: int         # current size in bytes
    location: str           # "messages" or "system" — where the item lives


def scan_reducible_items(
    body: dict,
    fmt: PayloadFormat,
) -> list[ReducibleItem]:
    """Scan the payload and return all items eligible for reduction.

    Uses format methods for cross-format compatibility:
    - ``fmt.group_into_turns()`` for the protection window
    - ``fmt.iter_tool_outputs()`` for tool results
    - ``fmt.iter_media_blocks()`` for images
    - ``fmt.extract_text_from_item()`` for conversational text
    - Anthropic-specific thinking block detection
    """
    items: list[ReducibleItem] = []

    messages = fmt.get_messages(body)
    if not messages:
        return items

    # Use turn groups to find the protection window (last 2 real turns).
    turns = fmt.group_into_turns(body)
    if len(turns) >= 3:
        _last2_start = turns[-2].indices[0]
    else:
        # With 1-2 turns everything is recent — protect all of it.
        _last2_start = 0

    # Track which msg_index+block_index pairs are already registered to
    # avoid duplicate entries (e.g. a tool_result also matching text scan).
    _seen: set[tuple[int, int]] = set()

    # -- VC context blocks in system prompt (Anthropic-specific) --------
    system = body.get("system", [])
    if isinstance(system, list):
        for si, block in enumerate(system):
            if not isinstance(block, dict):
                continue
            text = block.get("text", "")
            if "<context-topics" in text or ("system-reminder" in text and "context-topics" in text):
                items.append(ReducibleItem(
                    msg_index=-1, block_index=si, category="vc_context",
                    size_bytes=len(text), location="system",
                ))

    # -- Tool outputs (via format method) -------------------------------
    for tool_out in fmt.iter_tool_outputs(body):
        content_str = tool_out.content
        tc_bytes = len(content_str)
        if tc_bytes < 100:
            continue
        if "vc_restore_tool" in content_str:
            continue
        # For Anthropic, block_index is the position within content list.
        # For OpenAI Chat/Responses, block_index is -1 (whole message).
        if tool_out.carrier_type == "anthropic":
            # Find the block_index within the content list
            msg = messages[tool_out.msg_index]
            content_list = msg.get("content", [])
            bi = -1
            if isinstance(content_list, list):
                for _bi, _blk in enumerate(content_list):
                    if _blk is tool_out.carrier:
                        bi = _bi
                        break
        else:
            bi = -1
        category = "tool_result_last2" if tool_out.msg_index >= _last2_start else "tool_result"
        items.append(ReducibleItem(
            msg_index=tool_out.msg_index, block_index=bi, category=category,
            size_bytes=tc_bytes, location="messages",
        ))
        _seen.add((tool_out.msg_index, bi))

    # -- Media blocks (via format method) -------------------------------
    for media_info in fmt.iter_media_blocks(body):
        # Estimate data size from the source/URL — check all known shapes.
        msg = messages[media_info.msg_index]
        # Gemini stores blocks in "parts"; others use "content".
        content = msg.get("parts") or msg.get("content", [])
        data_size = 0
        if isinstance(content, list) and media_info.block_index < len(content):
            block = content[media_info.block_index]
            if isinstance(block, dict):
                # Anthropic: {"source": {"type": "base64", "data": "..."}}
                source = block.get("source", {})
                if isinstance(source, dict) and source.get("type") == "base64":
                    data_size = len(source.get("data", ""))
                # OpenAI Chat: {"type": "image_url", "image_url": {"url": "data:..."}}
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:") and ";base64," in url:
                        data_size = len(url.split(";base64,", 1)[1]) if ";base64," in url else 0
                # OpenAI Responses: {"type": "input_image", "image_url": "data:..."}
                elif block.get("type") == "input_image":
                    url = block.get("image_url", "")
                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                        data_size = len(url.split(";base64,", 1)[1])
                # Gemini: {"inline_data": {"mime_type": "...", "data": "..."}}
                elif "inline_data" in block:
                    inline = block.get("inline_data", {})
                    if isinstance(inline, dict):
                        data_size = len(inline.get("data", ""))
        if data_size > 10000:
            key = (media_info.msg_index, media_info.block_index)
            if key not in _seen:
                items.append(ReducibleItem(
                    msg_index=media_info.msg_index, block_index=media_info.block_index,
                    category="image", size_bytes=data_size, location="messages",
                ))
                _seen.add(key)

    # -- Thinking blocks (Anthropic only) -------------------------------
    if fmt.name == "anthropic":
        for mi, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "thinking":
                    sig = block.get("signature", "")
                    if sig:
                        key = (mi, bi)
                        if key not in _seen:
                            items.append(ReducibleItem(
                                msg_index=mi, block_index=bi, category="thinking_sig",
                                size_bytes=len(sig), location="messages",
                            ))
                            _seen.add(key)

    # -- Conversational text (via format method) ------------------------
    for mi in range(len(messages)):
        text = fmt.extract_text_from_item(body, mi)
        if not text or len(text) <= 100:
            continue
        if "[Compacted turn" in text:
            # Skip compacted stubs — check for vc_restore_tool presence too
            if "vc_restore_tool" in text:
                continue
            # Plain "[Compacted turn" string content → also skip
            content = messages[mi].get("content", "")
            if isinstance(content, str) and "[Compacted turn" in content:
                continue

        # Determine block_index: -1 for string content, specific index for
        # list content with text blocks.
        content = messages[mi].get("content", "")
        if isinstance(content, str):
            key = (mi, -1)
            if key not in _seen:
                items.append(ReducibleItem(
                    msg_index=mi, block_index=-1, category="conversation_text",
                    size_bytes=len(content), location="messages",
                ))
                _seen.add(key)
        elif isinstance(content, list):
            # Check for compacted stub (list content)
            all_text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if "[Compacted turn" in all_text and "vc_restore_tool" in all_text:
                continue
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype in ("text", "output_text"):
                    btext = block.get("text", "")
                    if "[Compacted turn" in btext:
                        continue
                    if len(btext) > 100:
                        key = (mi, bi)
                        if key not in _seen:
                            items.append(ReducibleItem(
                                msg_index=mi, block_index=bi, category="conversation_text",
                                size_bytes=len(btext), location="messages",
                            ))
                            _seen.add(key)
        else:
            # OpenAI Responses bare items with output field
            output = messages[mi].get("output", "")
            if isinstance(output, str) and len(output) > 100:
                key = (mi, -1)
                if key not in _seen:
                    items.append(ReducibleItem(
                        msg_index=mi, block_index=-1, category="conversation_text",
                        size_bytes=len(output), location="messages",
                    ))
                    _seen.add(key)

    return items


def apply_reduction(
    body: dict,
    item: ReducibleItem,
    fmt: PayloadFormat,
    store: "ContextStore | None" = None,
    conversation_id: str = "",
) -> int:
    """Apply a category-appropriate reduction to one item. Returns bytes freed.

    Uses format methods for cross-format compatibility:
    - ``fmt.remove_thinking_block()`` for thinking signatures
    - ``fmt.replace_tool_output_content()`` via ``fmt.iter_tool_outputs()``
    - ``fmt.get_messages()`` for message list access
    """
    if item.category == "thinking_sig":
        return _reduce_thinking_sig(body, item, fmt)
    elif item.category == "tool_result":
        return _reduce_tool_result(body, item, fmt, store, conversation_id)
    elif item.category == "tool_result_last2":
        return _reduce_tool_result_last2(body, item, fmt)
    elif item.category == "vc_context":
        return _reduce_vc_context(body, item)
    elif item.category == "conversation_text":
        return _reduce_conversation_text(body, item, fmt)
    elif item.category == "image":
        return _reduce_image(body, item, fmt)
    return 0


def _reduce_thinking_sig(body: dict, item: ReducibleItem, fmt: PayloadFormat) -> int:
    """Remove a thinking block using fmt.remove_thinking_block()."""
    msgs = fmt.get_messages(body)
    if item.msg_index >= len(msgs):
        return 0
    msg = msgs[item.msg_index]
    content = msg.get("content", [])
    if not isinstance(content, list) or item.block_index >= len(content):
        return 0
    block = content[item.block_index]
    freed = len(block.get("signature", "")) + len(block.get("thinking", ""))
    fmt.remove_thinking_block(body, msg_index=item.msg_index, block_index=item.block_index)
    return freed


def _find_tool_output_info(body, item, fmt):
    """Find the ToolOutputInfo matching a ReducibleItem's msg_index."""
    for tool_out in fmt.iter_tool_outputs(body):
        if tool_out.msg_index != item.msg_index:
            continue
        # For Anthropic, match by block_index (carrier identity)
        if item.block_index >= 0 and tool_out.carrier_type == "anthropic":
            msgs = fmt.get_messages(body)
            msg = msgs[item.msg_index]
            content_list = msg.get("content", [])
            if isinstance(content_list, list) and item.block_index < len(content_list):
                if content_list[item.block_index] is tool_out.carrier:
                    return tool_out
        elif item.block_index == -1:
            # OpenAI Chat / Responses: whole-message carriers
            return tool_out
    return None


def _reduce_tool_result(
    body: dict, item: ReducibleItem, fmt: PayloadFormat,
    store, conversation_id: str,
) -> int:
    """Stub a tool result using fmt.replace_tool_output_content()."""
    tool_out = _find_tool_output_info(body, item, fmt)
    if tool_out is None:
        return 0

    tc_str = tool_out.content
    ref = f"tool_{hashlib.sha256(tc_str.encode()).hexdigest()[:12]}"

    if store is not None:
        try:
            tool_name = ""
            # Find tool name via fmt.iter_tool_calls()
            for tc_info in fmt.iter_tool_calls(body):
                if tc_info.call_id == tool_out.call_id:
                    tool_name = tc_info.name
                    break
            store.store_tool_output(
                ref=ref, conversation_id=conversation_id,
                tool_name=tool_name, command="", turn=-1,
                content=tc_str, original_bytes=len(tc_str.encode("utf-8")),
            )
        except Exception:
            logger.warning("BUDGET_ENFORCE: failed to store ref=%s", ref, exc_info=True)

    stub_text = f'[budget-reduced tool output ref={ref} | call vc_restore_tool(ref="{ref}")]'
    fmt.replace_tool_output_content(body, tool_out, stub_text)
    return item.size_bytes - len(stub_text)


def _reduce_tool_result_last2(body: dict, item: ReducibleItem, fmt: PayloadFormat) -> int:
    """Truncate a recent tool result (head+tail) using fmt.replace_tool_output_content()."""
    tool_out = _find_tool_output_info(body, item, fmt)
    if tool_out is None:
        return 0

    tc_str = tool_out.content
    if len(tc_str) <= 500:
        return 0
    head = tc_str[:200]
    tail = tc_str[-200:]
    truncated = f"{head}\n\n[... {len(tc_str) - 400} chars truncated ...]\n\n{tail}"
    fmt.replace_tool_output_content(body, tool_out, truncated)
    return len(tc_str) - len(truncated)


def _reduce_vc_context(body: dict, item: ReducibleItem) -> int:
    system = body.get("system", [])
    if not isinstance(system, list) or item.block_index >= len(system):
        return 0
    block = system[item.block_index]
    text = block.get("text", "")
    if len(text) <= 500:
        return 0
    cut_point = len(text) // 2
    truncated = text[:cut_point] + "\n[... VC context truncated to fit budget ...]"
    block["text"] = truncated
    return len(text) - len(truncated)


def _reduce_conversation_text(body: dict, item: ReducibleItem, fmt: PayloadFormat) -> int:
    msgs = fmt.get_messages(body)
    msg = msgs[item.msg_index]

    if item.block_index == -1:
        text = msg.get("content", "")
        if not isinstance(text, str) or len(text) <= 500:
            return 0
        head = text[:200]
        tail = text[-200:]
        truncated = f"{head}\n\n[... {len(text) - 400} chars truncated ...]\n\n{tail}"
        msg["content"] = truncated
        return len(text) - len(truncated)
    else:
        content = msg.get("content", [])
        if not isinstance(content, list) or item.block_index >= len(content):
            return 0
        block = content[item.block_index]
        text = block.get("text", "")
        if len(text) <= 500:
            return 0
        head = text[:200]
        tail = text[-200:]
        truncated = f"{head}\n\n[... {len(text) - 400} chars truncated ...]\n\n{tail}"
        block["text"] = truncated
        return len(text) - len(truncated)


def _reduce_image(body: dict, item: ReducibleItem, fmt: PayloadFormat) -> int:
    msgs = fmt.get_messages(body)
    # Skip if likely already compressed by the media pipeline.
    # Check: small size AND already JPEG media_type.
    if item.size_bytes < 200000:
        if item.msg_index < len(msgs):
            content = msgs[item.msg_index].get("content", [])
            if isinstance(content, list) and item.block_index < len(content):
                block = content[item.block_index]
                source = block.get("source", {})
                if isinstance(source, dict) and "jpeg" in source.get("media_type", "").lower():
                    return 0
    import base64
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        logger.warning("BUDGET_ENFORCE: Pillow not installed, cannot compress image")
        return 0

    MAX_WIDTH = 1024
    MAX_HEIGHT = 1024
    JPEG_QUALITY = 75

    msg = msgs[item.msg_index]
    content = msg.get("content", [])
    if not isinstance(content, list) or item.block_index >= len(content):
        return 0
    block = content[item.block_index]
    source = block.get("source", {})
    if not isinstance(source, dict) or source.get("type") != "base64":
        return 0

    original_data = source.get("data", "")
    original_bytes = len(original_data)

    try:
        img_bytes = base64.b64decode(original_data)
        img = Image.open(BytesIO(img_bytes))
        if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
            img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Try JPEG at decreasing quality until smaller than original
        best_b64 = None
        for quality in (JPEG_QUALITY, 50, 30):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            candidate = base64.b64encode(buf.getvalue()).decode("ascii")
            if len(candidate) < original_bytes:
                best_b64 = candidate
                break

        # Fall back to resized PNG if JPEG doesn't help
        if best_b64 is None:
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            candidate = base64.b64encode(buf.getvalue()).decode("ascii")
            if len(candidate) < original_bytes:
                best_b64 = candidate
                source["data"] = best_b64
                source["media_type"] = "image/png"
                return original_bytes - len(best_b64)
            return 0

        source["data"] = best_b64
        source["media_type"] = "image/jpeg"
        return original_bytes - len(best_b64)
    except Exception as e:
        logger.warning("BUDGET_ENFORCE: image compression failed: %s", e)
        return 0


def enforce_payload_budget(
    body: dict,
    fmt: PayloadFormat,
    context_window: int,
    store: "ContextStore | None" = None,
    conversation_id: str = "",
) -> tuple[dict, int, int]:
    """Enforce the VC context window budget on the assembled payload.

    Iteratively reduces the largest reducible item until the payload fits.
    Only meaningful when outbound_tokens > context_window.

    Returns (modified_body, reductions_applied, total_bytes_freed).
    """
    _MAX_ITERATIONS = 200

    outbound_tokens = fmt.estimate_payload_tokens(body)
    if outbound_tokens <= context_window:
        return body, 0, 0

    logger.info(
        "BUDGET_ENFORCE: payload %dt exceeds budget %dt — starting reduction loop",
        outbound_tokens, context_window,
    )

    total_freed = 0
    reductions = 0

    for iteration in range(_MAX_ITERATIONS):
        if outbound_tokens <= context_window:
            break

        items = scan_reducible_items(body, fmt)
        if not items:
            logger.warning(
                "BUDGET_ENFORCE: no reducible items left, still over budget (%dt > %dt)",
                outbound_tokens, context_window,
            )
            break

        # Try items from largest to smallest — skip any that can't be reduced
        items.sort(key=lambda x: x.size_bytes, reverse=True)
        freed = 0
        largest = None
        for candidate in items:
            freed = apply_reduction(body, candidate, fmt, store=store, conversation_id=conversation_id)
            if freed > 0:
                largest = candidate
                break
            logger.debug(
                "BUDGET_ENFORCE: skipping %s (%d bytes at msg %d) — not reducible",
                candidate.category, candidate.size_bytes, candidate.msg_index,
            )

        if freed <= 0 or largest is None:
            logger.info(
                "BUDGET_ENFORCE: no items could be reduced — stopping",
            )
            break

        reductions += 1
        total_freed += freed
        outbound_tokens = fmt.estimate_payload_tokens(body)

        logger.info(
            "BUDGET_ENFORCE: [%d] cut %s at msg %d (%d bytes freed) — now %dt/%dt",
            iteration + 1, largest.category, largest.msg_index, freed,
            outbound_tokens, context_window,
        )

    if outbound_tokens > context_window:
        logger.warning(
            "BUDGET_ENFORCE: exhausted after %d reductions (%d bytes freed) — "
            "still %dt > %dt, deferring to bloat fallback",
            reductions, total_freed, outbound_tokens, context_window,
        )
    else:
        logger.info(
            "BUDGET_ENFORCE: done — %d reductions, %d bytes freed, final %dt/%dt",
            reductions, total_freed, outbound_tokens, context_window,
        )

    return body, reductions, total_freed


def fill_pass(
    body: dict,
    fmt: PayloadFormat,
    outbound_tokens: int,
    target_tokens: int,
    assembled: "AssembledContext | None",
    pre_filter_body: dict,
    store: "ContextStore | None",
    conversation_id: str,
    summary_ratio: float = 0.60,
    client_truncated: bool = False,
    turn_tag_index: "TurnTagIndex | None" = None,
) -> tuple[dict, int, int]:
    """Fill payload from VC floor up to target threshold.

    The caller clamps *target_tokens* before calling:
        target = min(soft_threshold_tokens, inbound_tokens, upstream_limit - max_tokens)

    If *assembled* is None or has no retrieval_result, phase 1a (overflow)
    is skipped. Phase 1b (breadth) and phase 2 (turns) still run.

    Returns (modified_body, summaries_added, turns_added).
    """
    headroom = target_tokens - outbound_tokens
    if headroom <= 0:
        return body, 0, 0

    presented_refs: set[str] = set(assembled.presented_segment_refs) if assembled else set()
    covered_tags: set[str] = set(assembled.presented_tags) if assembled else set()
    tokens_used = 0
    summaries_added = 0

    # Phase 1: Tag summaries (up to summary_ratio of headroom)
    summary_budget = int(headroom * summary_ratio)

    # 1a. Overflow candidates (relevant but budget-excluded, new tags only)
    if (
        assembled
        and getattr(assembled, "retrieval_result", None)
        and getattr(assembled.retrieval_result, "overflow_summaries", None)
    ):
        from ..core.assembler import format_tag_section

        for summary in assembled.retrieval_result.overflow_summaries:
            if summary.primary_tag in covered_tags:
                continue
            if summary.ref in presented_refs:
                continue
            # Don't pass store — tool hint enrichment would add vc_restore_tool
            # references, but the fill pass runs after tool injection decisions.
            text = format_tag_section(summary.primary_tag, [summary])
            tokens = len(text) // 4  # rough estimate
            if tokens_used + tokens > summary_budget:
                continue
            body = _append_to_context(body, fmt, text)
            tokens_used += tokens
            summaries_added += 1
            presented_refs.add(summary.ref)
            covered_tags.add(summary.primary_tag)
            covered_tags.update(summary.tags)

    # 1b. Breadth summaries (remaining conversation tag summaries, recency-sorted)
    if store is not None and tokens_used < summary_budget:
        all_tag_summaries = store.get_all_tag_summaries(conversation_id=conversation_id)
        for ts in sorted(all_tag_summaries, key=lambda s: s.updated_at, reverse=True):
            if ts.tag in covered_tags:
                continue
            source_refs = set(ts.source_segment_refs)
            if source_refs and source_refs <= presented_refs:
                continue
            text = _format_breadth_section(ts)
            tokens = len(text) // 4
            if tokens_used + tokens > summary_budget:
                continue
            body = _append_to_context(body, fmt, text)
            tokens_used += tokens
            summaries_added += 1
            presented_refs.update(source_refs)
            covered_tags.add(ts.tag)

    # Phase 2: Recent turns (remaining headroom)
    turn_budget = headroom - tokens_used
    turns_added = 0

    if turn_budget > 200:
        if client_truncated and store is not None:
            # Store-backed: restore from turn_messages (unpruned suffix)
            import hashlib as _hl
            store_turns = store.load_recent_turn_messages(conversation_id, limit=200)

            # Build set of canonical turn numbers already in the payload
            # using hash lookup (same approach as collapse_turn_chains)
            _payload_canonical_turns: set[int] = set()
            if turn_tag_index is not None:
                for g in fmt.group_into_turns(body):
                    msgs = fmt.get_messages(body)
                    user_text_hash = ""
                    asst_text_hash = ""
                    for gi in g.indices:
                        msg = msgs[gi]
                        if msg.get("role") in ("user", "human"):
                            user_text_hash = _extract_text_for_stub_hash(msg)
                        elif msg.get("role") in ("assistant", "model"):
                            if not asst_text_hash:
                                asst_text_hash = _extract_text_for_stub_hash(msg)
                    if user_text_hash:
                        combined = f"{user_text_hash} {asst_text_hash}"
                        h = _hl.sha256(combined.encode()).hexdigest()[:16]
                        entry = turn_tag_index.get_entry_by_hash(h)
                        if entry is not None:
                            _payload_canonical_turns.add(entry.turn_number)

            # Restore newest first
            for turn_num, user_text, asst_text in reversed(store_turns):
                if not user_text.strip():
                    continue
                # Skip turns already in the payload
                if turn_num in _payload_canonical_turns:
                    continue
                # Skip tool-bearing turns (chain recovery handles those)
                try:
                    tool_refs = store.get_tool_outputs_for_turn(conversation_id, turn_num)
                    if tool_refs:
                        continue
                except Exception:
                    pass
                # Build format-appropriate message pair
                _fname = fmt.name
                if _fname == "gemini":
                    turn_msgs = [
                        {"role": "user", "parts": [{"text": user_text}]},
                        {"role": "model", "parts": [{"text": asst_text}]},
                    ]
                elif _fname == "openai_responses":
                    turn_msgs = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": [{"type": "output_text", "text": asst_text}]},
                    ]
                else:
                    turn_msgs = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
                    ]
                turn_text = json.dumps(turn_msgs, default=str)
                turn_tokens = len(turn_text) // 4
                if turn_tokens > turn_budget:
                    continue
                # Insert after system/developer prefix
                cur_msgs = fmt.get_messages(body)
                insert_at = 0
                for i, m in enumerate(cur_msgs):
                    if m.get("role") in ("system", "developer"):
                        insert_at = i + 1
                    else:
                        break
                fmt.insert_items(body, insert_at, turn_msgs)
                turn_budget -= turn_tokens
                turns_added += 1

        elif pre_filter_body is not None:
            # Normal path — restore from pre_filter_body snapshot
            pre_turns = fmt.group_into_turns(pre_filter_body)
            cur_turn_count = len(fmt.group_into_turns(body))
            pre_messages = fmt.get_messages(pre_filter_body)

            if len(pre_turns) > cur_turn_count:
                dropped_end = len(pre_turns) - cur_turn_count
                for tidx in range(dropped_end - 1, -1, -1):
                    turn = pre_turns[tidx]
                    if turn.has_tool_activity:
                        continue
                    turn_msgs = [pre_messages[i] for i in turn.indices if 0 <= i < len(pre_messages)]
                    turn_msgs = _sanitize_restored_turn(turn_msgs)
                    turn_text = json.dumps(turn_msgs, default=str)
                    turn_tokens = len(turn_text) // 4
                    if turn_tokens > turn_budget:
                        continue
                    cur_msgs = fmt.get_messages(body)
                    insert_at = 0
                    for i, m in enumerate(cur_msgs):
                        if m.get("role") in ("system", "developer"):
                            insert_at = i + 1
                        else:
                            break
                    fmt.insert_items(body, insert_at, turn_msgs)
                    turn_budget -= turn_tokens
                    turns_added += 1

    if summaries_added or turns_added:
        logger.info(
            "FILL-PASS: added %d summaries (%d tokens) + %d turns, headroom=%d target=%d",
            summaries_added, tokens_used, turns_added, headroom, target_tokens,
        )

    return body, summaries_added, turns_added


def _append_to_context(body: dict, fmt: PayloadFormat, text: str) -> dict:
    """Append text to the existing context injection point.

    Returns the updated body. ``fmt.inject_context()`` returns a new
    deep-copied body and already wraps input in ``<system-reminder>``
    tags, so we pass raw section text (no pre-wrapping).
    """
    return fmt.inject_context(body, text)


def _format_breadth_section(ts) -> str:
    """Render a TagSummary breadth item as a simple tag section."""
    return (
        f'<virtual-context tags="{ts.tag}" type="breadth">\n'
        f"{ts.summary}\n"
        f"</virtual-context>"
    )


def _sanitize_restored_turn(messages: list[dict]) -> list[dict]:
    """Sanitize restored turn messages: strip thinking, media, and tool scaffolding.

    Restored turns may contain tool_result blocks whose matching tool_use
    is in a different (non-restored) turn.  Sending an orphaned tool_result
    causes Anthropic API 400.  Strip all tool scaffolding from restored turns.
    """
    sanitized = []
    for msg in messages:
        msg = dict(msg)  # shallow copy
        content = msg.get("content", "")
        if isinstance(content, list):
            cleaned = []
            for block in content:
                if not isinstance(block, dict):
                    cleaned.append(block)
                    continue
                btype = block.get("type", "")
                # Strip thinking blocks
                if btype == "thinking":
                    continue
                # Strip tool_use and tool_result �� their partners may not be restored
                if btype in ("tool_use", "tool_result"):
                    continue
                # Replace media with passive placeholder
                if btype in ("image", "image_url") or block.get("source", {}).get("type") == "base64":
                    cleaned.append({"type": "text", "text": "[image removed from restored turn]"})
                    continue
                if btype == "input_image":
                    cleaned.append({"type": "input_text", "text": "[image removed from restored turn]"})
                    continue
                cleaned.append(block)
            # If stripping left no content, add a placeholder
            if not cleaned:
                cleaned = [{"type": "text", "text": "[restored turn — tool content removed]"}]
            msg["content"] = cleaned
        # Strip OpenAI tool_calls from assistant messages
        msg.pop("tool_calls", None)
        if msg.get("role") == "tool":
            continue  # skip entire tool-role messages
        parts = msg.get("parts", [])
        if isinstance(parts, list) and parts:
            cleaned_parts = []
            for part in parts:
                if isinstance(part, dict) and "inlineData" in part:
                    cleaned_parts.append({"text": "[image removed from restored turn]"})
                else:
                    cleaned_parts.append(part)
            msg["parts"] = cleaned_parts
        sanitized.append(msg)
    return sanitized
