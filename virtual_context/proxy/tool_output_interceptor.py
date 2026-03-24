"""Tool output interception: truncate large tool outputs + index into FTS5.

Scans all messages/items in the request body for tool outputs across all
supported provider formats:

- **Anthropic**: ``tool_result`` content blocks in user messages
- **OpenAI Chat Completions**: ``role: "tool"`` messages
- **OpenAI Responses**: bare ``function_call_output`` items

Small outputs pass through unchanged.  Large outputs are truncated (head +
tail on line boundaries) and the full content is stored in FTS5 for
find_quote search.

VC tool results (vc_find_quote, vc_expand_topic, etc.) are always skipped to
prevent feedback loops.
"""

from __future__ import annotations

import fnmatch
from uuid import uuid4

from ..core.store import ContextStore
from ..core.tool_loop import VC_TOOL_NAMES
from ..types import ToolOutputConfig, ToolOutputRule, ToolOutputStats
from .formats import PayloadFormat


class ToolOutputInterceptor:
    """Intercepts large tool outputs across all provider formats.

    Uses ``PayloadFormat.iter_tool_calls()`` and
    ``PayloadFormat.iter_tool_outputs()`` to enumerate tool calls and tool
    outputs in a format-agnostic way.  Outputs above the configured threshold
    are truncated and indexed into FTS5 for ``vc_find_quote`` recovery.
    """

    def __init__(
        self,
        config: ToolOutputConfig,
        store: ContextStore,
        conversation_id: str,
    ) -> None:
        self.config = config
        self.store = store
        self.conversation_id = conversation_id
        self.stats = ToolOutputStats()
        self._turn_counter = 0
        self.intercepted_refs: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        body: dict,
        fmt: PayloadFormat,
        vc_tool_ids: frozenset[str] = frozenset(),
    ) -> dict:
        """Intercept large tool outputs across all messages/items.

        Returns the (potentially mutated) body.  Scans all messages for
        tool outputs using the normalized iterators on ``PayloadFormat``,
        covering Anthropic, OpenAI Chat, and OpenAI Responses formats.
        """
        if not self.config.enabled:
            return body

        # Build call_id -> {name, arguments} map from all tool calls
        tool_use_map: dict[str, dict] = {}
        for tc in fmt.iter_tool_calls(body):
            if tc.call_id:
                tool_use_map[tc.call_id] = {
                    "name": tc.name,
                    "input": tc.arguments,
                }

        # Process all tool outputs
        for output in fmt.iter_tool_outputs(body):
            call_id = output.call_id
            if call_id in vc_tool_ids:
                continue
            tool_name = tool_use_map.get(call_id, {}).get("name", "")
            if tool_name in VC_TOOL_NAMES:
                continue

            content_text = output.content
            if not content_text:
                continue
            content_bytes = len(content_text.encode("utf-8"))
            rule = self._match_rule(tool_name)
            threshold = (
                rule.truncate_threshold
                if rule.truncate_threshold is not None
                else self.config.default_truncate_threshold
            )

            if content_bytes < threshold:
                continue  # passthrough

            # Truncate + Index
            head_text, tail_text = self._truncate(
                content_text, threshold, rule.head_ratio, rule.tail_ratio,
            )
            ref = self._index(content_text, tool_name, rule)

            truncated_bytes = (
                len(head_text.encode("utf-8")) + len(tail_text.encode("utf-8"))
            )
            saved_bytes = content_bytes - truncated_bytes
            notice = (
                f"\n... [{saved_bytes} bytes truncated"
                f" — call vc_find_quote(query) to search the full output] ...\n"
            )
            replacement = head_text + notice + tail_text

            # Replace content in-place based on carrier type
            carrier = output.carrier
            carrier_type = output.carrier_type
            if carrier_type == "anthropic":
                self._replace_content(carrier, replacement)
            elif carrier_type == "openai_chat":
                carrier["content"] = replacement
            elif carrier_type == "openai_responses":
                carrier["output"] = replacement

            self._record_stats(
                tool_name, content_bytes,
                len(replacement.encode("utf-8")),
                content_bytes,
            )

            # Record provenance for later ingestion matching
            self.intercepted_refs.append({
                "ref": ref,
                "call_id": call_id,
                "msg_index": output.msg_index,
                "format": carrier_type,
            })

        return body

    def increment_turn(self) -> None:
        self._turn_counter += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_rule(self, tool_name: str) -> ToolOutputRule:
        """Find first matching rule by fnmatch, or return defaults."""
        for rule in self.config.rules:
            if fnmatch.fnmatch(tool_name, rule.match):
                return rule
        return ToolOutputRule(
            head_ratio=self.config.default_head_ratio,
            tail_ratio=self.config.default_tail_ratio,
        )

    @staticmethod
    def _truncate(
        text: str,
        budget: int,
        head_ratio: float,
        tail_ratio: float,
    ) -> tuple[str, str]:
        """Split on line boundaries. Returns (head, tail) within budget."""
        # Normalize ratios
        total = head_ratio + tail_ratio
        if total <= 0:
            head_ratio, tail_ratio = 0.5, 0.5
            total = 1.0
        head_budget = int(budget * (head_ratio / total))
        tail_budget = budget - head_budget

        lines = text.split("\n")

        # Build head
        head_lines: list[str] = []
        head_size = 0
        for line in lines:
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
            if head_size + line_bytes > head_budget and head_lines:
                break
            head_lines.append(line)
            head_size += line_bytes

        # Build tail (from end)
        tail_lines: list[str] = []
        tail_size = 0
        for line in reversed(lines):
            line_bytes = len(line.encode("utf-8")) + 1
            if tail_size + line_bytes > tail_budget and tail_lines:
                break
            tail_lines.append(line)
            tail_size += line_bytes
        tail_lines.reverse()

        return "\n".join(head_lines), "\n".join(tail_lines)

    def _index(
        self,
        content: str,
        tool_name: str,
        rule: ToolOutputRule,
    ) -> str:
        ref = f"tool_{uuid4().hex[:12]}"
        cap = (
            rule.max_index_bytes
            if rule.max_index_bytes is not None
            else self.config.max_index_bytes
        )
        self.store.store_tool_output(
            ref=ref,
            conversation_id=self.conversation_id,
            tool_name=tool_name,
            command="",
            turn=self._turn_counter,
            content=content[:cap],
            original_bytes=len(content.encode("utf-8")),
        )
        return ref

    @staticmethod
    def _replace_content(block: dict, new_text: str) -> None:
        """Replace content in an Anthropic tool_result block."""
        content = block.get("content", "")
        if isinstance(content, str):
            block["content"] = new_text
        elif isinstance(content, list):
            # Replace text in first text block, remove others
            replaced = False
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and not replaced:
                    new_content.append({"type": "text", "text": new_text})
                    replaced = True
                elif isinstance(item, dict) and item.get("type") == "text":
                    continue  # drop extra text blocks
                else:
                    new_content.append(item)  # preserve non-text blocks (images, etc.)
            if not replaced:
                new_content.insert(0, {"type": "text", "text": new_text})
            block["content"] = new_content

    def _record_stats(
        self,
        tool_name: str,
        original_bytes: int,
        returned_bytes: int,
        indexed_bytes: int,
    ) -> None:
        self.stats.total_intercepted += 1
        self.stats.total_bytes_original += original_bytes
        self.stats.total_bytes_returned += returned_bytes
        self.stats.total_bytes_indexed += indexed_bytes
        # Cap by_tool entries to prevent unbounded growth in long sessions
        if tool_name in self.stats.by_tool or len(self.stats.by_tool) < 200:
            tool_stats = self.stats.by_tool.setdefault(tool_name, {
                "count": 0, "original_bytes": 0, "returned_bytes": 0,
            })
            tool_stats["count"] += 1
            tool_stats["original_bytes"] += original_bytes
            tool_stats["returned_bytes"] += returned_bytes
