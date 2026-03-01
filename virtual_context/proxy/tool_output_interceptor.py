"""Tool output interception: truncate large tool_result blocks + index into FTS5.

Scans only the **last user message** for tool_result blocks.  Small outputs
pass through unchanged.  Large outputs are truncated (head + tail on line
boundaries) and the full content is stored in FTS5 for find_quote search.

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
    """Intercepts large tool_result blocks: truncate + index for find_quote."""

    def __init__(
        self,
        config: ToolOutputConfig,
        store: ContextStore,
        session_id: str,
    ) -> None:
        self.config = config
        self.store = store
        self.session_id = session_id
        self.stats = ToolOutputStats()
        self._turn_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        body: dict,
        fmt: PayloadFormat,
        vc_tool_ids: frozenset[str] = frozenset(),
    ) -> dict:
        """Intercept large tool_result blocks across all user messages.

        Returns the (potentially mutated) body.  Scans all user messages
        for tool_result blocks — Claude Code sends tool results in user
        messages throughout the conversation, not just the last one.
        """
        if not self.config.enabled:
            return body

        messages = fmt.get_messages(body)

        # Build tool_use_id -> {name, input} map from ALL assistant messages
        tool_use_map: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_use_map.update(self._build_tool_use_map(msg))

        # Process ALL user messages with tool_result blocks
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for block in self._iter_tool_result_blocks(msg):
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in vc_tool_ids:
                    continue
                tool_name = tool_use_map.get(tool_use_id, {}).get("name", "")
                if tool_name in VC_TOOL_NAMES:
                    continue

                content_text = self._extract_text(block)
                if not content_text:
                    continue
                content_bytes = len(content_text.encode("utf-8"))
                rule = self._match_rule(tool_name)
                threshold = rule.truncate_threshold if rule.truncate_threshold is not None else self.config.default_truncate_threshold

                if content_bytes < threshold:
                    continue  # passthrough

                # Truncate + Index
                head_ratio = rule.head_ratio
                tail_ratio = rule.tail_ratio
                head_text, tail_text = self._truncate(
                    content_text, threshold, head_ratio, tail_ratio,
                )
                ref = self._index(content_text, tool_name, rule)

                truncated_bytes = len(head_text.encode("utf-8")) + len(tail_text.encode("utf-8"))
                saved_bytes = content_bytes - truncated_bytes
                notice = (
                    f"\n... [{saved_bytes} bytes truncated"
                    f" — call vc_find_quote(query) to search the full output] ...\n"
                )
                replacement = head_text + notice + tail_text
                self._replace_content(block, replacement)
                self._record_stats(
                    tool_name, content_bytes,
                    len(replacement.encode("utf-8")),
                    content_bytes,
                )

        return body

    def increment_turn(self) -> None:
        """Call when a new turn starts."""
        self._turn_counter += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_user_msg(messages: list[dict]) -> dict | None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg
        return None

    @staticmethod
    def _find_last_assistant_msg(messages: list[dict]) -> dict | None:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg
        return None

    @staticmethod
    def _build_tool_use_map(assistant_msg: dict | None) -> dict[str, dict]:
        """Map tool_use_id -> {name, input} from assistant message blocks."""
        result: dict[str, dict] = {}
        if not assistant_msg:
            return result
        content = assistant_msg.get("content", [])
        if not isinstance(content, list):
            return result
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tid = block.get("id", "")
                if tid:
                    result[tid] = {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
        return result

    @staticmethod
    def _iter_tool_result_blocks(user_msg: dict):
        """Yield tool_result content blocks from a user message."""
        content = user_msg.get("content", [])
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block

    @staticmethod
    def _extract_text(block: dict) -> str:
        """Extract text content from a tool_result block."""
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

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
        """Store full content in FTS5. Returns ref for the notice."""
        ref = f"tool_{uuid4().hex[:12]}"
        cap = rule.max_index_bytes if rule.max_index_bytes is not None else self.config.max_index_bytes
        self.store.store_tool_output(
            ref=ref,
            session_id=self.session_id,
            tool_name=tool_name,
            command="",
            turn=self._turn_counter,
            content=content[:cap],
            original_bytes=len(content.encode("utf-8")),
        )
        return ref

    @staticmethod
    def _replace_content(block: dict, new_text: str) -> None:
        """Replace the text content of a tool_result block in-place."""
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
        tool_stats = self.stats.by_tool.setdefault(tool_name, {
            "count": 0, "original_bytes": 0, "returned_bytes": 0,
        })
        tool_stats["count"] += 1
        tool_stats["original_bytes"] += original_bytes
        tool_stats["returned_bytes"] += returned_bytes
