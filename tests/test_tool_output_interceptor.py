"""Tests for tool output interception: truncate + index into FTS5."""

from __future__ import annotations

import pytest

from virtual_context.proxy.tool_output_interceptor import ToolOutputInterceptor
from virtual_context.proxy.formats import detect_format
from virtual_context.types import ToolOutputConfig, ToolOutputRule, ToolOutputStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anthropic_body(
    assistant_content: list[dict],
    user_content: list[dict],
    *,
    older_user_content: list[dict] | None = None,
) -> dict:
    """Build a minimal Anthropic-format request body."""
    messages = []
    if older_user_content is not None:
        # Add an assistant message with tool_use blocks for any tool_results
        # in the older user content, so the tool_use_id map is populated.
        older_tool_uses = []
        for block in older_user_content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                older_tool_uses.append(_tool_use_block(block["tool_use_id"], "Read"))
        if older_tool_uses:
            messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
            messages.append({"role": "assistant", "content": older_tool_uses})
        messages.append({"role": "user", "content": older_user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
    messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": user_content})
    return {"model": "test", "messages": messages}


def _tool_use_block(tool_id: str, name: str, input_data: dict | None = None) -> dict:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": input_data or {},
    }


def _tool_result_block(tool_id: str, content: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
    }


def _tool_result_block_structured(tool_id: str, text: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": [{"type": "text", "text": text}],
    }


class FakeStore:
    """Minimal store that records calls and supports search."""

    def __init__(self):
        self.stored: list[dict] = []

    def store_tool_output(self, ref, session_id, tool_name, command, turn, content, original_bytes):
        self.stored.append({
            "ref": ref,
            "session_id": session_id,
            "tool_name": tool_name,
            "command": command,
            "turn": turn,
            "content": content,
            "original_bytes": original_bytes,
        })

    def search_tool_outputs(self, query, limit=5):
        return []


# ---------------------------------------------------------------------------
# Passthrough tests
# ---------------------------------------------------------------------------

class TestPassthrough:
    """Small outputs should pass through unchanged."""

    def test_small_output_unchanged(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=1000)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        small_text = "just a small output"
        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", small_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        # Content should be unchanged
        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == small_text
        assert store.stored == []
        assert interceptor.stats.total_intercepted == 0

    def test_disabled_config_passthrough(self):
        config = ToolOutputConfig(enabled=False)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 50000
        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == big_text
        assert store.stored == []


# ---------------------------------------------------------------------------
# Truncation tests
# ---------------------------------------------------------------------------

class TestTruncation:
    """Large outputs should be truncated + indexed."""

    def test_large_output_truncated(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        # Generate multi-line content exceeding threshold
        lines = [f"line {i}: " + "x" * 20 for i in range(50)]
        big_text = "\n".join(lines)
        assert len(big_text.encode("utf-8")) > 100

        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        new_content = last_user["content"][0]["content"]
        # Should be smaller than original
        assert len(new_content.encode("utf-8")) < len(big_text.encode("utf-8"))
        # Should contain the notice
        assert "bytes truncated" in new_content
        assert "vc_find_quote" in new_content
        # Should have stored the full content
        assert len(store.stored) == 1
        assert store.stored[0]["content"] == big_text
        assert store.stored[0]["tool_name"] == "Bash"
        assert interceptor.stats.total_intercepted == 1

    def test_structured_content_truncated(self):
        """tool_result with list content (structured blocks)."""
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 500
        body = _anthropic_body(
            [_tool_use_block("tu1", "Read")],
            [_tool_result_block_structured("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        content_blocks = last_user["content"][0]["content"]
        assert isinstance(content_blocks, list)
        text_block = next(b for b in content_blocks if b.get("type") == "text")
        assert "find_quote" in text_block["text"]
        assert len(store.stored) == 1

    def test_truncation_preserves_lines(self):
        """Truncation should split on line boundaries."""
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=200)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        lines = [f"LINE_{i:03d}" for i in range(100)]
        big_text = "\n".join(lines)

        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        new_content = result["messages"][-1]["content"][0]["content"]
        # Head should start with LINE_000
        assert new_content.startswith("LINE_000")
        # Tail should end with LINE_099
        assert new_content.rstrip().endswith("LINE_099")


# ---------------------------------------------------------------------------
# VC tool exclusion
# ---------------------------------------------------------------------------

class TestVCToolExclusion:
    """VC tool results must never be intercepted."""

    def test_vc_tool_by_name_excluded(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=10)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 1000
        body = _anthropic_body(
            [_tool_use_block("tu1", "vc_find_quote")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        # Should NOT be truncated
        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == big_text
        assert store.stored == []

    def test_vc_tool_by_id_excluded(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=10)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 1000
        body = _anthropic_body(
            [_tool_use_block("tu1", "some_tool")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        vc_ids = frozenset(["tu1"])
        result = interceptor.process(body, fmt, vc_tool_ids=vc_ids)

        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == big_text
        assert store.stored == []

    def test_vc_expand_topic_excluded(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=10)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 1000
        body = _anthropic_body(
            [_tool_use_block("tu1", "vc_expand_topic")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == big_text


# ---------------------------------------------------------------------------
# History preservation
# ---------------------------------------------------------------------------

class TestHistoryPreservation:
    """All user messages with large tool_results should be truncated."""

    def test_all_large_tool_results_truncated(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=10)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        old_big_text = "y" * 1000
        new_big_text = "z" * 1000
        body = _anthropic_body(
            [_tool_use_block("tu2", "Bash")],
            [_tool_result_block("tu2", new_big_text)],
            older_user_content=[_tool_result_block("tu1", old_big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        # Both old and new user messages should be truncated
        # [0]=user(text), [1]=asst(tool_use tu1), [2]=user(tool_result tu1),
        # [3]=asst(text), [4]=asst(tool_use tu2), [5]=user(tool_result tu2)
        old_user = result["messages"][2]
        assert old_user["content"] != old_big_text, "old tool_result should be truncated"
        assert "find_quote" in str(old_user["content"])
        new_user = result["messages"][5]
        assert new_user["content"] != new_big_text, "new tool_result should be truncated"
        assert "find_quote" in str(new_user["content"])
        assert len(store.stored) == 2  # both outputs indexed


# ---------------------------------------------------------------------------
# Per-tool rules
# ---------------------------------------------------------------------------

class TestPerToolRules:
    """Per-tool rules should override defaults."""

    def test_custom_threshold(self):
        config = ToolOutputConfig(
            enabled=True,
            default_truncate_threshold=100,
            rules=[
                ToolOutputRule(match="Read", truncate_threshold=5000),
            ],
        )
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        # 500 bytes — above default but below Read threshold
        text = "x" * 500
        body = _anthropic_body(
            [_tool_use_block("tu1", "Read")],
            [_tool_result_block("tu1", text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        # Should pass through (500 < 5000)
        last_user = result["messages"][-1]
        assert last_user["content"][0]["content"] == text
        assert store.stored == []

    def test_custom_head_ratio(self):
        config = ToolOutputConfig(
            enabled=True,
            default_truncate_threshold=200,
            rules=[
                ToolOutputRule(match="Read", head_ratio=0.8, tail_ratio=0.2),
            ],
        )
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        lines = [f"LINE_{i:03d}_{'x' * 10}" for i in range(100)]
        big_text = "\n".join(lines)

        body = _anthropic_body(
            [_tool_use_block("tu1", "Read")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        new_content = result["messages"][-1]["content"][0]["content"]
        notice_idx = new_content.index("... [")
        head = new_content[:notice_idx]
        # Head should start with LINE_000
        assert head.startswith("LINE_000")

    def test_fnmatch_wildcard(self):
        config = ToolOutputConfig(
            enabled=True,
            default_truncate_threshold=100,
            rules=[
                ToolOutputRule(match="Bash*", truncate_threshold=50),
            ],
        )
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        text = "x" * 80  # Above 50 (Bash* rule) but below 100 (default)
        body = _anthropic_body(
            [_tool_use_block("tu1", "BashTool")],
            [_tool_result_block("tu1", text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        # Should be truncated (matches Bash* rule, threshold=50)
        assert len(store.stored) == 1


# ---------------------------------------------------------------------------
# tool_use_id preservation
# ---------------------------------------------------------------------------

class TestToolUseIdPreservation:
    """tool_use_id must be preserved after truncation."""

    def test_tool_use_id_preserved(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        big_text = "x" * 500
        body = _anthropic_body(
            [_tool_use_block("tu_abc123", "Bash")],
            [_tool_result_block("tu_abc123", big_text)],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        block = last_user["content"][0]
        assert block["tool_use_id"] == "tu_abc123"
        assert block["type"] == "tool_result"


# ---------------------------------------------------------------------------
# Stats accumulation
# ---------------------------------------------------------------------------

class TestStats:
    """Stats should accumulate correctly."""

    def test_stats_accumulate(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        for i in range(3):
            big_text = "x" * 500
            body = _anthropic_body(
                [_tool_use_block(f"tu{i}", "Bash")],
                [_tool_result_block(f"tu{i}", big_text)],
            )
            fmt = detect_format(body)
            interceptor.process(body, fmt)

        assert interceptor.stats.total_intercepted == 3
        assert interceptor.stats.total_bytes_original == 500 * 3
        assert "Bash" in interceptor.stats.by_tool
        assert interceptor.stats.by_tool["Bash"]["count"] == 3

    def test_mixed_passthrough_and_truncation(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        # Small output — passthrough
        small_body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", "small")],
        )
        fmt = detect_format(small_body)
        interceptor.process(small_body, fmt)
        assert interceptor.stats.total_intercepted == 0

        # Large output — truncated
        big_body = _anthropic_body(
            [_tool_use_block("tu2", "Bash")],
            [_tool_result_block("tu2", "x" * 500)],
        )
        fmt = detect_format(big_body)
        interceptor.process(big_body, fmt)
        assert interceptor.stats.total_intercepted == 1


# ---------------------------------------------------------------------------
# Multiple tool results in one message
# ---------------------------------------------------------------------------

class TestMultipleToolResults:
    """Multiple tool results in a single user message."""

    def test_multiple_results_some_truncated(self):
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        store = FakeStore()
        interceptor = ToolOutputInterceptor(config, store, "sess1")

        small_text = "small output"
        big_text = "x" * 500

        body = _anthropic_body(
            [
                _tool_use_block("tu1", "Bash"),
                _tool_use_block("tu2", "Read"),
            ],
            [
                _tool_result_block("tu1", small_text),
                _tool_result_block("tu2", big_text),
            ],
        )
        fmt = detect_format(body)
        result = interceptor.process(body, fmt)

        last_user = result["messages"][-1]
        # First block should be unchanged
        assert last_user["content"][0]["content"] == small_text
        # Second block should be truncated
        assert "find_quote" in last_user["content"][1]["content"]
        assert len(store.stored) == 1
        assert store.stored[0]["tool_name"] == "Read"


# ---------------------------------------------------------------------------
# SQLite integration
# ---------------------------------------------------------------------------

class TestSQLiteIntegration:
    """End-to-end with real SQLiteStore."""

    def test_index_and_search(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore

        db = SQLiteStore(tmp_path / "test.db")
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        interceptor = ToolOutputInterceptor(config, db, "sess1")

        big_text = "The quick brown fox jumped over the lazy dog\n" * 50
        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        interceptor.process(body, fmt)

        # Should be searchable via search_tool_outputs
        results = db.search_tool_outputs("brown fox", limit=5)
        assert len(results) >= 1
        assert results[0].match_type == "tool_output"
        assert "brown fox" in results[0].text or ">>>" in results[0].text

    def test_persistence_across_turns(self, tmp_path):
        """Indexed tool output should be searchable on subsequent turns."""
        from virtual_context.storage.sqlite import SQLiteStore

        db = SQLiteStore(tmp_path / "test.db")
        config = ToolOutputConfig(enabled=True, default_truncate_threshold=100)
        interceptor = ToolOutputInterceptor(config, db, "sess1")

        # Turn 1: index a big output
        big_text = "unique_identifier_XYZ_12345 " * 100
        body = _anthropic_body(
            [_tool_use_block("tu1", "Bash")],
            [_tool_result_block("tu1", big_text)],
        )
        fmt = detect_format(body)
        interceptor.process(body, fmt)
        interceptor.increment_turn()

        # Turn 2: search should find it
        results = db.search_tool_outputs("unique_identifier_XYZ", limit=5)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:
    """ToolOutputConfig should parse from YAML-like dict."""

    def test_parse_from_raw(self):
        from virtual_context.config import load_config
        import tempfile
        import os

        yaml_content = """
version: "0.2"
storage_root: ".virtualcontext"
tool_output:
  enabled: true
  default_truncate_threshold: 4096
  rules:
    - match: "Read"
      truncate_threshold: 16384
      head_ratio: 0.7
      tail_ratio: 0.3
    - match: "Bash"
      head_ratio: 0.5
      tail_ratio: 0.5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                cfg = load_config(f.name)
                assert cfg.tool_output.enabled is True
                assert cfg.tool_output.default_truncate_threshold == 4096
                assert len(cfg.tool_output.rules) == 2
                assert cfg.tool_output.rules[0].match == "Read"
                assert cfg.tool_output.rules[0].truncate_threshold == 16384
                assert cfg.tool_output.rules[0].head_ratio == 0.7
                assert cfg.tool_output.rules[1].match == "Bash"
            finally:
                os.unlink(f.name)

    def test_default_config(self):
        config = ToolOutputConfig()
        assert config.enabled is False
        assert config.default_truncate_threshold == 8192
        assert config.max_index_bytes == 524288
        assert config.rules == []
