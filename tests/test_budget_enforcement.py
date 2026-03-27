import json
import pytest
from virtual_context.proxy.message_filter import scan_reducible_items
from virtual_context.proxy.formats import detect_format


def _make_payload(messages, system=None, tools=None):
    body = {"model": "claude-sonnet-4-6", "messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    return body


class TestScanReducibleItems:
    def test_finds_thinking_signatures(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "A" * 4000},
                {"type": "text", "text": "hi there"},
            ]},
            {"role": "user", "content": "bye"},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        thinking_items = [i for i in items if i.category == "thinking_sig"]
        assert len(thinking_items) == 1
        assert thinking_items[0].msg_index == 1
        assert thinking_items[0].block_index == 0
        assert thinking_items[0].size_bytes > 0

    def test_finds_tool_results(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": "thanks"},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        tr_items = [i for i in items if i.category == "tool_result"]
        assert len(tr_items) == 1
        assert tr_items[0].msg_index == 2

    def test_last_2_turns_tool_results_categorized_separately(self):
        msgs = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": "thanks"},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        last2_items = [i for i in items if i.category == "tool_result_last2"]
        assert len(last2_items) == 1

    def test_finds_conversation_text(self):
        msgs = [
            {"role": "user", "content": "hello " * 1000},
            {"role": "assistant", "content": "response " * 1000},
            {"role": "user", "content": "bye"},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        text_items = [i for i in items if i.category == "conversation_text"]
        assert len(text_items) >= 2

    def test_finds_vc_context_block(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        system = [
            {"type": "text", "text": "You are Claude Code."},
            {"type": "text", "text": "<system-reminder>\n<context-topics>VC content here</context-topics>\n</system-reminder>"},
        ]
        body = _make_payload(msgs, system=system)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        vc_items = [i for i in items if i.category == "vc_context"]
        assert len(vc_items) == 1

    def test_skips_compacted_stubs(self):
        msgs = [
            {"role": "user", "content": "[Compacted turn 0]"},
            {"role": "assistant", "content": [{"type": "text", "text": "[Compacted turn 0 | topics=foo | Read(...).\\nTo restore: ...]"}]},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        for item in items:
            assert item.category != "compacted_stub"

    def test_skips_system_prompt_non_vc(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        system = [{"type": "text", "text": "You are Claude Code. " * 500}]
        body = _make_payload(msgs, system=system)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        categories = {i.category for i in items}
        assert "system_prompt" not in categories

    def test_finds_large_image_blocks(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "A" * 50000}},
                {"type": "text", "text": "what is this?"},
            ]},
            {"role": "assistant", "content": "It's an image."},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        image_items = [i for i in items if i.category == "image"]
        assert len(image_items) == 1
        assert image_items[0].size_bytes == 50000

    def test_skips_small_image_blocks(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "A" * 100}},
            ]},
            {"role": "assistant", "content": "tiny image."},
        ]
        body = _make_payload(msgs)
        fmt = detect_format(body)
        items = scan_reducible_items(body, fmt)
        assert not any(i.category == "image" for i in items)
