"""Tests for upstream context window enforcement trimming."""
import copy

from virtual_context.proxy.message_filter import trim_to_upstream_limit
from virtual_context.proxy.formats import OpenAIResponsesFormat, get_format


def _make_body(n_pairs, system="System prompt.", tools=None):
    """Create an OpenAI-format body with n user/assistant pairs."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for i in range(n_pairs):
        messages.append({"role": "user", "content": f"User message {i}. " * 50})
        messages.append({"role": "assistant", "content": f"Assistant reply {i}. " * 50})
    body = {"model": "gpt-4o", "messages": messages, "max_tokens": 4096}
    if tools:
        body["tools"] = tools
    return body


class TestTrimToUpstreamLimit:
    def _fmt(self):
        """Create a fresh format instance to avoid singleton mutation."""
        fmt = get_format("openai")
        import copy
        fmt_copy = copy.copy(fmt)
        fmt_copy.set_token_counter(lambda text: len(text) // 4)
        return fmt_copy

    def test_no_trim_when_under_limit(self):
        body = _make_body(5)
        fmt = self._fmt()
        trimmed, removed = trim_to_upstream_limit(body, 100_000, fmt)
        assert removed == 0
        assert trimmed is body

    def test_trims_oldest_pairs_when_over(self):
        body = _make_body(100)
        fmt = self._fmt()
        trimmed, removed = trim_to_upstream_limit(body, 5000, fmt)
        assert removed > 0
        msgs = trimmed["messages"]
        last_user = [m for m in msgs if m["role"] == "user"][-1]
        assert "User message 99" in last_user["content"]

    def test_protects_last_2_pairs(self):
        body = _make_body(10)
        fmt = self._fmt()
        trimmed, removed = trim_to_upstream_limit(body, 500, fmt)
        msgs = trimmed["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 2

    def test_system_and_tools_preserved(self):
        body = _make_body(50, system="System instructions.")
        body["tools"] = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        fmt = self._fmt()
        trimmed, removed = trim_to_upstream_limit(body, 3000, fmt)
        assert trimmed["messages"][0]["role"] == "system"
        assert "tools" in trimmed
        assert len(trimmed["tools"]) == 1

    def test_accounts_for_max_tokens(self):
        body = _make_body(50, system="Sys.")
        body["max_tokens"] = 50_000
        fmt = self._fmt()
        total_before = fmt.estimate_payload_tokens(body)
        trimmed, removed = trim_to_upstream_limit(body, 60_000, fmt)
        if total_before > 10_000:
            assert removed > 0

    def test_anthropic_format(self):
        body = {
            "model": "claude-sonnet-4-6",
            "system": "System prompt.",
            "messages": [],
            "max_tokens": 4096,
        }
        for i in range(50):
            body["messages"].append({"role": "user", "content": f"User {i}. " * 50})
            body["messages"].append({"role": "assistant", "content": f"Asst {i}. " * 50})
        fmt = get_format("anthropic")
        import copy
        fmt_copy = copy.copy(fmt)
        fmt_copy.set_token_counter(lambda text: len(text) // 4)
        trimmed, removed = trim_to_upstream_limit(body, 3000, fmt_copy)
        assert removed > 0
        assert trimmed["system"] == "System prompt."

    def test_returns_body_unchanged_when_few_pairs(self):
        body = _make_body(2)
        fmt = self._fmt()
        trimmed, removed = trim_to_upstream_limit(body, 10, fmt)
        assert removed == 0
        assert trimmed is body

    def test_responses_tool_round_trim_is_atomic(self):
        body = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "User message 0. " * 60},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Assistant intro 0. " * 60}]},
                {"type": "function_call", "call_id": "fc0", "name": "search", "arguments": '{"q":"zero"}'},
                {"type": "function_call_output", "call_id": "fc0", "output": "Tool output 0. " * 60},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Assistant final 0. " * 60}]},
                {"role": "user", "content": "User message 1. " * 60},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Assistant 1. " * 60}]},
                {"role": "user", "content": "User message 2. " * 60},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Assistant 2. " * 60}]},
                {"role": "user", "content": "User message 3. " * 60},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Assistant 3. " * 60}]},
            ],
            "max_tokens": 512,
        }
        fmt = copy.copy(OpenAIResponsesFormat())
        fmt.set_token_counter(lambda text: len(text) // 4)

        trimmed, removed = trim_to_upstream_limit(body, 2500, fmt)

        assert removed > 0
        rendered = str(trimmed["input"])
        assert "fc0" not in rendered
        assert "Assistant final 0." not in rendered
