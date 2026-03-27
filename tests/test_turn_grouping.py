import pytest
from virtual_context.proxy.formats import detect_format, TurnGroup


class TestAnthropicTurnGrouping:
    def test_simple_turns(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1]
        assert turns[0].role == "user"
        assert not turns[0].has_tool_activity
        assert turns[1].indices == [2, 3]

    def test_tool_chain(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "welcome"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1, 2, 3]
        assert turns[0].has_tool_activity
        assert turns[1].indices == [4, 5]
        assert not turns[1].has_tool_activity

    def test_trailing_user(self):
        body = {"model": "claude-sonnet-4-6", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "current question"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[1].indices == [2]  # trailing user is its own group


class TestOpenAIChatTurnGrouping:
    def test_simple_turns(self):
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2

    def test_tool_chain(self):
        body = {"model": "gpt-4", "messages": [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "welcome"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1, 2, 3]
        assert turns[0].has_tool_activity


class TestOpenAIResponsesTurnGrouping:
    def test_simple_turns(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "goodbye"}]},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2

    def test_bare_function_items(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "calling tool"}]},
            {"type": "function_call", "call_id": "fc1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "result"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "welcome"}]},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1, 2, 3, 4]
        assert turns[0].has_tool_activity

    def test_reasoning_items(self):
        body = {"model": "gpt-5", "input": [
            {"role": "user", "content": "think about this"},
            {"type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "thought about it"}]},
            {"role": "user", "content": "ok"},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1, 2]  # reasoning belongs to the turn


class TestGeminiTurnGrouping:
    def test_simple_turns(self):
        body = {"model": "gemini-pro", "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": [{"text": "bye"}]},
            {"role": "model", "parts": [{"text": "goodbye"}]},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2

    def test_function_chain(self):
        body = {"model": "gemini-pro", "contents": [
            {"role": "user", "parts": [{"text": "do something"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "read", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "read", "response": {"result": "data"}}}]},
            {"role": "model", "parts": [{"text": "done"}]},
            {"role": "user", "parts": [{"text": "thanks"}]},
            {"role": "model", "parts": [{"text": "welcome"}]},
        ]}
        fmt = detect_format(body)
        turns = fmt.group_into_turns(body)
        assert len(turns) == 2
        assert turns[0].indices == [0, 1, 2, 3]
        assert turns[0].has_tool_activity
