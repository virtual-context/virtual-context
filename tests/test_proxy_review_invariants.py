"""Focused proxy regressions: request isolation, tool ownership and admission."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from starlette.responses import JSONResponse

from virtual_context.proxy.formats import get_format
from virtual_context.proxy.handlers import _handle_non_streaming, _handle_streaming
from virtual_context.proxy.message_filter import PayloadBudgetExceeded, admit_provider_payload
from virtual_context.proxy.server import create_app
from virtual_context.types import SpeakerRetrievalContext


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def _state():
    state = MagicMock()
    state.conversation_history = []
    state.is_conversation_deleted.return_value = False
    state.engine.reassemble_context.return_value = ""
    state.engine.config.conversation_id = "conversation-A"
    return state


def _response(fmt, *, tool=None, text=None):
    if fmt == "anthropic":
        content = ([{"type": "text", "text": text}] if text else [])
        if tool:
            content.append({"type": "tool_use", "id": "call-1", "name": tool, "input": {"query": "history"}})
        return {"id": "msg-1", "role": "assistant", "type": "message", "content": content, "stop_reason": "tool_use" if tool else "end_turn"}
    if fmt == "openai_responses":
        output = ([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}] if text else [])
        if tool:
            output.append({"type": "function_call", "call_id": "call-1", "name": tool, "arguments": '{"query":"history"}'})
        return {"id": "resp-1", "output": output, "status": "completed"}
    if fmt == "gemini":
        parts = ([{"text": text}] if text else [])
        if tool:
            parts.append({"functionCall": {"name": tool, "args": {"query": "history"}}})
        return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {"name": tool, "arguments": '{"query":"history"}'}}]
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if tool else "stop"}]}


def _body(fmt):
    if fmt == "openai_responses":
        return {"model": "gpt-4.1", "input": [{"role": "user", "content": "Recall history"}], "max_output_tokens": 200}
    if fmt == "gemini":
        return {"contents": [{"role": "user", "parts": [{"text": "Recall history"}]}], "generationConfig": {"maxOutputTokens": 200}}
    return {"model": "claude-test" if fmt == "anthropic" else "gpt-4o", "messages": [{"role": "user", "content": "Recall history"}], "max_tokens": 200}


def test_concurrent_requests_keep_tenant_metrics_local():
    async def run():
        metrics = {name: SimpleNamespace(owner=name) for name in ("default", "A", "B")}
        states = {name: SimpleNamespace(metrics=metrics[name], engine=SimpleNamespace(config=SimpleNamespace(conversation_id=name))) for name in ("A", "B")}
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare(body, state, fmt, request_metrics, **kwargs):
            assert request_metrics.owner == state.engine.config.conversation_id
            if request_metrics.owner == "A":
                entered.set()
                await release.wait()
            return SimpleNamespace(vc_command=False, is_passthrough=False, is_streaming=False, paging_enabled=False, tool_output_find_quote=False, restore_tool_injected=False, enriched_body=body, api_format="anthropic", turn=1, request_turn=1, turn_id="t", overhead_ms=0, conversation_id=state.engine.config.conversation_id, speaker_context=None, upstream_limit=200_000, speaker_roster_snapshot=None)

        async def handler(*args, **kwargs):
            return JSONResponse({"conversation_id": kwargs["conversation_id"], "metrics_owner": kwargs["metrics"].owner})

        with patch("virtual_context.proxy.server.VirtualContextEngine", side_effect=RuntimeError("no storage")), patch("virtual_context.proxy.server.prepare_payload", side_effect=prepare), patch("virtual_context.proxy.server._handle_non_streaming", side_effect=handler):
            app = create_app("http://upstream.invalid", shared_metrics=metrics["default"])
            app.state.state_resolver = lambda request, body, cid: (states[request.headers["x-tenant"]], False)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                a = asyncio.create_task(client.post("/v1/messages", headers={"x-tenant": "A"}, json=_body("anthropic")))
                await entered.wait()
                b = await client.post("/v1/messages", headers={"x-tenant": "B"}, json=_body("anthropic"))
                release.set()
                a_response = await a
                assert a_response.json() == {"conversation_id": "A", "metrics_owner": "A"}
                assert b.json() == {"conversation_id": "B", "metrics_owner": "B"}
    asyncio.run(run())


@pytest.mark.parametrize("character", ["é", "🍕", "中"])
def test_sse_unicode_chunk_boundaries_preserve_canonical_text(character):
    async def run():
        text = "café 🍕 中文"
        data = ("data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}, ensure_ascii=False) + "\n\n").encode()
        for split in range(1, len(character.encode())):
            boundary = data.index(character.encode()) + split
            transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([data[:boundary], data[boundary:]])))
            state = _state()
            async with httpx.AsyncClient(transport=transport) as client:
                response = await _handle_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", state, skip_marker_injection=True)
                outbound = b"".join([part async for part in response.body_iterator])
                assert outbound == data
                assert state.conversation_history[-1].content == text
                assert state.fire_turn_complete.call_args.args[0][-1].content == text
    asyncio.run(run())


@pytest.mark.parametrize("fmt,body,expected", [
    ("anthropic", {"max_tokens": 64}, 64),
    ("openai", {"max_completion_tokens": 128, "max_tokens": 64}, 128),
    ("openai", {"max_tokens": 64}, 64),
    ("openai_responses", {"max_output_tokens": 256}, 256),
    ("gemini", {"generationConfig": {"maxOutputTokens": 512}}, 512),
    ("gemini", {"generation_config": {"max_output_tokens": 512}}, 512),
    ("anthropic", {}, 4096),
])
def test_output_reservation_uses_provider_setting(fmt, body, expected):
    assert get_format(fmt).output_token_allowance(body) == expected


def test_admission_reserves_output_and_preserves_protected_turns():
    fmt = copy.copy(get_format("anthropic"))
    fmt.set_token_counter(lambda text: len(text) // 4)
    body = {"system": "Preserved instructions", "max_tokens": 200, "messages": sum(([{"role": "user", "content": f"user {index} " * 100}, {"role": "assistant", "content": f"answer {index} " * 100}] for index in range(5)), [])}
    total = fmt.estimate_payload_tokens(body)
    result, removed = admit_provider_payload(body, total + 100, fmt)
    assert removed > 0
    assert fmt.estimate_payload_tokens(result) <= total - 100
    assert result["messages"][-4:] == body["messages"][-4:]
    assert result["system"] == body["system"]


@pytest.mark.parametrize("streaming", [False, True])
def test_untrimmable_request_is_rejected_before_provider_call(streaming):
    async def run():
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, json={}))
        body = _body("anthropic")
        body["system"] = "protected " * 400
        handler = _handle_streaming if streaming else _handle_non_streaming
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(PayloadBudgetExceeded):
                await handler(client, "http://upstream", {}, body, "anthropic", None, upstream_limit=300)
        assert calls == []
    asyncio.run(run())


@pytest.mark.parametrize("fmt", ["anthropic", "openai_responses", "gemini", "openai"])
def test_nonstream_memory_tools_execute_with_request_authority(fmt):
    async def run():
        requests = []
        initial, final = _response(fmt, tool="vc_find_quote"), _response(fmt, text="Stored answer")
        def provider(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=initial if len(requests) == 1 else final)
        state = _state()
        context = SpeakerRetrievalContext.ineligible()
        roster = object()
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value='{"found":true,"quote":"stored evidence"}') as execute:
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                result = await _handle_non_streaming(client, "http://upstream", {"authorization": "test-auth"}, _body(fmt), fmt, state, intercept_vc_tools=True, skip_marker_injection=True, speaker_context=context, roster_snapshot=roster)
            assert json.loads(result.body) == final
            assert len(requests) == 2
            assert "stored evidence" in json.dumps(requests[1])
            assert execute.call_args.kwargs["speaker_context"] is context
            assert execute.call_args.kwargs["roster_snapshot"] is roster
            assert state.conversation_history[-1].content == "Stored answer"
    asyncio.run(run())


def test_nonstream_client_owned_tools_pass_through():
    async def run():
        upstream = _response("anthropic", tool="client_read_file")
        with patch("virtual_context.proxy.handlers.execute_vc_tool") as execute:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=upstream))) as client:
                result = await _handle_non_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", _state(), intercept_vc_tools=True)
            assert json.loads(result.body) == upstream
            execute.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["round_limit", "upstream_error", "budget"])
def test_nonstream_tool_failure_never_leaks_internal_calls(failure):
    async def run():
        calls = []
        upstream = _response("anthropic", tool="vc_find_quote")
        if failure == "mixed":
            upstream["content"].extend(_response("anthropic", tool="client_write_file")["content"])
        def provider(request):
            calls.append(request)
            if failure == "upstream_error" and len(calls) > 1:
                return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})
            return httpx.Response(200, json=upstream)
        tool_result = "evidence " * 2000 if failure == "budget" else '{"found":true}'
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value=tool_result) as execute:
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                if failure == "budget":
                    with pytest.raises(PayloadBudgetExceeded):
                        await _handle_non_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", _state(), intercept_vc_tools=True, upstream_limit=500)
                    assert len(calls) == 1
                    return
                result = await _handle_non_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", _state(), intercept_vc_tools=True)
            assert result.status_code == (429 if failure == "upstream_error" else 502)
            assert "tool_use" not in result.body.decode()
            assert len(calls) == (6 if failure == "round_limit" else 2 if failure == "upstream_error" else 1)
            if failure == "mixed":
                execute.assert_not_called()
    asyncio.run(run())


def test_app_returns_structured_413_for_protected_overflow():
    async def run():
        with patch("virtual_context.proxy.server.VirtualContextEngine", side_effect=RuntimeError("no storage")), patch("virtual_context.proxy.server.prepare_payload", side_effect=PayloadBudgetExceeded(1000, 800)):
            app = create_app("http://upstream.invalid", shared_metrics=MagicMock())
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                response = await client.post("/v1/messages", json=_body("anthropic"))
        assert response.status_code == 413
        assert response.json()["error"]["type"] == "context_budget_exceeded"
        assert response.json()["error"]["input_tokens"] == 1000
        assert response.json()["error"]["input_limit"] == 800
    asyncio.run(run())


def test_streaming_continuation_overflow_stops_before_second_provider_call():
    async def run():
        events = [
            {"type": "message_start", "message": {"id": "msg-1"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call-1", "name": "vc_find_quote", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"history"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
        data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events).encode()
        calls = []
        def provider(request):
            calls.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([data]))
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="evidence " * 2000):
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                response = await _handle_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", _state(), paging_enabled=True, upstream_limit=500, skip_marker_injection=True)
                outbound = b"".join([part async for part in response.body_iterator])
        assert len(calls) == 1
        assert b"context_budget_exceeded" in outbound
        assert b'"name": "vc_find_quote"' not in outbound
    asyncio.run(run())
