"""Speaker retrieval context threading through prepare, handlers, and tool loops.

Proves the request-owned ``SpeakerRetrievalContext`` is derived once in
``prepare_payload`` before command dispatch (without running ingest,
tagging, or assembly for commands), is carried on ``PreparedPayload``,
and reaches ``execute_vc_tool`` unchanged on the proxy streaming path,
the synchronous tool-loop path, and continuation rounds. Nothing reads
the context for ranking or filtering yet — these tests pin the plumbing
and the fail-closed audience rule (unproved audience is ineligible and
is never substituted with the resolved owner).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_context.config import load_config
from virtual_context.core.tool_loop import (
    AnthropicAdapter,
    _execute_pending_tools,
    run_tool_loop,
)
from virtual_context.core.tool_query import ToolQueryRunner
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy import handlers as handlers_module
from virtual_context.proxy import server as server_module
from virtual_context.proxy.formats import AnthropicFormat
from virtual_context.proxy.handlers import (
    _handle_vc_command,
    _handle_vc_command_rest,
    _ProxyToolRuntime,
)
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.server import ProxyState, create_app, prepare_payload
from virtual_context.types import (
    AssembledContext,
    EngineState,
    PagingConfig,
    SpeakerRetrievalContext,
    ToolLoopResult,
)


SENTINEL_CONTEXT = SpeakerRetrievalContext(
    tenant_id="tenant-A",
    owner_conversation_id="owner-conv",
    audience_conversation_id="audience-conv",
    audience_channel_id="chan-1",
    requester_actor_id="actor:test:secret-id",
    original_active_user_text="what did I say about the trip",
)


def _found_result_json() -> str:
    return json.dumps({"found": True, "results": [{"excerpt": "x"}]})


# ---------------------------------------------------------------------------
# prepare_payload: command-path derivation
# ---------------------------------------------------------------------------


def _make_command_state(tmp_path, *, resolver=None):
    """Minimal ProxyState around a MagicMock engine with a real config."""
    config = load_config(config_dict={
        "conversation_id": "conv-123",
        "context_window": 10000,
        "storage_root": str(tmp_path),
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
        "tool_output": {"enabled": False},
    })
    config.tenant_id = "tenant-A"
    engine = MagicMock()
    engine.config = config
    engine._turn_tag_index = TurnTagIndex()
    engine._engine_state = EngineState()
    engine._store = MagicMock()
    if resolver is not None:
        engine._store.resolve_request_audience.side_effect = resolver
    engine._session_state_provider = None
    state = ProxyState(engine)
    return state, engine


class TestCommandPathContextDerivation:
    def _run_command(self, state, text="VCRECALL project alpha"):
        body = {
            "model": "claude-3",
            "stream": False,
            "messages": [{"role": "user", "content": text}],
        }
        return asyncio.run(prepare_payload(
            body, state, AnthropicFormat(), ProxyMetrics(),
            body_bytes=b"",
            inbound_conversation_id="route-1",
        ))

    def test_command_derives_eligible_context_without_side_effects(self, tmp_path):
        state, engine = _make_command_state(
            tmp_path, resolver=lambda tenant, raw, owner: "aud-route-1",
        )
        state.handle_prepare_payload = MagicMock()

        result = self._run_command(state)

        assert result.vc_command == "recall"
        assert result.vc_command_arg == "project alpha"
        ctx = result.speaker_context
        assert isinstance(ctx, SpeakerRetrievalContext)
        assert ctx.eligible is True
        assert ctx.tenant_id == "tenant-A"
        assert ctx.owner_conversation_id == "conv-123"
        assert ctx.audience_conversation_id == "aud-route-1"
        assert ctx.original_active_user_text == "VCRECALL project alpha"

        # Derivation is read-only: no ingest, tagging, or assembly ran.
        state.handle_prepare_payload.assert_not_called()
        engine.on_message_inbound.assert_not_called()
        engine.tag_turn.assert_not_called()
        engine.ingest_history.assert_not_called()
        assert state.conversation_history == []

    def test_unproved_audience_is_ineligible_and_never_the_owner(self, tmp_path):
        def _raise(tenant, raw, owner):
            raise RuntimeError("resolver unavailable")

        state, engine = _make_command_state(tmp_path, resolver=_raise)
        state.handle_prepare_payload = MagicMock()

        result = self._run_command(state)

        ctx = result.speaker_context
        assert ctx.eligible is False
        assert ctx.audience_conversation_id == ""
        # Fail closed: the resolved owner never substitutes the audience.
        assert ctx.audience_conversation_id != ctx.owner_conversation_id
        assert ctx.owner_conversation_id == "conv-123"
        state.handle_prepare_payload.assert_not_called()
        engine.on_message_inbound.assert_not_called()


# ---------------------------------------------------------------------------
# Command dispatchers: context reaches _handle_vcrecall
# ---------------------------------------------------------------------------


class TestCommandDispatchersPassContext:
    def _result(self, streaming=False):
        return SimpleNamespace(
            vc_command="recall",
            vc_command_arg="alpha",
            conversation_id="conv-123",
            is_streaming=streaming,
            speaker_context=SENTINEL_CONTEXT,
        )

    def test_proxy_dispatcher_passes_context_to_vcrecall(self):
        fmt = MagicMock()
        fmt.build_fake_response.return_value = {"ok": True}
        with patch.object(
            handlers_module, "_handle_vcrecall", return_value="ok",
        ) as spy:
            asyncio.run(_handle_vc_command(self._result(), fmt, None, None))
        assert spy.call_count == 1
        assert spy.call_args.kwargs["speaker_context"] is SENTINEL_CONTEXT

    def test_rest_dispatcher_passes_context_to_vcrecall(self):
        with patch.object(
            handlers_module, "_handle_vcrecall", return_value="ok",
        ) as spy:
            _handle_vc_command_rest(
                self._result(), None, None, tenant_id="tenant-A", vcconv="",
            )
        assert spy.call_count == 1
        assert spy.call_args.kwargs["speaker_context"] is SENTINEL_CONTEXT


# ---------------------------------------------------------------------------
# Proxy streaming path: context reaches execute_vc_tool and the runtime
# ---------------------------------------------------------------------------


def _make_sse_event(event_type: str, data: dict) -> bytes:
    data_str = json.dumps(data)
    return f"event: {event_type}\r\ndata: {data_str}\r\n\r\n".encode()


def _build_tool_use_sse_stream() -> bytes:
    tool_input_json = json.dumps({"query": "trip", "mode": "lookup"})
    events = _make_sse_event("message_start", {"type": "message_start"})
    events += _make_sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use", "id": "toolu_01",
            "name": "vc_find_quote", "input": {},
        },
    })
    events += _make_sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": tool_input_json},
    })
    events += _make_sse_event("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    })
    events += _make_sse_event("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": "tool_use"},
    })
    events += _make_sse_event("message_stop", {"type": "message_stop"})
    return events


@pytest.fixture
def paging_client(tmp_path):
    """App with a paging-enabled MagicMock engine (autonomous mode)."""
    from starlette.testclient import TestClient

    with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
        config = load_config(config_dict={
            "context_window": 10000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "s.db")}},
            "tag_generator": {"type": "keyword"},
        })
        config.paging = PagingConfig(enabled=True, autonomous_models=["opus", "sonnet"])
        engine = MagicMock()
        engine.config = config
        engine.on_message_inbound.return_value = AssembledContext()
        engine.tag_turn.return_value = None
        engine._turn_tag_index = TurnTagIndex()
        engine._retrieval._resolve_paging_mode.return_value = "autonomous"
        engine._engine_state.compacted_prefix_messages = 0
        MockEngine.return_value = engine
        app = create_app(upstream="http://fake:9999", config_path=None)
    with TestClient(app) as client:
        yield client, engine


class TestProxyStreamingContextThreading:
    def test_context_reaches_execute_vc_tool_and_runtime(self, paging_client):
        client, engine = paging_client

        captured: dict = {}
        real_prepare = server_module.prepare_payload

        async def capturing_prepare(*args, **kwargs):
            result = await real_prepare(*args, **kwargs)
            captured["result"] = result
            return result

        tool_calls: list[dict] = []

        def spy_execute(engine_arg, name, tool_input, **kwargs):
            tool_calls.append(dict(kwargs, tool_name=name))
            return _found_result_json()

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def mock_aiter_bytes():
            yield _build_tool_use_sse_stream()

        mock_resp.aiter_bytes = mock_aiter_bytes
        mock_resp.aclose = AsyncMock()

        cont_response = MagicMock()
        cont_response.status_code = 200
        cont_response.json.return_value = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}],
        }

        async def mock_send(req, **kwargs):
            return mock_resp

        async def mock_post(url, **kwargs):
            return cont_response

        with patch("virtual_context.proxy.server.prepare_payload", new=capturing_prepare), \
                patch("virtual_context.proxy.handlers.execute_vc_tool", new=spy_execute), \
                patch("virtual_context.proxy.server.httpx.AsyncClient.send", side_effect=mock_send), \
                patch("virtual_context.proxy.server.httpx.AsyncClient.build_request"), \
                patch("virtual_context.proxy.server.httpx.AsyncClient.post", side_effect=mock_post):
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "system": "test",
                    "stream": True,
                    "messages": [{"role": "user", "content": "About the trip"}],
                },
            )

        assert resp.status_code == 200
        result = captured["result"]
        ctx = result.speaker_context
        assert isinstance(ctx, SpeakerRetrievalContext)
        assert ctx.owner_conversation_id == engine.config.conversation_id
        # No pre-alias route was proved for this request.
        assert ctx.eligible is False

        assert len(tool_calls) == 1
        call = tool_calls[0]
        assert call["tool_name"] == "vc_find_quote"
        # The exact object derived by prepare_payload reaches execution.
        assert call["speaker_context"] is ctx
        # The runtime retains the same request-owned object.
        assert call["tool_runtime"].speaker_context is ctx

    def test_no_user_skip_path_passes_explicit_ineligible_context(self, paging_client):
        client, _engine = paging_client

        handler_spy = AsyncMock(
            return_value=handlers_module.JSONResponse(content={}),
        )
        with patch("virtual_context.proxy.server._handle_non_streaming", new=handler_spy):
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3",
                    "stream": False,
                    "messages": [{"role": "assistant", "content": "done"}],
                },
            )

        assert resp.status_code == 200
        assert handler_spy.await_count == 1
        ctx = handler_spy.call_args.kwargs["speaker_context"]
        assert isinstance(ctx, SpeakerRetrievalContext)
        assert ctx.eligible is False
        assert ctx == SpeakerRetrievalContext.ineligible()


# ---------------------------------------------------------------------------
# Synchronous tool loop and continuation rounds
# ---------------------------------------------------------------------------


def _anthropic_response(content, stop_reason="end_turn"):
    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _mock_http_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = payload
    return resp


class TestSyncToolLoopContextThreading:
    def test_run_tool_loop_forwards_context_across_continuation_rounds(self):
        engine = MagicMock()
        engine.reassemble_context.return_value = ""

        initial = _anthropic_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote",
             "input": {"query": "a", "mode": "lookup"}},
        ], stop_reason="tool_use")
        round_two = _anthropic_response([
            {"type": "tool_use", "id": "t2", "name": "vc_find_quote",
             "input": {"query": "b", "mode": "lookup"}},
        ], stop_reason="tool_use")
        final = _anthropic_response([{"type": "text", "text": "answer"}])

        seen: list[dict] = []

        def spy_execute(engine_arg, name, tool_input, **kwargs):
            seen.append(dict(kwargs))
            return _found_result_json()

        original_request = {
            "model": "m", "max_tokens": 100,
            "messages": [{"role": "user", "content": "q"}],
        }

        with patch("virtual_context.core.tool_loop.execute_vc_tool", new=spy_execute), \
                patch("virtual_context.core.tool_loop.httpx.Client") as client_cls:
            client = MagicMock()
            client.post.side_effect = [
                _mock_http_response(round_two),
                _mock_http_response(final),
            ]
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client_cls.return_value = client

            result = run_tool_loop(
                engine, initial, original_request, AnthropicAdapter("k"),
                speaker_context=SENTINEL_CONTEXT,
            )

        assert result.text == "answer"
        # Both the initial round and the continuation round forwarded the
        # exact request-owned object.
        assert len(seen) == 2
        assert all(call["speaker_context"] is SENTINEL_CONTEXT for call in seen)

    def test_execute_pending_tools_forwards_context(self):
        engine = MagicMock()
        seen: list[dict] = []

        def spy_execute(engine_arg, name, tool_input, **kwargs):
            seen.append(dict(kwargs))
            return _found_result_json()

        with patch("virtual_context.core.tool_loop.execute_vc_tool", new=spy_execute):
            _execute_pending_tools(
                engine,
                [{"id": "t1", "name": "vc_find_quote", "input": {}}],
                AnthropicAdapter("k"),
                ToolLoopResult(),
                "",
                set(),
                set(),
                tool_runtime=None,
                speaker_context=SENTINEL_CONTEXT,
            )

        assert len(seen) == 1
        assert seen[0]["speaker_context"] is SENTINEL_CONTEXT

    def test_query_with_tools_forwards_context_to_tool_loop(self):
        engine = MagicMock()
        runner = ToolQueryRunner(engine, MagicMock())

        initial = _anthropic_response([
            {"type": "tool_use", "id": "t1", "name": "vc_find_quote",
             "input": {"query": "a", "mode": "lookup"}},
        ], stop_reason="tool_use")

        loop_result = ToolLoopResult()

        with patch("httpx.Client") as client_cls, \
                patch(
                    "virtual_context.core.tool_loop.run_tool_loop",
                    return_value=loop_result,
                ) as loop_spy:
            client = MagicMock()
            client.post.return_value = _mock_http_response(initial)
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client_cls.return_value = client

            result = runner.query_with_tools(
                [{"role": "user", "content": "q"}],
                model="claude-x",
                api_key="k",
                force_tools=True,
                max_loops=2,
                speaker_context=SENTINEL_CONTEXT,
            )

        assert result is loop_result
        assert loop_spy.call_count == 1
        assert loop_spy.call_args.kwargs["speaker_context"] is SENTINEL_CONTEXT


# ---------------------------------------------------------------------------
# Runtime retention and containment
# ---------------------------------------------------------------------------


class TestContextContainment:
    def test_proxy_tool_runtime_retains_context(self):
        runtime = _ProxyToolRuntime(
            engine=MagicMock(),
            api_format="anthropic",
            conversation_id="conv-1",
            get_target_body=lambda: {},
            speaker_context=SENTINEL_CONTEXT,
        )
        assert runtime.speaker_context is SENTINEL_CONTEXT

    def test_context_repr_never_exposes_actor_id_or_user_text(self):
        text = repr(SENTINEL_CONTEXT)
        assert "actor:test:secret-id" not in text
        assert "what did I say about the trip" not in text
        # Non-identity routing fields remain visible for debugging.
        assert "owner-conv" in text
