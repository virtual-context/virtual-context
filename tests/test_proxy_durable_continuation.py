"""Real SQLite, fake provider: durable mixed tool handoff across proxy workers."""
import asyncio
import json
import threading
import time
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from test_proxy_review_invariants import _body, _response, _state, _Chunks
from virtual_context.proxy.continuation import ContinuationError, assistant_items, messages
from virtual_context.proxy.handlers import _continuation_session, _handle_non_streaming, _handle_streaming
from virtual_context.proxy.request_context import RequestContext
from virtual_context.proxy.response_codec import collect_response, response_events
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SpeakerRetrievalContext


def context(fmt, state, **kwargs):
    return RequestContext.create(body=_body(fmt), state=state, provider="http://upstream", api_format=fmt, tenant_id="tenant-A", conversation_id="conversation-A", audience_route="audience-A", upstream_limit=100_000, output_allowance=200, speaker_context=SpeakerRetrievalContext(tenant_id="tenant-A", owner_conversation_id="conversation-A", audience_conversation_id="audience-A"), metrics=state.metrics, **kwargs)


def mixed_response(fmt):
    response = _response(fmt, tool="vc_find_quote")
    external = _response(fmt, tool="client_read_file")
    if fmt == "anthropic":
        external["content"][0]["id"] = "external-1"
        response["content"].extend(external["content"])
    elif fmt == "openai":
        external["choices"][0]["message"]["tool_calls"][0]["id"] = "external-1"
        response["choices"][0]["message"]["tool_calls"].extend(external["choices"][0]["message"]["tool_calls"])
    elif fmt == "openai_responses":
        external["output"][0]["call_id"] = "external-1"
        response["output"].extend(external["output"])
    else:
        external["candidates"][0]["content"]["parts"][0]["functionCall"]["id"] = "external-1"
        response["candidates"][0]["content"]["parts"].extend(external["candidates"][0]["content"]["parts"])
    return response


def next_request(fmt, visible, *, idless=False):
    body = _body(fmt)
    items = messages(body, fmt) + assistant_items(visible, fmt)
    if fmt == "anthropic":
        token = next(item["id"] for item in visible["content"] if item.get("type") == "tool_use")
        items.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": token, "content": "real client result"}]})
    elif fmt == "openai":
        token = visible["choices"][0]["message"]["tool_calls"][0]["id"]
        items.append({"role": "tool", "tool_call_id": token, "content": "real client result"})
    elif fmt == "openai_responses":
        token = next(item["call_id"] for item in visible["output"] if item.get("type") == "function_call")
        items.append({"type": "function_call_output", "call_id": token, "output": "real client result"})
    else:
        call = next(part["functionCall"] for part in visible["candidates"][0]["content"]["parts"] if "functionCall" in part)
        result = {"name": call["name"], "response": {"content": "real client result"}}
        if not idless:
            result["id"] = call["id"]
        items.append({"role": "user", "parts": [{"functionResponse": result}]})
    body[{"gemini": "contents", "openai_responses": "input"}.get(fmt, "messages")] = items
    return body


@pytest.mark.parametrize("fmt", ["anthropic", "openai", "openai_responses", "gemini"])
@pytest.mark.parametrize("streaming", [False, True])
def test_mixed_exchange_resumes_on_another_worker_with_only_real_client_results(tmp_path, fmt, streaming):
    async def run():
        store_a, store_b = SQLiteStore(tmp_path / "store.db"), SQLiteStore(tmp_path / "store.db")
        state_a, state_b = _state(), _state()
        state_a.engine._store, state_b.engine._store = store_a, store_b
        original = mixed_response(fmt)
        sent = []
        def provider(request):
            sent.append(json.loads(request.content))
            response = original if len(sent) == 1 else _response(fmt, text="finished")
            if streaming:
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks(list(response_events(response, fmt))))
            return httpx.Response(200, json=response)
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private memory evidence") as execute:
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                handler = _handle_streaming if streaming else _handle_non_streaming
                opts = {"paging_enabled": True} if streaming else {"intercept_vc_tools": True}
                first = await handler(client, "http://upstream", {}, _body(fmt), fmt, state_a, request_context=context(fmt, state_a), skip_marker_injection=True, **opts)
                if streaming:
                    raw = b"".join([part async for part in first.body_iterator])
                    visible = await collect_response(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([raw])), fmt)
                else:
                    raw, visible = first.body, json.loads(first.body)
                assert b"vc_find_quote" not in raw and b"private memory evidence" not in raw
                assert b"client_read_file" in raw and b"vcx_" in raw
                assert len(sent) == 1
                assert len(state_a.conversation_history) == 0
                followup = next_request(fmt, visible)
                session = _continuation_session(context(fmt, state_b), state_b)
                restored = await session.resume(followup)
                assert "private memory evidence" in json.dumps(restored)
                assert "real client result" in json.dumps(restored)
                assert "vcx_" not in json.dumps(restored)
                assert session.context.metrics is state_b.metrics
                assert session.context.speaker_context.audience_conversation_id == "audience-A"
                second = await handler(client, "http://upstream", {}, restored, fmt, state_b, request_context=session.context, continuation_session=session, skip_marker_injection=True, **opts)
                if streaming:
                    await collect_response(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([b"".join([part async for part in second.body_iterator])])), fmt)
                assert len(sent) == 2
                assert execute.call_count == 1
                assert state_b.conversation_history[-1].content == "finished"
                assert not store_b.list_pending_exchanges("conversation-A", now=0)
                with pytest.raises(ContinuationError, match="expired|progress"):
                    await _continuation_session(context(fmt, state_b), state_b).resume(followup)
        store_a.close()
        store_b.close()
    asyncio.run(run())


@pytest.mark.parametrize("alter", ["tenant", "conversation", "audience", "model", "source", "arguments", "partial", "duplicate", "lifecycle"])
def test_deferred_authority_transcript_and_exact_result_batch_are_required(tmp_path, alter):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        state = _state()
        state.engine._store = store
        ctx = context("anthropic", state)
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="secret"):
            session = _continuation_session(ctx, state)
            _, visible = await session.advance(_body("anthropic"), mixed_response("anthropic"))
        incoming = next_request("anthropic", visible)
        current = ctx
        if alter in ("tenant", "conversation", "audience", "model", "lifecycle"):
            field = {"tenant": "tenant_id", "conversation": "conversation_id", "audience": "audience_route", "model": "model", "lifecycle": "lifecycle_epoch"}[alter]
            current = replace(current, **{field: 2 if alter == "lifecycle" else "different"})
        elif alter == "source":
            incoming["messages"][0]["content"] = "forged user source"
        elif alter == "arguments":
            incoming["messages"][-2]["content"][0]["input"] = {"path": "/different"}
        elif alter == "partial":
            incoming["messages"][-1]["content"].append({"type": "text", "text": "new prompt"})
        else:
            incoming["messages"][-1]["content"] *= 2
        with pytest.raises(ContinuationError):
            await _continuation_session(current, state).resume(incoming)
        # A rejected request releases its claim; the exact original still works.
        valid = _continuation_session(ctx, state)
        assert await valid.resume(next_request("anthropic", visible))
        await valid.close()
        store.close()
    asyncio.run(run())


def test_gemini_idless_results_and_responses_previous_response_id(tmp_path):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        for fmt in ("gemini", "openai_responses"):
            state = _state()
            state.engine._store = store
            ctx = context(fmt, state)
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="secret"):
                first = _continuation_session(ctx, state)
                _, visible = await first.advance(_body(fmt), mixed_response(fmt))
            incoming = next_request(fmt, visible, idless=True)
            if fmt == "openai_responses":
                incoming["input"] = incoming["input"][-1:]
                incoming["previous_response_id"] = visible["id"]
            next_session = _continuation_session(ctx, state)
            assert "secret" in json.dumps(await next_session.resume(incoming))
            await next_session.close(consume=True)
        store.close()
    asyncio.run(run())


def test_request_context_is_frozen_and_excludes_metrics_from_checkpoint(tmp_path):
    state = _state()
    ctx = context("anthropic", state)
    with pytest.raises(FrozenInstanceError):
        ctx.tenant_id = "other"
    assert "metrics" not in repr(ctx) and "audience-A" not in repr(ctx.speaker_context.requester_actor_id)


@pytest.mark.parametrize("fmt", ["anthropic", "openai", "openai_responses", "gemini"])
@pytest.mark.parametrize("mixed", [False, True])
def test_public_text_survives_internal_rounds_and_is_persisted_once(tmp_path, fmt, mixed):
    async def run():
        store = SQLiteStore(tmp_path / "text.db")
        try:
            state = _state()
            state.engine._store = store
            ctx = context(fmt, state)
            session = _continuation_session(ctx, state)
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private evidence"):
                continuation, _ = await session.advance(_body(fmt), _response(fmt, tool="vc_find_quote", text="First public answer."))
                if mixed:
                    response = mixed_response(fmt)
                    # Include a second public fragment beside the mixed calls.
                    from virtual_context.proxy.continuation import _prepend_public_text
                    response = _prepend_public_text(response, fmt, ["Second public answer."], native_messages=_response("openai_responses", text="Second public answer.")["output"])
                    _, visible = await session.advance(continuation, response)
                    wire = json.dumps(visible)
                    assert "First public answer." in wire and "Second public answer." in wire
                    assert "private evidence" not in wire and "vc_find_quote" not in wire
                    followup = next_request(fmt, visible)
                    session = _continuation_session(ctx, state)
                    continuation = await session.resume(followup)
                _, final = await session.advance(continuation, _response(fmt, text="Final public answer."))
            text = session.adapter.extract_text(final)
            assert ("First public answer." in text) is (not mixed)
            session.persist_completed(text, final.get("content"))
            saved = state.fire_turn_complete.call_args.args[0][-1]
            assert saved.content.count("First public answer.") == 1
            assert saved.content.count("Final public answer.") == 1
            assert ("Second public answer." in saved.content) is mixed
            assert "private evidence" not in saved.content
            if mixed:
                await session.close(consume=True)
        finally:
            store.close()
    asyncio.run(run())


def test_cancelled_provider_collection_releases_durable_claim(tmp_path):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        state = _state()
        state.engine._store = store
        ctx = context("anthropic", state)
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="secret"):
            _, visible = await _continuation_session(ctx, state).advance(_body("anthropic"), mixed_response("anthropic"))
        incoming = next_request("anthropic", visible)
        session = _continuation_session(ctx, state)
        restored = await session.resume(incoming)
        entered = asyncio.Event()
        async def provider(request):
            entered.set()
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            task = asyncio.create_task(_handle_non_streaming(client, "http://upstream", {}, restored, "anthropic", state, request_context=session.context, continuation_session=session, intercept_vc_tools=True))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        retry = _continuation_session(ctx, state)
        assert await retry.resume(incoming)
        await retry.close()
        assert not state.conversation_history
        store.close()
    asyncio.run(run())


def test_anthropic_sdk_consumes_text_tools_and_signed_thinking_events():
    from anthropic import AsyncAnthropic

    async def run():
        response = {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [
                {"type": "thinking", "thinking": "opaque reasoning", "signature": "signed-proof"},
                {"type": "text", "text": "café 🍕"},
                {"type": "tool_use", "id": "client-1", "name": "client_read_file", "input": {"path": "a.py"}},
            ],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        from virtual_context.proxy.continuation import _prepend_public_text
        response = _prepend_public_text(response, "anthropic", ["Earlier public text. "])
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks(list(response_events(response, "anthropic")))))
        async with httpx.AsyncClient(transport=transport) as http:
            client = AsyncAnthropic(api_key="local-test", http_client=http)
            async with client.messages.stream(model="claude-test", max_tokens=100, messages=[{"role": "user", "content": "test"}]) as stream:
                text = "".join([text async for text in stream.text_stream])
                final = await stream.get_final_message()
        assert text == "Earlier public text. \n\ncafé 🍕"
        assert final.content[0].signature == "signed-proof"
        assert final.content[3].input == {"path": "a.py"}
    asyncio.run(run())


def test_openai_responses_sdk_consumes_text_and_client_calls():
    from openai import AsyncOpenAI

    async def run():
        response = {
            "id": "resp_test", "object": "response", "created_at": 1, "status": "completed",
            "model": "gpt-4.1", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [
                {"id": "msg_test", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "café 🍕", "annotations": [], "logprobs": []}]},
                {"id": "fc_test", "type": "function_call", "call_id": "client-1", "name": "client_read_file", "arguments": '{"path":"a.py"}', "status": "completed"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        }
        from virtual_context.proxy.continuation import _prepend_public_text
        response["output"][:0] = [{"type": "reasoning", "id": "rs_native", "summary": [], "encrypted_content": "opaque-proof"}]
        prior_message = {"id": "msg_prior_native", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "Earlier public text. ", "annotations": []}]}
        response = _prepend_public_text(response, "openai_responses", ["Earlier public text. "], native_messages=[prior_message])
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks(list(response_events(response, "openai_responses")))))
        async with httpx.AsyncClient(transport=transport) as http:
            client = AsyncOpenAI(api_key="local-test", http_client=http)
            async with client.responses.stream(model="gpt-4.1", input="test") as stream:
                text = "".join([event.delta async for event in stream if event.type == "response.output_text.delta"])
                final = await stream.get_final_response()
        assert text == "Earlier public text. \n\ncafé 🍕"
        assert final.output[0].id == "rs_native"
        assert final.output[0].encrypted_content == "opaque-proof"
        assert final.output[1].id == "msg_prior_native"
        assert final.output[3].arguments == '{"path":"a.py"}'
        assert final.output[3].call_id == "client-1"
    asyncio.run(run())


def test_completion_uses_original_user_snapshot_when_another_request_replaces_history():
    from virtual_context.types import Message

    state = _state()
    state.conversation_history = [Message(role="user", content="original request")]
    ctx = context("anthropic", state)
    state.conversation_history = [Message(role="user", content="concurrent request")]
    _continuation_session(ctx, state).persist_completed("original answer", None)
    saved = state.fire_turn_complete.call_args.args[0]
    assert [(item.role, item.content) for item in saved] == [("user", "original request"), ("assistant", "original answer")]
    assert state.conversation_history[0].content == "concurrent request"


def test_collection_rejects_truncation_and_oversized_unterminated_event():
    async def run():
        with pytest.raises(ContinuationError, match="completion event"):
            await collect_response(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([b'data: {"type":"message_start","message":{}}\n\n'])), "anthropic")
        with patch("virtual_context.proxy.response_codec.MAX_RESPONSE_BYTES", 16):
            with pytest.raises(ContinuationError, match="collection limit"):
                await collect_response(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Chunks([b"data: ", b"x" * 17])), "anthropic")
    asyncio.run(run())


def test_completed_mixed_exchange_in_history_does_not_block_the_next_user_request(tmp_path):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        state = _state()
        state.engine._store = store
        ctx = context("anthropic", state)
        with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="secret"):
            _, visible = await _continuation_session(ctx, state).advance(_body("anthropic"), mixed_response("anthropic"))
        incoming = next_request("anthropic", visible)
        session = _continuation_session(ctx, state)
        assert await session.resume(incoming)
        await session.close(consume=True)
        incoming["messages"].extend([{"role": "assistant", "content": "finished"}, {"role": "user", "content": "next question"}])
        assert await _continuation_session(ctx, state).resume(incoming) is None
        store.close()
    asyncio.run(run())


@pytest.mark.parametrize('fmt,expected_input', [('openai', 30), ('anthropic', 44)])
def test_request_metrics_accumulate_all_internal_provider_rounds(fmt, expected_input):
    async def run():
        state = _state()
        replies = [_response(fmt, tool='vc_find_quote'), _response(fmt, text='finished')]
        if fmt == 'openai':
            replies[0]['usage'] = {'prompt_tokens': 10, 'completion_tokens': 2}
            replies[1]['usage'] = {'prompt_tokens': 20, 'completion_tokens': 3}
        else:
            replies[0]['usage'] = {'input_tokens': 10, 'output_tokens': 2, 'cache_read_input_tokens': 3, 'cache_creation_input_tokens': 7}
            replies[1]['usage'] = {'input_tokens': 20, 'output_tokens': 3, 'cache_read_input_tokens': 4}
        with patch('virtual_context.proxy.handlers.execute_vc_tool', return_value='memory evidence'):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=replies.pop(0)))) as client:
                response = await _handle_non_streaming(client, 'http://upstream', {}, _body(fmt), fmt, state, request_context=context(fmt, state), intercept_vc_tools=True)
        assert response.status_code == 200
        assert state.metrics.capture_response.call_args.kwargs['upstream_input_tokens'] == expected_input
        assert state.metrics.capture_response.call_args.kwargs['upstream_output_tokens'] == 5
    asyncio.run(run())


def _gemini_without_call_ids(visible):
    incoming = next_request("gemini", visible, idless=True)
    for part in incoming["contents"][-2]["parts"]:
        part.get("functionCall", {}).pop("id", None)
    return incoming


def _ordinary_gemini_result():
    incoming = next_request("gemini", _response("gemini", tool="client_read_file"), idless=True)
    incoming["contents"][0]["parts"][0]["text"] = "Read my current file, from another concurrent client"
    return incoming


@pytest.mark.parametrize("expired", [False, True])
def test_unrelated_gemini_tool_result_never_claims_pending_or_expired_exchange(tmp_path, expired):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        try:
            state = _state()
            state.engine._store = store
            ctx = context("gemini", state)
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private"):
                await _continuation_session(ctx, state).advance(_body("gemini"), mixed_response("gemini"))
            if expired:
                store._get_conn().execute("UPDATE pending_tool_exchanges SET expires_at=?", (time.time() - 1,))
            assert store.list_pending_exchanges("conversation-A", now=time.time())
            with patch.object(store, "claim_pending_exchange", wraps=store.claim_pending_exchange) as claim:
                assert await _continuation_session(ctx, state).resume(_ordinary_gemini_result()) is None
                claim.assert_not_called()
        finally:
            store.close()
    asyncio.run(run())


def test_ordinary_gemini_tool_result_does_not_require_continuation_capability():
    async def run():
        state = _state()
        state.engine._store = SimpleNamespace()
        assert await _continuation_session(context("gemini", state), state).resume(_ordinary_gemini_result()) is None
    asyncio.run(run())


def test_ordinary_gemini_ignores_inherited_unsupported_store_methods():
    from virtual_context.core.store import ContextStore

    # Concrete legacy adapter: it implements the abstract CRUD surface but
    # inherits the base continuation methods, which raise NotImplementedError.
    legacy_type = type('LegacyStore', (ContextStore,), {
        name: (lambda *args, **kwargs: None) for name in ContextStore.__abstractmethods__
    })

    async def run():
        state = _state()
        state.engine._store = legacy_type()
        assert callable(state.engine._store.list_pending_exchanges)
        ctx = context('gemini', state)
        assert await _continuation_session(ctx, state).resume(_ordinary_gemini_result()) is None
        with patch('virtual_context.proxy.handlers.execute_vc_tool') as execute:
            with pytest.raises(ContinuationError, match='does not support durable'):
                await _continuation_session(ctx, state).advance(_body('gemini'), mixed_response('gemini'))
            execute.assert_not_called()
    asyncio.run(run())


def test_responses_missing_response_id_does_not_replace_transcript_proof(tmp_path):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        try:
            state = _state()
            state.engine._store = store
            ctx = context("openai_responses", state)
            response = mixed_response("openai_responses")
            response.pop("id")
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private"):
                _, visible = await _continuation_session(ctx, state).advance(_body("openai_responses"), response)
            incoming = next_request("openai_responses", visible)
            incoming["input"] = incoming["input"][-1:]
            with pytest.raises(ContinuationError, match="transcript"):
                await _continuation_session(ctx, state).resume(incoming)
        finally:
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("ambiguous", [False, True])
def test_gemini_without_both_native_ids_requires_unique_exact_transcript(tmp_path, ambiguous):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        try:
            state = _state()
            state.engine._store = store
            ctx = context("gemini", state)
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private"):
                _, visible = await _continuation_session(ctx, state).advance(_body("gemini"), mixed_response("gemini"))
                if ambiguous:
                    await _continuation_session(ctx, state).advance(_body("gemini"), mixed_response("gemini"))
            session = _continuation_session(ctx, state)
            with patch.object(store, "claim_pending_exchange", wraps=store.claim_pending_exchange) as claim:
                if ambiguous:
                    with pytest.raises(ContinuationError, match="multiple pending"):
                        await session.resume(_gemini_without_call_ids(visible))
                    claim.assert_not_called()
                else:
                    restored = await session.resume(_gemini_without_call_ids(visible))
                    assert "private" in json.dumps(restored) and "real client result" in json.dumps(restored)
                    assert claim.call_count == 1
                    await session.close(consume=True)
        finally:
            store.close()
    asyncio.run(run())


def test_gemini_discovery_does_not_steal_a_concurrent_legitimate_claim(tmp_path):
    async def run():
        store = SQLiteStore(tmp_path / "store.db")
        release = threading.Event()
        try:
            state = _state()
            state.engine._store = store
            ctx = context("gemini", state)
            with patch("virtual_context.proxy.handlers.execute_vc_tool", return_value="private"):
                _, visible = await _continuation_session(ctx, state).advance(_body("gemini"), mixed_response("gemini"))
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()
            original_peek = store.get_pending_exchange

            def paused_peek(*args, **kwargs):
                raw = original_peek(*args, **kwargs)
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "test did not release read-only discovery"
                return raw

            with patch.object(store, "get_pending_exchange", side_effect=paused_peek):
                unrelated = asyncio.create_task(_continuation_session(ctx, state).resume(_ordinary_gemini_result()))
                await asyncio.wait_for(entered.wait(), 2)
                legitimate = _continuation_session(ctx, state)
                try:
                    assert await legitimate.resume(next_request("gemini", visible))
                    assert legitimate.claim is not None
                finally:
                    release.set()
                assert await unrelated is None
                assert store.claim_pending_exchange("conversation-A", legitimate.claim[0], "third", now=time.time()) is None
                await legitimate.close(consume=True)
        finally:
            release.set()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["deleted", "epoch"])
def test_raw_sse_lifecycle_change_preserves_bytes_without_persisting_new_epoch(change):
    async def run():
        state = _state()
        state.engine._engine_state.lifecycle_epoch = 1
        ctx = context("anthropic", state)
        events = list(response_events(_response("anthropic", text="complete answer"), "anthropic"))

        class ChangingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield events[0]
                if change == "deleted":
                    state.is_conversation_deleted.return_value = True
                else:
                    state.engine._engine_state.lifecycle_epoch = 2
                for event in events[1:]:
                    yield event

        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=ChangingStream(),
        ))) as client:
            response = await _handle_streaming(client, "http://upstream", {}, _body("anthropic"), "anthropic", state, request_context=ctx)
            outbound = b"".join([part async for part in response.body_iterator])
        assert outbound == b"".join(events)
        assert state.conversation_history == []
        state.fire_turn_complete.assert_not_called()
    asyncio.run(run())
