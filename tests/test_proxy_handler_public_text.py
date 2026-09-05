"""Exercise completion extraction through real JSON and SSE proxy handlers."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from test_proxy_durable_continuation import context, mixed_response, next_request
from test_proxy_review_invariants import _body, _response, _state, _Chunks
from virtual_context.proxy.continuation import _prepend_public_text
from virtual_context.proxy.handlers import _continuation_session, _handle_non_streaming, _handle_streaming
from virtual_context.proxy.response_codec import collect_response, response_events
from virtual_context.storage.sqlite import SQLiteStore


def provider_response(fmt, text, *, tool=None, mixed=False, reasoning=False):
    response = mixed_response(fmt) if mixed else _response(fmt, tool=tool, text=text)
    native = _response('openai_responses', text=text)['output']
    native[0]['id'] = 'msg_' + text.replace(' ', '_')
    if mixed:
        response = _prepend_public_text(response, fmt, [text], native_messages=native)
    elif fmt == 'openai_responses':
        response['output'][0]['id'] = native[0]['id']
    # Provider text parts inside one round must concatenate without inserting
    # paragraph separators between arbitrary deltas or content blocks.
    if fmt == 'anthropic':
        parts = response['content']
    elif fmt == 'gemini':
        parts = response['candidates'][0]['content']['parts']
    elif fmt == 'openai_responses':
        parts = next(item['content'] for item in response['output'] if item['type'] == 'message')
    else:
        parts = []
    for index, part in reversed(list(enumerate(parts))):
        if part.get('text') == text:
            midpoint = len(text) // 2
            parts[index:index + 1] = [{**part, 'text': text[:midpoint]}, {**part, 'text': text[midpoint:]}]
    if reasoning:
        if fmt == 'anthropic':
            response['content'][:0] = [{'type': 'thinking', 'thinking': 'opaque reasoning', 'signature': 'native-signature'}]
        elif fmt == 'openai_responses':
            response['output'][:0] = [{'type': 'reasoning', 'id': 'rs_native', 'summary': [], 'encrypted_content': 'opaque-encrypted-proof'}]
        elif fmt == 'gemini':
            response['candidates'][0]['content']['parts'][:0] = [{'thought': True, 'text': 'opaque reasoning', 'thoughtSignature': 'native-signature'}]
    return response


@pytest.mark.parametrize('fmt', ['anthropic', 'openai', 'openai_responses', 'gemini'])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_handlers_persist_every_public_fragment_once_after_resume_and_new_internal_round(tmp_path, fmt, streaming, mixed):
    async def run():
        store = SQLiteStore(tmp_path / 'handler.db')
        try:
            state = _state()
            state.engine._store = store
            replies = [provider_response(fmt, 'before lookup', tool='vc_find_quote', reasoning=True)]
            if mixed:
                replies += [provider_response(fmt, 'client handoff', mixed=True), provider_response(fmt, 'after resume', tool='vc_find_quote', reasoning=True)]
            replies += [provider_response(fmt, 'final answer', reasoning=True)]
            original_final = replies[-1]
            sent = []

            def provider(request):
                sent.append(json.loads(request.content))
                reply = replies.pop(0)
                if streaming:
                    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=_Chunks(list(response_events(reply, fmt))))
                return httpx.Response(200, json=reply)

            handler = _handle_streaming if streaming else _handle_non_streaming
            options = {'paging_enabled': True} if streaming else {'intercept_vc_tools': True}

            async def read(response):
                assert response.status_code == 200
                if streaming:
                    raw = b''.join([chunk async for chunk in response.body_iterator])
                    return await collect_response(httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=_Chunks([raw])), fmt)
                return json.loads(response.body)

            with patch('virtual_context.proxy.handlers.execute_vc_tool', return_value='hidden memory result'):
                async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                    visible = await read(await handler(client, 'http://upstream', {}, _body(fmt), fmt, state, request_context=context(fmt, state), skip_marker_injection=True, **options))
                    if mixed:
                        assert _continuation_session(context(fmt, state), state).public_text(visible) == 'before lookup\n\nclient handoff'
                        state.fire_turn_complete.assert_not_called()
                        successor = _state()
                        successor.engine._store = store
                        session = _continuation_session(context(fmt, successor), successor)
                        restored = await session.resume(next_request(fmt, visible))
                        visible = await read(await handler(client, 'http://upstream', {}, restored, fmt, successor, request_context=session.context, continuation_session=session, skip_marker_injection=True, **options))
                        state = successor
                        assert 'before lookup' not in json.dumps(visible)
                        assert 'client handoff' not in json.dumps(visible)
            assert not replies
            saved = state.fire_turn_complete.call_args.args[0][-1]
            expected = ['before lookup', *(['client handoff', 'after resume'] if mixed else []), 'final answer']
            assert saved.content == '\n\n'.join(expected)
            expected_visible = 'after resume\n\nfinal answer' if mixed else 'before lookup\n\nfinal answer'
            assert _continuation_session(context(fmt, state), state).public_text(visible) == expected_visible
            assert 'hidden memory result' not in saved.content
            assert 'opaque reasoning' not in saved.content
            assert 'vc_find_quote' not in json.dumps(visible)
            assert 'hidden memory result' not in json.dumps(visible)
            if fmt == 'anthropic':
                assert visible['content'][0] == original_final['content'][0]
                assert saved.raw_content[0] == original_final['content'][0]
                raw_text = ''.join(block.get('text', '') for block in saved.raw_content if block['type'] == 'text')
                assert raw_text == saved.content
            elif fmt == 'openai_responses':
                assert visible['output'][0] == original_final['output'][0]
                message_ids = [item['id'] for item in visible['output'] if item['type'] == 'message']
                assert message_ids == ['msg_after_resume' if mixed else 'msg_before_lookup', 'msg_final_answer']
            elif fmt == 'gemini':
                assert visible['candidates'][0]['content']['parts'][0] == original_final['candidates'][0]['content']['parts'][0]
            # The provider continuation receives its signed original blocks,
            # while completion stores only public prose as canonical text.
            assert 'hidden memory result' in json.dumps(sent[-1])
        finally:
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize('fmt', ['anthropic', 'openai', 'openai_responses', 'gemini'])
def test_multiple_internal_rounds_preserve_paragraph_boundaries_and_native_parts(fmt):
    async def run():
        state = _state()
        replies = [
            provider_response(fmt, 'before lookup', tool='vc_find_quote'),
            provider_response(fmt, 'second lookup', tool='vc_find_quote'),
            provider_response(fmt, 'final answer', reasoning=True),
        ]
        with patch('virtual_context.proxy.handlers.execute_vc_tool', return_value='hidden result'):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=replies.pop(0)))) as client:
                response = await _handle_non_streaming(client, 'http://upstream', {}, _body(fmt), fmt, state, request_context=context(fmt, state), intercept_vc_tools=True, skip_marker_injection=True)
        assert response.status_code == 200
        expected = 'before lookup\n\nsecond lookup\n\nfinal answer'
        assert state.fire_turn_complete.call_args.args[0][-1].content == expected
        assert _continuation_session(context(fmt, state), state).public_text(json.loads(response.body)) == expected
    asyncio.run(run())
