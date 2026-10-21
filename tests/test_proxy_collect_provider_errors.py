"""A provider that reports its own failure mid-stream is a gateway error, not a client error."""
import asyncio
import json

import httpx
import pytest

from virtual_context.proxy.continuation import ContinuationError
from virtual_context.proxy.response_codec import collect_response

pytestmark = pytest.mark.regression("BUG-141")


class _Resp:
    def __init__(self, chunks, headers=None, status=200):
        self._chunks = chunks
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})
        self.status_code = status
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _collect(events, api_format="openai_responses"):
    with pytest.raises(ContinuationError) as exc:
        asyncio.run(collect_response(_Resp([_sse(events)]), api_format))
    return exc.value


CREATED = {"type": "response.created", "response": {"id": "r", "status": "in_progress", "output": []}}
PROVIDER_ERROR = {"message": "response protection is unavailable", "type": "internal_error", "param": None, "code": None}


def test_provider_error_event_is_a_gateway_error_carrying_the_provider_message():
    exc = _collect([CREATED, {"type": "error", "error": PROVIDER_ERROR}])
    assert exc.status_code == 502
    assert "response protection is unavailable" in str(exc)


def test_error_field_on_any_event_is_a_gateway_error():
    exc = _collect([CREATED, {"type": "response.output_text.delta", "delta": "", "error": PROVIDER_ERROR}])
    assert exc.status_code == 502
    assert "response protection is unavailable" in str(exc)


def test_failed_response_is_a_gateway_error_carrying_the_failure():
    failed = {"type": "response.failed", "response": {"id": "r", "status": "failed", "output": [],
                                                       "error": {"code": "server_error", "message": "The model produced no output."}}}
    exc = _collect([CREATED, failed])
    assert exc.status_code == 502
    assert "The model produced no output." in str(exc)


def test_incomplete_response_stays_a_client_error():
    incomplete = {"type": "response.incomplete", "response": {"id": "r", "status": "incomplete", "output": [],
                                                               "incomplete_details": {"reason": "max_output_tokens"}}}
    exc = _collect([CREATED, incomplete])
    assert exc.status_code == 422
    assert "max_output_tokens" in str(exc)


def test_anthropic_error_event_is_a_gateway_error():
    events = [{"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}},
              {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]
    exc = _collect(events, "anthropic")
    assert exc.status_code == 502
    assert "Overloaded" in str(exc)
