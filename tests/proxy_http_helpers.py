"""Offline upstream transport fixtures for the shared JSON/SSE continuation path."""

from contextlib import contextmanager
import json
from unittest.mock import patch

import httpx


@contextmanager
def upstream_rounds(initial, continuation):
    """Route real built requests through one initial SSE and JSON continuations."""
    requests = []

    class InitialStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for chunk in initial.aiter_bytes():
                yield chunk

    async def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(initial.status_code, headers=dict(initial.headers), stream=InitialStream())
        response = await continuation(str(request.url), json=json.loads(request.content))
        return httpx.Response(response.status_code, json=response.json())

    with patch('httpx.AsyncClient._transport_for_url', return_value=httpx.MockTransport(respond)):
        yield requests
