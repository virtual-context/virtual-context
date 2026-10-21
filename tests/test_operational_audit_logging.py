"""Deletions, client compactions and forwarded provider failures leave a log line."""
import asyncio
import json
import logging

import httpx
import pytest

from virtual_context.proxy.handlers import _handle_streaming, upstream_request_kind
from virtual_context.storage.sqlite import SQLiteStore

pytestmark = pytest.mark.regression("BUG-142")

COMPACTION = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary."


def _responses_body(text):
    return {"model": "gpt-test", "stream": True,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]}


def test_request_kind_names_client_compactions():
    assert upstream_request_kind(_responses_body(COMPACTION)) == "compaction"
    assert upstream_request_kind({"messages": [{"role": "user", "content": COMPACTION}]}) == "compaction"
    assert upstream_request_kind(_responses_body("hello")) == "turn"
    assert upstream_request_kind({"input": "plain"}) == "turn"


def test_deletion_logs_full_conversation_id_and_caller(tmp_path, caplog):
    store = SQLiteStore(str(tmp_path / "store.db"))
    conv = "sk:agent:example:discord:channel:123456789012345678"
    with caplog.at_level(logging.WARNING, logger="virtual_context.deletion_audit"):
        generation = store.begin_conversation_deletion(conv)
    lines = [r.getMessage() for r in caplog.records if "CONVERSATION_DELETE_BEGIN" in r.getMessage()]
    assert len(lines) == 1
    assert f"conv={conv} " in lines[0]
    assert f"generation={generation} " in lines[0]
    assert "test_operational_audit_logging.py" in lines[0]


def test_forwarded_stream_logs_provider_error_and_compaction(caplog):
    error = {"type": "error", "error": {"message": "response protection is unavailable", "type": "internal_error"}}
    raw = f"event: error\ndata: {json.dumps(error)}\n\n".encode()

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await _handle_streaming(client, "https://upstream.test/responses", {}, _responses_body(COMPACTION),
                                           "openai_responses", None, conversation_id="conv-xyz", turn=3, passthrough=True)
            return b"".join([chunk async for chunk in resp.body_iterator])

    with caplog.at_level(logging.INFO, logger="virtual_context.proxy.handlers"):
        forwarded = asyncio.run(run())
    assert forwarded == raw
    messages = [r.getMessage() for r in caplog.records]
    assert any("UPSTREAM_REQUEST kind=compaction conv=conv-xyz turn=3" in m for m in messages)
    errors = [m for m in messages if m.startswith("PROVIDER_STREAM_ERROR path=forwarded")]
    assert len(errors) == 1
    assert "kind=compaction" in errors[0] and "conv=conv-xyz" in errors[0]
    assert "response protection is unavailable" in errors[0]
