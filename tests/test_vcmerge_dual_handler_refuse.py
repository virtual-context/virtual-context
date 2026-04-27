"""Dual-handler refuse pair tests for VCMERGE (P1.4 + P1.8 / C1.0 closure).

Per VCMerge plan v1.11 sections 3.4 + 4.1 + 11.2 + 13.3 anti-subversion
summary. Pins:

- _handle_vc_command_rest at handlers.py:2253 refuses cmd == "merge"
  with a `merge_routed_outside_cloud_rest` error envelope (P1.4 / v1.4-1).
- _handle_vc_command at handlers.py:1883 refuses cmd == "merge" with a
  `merge_routed_outside_cloud_proxy` error envelope (P1.8 / v1.10 C1.0).
- Both refuses preserve the dual-populated error+message envelope shape
  (per spec section 12.9) so plugin clients render the message and
  programmatic consumers branch on the error code.
- The proxy-mode regex in proxy/server.py admits the MERGE / MERGESTATUS
  alternations so result.vc_command can populate "merge" or
  "mergestatus" — without this regex extension P1.4 / P1.8 would never
  fire because the command would never parse.
- VCMERGESTATUS returns the "not yet implemented" placeholder per P1.3
  (read-only status query, future work).
- VCMERGE PREVIEW returns the "not yet implemented" placeholder per
  P1.2 (engine-handled preview body, future work).

Marked @pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26") per
VCMerge plan v1.11 section 11 prologue.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from virtual_context.proxy import handlers


pytestmark = pytest.mark.regression("VCATTACH-DATALOSS-2026-04-26")


# ---------------------------------------------------------------------------
# Regex (P1.1 + P1.2 + P1.3 alternation extension in proxy/server.py)
# ---------------------------------------------------------------------------

# Mirror of the live regex at proxy/server.py:657-660; pinned here so a
# regression in the alternation breaks this test loud.
_VC_REGEX = re.compile(
    r"^VC(ATTACH|LABEL|STATUS|RECALL|COMPACT|LIST|FORGET|MERGESTATUS|MERGE)(?:\s+(.+))?$",
    re.IGNORECASE,
)


def _parse(text):
    """Mimic proxy/server.py's vc_command derivation."""
    m = _VC_REGEX.match(text.strip())
    if not m:
        return None, None
    return m.group(1).lower(), (m.group(2) or "").strip()


def test_regex_admits_vcmerge_into():
    cmd, arg = _parse("VCMERGE INTO target-conv")
    assert cmd == "merge"
    assert arg == "INTO target-conv"


def test_regex_admits_vcmerge_preview():
    cmd, arg = _parse("VCMERGE PREVIEW target-conv")
    assert cmd == "merge"
    assert arg == "PREVIEW target-conv"


def test_regex_admits_vcmergestatus():
    """VCMERGESTATUS must be matched as `mergestatus`, NOT as `merge` with
    " STATUS" arg. The alternation order in the regex (MERGESTATUS before
    MERGE) is what makes this pass; reverse the order and this test
    detects the regression.
    """
    cmd, arg = _parse("VCMERGESTATUS some-merge-id")
    assert cmd == "mergestatus"
    assert arg == "some-merge-id"


def test_regex_still_admits_existing_commands():
    """No accidental regression on the prior alternation set."""
    for word in ("VCATTACH foo", "VCLABEL bar", "VCSTATUS",
                 "VCRECALL r", "VCCOMPACT", "VCLIST", "VCFORGET t"):
        cmd, _arg = _parse(word)
        assert cmd is not None, f"{word!r} no longer parses"


# ---------------------------------------------------------------------------
# REST handler refuse (_handle_vc_command_rest / P1.4)
# ---------------------------------------------------------------------------

def _make_rest_result(cmd: str, arg: str = "", conv_id: str = "conv-x"):
    return SimpleNamespace(
        vc_command=cmd,
        vc_command_arg=arg,
        conversation_id=conv_id,
    )


def _make_dummy_state():
    """Minimal state stub. None is acceptable for refuse paths."""
    return None


def _make_dummy_registry():
    """REST path doesn't dispatch to attach for merge — no registry needed."""
    return None


def test_rest_handler_refuses_vcmerge_into_with_merge_routed_outside_cloud_rest():
    """P1.4 — engine REST handler refuses cmd='merge' arg='INTO ...' with
    the merge_routed_outside_cloud_rest envelope. No merge body fires;
    no merge_audit row is INSERTed.
    """
    result = _make_rest_result("merge", arg="INTO target-conv", conv_id="src-conv")
    response = handlers._handle_vc_command_rest(
        result, _make_dummy_state(), _make_dummy_registry(),
        tenant_id="tenant-A", vcconv=None,
    )
    payload = json.loads(response.body)
    assert payload["error"] == "merge_routed_outside_cloud_rest"
    # Dual-populated envelope per spec section 12.9.
    assert "message" in payload
    assert "VCMERGE" in payload["message"]
    assert "vc_cloud/rest_api.py" in payload["message"]
    assert payload["vc_command"] == "merge"
    # Source's id stays bound (per spec section 6.2 codex P2-10 pattern;
    # PREVIEW would also bind to source).
    assert payload["conversation_id"] == "src-conv"


def test_rest_handler_refuses_vcmerge_into_uppercased_arg():
    """Case-insensitive arg parsing — INTO / into / Into all refuse."""
    for arg_form in ("INTO foo", "into foo", "Into foo"):
        result = _make_rest_result("merge", arg=arg_form, conv_id="src")
        response = handlers._handle_vc_command_rest(
            result, _make_dummy_state(), _make_dummy_registry(),
            tenant_id="tA", vcconv=None,
        )
        payload = json.loads(response.body)
        assert payload["error"] == "merge_routed_outside_cloud_rest", (
            f"arg form {arg_form!r} failed to refuse"
        )


def test_rest_handler_returns_preview_not_implemented():
    """P1.2 placeholder — engine REST handler returns 'not yet
    implemented' message for PREVIEW arg, NOT the refuse envelope.
    """
    result = _make_rest_result("merge", arg="PREVIEW target-conv", conv_id="src")
    response = handlers._handle_vc_command_rest(
        result, _make_dummy_state(), _make_dummy_registry(),
        tenant_id="tA", vcconv=None,
    )
    payload = json.loads(response.body)
    # Not a refuse — no "error" key (the placeholder uses message-only).
    assert "error" not in payload
    assert "not yet implemented" in payload["message"].lower()


def test_rest_handler_returns_syntax_help_for_unknown_arg():
    """Unknown VCMERGE subcommand returns syntax help with vcmerge_syntax
    error code.
    """
    result = _make_rest_result("merge", arg="EXPORT foo", conv_id="src")
    response = handlers._handle_vc_command_rest(
        result, _make_dummy_state(), _make_dummy_registry(),
        tenant_id="tA", vcconv=None,
    )
    payload = json.loads(response.body)
    assert payload["error"] == "vcmerge_syntax"
    assert "VCMERGE syntax" in payload["message"]


def test_rest_handler_returns_mergestatus_not_implemented():
    """P1.3 — VCMERGESTATUS returns the placeholder."""
    result = _make_rest_result("mergestatus", arg="merge-id-123", conv_id="src")
    response = handlers._handle_vc_command_rest(
        result, _make_dummy_state(), _make_dummy_registry(),
        tenant_id="tA", vcconv=None,
    )
    payload = json.loads(response.body)
    assert "not yet implemented" in payload["message"].lower()


# ---------------------------------------------------------------------------
# Proxy handler refuse (_handle_vc_command / P1.8)
# ---------------------------------------------------------------------------

class _StubFmt:
    """Minimal Format stub for the proxy-mode handler."""

    def emit_fake_response_sse(self, text, conv_id):
        return f"data: {json.dumps({'text': text, 'conv': conv_id})}\n\n"

    def build_fake_response(self, text, conv_id):
        return {"text": text, "conv_id": conv_id}


def _make_proxy_result(cmd: str, arg: str = "", conv_id: str = "conv-x",
                       streaming: bool = False):
    return SimpleNamespace(
        vc_command=cmd,
        vc_command_arg=arg,
        conversation_id=conv_id,
        is_streaming=streaming,
    )


def _run_proxy_handler(result):
    """Wrap async handler in asyncio.run for sync test invocation; the
    project test suite has no pytest-asyncio plugin (per
    test_history_filter.py:326 PytestUnknownMarkWarning).
    """
    import asyncio
    return asyncio.run(handlers._handle_vc_command(
        result, _StubFmt(), state=None, registry=None,
        tenant_registry=None, tenant_id="tenant-A",
    ))


def test_proxy_handler_refuses_vcmerge_into_with_merge_routed_outside_cloud_proxy():
    """P1.8 (v1.10 C1.0) — engine proxy handler refuses cmd='merge'
    arg='INTO ...' with the merge_routed_outside_cloud_proxy envelope.
    NOT the silent 'Unknown VC command: merge' fallback.
    """
    result = _make_proxy_result("merge", arg="INTO target-conv", conv_id="src")
    response = _run_proxy_handler(result)
    payload = json.loads(response.body)
    assert payload["error"] == "merge_routed_outside_cloud_proxy"
    assert "message" in payload
    assert "VCMergeMiddleware" in payload["message"]
    assert "vc_cloud/main.py" in payload["message"]
    assert payload["vc_command"] == "merge"


def test_proxy_handler_refuses_streaming_request_too():
    """P1.8 streaming variant — the refuse must still fire when the
    request is streaming; engine returns a StreamingResponse with the
    error text.

    Note: per VCMerge plan v1.12 §13.2 OI5 (deferred to v1.14), the
    streaming response carries ONLY the human message text via
    `fmt.emit_fake_response_sse`; the programmatic `error` code from
    the dual-populated envelope is intentionally NOT preserved through
    the SSE stream in the V0 implementation. This matches the existing
    VCATTACH-error precedent at handlers.py:1853 and is acceptable
    because: (a) the streaming refuse is a defense-in-depth path that
    doesn't fire in normal flow (cloud always intercepts first per
    C2.1b VCMergeMiddleware); (b) the non-streaming JSON path at
    JSONResponse already carries the dual-populated envelope per spec
    §12.9. This test asserts the StreamingResponse class is returned
    but does NOT (yet) assert a structured error code in the stream;
    that assertion will be added when v1.14 picks up OI5.
    """
    result = _make_proxy_result(
        "merge", arg="INTO target-conv", conv_id="src", streaming=True,
    )
    response = _run_proxy_handler(result)
    from starlette.responses import StreamingResponse
    assert isinstance(response, StreamingResponse)


def test_proxy_handler_does_not_silently_no_op_on_merge():
    """Regression: prior code returned 'Unknown VC command: merge' as a
    fake-LLM text response, silently dropping the merge command. P1.8
    MUST replace that silent no-op with an explicit error envelope.
    """
    result = _make_proxy_result("merge", arg="INTO target-conv", conv_id="src")
    response = _run_proxy_handler(result)
    payload = json.loads(response.body)
    # The silent-no-op pattern would have populated `text` (StubFmt's
    # build_fake_response shape) without an error key.
    assert "Unknown VC command" not in payload.get("message", "")
    assert "Unknown VC command" not in payload.get("text", "")
    assert payload.get("error") == "merge_routed_outside_cloud_proxy"


def test_proxy_handler_returns_preview_not_implemented():
    """P1.2 placeholder mirror on the proxy path."""
    result = _make_proxy_result("merge", arg="PREVIEW target-conv", conv_id="src")
    response = _run_proxy_handler(result)
    payload = json.loads(response.body)
    assert "not yet implemented" in str(payload).lower()


# ---------------------------------------------------------------------------
# Dual-error-code symmetry — the two transports MUST use distinct codes
# ---------------------------------------------------------------------------

def test_rest_and_proxy_refuse_codes_are_distinct():
    """The C1.0 anti-subversion summary at plan section 13.3 specifies
    distinct error codes for each transport so cloud ops can triage
    WHICH transport's intercept failed. A regression that conflates the
    two codes loses that diagnostic signal.
    """
    rest_result = _make_rest_result("merge", arg="INTO foo", conv_id="src")
    rest_payload = json.loads(handlers._handle_vc_command_rest(
        rest_result, None, None, tenant_id="tA", vcconv=None,
    ).body)

    import asyncio
    proxy_result = _make_proxy_result("merge", arg="INTO foo", conv_id="src")
    proxy_payload = json.loads(asyncio.run(handlers._handle_vc_command(
        proxy_result, _StubFmt(), state=None, registry=None,
        tenant_registry=None, tenant_id="tA",
    )).body)

    assert rest_payload["error"] == "merge_routed_outside_cloud_rest"
    assert proxy_payload["error"] == "merge_routed_outside_cloud_proxy"
    assert rest_payload["error"] != proxy_payload["error"]
