"""Core tool loop: VC paging tool definitions, execution, and synchronous tool loop.

Extracted from proxy/server.py so that the engine, benchmark, and CLI
can use VC paging tools without going through the HTTP proxy.

Provider adapters live in provider_adapters.py and are re-exported here
for backward compatibility.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import TYPE_CHECKING

import httpx

from ..types import ToolCallRecord, ToolLoopResult

# Re-export adapter classes for backward compatibility
from .provider_adapters import (  # noqa: F401
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAICodexAdapter,
    ProviderAdapter,
    get_adapter,
)

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

VC_TOOL_NAMES: frozenset[str] = frozenset({
    "vc_expand_topic",
    "vc_collapse_topic",
    "vc_find_quote",
    "vc_recall_all",
    "vc_remember_when",
})


def vc_tool_definitions() -> list[dict]:
    """Return Anthropic-format tool definitions for VC context tools.

    This is the canonical format. Provider adapters convert as needed.
    """
    return [
        {
            "name": "vc_expand_topic",
            "description": (
                "Load the full original conversation text for a topic. "
                "Use when a topic summary covers the area you need — expanding "
                "reveals the complete conversation including details the summary "
                "may have compressed. Also use after vc_find_quote returns "
                "snippets — expand the matching tag to read surrounding context "
                "before answering. For specific facts when you don't know which "
                "topic holds them, use vc_find_quote first to locate them."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag from the context-topics list to expand.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["segments", "full"],
                        "description": (
                            "Target depth: 'segments' for individual summaries, "
                            "'full' for original conversation text."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_collapse_topic",
            "description": (
                "Collapse an expanded topic back to its summary to free context "
                "budget. Use after you've retrieved what you need from an expanded "
                "topic, or to make room before expanding a different one."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Topic tag to collapse.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["summary", "none"],
                        "description": (
                            "Target depth: 'summary' for brief overview, "
                            "'none' to remove from context entirely."
                        ),
                    },
                },
                "required": ["tag"],
            },
        },
        {
            "name": "vc_find_quote",
            "description": (
                "Search the full original conversation text for a specific word, "
                "phrase, or detail. Use this as your first tool when the user asks "
                "about a specific fact — a name, number, dosage, recommendation, "
                "date, or decision — especially when no topic summary mentions it "
                "or you don't know which topic it falls under. This bypasses tags "
                "entirely and searches raw text, so it finds content even when "
                "it's filed under an unexpected topic. Returns short excerpts — "
                "use vc_expand_topic on a matching tag if you need more context."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The word or phrase to search for. Use the most specific and "
                            "distinctive terms — e.g. 'magnesium glycinate' rather than "
                            "'supplement', or 'reservation 7pm' rather than 'dinner'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "vc_recall_all",
            "description": (
                "Load summaries of ALL stored conversation topics at once. "
                "Use when the user asks for a broad overview, wants to know "
                "everything discussed, needs a full summary, or asks a vague "
                "question that spans multiple topics. Returns all tag summaries "
                "within the token budget. After reviewing, use vc_expand_topic "
                "on specific tags if you need more detail."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "vc_remember_when",
            "description": (
                "Find memory by topic within a time window. "
                "Use for requests like 'last week', 'last month', "
                "'3 days ago', or 'between June and July'. "
                "Do not compute calendar dates yourself; prefer relative "
                "presets and let the backend resolve exact dates."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic/fact query to search for within a time window.",
                    },
                    "time_range": {
                        "type": "object",
                        "description": (
                            "Time window selector. One of: "
                            "{kind:'relative', preset:'last_24_hours|last_7_days|last_30_days|"
                            "this_week|last_week|this_month|last_month|this_year|last_year'} "
                            "or {kind:'between_dates', start:'YYYY-MM-DD or YYYY-MM', "
                            "end:'YYYY-MM-DD or YYYY-MM'}."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5).",
                    },
                },
                "required": ["query", "time_range"],
            },
        },
    ]


def is_vc_tool(name: str) -> bool:
    """Return True if *name* is a known VC paging tool."""
    return name in VC_TOOL_NAMES


def execute_vc_tool(
    engine: VirtualContextEngine, name: str, tool_input: dict,
) -> str:
    """Execute a VC paging tool and return a JSON result string."""
    try:
        if name == "vc_expand_topic":
            result = engine.expand_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "full"),
            )
        elif name == "vc_collapse_topic":
            result = engine.collapse_topic(
                tag=tool_input.get("tag", ""),
                depth=tool_input.get("depth", "summary"),
            )
        elif name == "vc_find_quote":
            result = engine.find_quote(
                query=tool_input.get("query", ""),
                max_results=tool_input.get("max_results", 5),
            )
        elif name == "vc_recall_all":
            result = engine.recall_all()
        elif name == "vc_remember_when":
            result = engine.remember_when(
                query=tool_input.get("query", ""),
                time_range=tool_input.get("time_range", {}),
                max_results=tool_input.get("max_results", 5),
            )
        else:
            result = {"error": f"unknown VC tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        logger.error("VC tool %s raised %s: %s", name, type(e).__name__, e, exc_info=True)
        return json.dumps({"is_error": True, "content": str(e)})


# ---------------------------------------------------------------------------
# Synchronous tool loop
# ---------------------------------------------------------------------------

_MAX_LOOPS = 5


def _parse_sse_response(text: str) -> dict:
    """Parse SSE text and return the final `response` object."""
    last_response: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload_raw = line[6:]
        if not payload_raw or payload_raw == "[DONE]":
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type", "")
        if event_type in {"response.created", "response.in_progress"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                last_response = resp
        elif event_type in {"response.completed", "response.failed"}:
            resp = payload.get("response")
            if isinstance(resp, dict):
                return resp
        elif event_type == "error":
            return {"error": payload.get("error", payload)}
    return last_response or {}


def _parse_provider_http_response(resp: httpx.Response) -> dict:
    """Parse provider HTTP response, supporting JSON and SSE payloads."""
    content_type = (resp.headers.get("content-type", "") or "").lower()
    if "text/event-stream" in content_type:
        return _parse_sse_response(resp.text)
    try:
        return resp.json()
    except json.JSONDecodeError:
        return _parse_sse_response(resp.text)


def run_tool_loop(
    engine: VirtualContextEngine,
    initial_response: dict,
    original_request: dict,
    adapter: ProviderAdapter,
    *,
    url: str = "",
    max_loops: int = _MAX_LOOPS,
) -> ToolLoopResult:
    """Run a synchronous non-streaming tool loop.

    Given an initial LLM response that contains VC tool calls,
    execute the tools, send continuation requests, and return the final
    text result.  Works with any provider via the adapter pattern.

    Parameters
    ----------
    engine : VirtualContextEngine
        Engine instance for tool execution and context reassembly.
    initial_response : dict
        The first LLM response body (parsed JSON).
    original_request : dict
        The original request body (for building continuations).
    adapter : ProviderAdapter
        Provider-specific adapter for format conversion.
    url : str
        API endpoint URL (if empty, uses ``adapter.get_url()``).
    max_loops : int
        Maximum continuation rounds (default 5).

    Returns
    -------
    ToolLoopResult
        Final text, tool call records, and usage metrics.
    """
    if not url:
        url = adapter.get_url()

    result = ToolLoopResult()
    result.raw_responses.append(initial_response)

    # Accumulate usage from initial response
    input_toks, output_toks = adapter.extract_usage(initial_response)
    result.input_tokens += input_toks
    result.output_tokens += output_toks

    # Collect text from the initial response
    result.text += adapter.extract_text(initial_response)

    stop_reason = adapter.get_stop_reason(initial_response)

    # Identify tool calls
    all_tool_calls = adapter.extract_tool_calls(initial_response)
    vc_tools = [tc for tc in all_tool_calls if is_vc_tool(tc["name"])]
    non_vc_tools = [tc for tc in all_tool_calls if not is_vc_tool(tc["name"])]

    # BAIL: no VC tools — just return text
    if not vc_tools:
        result.stop_reason = stop_reason
        return result

    # BAIL: mixed VC + non-VC tools
    if non_vc_tools:
        logger.warning("Mixed VC + non-VC tools in response — returning as-is")
        result.stop_reason = stop_reason
        return result

    headers = adapter.get_headers()
    cont_body: dict | None = None
    current_response = initial_response

    with httpx.Client(timeout=300.0) as client:
        for loop_i in range(max_loops):
            # Execute VC tools
            tool_results: list[dict] = []
            for tc in vc_tools:
                t0 = time.monotonic()
                result_str = execute_vc_tool(
                    engine, tc["name"], tc["input"],
                )
                duration_ms = round((time.monotonic() - t0) * 1000, 1)

                result.tool_calls.append(ToolCallRecord(
                    tool_name=tc["name"],
                    tool_input=tc["input"],
                    result_json=result_str,
                    duration_ms=duration_ms,
                ))

                tool_results.append(
                    adapter.build_tool_result(tc["id"], tc["name"], result_str)
                )

            # Re-assemble context with updated working set
            new_prepend = engine.reassemble_context()

            # Build continuation request
            cont_body = adapter.build_continuation(
                cont_body, original_request, current_response, tool_results,
            )

            # Inject updated context into system prompt
            if new_prepend:
                adapter.inject_context(cont_body, new_prepend)

            result.continuation_count += 1

            # Capture the continuation request body (deep copy to freeze state)
            result.raw_requests.append(copy.deepcopy(cont_body))

            # Send continuation
            resp = client.post(url, headers=headers, json=cont_body)
            if resp.status_code >= 300:
                logger.error(
                    "Continuation %d failed: HTTP %d",
                    result.continuation_count, resp.status_code,
                )
                result.stop_reason = "error"
                break

            cont_data = _parse_provider_http_response(resp)
            current_response = cont_data
            result.raw_responses.append(cont_data)

            # Accumulate usage
            input_toks, output_toks = adapter.extract_usage(cont_data)
            result.input_tokens += input_toks
            result.output_tokens += output_toks

            # Extract text
            result.text += adapter.extract_text(cont_data)

            stop_reason = adapter.get_stop_reason(cont_data)

            # Check for more VC tool calls
            all_tool_calls = adapter.extract_tool_calls(cont_data)
            vc_tools = [tc for tc in all_tool_calls if is_vc_tool(tc["name"])]
            non_vc_in_cont = [
                tc for tc in all_tool_calls if not is_vc_tool(tc["name"])
            ]

            if (adapter.is_tool_use_stop(cont_data)
                    and vc_tools and not non_vc_in_cont):
                continue  # loop again

            # Done
            result.stop_reason = stop_reason
            break
        else:
            # Exhausted max_loops — force a text answer if none produced
            if not result.text.strip():
                logger.warning(
                    "Tool loop exhausted %d iterations with no text — "
                    "sending forced text continuation",
                    max_loops,
                )
                # Execute the last batch of tool calls so we can include
                # their results in the forced continuation
                tool_results_final: list[dict] = []
                for tc in vc_tools:
                    t0 = time.monotonic()
                    result_str = execute_vc_tool(
                        engine, tc["name"], tc["input"],
                    )
                    duration_ms = round((time.monotonic() - t0) * 1000, 1)
                    result.tool_calls.append(ToolCallRecord(
                        tool_name=tc["name"],
                        tool_input=tc["input"],
                        result_json=result_str,
                        duration_ms=duration_ms,
                    ))
                    tool_results_final.append(
                        adapter.build_tool_result(
                            tc["id"], tc["name"], result_str,
                        )
                    )

                # Re-assemble context with updated working set
                forced_prepend = engine.reassemble_context()

                # Build continuation WITHOUT tools to force text output
                cont_body = adapter.build_continuation(
                    cont_body, original_request, current_response,
                    tool_results_final,
                )
                if forced_prepend:
                    adapter.inject_context(cont_body, forced_prepend)
                adapter.strip_tools(cont_body)

                result.continuation_count += 1
                result.raw_requests.append(copy.deepcopy(cont_body))
                resp = client.post(url, headers=headers, json=cont_body)
                if resp.status_code < 300:
                    forced_data = _parse_provider_http_response(resp)
                    result.raw_responses.append(forced_data)
                    fi, fo = adapter.extract_usage(forced_data)
                    result.input_tokens += fi
                    result.output_tokens += fo
                    result.text += adapter.extract_text(forced_data)
                    result.stop_reason = adapter.get_stop_reason(forced_data)
                else:
                    logger.error(
                        "Forced text continuation failed: HTTP %d",
                        resp.status_code,
                    )
                    result.stop_reason = "error"
            else:
                result.stop_reason = stop_reason

    return result
