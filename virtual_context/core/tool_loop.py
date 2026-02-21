"""Core tool loop: VC paging tool definitions, execution, and synchronous tool loop.

Extracted from proxy/server.py so that the engine, benchmark, and CLI
can use VC paging tools without going through the HTTP proxy.

Supports multiple LLM providers via the ProviderAdapter pattern:
- AnthropicAdapter: Anthropic Messages API
- OpenAIAdapter: OpenAI Chat Completions API
- GeminiAdapter: Google Gemini API
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

from ..types import ToolCallRecord, ToolLoopResult

if TYPE_CHECKING:
    from ..engine import VirtualContextEngine

logger = logging.getLogger(__name__)

# Regex to match an existing <virtual-context> block (including its content)
_VC_BLOCK_RE = re.compile(
    r"<virtual-context>\n.*?\n</virtual-context>", re.DOTALL,
)

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

VC_TOOL_NAMES: frozenset[str] = frozenset({
    "vc_expand_topic",
    "vc_collapse_topic",
    "vc_find_quote",
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
        else:
            result = {"error": f"unknown VC tool: {name}"}
        return json.dumps(result)
    except Exception as e:
        logger.error("VC tool %s raised %s: %s", name, type(e).__name__, e, exc_info=True)
        return json.dumps({"is_error": True, "content": str(e)})


# ---------------------------------------------------------------------------
# Provider Adapter ABC
# ---------------------------------------------------------------------------

class ProviderAdapter(ABC):
    """Abstracts provider-specific API format details for the tool loop.

    Each adapter encapsulates: headers, URL construction, request body
    format, response parsing, tool definition conversion, and continuation
    building for a specific LLM provider.
    """

    def __init__(self, api_key: str, api_url: str = ""):
        self.api_key = api_key
        self._base_url = api_url

    @abstractmethod
    def get_headers(self) -> dict:
        """Return HTTP headers for API requests."""

    @abstractmethod
    def get_url(self, model: str = "") -> str:
        """Return the API endpoint URL."""

    @abstractmethod
    def build_request_body(
        self, *, model: str, messages: list[dict], system: str,
        max_tokens: int, temperature: float, tools: list[dict] | None,
    ) -> dict:
        """Build the provider-specific request body."""

    @abstractmethod
    def convert_tool_defs(self, anthropic_defs: list[dict]) -> list[dict]:
        """Convert Anthropic-format tool definitions to provider format."""

    @abstractmethod
    def extract_text(self, response: dict) -> str:
        """Extract text content from a response."""

    @abstractmethod
    def extract_tool_calls(self, response: dict) -> list[dict]:
        """Extract tool calls, normalized to ``[{id, name, input}]``."""

    @abstractmethod
    def extract_usage(self, response: dict) -> tuple[int, int]:
        """Return ``(input_tokens, output_tokens)`` from a response."""

    @abstractmethod
    def is_tool_use_stop(self, response: dict) -> bool:
        """Return True if the response stopped due to tool use."""

    @abstractmethod
    def get_stop_reason(self, response: dict) -> str:
        """Return a normalized stop reason string."""

    @abstractmethod
    def build_tool_result(
        self, tool_call_id: str, tool_name: str, content: str,
    ) -> dict:
        """Build a single provider-specific tool result entry."""

    @abstractmethod
    def build_continuation(
        self, cont_body: dict | None, original_body: dict,
        raw_response: dict, tool_results: list[dict],
    ) -> dict:
        """Build or update the continuation request body.

        If *cont_body* is None, creates a fresh body from *original_body*.
        Otherwise appends assistant response + tool results to *cont_body*.
        Returns the (possibly new) continuation body.
        """

    @abstractmethod
    def inject_context(self, body: dict, prepend_text: str) -> None:
        """Replace the <virtual-context> block in *body*'s system prompt (in-place).

        If an existing ``<virtual-context>...</virtual-context>`` block is
        found, it is replaced.  Otherwise the new block is prepended.
        """

    @abstractmethod
    def strip_tools(self, body: dict) -> None:
        """Remove tool definitions from a request body (in-place)."""


# ---------------------------------------------------------------------------
# Anthropic Adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str, api_url: str = ""):
        super().__init__(api_key, api_url or "https://api.anthropic.com/v1/messages")

    def get_headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def get_url(self, model: str = "") -> str:
        return self._base_url

    def build_request_body(self, *, model, messages, system, max_tokens,
                           temperature, tools):
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "messages": list(messages),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        return body

    def convert_tool_defs(self, anthropic_defs):
        return anthropic_defs  # already canonical format

    def extract_text(self, response):
        text = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return text

    def extract_tool_calls(self, response):
        calls = []
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "input": block.get("input", {}),
                })
        return calls

    def extract_usage(self, response):
        usage = response.get("usage", {})
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def is_tool_use_stop(self, response):
        return response.get("stop_reason") == "tool_use"

    def get_stop_reason(self, response):
        return response.get("stop_reason", "end_turn")

    def build_tool_result(self, tool_call_id, tool_name, content):
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }

    def build_continuation(self, cont_body, original_body, raw_response,
                           tool_results):
        content = raw_response.get("content", [])
        if cont_body is None:
            body: dict = {
                "model": original_body.get("model"),
                "max_tokens": original_body.get("max_tokens", 4096),
                "stream": False,
                "messages": list(original_body.get("messages", [])),
            }
            if "system" in original_body:
                body["system"] = original_body["system"]
            if "tools" in original_body:
                body["tools"] = original_body["tools"]
            body["messages"].append({"role": "assistant", "content": content})
            body["messages"].append({"role": "user", "content": tool_results})
            return body
        else:
            cont_body["messages"].append({"role": "assistant", "content": content})
            cont_body["messages"].append({"role": "user", "content": tool_results})
            return cont_body

    def inject_context(self, body, prepend_text):
        new_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        system = body.get("system", "")
        if isinstance(system, str):
            if _VC_BLOCK_RE.search(system):
                body["system"] = _VC_BLOCK_RE.sub(new_block, system, count=1)
            else:
                body["system"] = f"{new_block}\n\n{system}" if system else new_block
        elif isinstance(system, list):
            replaced = False
            for entry in system:
                if isinstance(entry, dict) and entry.get("type") == "text":
                    text = entry.get("text", "")
                    if _VC_BLOCK_RE.search(text):
                        entry["text"] = _VC_BLOCK_RE.sub(new_block, text, count=1)
                        replaced = True
                        break
            if not replaced:
                system.insert(0, {"type": "text", "text": new_block})

    def strip_tools(self, body):
        body.pop("tools", None)


# ---------------------------------------------------------------------------
# OpenAI Adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(ProviderAdapter):
    """Adapter for the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, api_url: str = ""):
        super().__init__(
            api_key, api_url or "https://api.openai.com/v1/chat/completions",
        )

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_url(self, model: str = "") -> str:
        return self._base_url

    def build_request_body(self, *, model, messages, system, max_tokens,
                           temperature, tools):
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        body: dict = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "messages": oai_messages,
        }
        if tools:
            body["tools"] = tools
        return body

    def convert_tool_defs(self, anthropic_defs):
        """Convert Anthropic tool defs to OpenAI function-calling format."""
        oai_tools = []
        for tool in anthropic_defs:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return oai_tools

    def extract_text(self, response):
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content") or ""

    def extract_tool_calls(self, response):
        choices = response.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        raw_calls = msg.get("tool_calls") or []
        calls = []
        for tc in raw_calls:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            calls.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": args,
            })
        return calls

    def extract_usage(self, response):
        usage = response.get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def is_tool_use_stop(self, response):
        choices = response.get("choices", [])
        if not choices:
            return False
        return choices[0].get("finish_reason") == "tool_calls"

    def get_stop_reason(self, response):
        choices = response.get("choices", [])
        if not choices:
            return "error"
        reason = choices[0].get("finish_reason", "stop")
        # Normalize to Anthropic-like stop reasons
        if reason == "stop":
            return "end_turn"
        if reason == "tool_calls":
            return "tool_use"
        return reason

    def build_tool_result(self, tool_call_id, tool_name, content):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def build_continuation(self, cont_body, original_body, raw_response,
                           tool_results):
        choices = raw_response.get("choices", [])
        assistant_msg = (
            choices[0]["message"] if choices
            else {"role": "assistant", "content": ""}
        )

        if cont_body is None:
            body: dict = {
                "model": original_body.get("model"),
                "max_completion_tokens": original_body.get(
                    "max_completion_tokens", 4096,
                ),
                "stream": False,
                "messages": list(original_body.get("messages", [])),
            }
            if "tools" in original_body:
                body["tools"] = original_body["tools"]
            if "temperature" in original_body:
                body["temperature"] = original_body["temperature"]
        else:
            body = cont_body

        body["messages"].append(assistant_msg)
        body["messages"].extend(tool_results)
        return body

    def inject_context(self, body, prepend_text):
        new_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if isinstance(content, str) and _VC_BLOCK_RE.search(content):
                messages[0]["content"] = _VC_BLOCK_RE.sub(
                    new_block, content, count=1,
                )
            elif isinstance(content, str):
                messages[0]["content"] = (
                    f"{new_block}\n\n{content}" if content else new_block
                )
        else:
            messages.insert(0, {"role": "system", "content": new_block})

    def strip_tools(self, body):
        body.pop("tools", None)


# ---------------------------------------------------------------------------
# Gemini Adapter
# ---------------------------------------------------------------------------

class GeminiAdapter(ProviderAdapter):
    """Adapter for the Google Gemini API."""

    _DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, api_url: str = ""):
        super().__init__(api_key, api_url)

    def get_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def get_url(self, model: str = "") -> str:
        if self._base_url:
            return self._base_url
        return (
            f"{self._DEFAULT_BASE}/{model}:generateContent"
            f"?key={self.api_key}"
        )

    def build_request_body(self, *, model, messages, system, max_tokens,
                           temperature, tools):
        contents: list[dict] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            raw_content = msg.get("content", "")
            if isinstance(raw_content, str):
                parts = [{"text": raw_content}]
            elif isinstance(raw_content, list):
                parts = []
                for block in raw_content:
                    if isinstance(block, str):
                        parts.append({"text": block})
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                if not parts:
                    parts = [{"text": ""}]
            else:
                parts = [{"text": str(raw_content)}]
            contents.append({"role": role, "parts": parts})

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = tools
        return body

    def convert_tool_defs(self, anthropic_defs):
        """Convert Anthropic tool defs to Gemini functionDeclarations."""
        declarations = []
        for tool in anthropic_defs:
            declarations.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            })
        return [{"functionDeclarations": declarations}]

    def extract_text(self, response):
        candidates = response.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if "text" in p)

    def extract_tool_calls(self, response):
        candidates = response.get("candidates", [])
        if not candidates:
            return []
        parts = candidates[0].get("content", {}).get("parts", [])
        calls = []
        for part in parts:
            fc = part.get("functionCall")
            if fc:
                calls.append({
                    "id": str(uuid.uuid4()),
                    "name": fc["name"],
                    "input": fc.get("args", {}),
                })
        return calls

    def extract_usage(self, response):
        usage = response.get("usageMetadata", {})
        return (
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
        )

    def is_tool_use_stop(self, response):
        candidates = response.get("candidates", [])
        if not candidates:
            return False
        parts = candidates[0].get("content", {}).get("parts", [])
        return any("functionCall" in p for p in parts)

    def get_stop_reason(self, response):
        if self.is_tool_use_stop(response):
            return "tool_use"
        candidates = response.get("candidates", [])
        if not candidates:
            return "error"
        reason = candidates[0].get("finishReason", "STOP")
        if reason == "STOP":
            return "end_turn"
        return reason.lower()

    def build_tool_result(self, tool_call_id, tool_name, content):
        return {
            "functionResponse": {
                "name": tool_name,
                "response": {"content": content},
            },
        }

    def build_continuation(self, cont_body, original_body, raw_response,
                           tool_results):
        candidates = raw_response.get("candidates", [])
        model_parts = (
            candidates[0].get("content", {}).get("parts", [])
            if candidates else []
        )

        if cont_body is None:
            body = copy.deepcopy(original_body)
        else:
            body = cont_body

        body["contents"].append({"role": "model", "parts": model_parts})
        body["contents"].append({"role": "user", "parts": tool_results})
        return body

    def inject_context(self, body, prepend_text):
        new_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        si = body.get("system_instruction")
        if isinstance(si, dict):
            parts = si.get("parts", [])
            replaced = False
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    if _VC_BLOCK_RE.search(part["text"]):
                        part["text"] = _VC_BLOCK_RE.sub(
                            new_block, part["text"], count=1,
                        )
                        replaced = True
                        break
            if not replaced:
                parts.insert(0, {"text": new_block})
        else:
            body["system_instruction"] = {"parts": [{"text": new_block}]}

    def strip_tools(self, body):
        body.pop("tools", None)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def get_adapter(
    provider: str, api_key: str, api_url: str = "",
) -> ProviderAdapter:
    """Create a ProviderAdapter for the given provider name.

    Parameters
    ----------
    provider : str
        One of ``"anthropic"``, ``"openai"``, ``"gemini"``.
    api_key : str
        API key for the provider.
    api_url : str
        Override for the API endpoint URL.
    """
    if provider == "anthropic":
        return AnthropicAdapter(api_key, api_url)
    elif provider == "openai":
        return OpenAIAdapter(api_key, api_url)
    elif provider == "gemini":
        return GeminiAdapter(api_key, api_url)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Use 'anthropic', 'openai', or 'gemini'."
        )


# ---------------------------------------------------------------------------
# Synchronous tool loop
# ---------------------------------------------------------------------------

_MAX_LOOPS = 5


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

            cont_data = resp.json()
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
                    forced_data = resp.json()
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
