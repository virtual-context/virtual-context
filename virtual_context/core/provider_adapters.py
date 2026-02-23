"""Provider adapters for the VC tool loop.

Each adapter encapsulates provider-specific API format details:
headers, URL construction, request body format, response parsing,
tool definition conversion, context injection, and continuation building.

Supports: Anthropic, OpenAI, OpenAI Codex (ChatGPT backend), Gemini.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from abc import ABC, abstractmethod

# Regex to match an existing <virtual-context> block (including its content)
_VC_BLOCK_RE = re.compile(
    r"<virtual-context>\n.*?\n</virtual-context>", re.DOTALL,
)


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
# OpenAI Codex Adapter (ChatGPT backend Responses API)
# ---------------------------------------------------------------------------

class OpenAICodexAdapter(ProviderAdapter):
    """Adapter for ChatGPT Codex Responses API (`chatgpt.com/backend-api`)."""

    def __init__(self, api_key: str, api_url: str = ""):
        super().__init__(
            api_key, api_url or "https://chatgpt.com/backend-api/codex/responses",
        )

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_url(self, model: str = "") -> str:
        return self._base_url

    def _to_input_item(self, msg: dict) -> dict:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block and isinstance(block["text"], str):
                        text_parts.append(block["text"])
            text = "\n".join(p for p in text_parts if p)
        else:
            text = str(content)
        return {
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        }

    def build_request_body(self, *, model, messages, system, max_tokens,
                           temperature, tools):
        body: dict = {
            "model": model,
            "instructions": system or "You are a helpful assistant.",
            "stream": True,
            "store": False,
            "input": [self._to_input_item(m) for m in messages],
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def convert_tool_defs(self, anthropic_defs):
        codex_tools = []
        for tool in anthropic_defs:
            codex_tools.append({
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            })
        return codex_tools

    def extract_text(self, response):
        output = response.get("output", [])
        text_parts: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
        return "".join(text_parts)

    def extract_tool_calls(self, response):
        calls = []
        for item in response.get("output", []):
            if item.get("type") != "function_call":
                continue
            args_raw = item.get("arguments", "")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({
                "id": item.get("call_id", item.get("id", str(uuid.uuid4()))),
                "name": item.get("name", ""),
                "input": args,
            })
        return calls

    def extract_usage(self, response):
        usage = response.get("usage", {})
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def is_tool_use_stop(self, response):
        return any(item.get("type") == "function_call"
                   for item in response.get("output", []))

    def get_stop_reason(self, response):
        if response.get("error"):
            return "error"
        if self.is_tool_use_stop(response):
            return "tool_use"
        return "end_turn"

    def build_tool_result(self, tool_call_id, tool_name, content):
        return {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": content,
        }

    def build_continuation(self, cont_body, original_body, raw_response,
                           tool_results):
        if cont_body is None:
            body = copy.deepcopy(original_body)
            body["input"] = list(original_body.get("input", []))
            body["stream"] = True
            body["store"] = False
        else:
            body = cont_body

        for item in raw_response.get("output", []):
            if item.get("type") == "function_call":
                body["input"].append({
                    "type": "function_call",
                    "call_id": item.get("call_id", item.get("id", "")),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                })

        body["input"].extend(tool_results)
        return body

    def inject_context(self, body, prepend_text):
        new_block = f"<virtual-context>\n{prepend_text}\n</virtual-context>"
        instructions = body.get("instructions", "")
        if isinstance(instructions, str) and _VC_BLOCK_RE.search(instructions):
            body["instructions"] = _VC_BLOCK_RE.sub(new_block, instructions, count=1)
        elif isinstance(instructions, str):
            body["instructions"] = (
                f"{new_block}\n\n{instructions}" if instructions else new_block
            )

    def strip_tools(self, body):
        body.pop("tools", None)
        body.pop("tool_choice", None)


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
        One of ``"anthropic"``, ``"openai"``, ``"openai-codex"``,
        ``"gemini"``.
    api_key : str
        API key for the provider.
    api_url : str
        Override for the API endpoint URL.
    """
    if provider == "anthropic":
        return AnthropicAdapter(api_key, api_url)
    elif provider == "openai":
        return OpenAIAdapter(api_key, api_url)
    elif provider in {"openai-codex", "openai_codex"}:
        return OpenAICodexAdapter(api_key, api_url)
    elif provider == "gemini":
        return GeminiAdapter(api_key, api_url)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Use 'anthropic', 'openai', 'openai-codex', or 'gemini'."
        )
