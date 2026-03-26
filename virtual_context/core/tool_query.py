"""ToolQueryRunner: sync tool loop for non-proxy callers.

Builds a provider-specific request, optionally injects VC paging tools,
sends a non-streaming POST, and runs a synchronous tool loop if the model
invokes any VC tools.  Extracted from engine.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import ToolLoopResult, VirtualContextConfig

logger = logging.getLogger(__name__)


class ToolQueryRunner:
    """Sync tool loop for non-proxy callers.

    Constructor takes:
        engine:  a VirtualContextEngine instance (back-reference for tool handlers)
        config:  a VirtualContextConfig instance
    """

    def __init__(self, engine, config: VirtualContextConfig) -> None:
        self._engine = engine
        self._config = config

    def query_with_tools(
        self,
        messages: list[dict],
        *,
        model: str = "",
        system: str = "",
        max_tokens: int = 4096,
        api_key: str = "",
        api_url: str = "",
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        force_tools: bool = False,
        require_tools: bool | None = None,
        max_loops: int | None = None,
        provider: str = "anthropic",
        extended_thinking: bool = False,
        tool_runtime=None,
    ) -> ToolLoopResult:
        """Send a query to an LLM with VC tool support.

        Builds a provider-specific request, optionally injects VC paging
        tools, sends a non-streaming POST, and runs a synchronous tool
        loop if the model invokes any VC tools.

        Supports Anthropic, OpenAI, OpenAI Codex, and Gemini providers via the adapter
        pattern.

        Parameters
        ----------
        messages : list[dict]
            Messages in ``[{"role": "user", "content": "..."}]`` format.
        model : str
            Model ID (e.g. ``"claude-sonnet-4-5-20250929"``, ``"gpt-4o"``).
        system : str
            System prompt.
        max_tokens : int
            Maximum tokens for the response.
        api_key : str
            API key for the provider.
        api_url : str
            Override for the API endpoint URL (default per provider).
        temperature : float
            Sampling temperature.
        tools : list[dict] | None
            Additional (non-VC) tool definitions to include (Anthropic format).
        force_tools : bool
            If True, inject VC tools even when the normal gate (paging
            enabled + compaction occurred) is not met.
        require_tools : bool | None
            If set, overrides provider tool policy: ``True`` requires at
            least one tool call, ``False`` leaves tool use optional.
        max_loops : int
            Maximum continuation rounds for the tool loop.
        provider : str
            LLM provider: ``"anthropic"``, ``"openai"``,
            ``"openai-codex"``, or ``"gemini"``.

        Returns
        -------
        ToolLoopResult
            Final text, tool call records, and usage metrics.
        """
        import httpx

        from .tool_loop import (
            _parse_provider_http_response,
            get_adapter,
            run_tool_loop,
            vc_tool_definitions_for_runtime,
        )
        from ..types import ToolLoopResult as _ToolLoopResult

        if not model:
            from ..types import DEFAULT_CHAT_MODEL
            model = DEFAULT_CHAT_MODEL

        adapter = get_adapter(provider, api_key, api_url)

        # Decide whether to inject VC tools
        inject_vc = force_tools or (
            self._config.paging.enabled and self._engine._engine_state.compacted_through > 0
        )
        all_tools: list[dict] = []
        if inject_vc:
            all_tools.extend(vc_tool_definitions_for_runtime(tool_runtime))
        if tools:
            all_tools.extend(tools)

        # Convert tool definitions to provider format
        converted_tools = adapter.convert_tool_defs(all_tools) if all_tools else None

        # Wrap VC context in <system-reminder> so inject_context can
        # find-and-replace on reassembly instead of stacking.
        wrapped_system = f"<system-reminder>\n{system}\n</system-reminder>" if system else ""

        # Build provider-specific request body
        body = adapter.build_request_body(
            model=model,
            messages=messages,
            system=wrapped_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=converted_tools,
        )

        # Optional tool-policy override for thresholded behavior.
        if converted_tools and require_tools is not None:
            if provider == "anthropic":
                if require_tools:
                    body["tool_choice"] = {"type": "any"}
                else:
                    body.pop("tool_choice", None)
            elif provider in {"openai", "openrouter", "openai-codex", "openai_codex"}:
                if require_tools:
                    body["tool_choice"] = "required"
                else:
                    body.pop("tool_choice", None)
            elif provider == "gemini":
                if require_tools:
                    body["tool_config"] = {
                        "function_calling_config": {"mode": "ANY"}
                    }
                else:
                    body.pop("tool_config", None)

        # Extended thinking: inject thinking params for Anthropic
        if extended_thinking and provider == "anthropic":
            body["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            body["temperature"] = 1
            # max_tokens must exceed thinking budget
            if body.get("max_tokens", 0) <= 10000:
                body["max_tokens"] = 16000
            # tool_choice "any" is incompatible with thinking
            if body.get("tool_choice", {}).get("type") == "any":
                body["tool_choice"] = {"type": "auto"}
        # Extended thinking: inject reasoning effort for OpenAI Responses API
        if extended_thinking and provider == "openai-responses":
            body["reasoning"] = {"effort": "high", "summary": "auto"}

        url = adapter.get_url(model)
        headers = adapter.get_headers()
        if extended_thinking and provider == "anthropic":
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, headers=headers, json=body)
            # Retry on 503 (Gemini overloaded) up to 2 times
            for _retry in range(2):
                if resp.status_code != 503:
                    break
                import time as _time
                _time.sleep(5)
                resp = client.post(url, headers=headers, json=body)

        if resp.status_code >= 300:
            raise RuntimeError(
                f"{provider} API error {resp.status_code}: {resp.text[:500]}"
            )

        data = _parse_provider_http_response(resp)

        # Check for VC tool calls
        tool_calls = adapter.extract_tool_calls(data)
        has_vc_tools = any(
            tc["name"].startswith("vc_") for tc in tool_calls
        )

        if has_vc_tools:
            effective_loops = (
                max_loops
                if max_loops is not None
                else self._config.paging.max_tool_loops
            )
            # Pass extra headers (e.g. anthropic-beta for extended thinking)
            _extra_hdrs = {}
            if extended_thinking and provider == "anthropic":
                _extra_hdrs["anthropic-beta"] = "interleaved-thinking-2025-05-14"
            loop_result = run_tool_loop(
                self._engine, data, body, adapter,
                url=url, max_loops=effective_loops,
                extra_headers=_extra_hdrs or None,
                tool_runtime=tool_runtime,
            )
            # Prepend the initial request to raw_requests
            loop_result.raw_requests.insert(0, body)
            return loop_result

        # No tool calls — return text directly
        result = _ToolLoopResult()
        result.raw_requests.append(body)
        result.raw_responses.append(data)
        input_toks, output_toks = adapter.extract_usage(data)
        result.input_tokens = input_toks
        result.output_tokens = output_toks
        result.stop_reason = adapter.get_stop_reason(data)
        result.text = adapter.extract_text(data)
        return result
