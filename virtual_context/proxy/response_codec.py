"""Bounded provider response collection and lossless SSE serialization.

Internal-tool turns are collected before exposure so a late mixed batch cannot
leak a private call. Opaque provider blocks (including signed thinking) survive
collection and continuation; only transport framing is regenerated.
"""
from __future__ import annotations

import copy
import codecs
import json

from .continuation import MAX_RESPONSE_BYTES, ContinuationError


async def _bounded_lines(response):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    buffer = ""
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise ContinuationError("The provider response exceeded the collection limit.")
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")


async def collect_response(response, api_format):
    try:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ContinuationError("The provider response exceeded the collection limit.")
                chunks.append(chunk)
            try:
                value = json.loads(b"".join(chunks))
            except ValueError as exc:
                raise ContinuationError("The provider returned an invalid JSON response.") from exc
            if not isinstance(value, dict):
                raise ContinuationError("The provider returned an invalid response object.")
            return value
        result, blocks, calls = {}, {}, {}
        size = 0
        terminal = False
        async for line in _bounded_lines(response):
            size += len(line.encode("utf-8")) + 1
            if size > MAX_RESPONSE_BYTES:
                raise ContinuationError("The provider response exceeded the collection limit.")
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                terminal = True
                continue
            try:
                event = json.loads(raw)
            except ValueError as exc:
                raise ContinuationError("The provider returned malformed SSE data.") from exc
            kind = event.get("type", "")
            if kind == "error" or event.get("error"):
                raise ContinuationError("The provider interrupted the response with an error.")
            if api_format == "anthropic":
                if kind == "message_start":
                    result = event.get("message", {})
                elif kind == "content_block_start":
                    blocks[event["index"]] = copy.deepcopy(event["content_block"])
                elif kind == "content_block_delta":
                    block = blocks.setdefault(event["index"], {})
                    delta = event.get("delta", {})
                    key = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature", "input_json_delta": "_partial_json"}.get(delta.get("type"))
                    if key:
                        field = "partial_json" if key == "_partial_json" else key
                        block[key] = block.get(key, "") + delta.get(field, "")
                    elif delta.get("type") == "citations_delta":
                        block.setdefault("citations", []).append(delta.get("citation", {}))
                    else:
                        raise ContinuationError("The provider returned an unsupported content delta.")
                elif kind == "message_delta":
                    result.update(event.get("delta", {}))
                    result.setdefault("usage", {}).update(event.get("usage", {}))
                elif kind == "message_stop":
                    terminal = True
            elif api_format == "openai_responses":
                if kind == "response.completed":
                    result = event["response"]
                    terminal = True
                elif kind in ("response.failed", "response.incomplete"):
                    raise ContinuationError("The provider did not complete the response.")
            elif api_format == "openai":
                result.update({key: value for key, value in event.items() if key != "choices"})
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ContinuationError("Memory tool interception supports one response choice.")
                    message = result.setdefault("choices", [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}])[0]
                    delta = choice.get("delta", {})
                    for key, value in delta.items():
                        if key == "tool_calls":
                            for call in value:
                                target = calls.setdefault(call["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                                if "id" in call:
                                    target["id"] = call["id"]
                                for field, text in call.get("function", {}).items():
                                    target["function"][field] = target["function"].get(field, "") + text
                        elif isinstance(value, str) and key != "role":
                            message["message"][key] = (message["message"].get(key) or "") + value
                        elif value is not None:
                            message["message"][key] = value
                    if choice.get("finish_reason"):
                        message["finish_reason"] = choice["finish_reason"]
                        terminal = True
            else:
                for candidate in event.get("candidates", []):
                    if candidate.get("index", 0) != 0:
                        raise ContinuationError("Memory tool interception supports one response candidate.")
                    target = result.setdefault("candidates", [{"content": {"role": "model", "parts": []}}])[0]
                    target["content"]["parts"].extend(candidate.get("content", {}).get("parts", []))
                    target.update({key: value for key, value in candidate.items() if key != "content"})
                    if candidate.get("finishReason"):
                        terminal = True
                result.update({key: value for key, value in event.items() if key != "candidates"})
        if not terminal:
            raise ContinuationError("The provider stream ended before its completion event.")
        if api_format == "anthropic":
            for block in blocks.values():
                if "_partial_json" in block:
                    try:
                        block["input"] = json.loads(block.pop("_partial_json"))
                    except ValueError as exc:
                        raise ContinuationError("The provider returned incomplete tool arguments.") from exc
            result["content"] = [blocks[index] for index in sorted(blocks)]
        elif api_format == "openai" and calls:
            result["choices"][0]["message"]["tool_calls"] = [calls[index] for index in sorted(calls)]
        return result
    finally:
        await response.aclose()


def _event(value, *, named=True):
    prefix = f"event: {value['type']}\n" if named and "type" in value else ""
    return (prefix + "data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def response_events(response, api_format):
    """Emit complete SDK-compatible provider events from the admitted response."""
    if api_format == "anthropic":
        start = copy.deepcopy(response)
        start.update(content=[], stop_reason=None, stop_sequence=None)
        yield _event({"type": "message_start", "message": start})
        for index, block in enumerate(response.get("content", [])):
            start = copy.deepcopy(block)
            deltas = []
            if block.get("type") == "text":
                start["text"] = ""
                deltas.append({"type": "text_delta", "text": block.get("text", "")})
                if block.get("citations"):
                    start["citations"] = []
                    deltas.extend({"type": "citations_delta", "citation": item} for item in block["citations"])
            elif block.get("type") in ("tool_use", "server_tool_use"):
                start["input"] = {}
                deltas.append({"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False)})
            elif block.get("type") == "thinking":
                start.update(thinking="", signature="")
                deltas.extend([{"type": "thinking_delta", "thinking": block.get("thinking", "")}, {"type": "signature_delta", "signature": block.get("signature", "")}])
            yield _event({"type": "content_block_start", "index": index, "content_block": start})
            for delta in deltas:
                yield _event({"type": "content_block_delta", "index": index, "delta": delta})
            yield _event({"type": "content_block_stop", "index": index})
        yield _event({"type": "message_delta", "delta": {"stop_reason": response.get("stop_reason", "end_turn"), "stop_sequence": response.get("stop_sequence")}, "usage": response.get("usage", {})})
        yield _event({"type": "message_stop"})
    elif api_format == "openai_responses":
        sequence = 0
        base = {**response, "output": [], "status": "in_progress"}
        for kind in ("response.created", "response.in_progress"):
            yield _event({"type": kind, "sequence_number": sequence, "response": base})
            sequence += 1
        for index, item in enumerate(response.get("output", [])):
            start = copy.deepcopy(item)
            if item.get("type") == "message":
                start["content"] = []
            elif item.get("type") == "function_call":
                start["arguments"] = ""
            yield _event({"type": "response.output_item.added", "sequence_number": sequence, "output_index": index, "item": start})
            sequence += 1
            shared = {"item_id": item.get("id", ""), "output_index": index}
            if item.get("type") == "message":
                for content_index, part in enumerate(item.get("content", [])):
                    part_start = copy.deepcopy(part)
                    if part.get("type") == "output_text":
                        part_start["text"] = ""
                    meta = {**shared, "content_index": content_index}
                    yield _event({"type": "response.content_part.added", "sequence_number": sequence, **meta, "part": part_start})
                    sequence += 1
                    if part.get("type") == "output_text":
                        yield _event({"type": "response.output_text.delta", "sequence_number": sequence, **meta, "delta": part.get("text", ""), "logprobs": []})
                        sequence += 1
                        yield _event({"type": "response.output_text.done", "sequence_number": sequence, **meta, "text": part.get("text", ""), "logprobs": []})
                        sequence += 1
                    yield _event({"type": "response.content_part.done", "sequence_number": sequence, **meta, "part": part})
                    sequence += 1
            elif item.get("type") == "function_call":
                yield _event({"type": "response.function_call_arguments.delta", "sequence_number": sequence, **shared, "delta": item.get("arguments", "")})
                sequence += 1
                yield _event({"type": "response.function_call_arguments.done", "sequence_number": sequence, **shared, "arguments": item.get("arguments", "")})
                sequence += 1
            yield _event({"type": "response.output_item.done", "sequence_number": sequence, "output_index": index, "item": item})
            sequence += 1
        yield _event({"type": "response.completed", "sequence_number": sequence, "response": response})
    elif api_format == "openai":
        base = {key: value for key, value in response.items() if key not in ("choices", "usage")}
        base["object"] = "chat.completion.chunk"
        for choice in response.get("choices", []):
            delta = copy.deepcopy(choice["message"])
            for index, call in enumerate(delta.get("tool_calls", [])):
                call["index"] = index
            yield _event({**base, "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}]}, named=False)
            yield _event({**base, "choices": [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}], "usage": response.get("usage", {})}, named=False)
        yield b"data: [DONE]\n\n"
    else:
        yield _event(response, named=False)
