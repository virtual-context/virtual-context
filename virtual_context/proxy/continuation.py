"""Transport-independent, durable internal tool continuations.

No client tool is executed here. A mixed batch checkpoints the original provider
calls and memory results, exposes client calls with opaque correlation IDs, and
waits for the client's real results. A successor must prove the same authenticated
route and exact visible transcript before hidden provider evidence is restored.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone

from ..core.provider_adapters import (
    AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAIResponsesAdapter,
)
from ..core.tool_loop import _extract_last_user_intent_text, is_vc_tool
from ..core.store_capabilities import capabilities_of
from ..types import Message, SpeakerRetrievalContext, SpeakerRosterEntry, SpeakerRosterSnapshot
from .formats import get_format
from .message_filter import admit_provider_payload
from .request_context import RequestContext

logger = logging.getLogger(__name__)
_TOKEN = re.compile(r"^vcx_([0-9a-f]{32})_([0-9]+)$")
MAX_ROUNDS = 5
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
EXCHANGE_TTL = 600


class ContinuationError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def adapter_for(api_format):
    return {"anthropic": AnthropicAdapter, "openai": OpenAIAdapter,
            "openai_responses": OpenAIResponsesAdapter, "gemini": GeminiAdapter}[api_format](api_key="")


def messages(body, api_format):
    key = {"openai_responses": "input", "gemini": "contents"}.get(api_format, "messages")
    value = body.get(key, [])
    return value if isinstance(value, list) else [{"role": "user", "content": value}]


def assistant_items(response, api_format):
    if api_format == "anthropic":
        return [{"role": "assistant", "content": copy.deepcopy(response.get("content", []))}]
    if api_format == "openai":
        return [copy.deepcopy(response["choices"][0]["message"])]
    if api_format == "openai_responses":
        return copy.deepcopy(response.get("output", []))
    return [copy.deepcopy(response["candidates"][0]["content"])]


def result_items(body, api_format):
    """Only actual client result carriers, with no inferred or fabricated data."""
    result = []
    for message in messages(body, api_format):
        if api_format == "anthropic" and message.get("role") == "user":
            result.extend(part for part in message.get("content", []) if isinstance(part, dict) and part.get("type") == "tool_result")
        elif api_format == "openai" and message.get("role") == "tool":
            result.append(message)
        elif api_format == "openai_responses" and message.get("type") == "function_call_output":
            result.append(message)
        elif api_format == "gemini" and message.get("role") == "user":
            result.extend(part for part in message.get("parts", []) if "functionResponse" in part)
    return result


def result_id(item, api_format):
    if api_format == "gemini":
        return item["functionResponse"].get("id", "")
    return item.get({"anthropic": "tool_use_id", "openai": "tool_call_id", "openai_responses": "call_id"}[api_format], "")


def trailing_results(body, api_format):
    """Completed older exchanges in full client history are not new obligations."""
    key = {"openai_responses": "input", "gemini": "contents"}.get(api_format, "messages")
    tail = []
    for item in reversed(messages(body, api_format)):
        extracted = result_items({key: [item]}, api_format)
        if not extracted:
            break
        tail[:0] = extracted
    return tail


def externalize(response, api_format, exchange_id):
    """Strip internal calls; retain exact client names/arguments and provider parts."""
    visible = copy.deepcopy(response)
    mapping = []
    if api_format == "anthropic":
        parent, key = visible, "content"
        id_key = "id"
    elif api_format == "openai":
        parent, key = visible["choices"][0]["message"], "tool_calls"
        id_key = "id"
    elif api_format == "openai_responses":
        parent, key = visible, "output"
        id_key = "call_id"
    else:
        parent, key = visible["candidates"][0]["content"], "parts"
        id_key = "id"
    retained = []
    for item in parent.get(key, []):
        is_call = (api_format == "openai" or item.get("type") in ("tool_use", "function_call") or "functionCall" in item)
        if not is_call:
            retained.append(item)
            continue
        call = item["function"] if api_format == "openai" else item["functionCall"] if api_format == "gemini" else item
        if is_vc_tool(call.get("name", "")):
            continue
        carrier = call if api_format == "gemini" else item
        token = f"vcx_{exchange_id}_{len(mapping)}"
        mapping.append({"token": token, "id": carrier.get(id_key, ""), "name": call.get("name", "")})
        # Gemini preserves an existing native ID. Older Gemini clients return
        # names only; bounded exact-transcript matching handles that shape.
        carrier[id_key] = token
        retained.append(item)
    parent[key] = retained
    return visible, mapping


def append_exchange(body, response, results, api_format):
    """Keep all signed/thinking/opaque provider items in their original roles."""
    result = copy.deepcopy(body)
    key = {"openai_responses": "input", "gemini": "contents"}.get(api_format, "messages")
    result[key] = copy.deepcopy(messages(body, api_format)) + assistant_items(response, api_format)
    if api_format == "anthropic":
        result[key].append({"role": "user", "content": results})
    elif api_format == "gemini":
        result[key].append({"role": "user", "parts": results})
    else:
        result[key].extend(results)
    return result


def _context_checkpoint(context):
    return {key: value for key, value in asdict(replace(context, metrics=None)).items() if key != "metrics"}


def _insert_after_reasoning(items, prefix, is_reasoning):
    index = 0
    while index < len(items) and is_reasoning(items[index]):
        index += 1
    items[index:index] = prefix


def _round_prefix_texts(texts, has_following_text):
    """Separate completed provider rounds, never blocks within one round."""
    return [text + ("\n\n" if index < len(texts) - 1 or has_following_text else "")
            for index, text in enumerate(texts)]


def _prepend_public_text(response, api_format, texts, *, native_messages=(), native_message_groups=()):
    """Retain public text from internal rounds without exposing their tools."""
    if not texts:
        return response
    result = copy.deepcopy(response)
    if api_format == "anthropic":
        content = result.setdefault("content", [])
        prefixes = _round_prefix_texts(texts, any(block.get("type") == "text" and block.get("text") for block in content))
        _insert_after_reasoning(content, [{"type": "text", "text": text} for text in prefixes], lambda block: block.get("type") in ("thinking", "redacted_thinking"))
    elif api_format == "gemini":
        parts = result["candidates"][0]["content"].setdefault("parts", [])
        prefixes = _round_prefix_texts(texts, any(part.get("text") and part.get("thought") is not True for part in parts))
        _insert_after_reasoning(parts, [{"text": text} for text in prefixes], lambda part: part.get("thought") is True)
    elif api_format == "openai":
        message = result["choices"][0]["message"]
        message["content"] = "\n\n".join([*texts, *([message["content"]] if message.get("content") else [])])
    else:
        # Reuse provider-issued public message items and IDs. Opaque reasoning
        # remains leading and unchanged; no invented message ID is echoed on
        # the client's next request.
        groups = copy.deepcopy(list(native_message_groups) or [list(native_messages)])
        if len(groups) != len(texts) or any(not group for group in groups):
            raise ContinuationError("Responses public text is missing its native message items.")
        output = result.setdefault("output", [])
        has_following = any(item.get("type") == "message" and any(part.get("type") == "output_text" and part.get("text") for part in item.get("content", [])) for item in output)
        for index, group in enumerate(groups):
            if index < len(groups) - 1 or has_following:
                # One round can contain several message items and text parts.
                # Append only to its last public part, preserving their IDs
                # and every internal block/delta boundary.
                parts = [part for item in group for part in item.get("content", []) if part.get("type") == "output_text"]
                parts[-1]["text"] = parts[-1].get("text", "") + "\n\n"
        _insert_after_reasoning(output, [item for group in groups for item in group], lambda item: item.get("type") == "reasoning")
    return result


def _restore_context(data, current):
    data = dict(data)
    data["speaker_context"] = SpeakerRetrievalContext(**data["speaker_context"]) if data.get("speaker_context") else None
    if data.get("roster_snapshot"):
        roster = dict(data["roster_snapshot"])
        roster["entries"] = tuple(SpeakerRosterEntry(**entry) for entry in roster["entries"])
        data["roster_snapshot"] = SpeakerRosterSnapshot(**roster)
    data.update(request_id=current.request_id, metrics=current.metrics)
    data["upstream_limit"] = min(data["upstream_limit"], current.upstream_limit)
    return RequestContext(**data)


def _authority_matches(saved, context):
    return all(saved["context"][field] == getattr(context, field) for field in (
        "tenant_id", "conversation_id", "audience_route", "provider",
        "api_format", "model", "lifecycle_epoch",
    ))


def _continuation_suffix(incoming, saved, api_format, *, idless=False):
    """Match exact history, with only the last Gemini call IDs optional."""
    items = messages(incoming, api_format)
    previous_id = incoming.get("previous_response_id")
    if api_format == "openai_responses" and previous_id and previous_id == saved.get("response_id"):
        return items
    expected = saved["visible_prefix"]
    prefix = items[:len(expected)]
    if prefix == expected:
        return items[len(expected):]
    if idless and api_format == "gemini" and expected and len(prefix) == len(expected):
        # Old clients may omit BOTH call and response IDs. Keep every earlier
        # transcript byte and native field bound; only this final model turn's
        # proxy-minted IDs may be absent. Ambiguous matches remain an error.
        without_ids = copy.deepcopy(expected[-1])
        for part in without_ids.get("parts", []):
            call = part.get("functionCall", {})
            if _TOKEN.match(call.get("id", "")):
                call.pop("id")
        if prefix[:-1] == expected[:-1] and prefix[-1] == without_ids:
            return items[len(expected):]
    return None


class ContinuationSession:
    """Own one tool operation, including lease, cancellation and completion."""
    def __init__(self, context, state, *, runtime_factory, execute_tool):
        self.context = context
        self.state = state
        self.runtime_factory = runtime_factory
        self.execute_tool = execute_tool
        self.adapter = adapter_for(context.api_format)
        self.body = None
        self.round = 0
        self.presented_refs = set()
        self.presented_facts = set()
        self.claim = None
        self.renewal = None
        self.lease_lost = False
        self.deferred = False
        self.completed = False
        self.usage = [0, 0]
        self.cache_usage = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self.pending_text = []
        self.pending_response_message_groups = []
        self.completed_prefix = []

    def public_text(self, response):
        if self.context.api_format == "gemini":
            candidates = response.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            return "".join(part.get("text", "") for part in parts if part.get("thought") is not True)
        return self.adapter.extract_text(response)

    def _store(self):
        store = self.state.engine._store
        required = ("put_pending_exchange", "claim_pending_exchange", "finish_pending_exchange", "renew_pending_exchange", "list_pending_exchanges", "get_pending_exchange")
        if not capabilities_of(store).durable_exchanges or not all(callable(getattr(store, name, None)) for name in required):
            raise ContinuationError("This storage backend does not support durable tool continuations.", 503)
        return store

    def assert_live(self):
        if self.lease_lost:
            raise ContinuationError("The tool continuation lease expired; retry the request.", 409)
        if self.state.is_conversation_deleted():
            raise ContinuationError("The conversation was deleted during tool continuation.", 409)
        epoch = getattr(getattr(self.state.engine, "_engine_state", None), "lifecycle_epoch", 0)
        if type(epoch) is int and epoch != self.context.lifecycle_epoch:
            raise ContinuationError("The conversation changed during tool continuation.", 409)

    async def _renew(self):
        while True:
            await asyncio.sleep(30)
            try:
                ok = await asyncio.to_thread(self._store().renew_pending_exchange, self.context.conversation_id, *self.claim, now=time.time(), lease_seconds=120)
            except Exception:
                ok = False
            if not ok:
                self.lease_lost = True
                return

    async def close(self, *, consume=False):
        if self.renewal:
            self.renewal.cancel()
            try:
                await self.renewal
            except asyncio.CancelledError:
                pass
            self.renewal = None
        if self.claim:
            claim, self.claim = self.claim, None
            accepted = await asyncio.shield(asyncio.to_thread(self._store().finish_pending_exchange, self.context.conversation_id, *claim, consume=consume))
            if consume and not accepted:
                raise ContinuationError("The tool continuation lease was lost before completion.", 409)

    async def resume(self, incoming):
        """Return restored provider body, or None for an ordinary request."""
        api_format = self.context.api_format
        incoming_results = trailing_results(incoming, api_format)
        tokens = {_TOKEN.match(result_id(item, api_format)) for item in incoming_results}
        if api_format == "gemini" and incoming_results and not any(tokens):
            # Older Gemini clients omit the functionResponse ID while retaining
            # the original functionCall in their transcript. Its opaque ID also
            # makes expiry/restart detection deterministic without a new header.
            for item in reversed(messages(incoming, api_format)):
                if item.get("role") == "model":
                    tokens.update(_TOKEN.match(part["functionCall"].get("id", "")) for part in item.get("parts", []) if "functionCall" in part)
                    break
        ids = {match.group(1) for match in tokens if match}
        if len(ids) > 1:
            raise ContinuationError("Tool results span different continuation requests.", 409)
        if not ids and api_format != "gemini":
            return None
        if not incoming_results:
            return None
        explicit_identity = bool(ids)
        if not ids:
            # An ordinary Gemini function response proves no VC obligation.
            # Optional discovery is read-only: never require a continuation
            # backend or steal a lease just to check an unrelated transcript.
            store = self.state.engine._store
            if not capabilities_of(store).durable_exchanges or not all(callable(getattr(store, name, None)) for name in ("list_pending_exchanges", "get_pending_exchange")):
                return None
            candidates = await asyncio.to_thread(store.list_pending_exchanges, self.context.conversation_id, now=time.time())
            for candidate in candidates:
                raw = await asyncio.to_thread(store.get_pending_exchange, self.context.conversation_id, candidate, now=time.time())
                if raw is None:
                    continue
                saved = json.loads(raw)
                if _authority_matches(saved, self.context) and _continuation_suffix(incoming, saved, api_format, idless=True) is not None:
                    ids.add(candidate)
            if not ids:
                return None
            if len(ids) > 1:
                raise ContinuationError("The tool continuation transcript matches multiple pending requests; preserve the call ID.", 409)
        store = self._store()
        for exchange_id in ids:
            claim_id = uuid.uuid4().hex
            raw = await asyncio.to_thread(store.claim_pending_exchange, self.context.conversation_id, exchange_id, claim_id, now=time.time(), lease_seconds=120)
            if not raw:
                raise ContinuationError("The tool continuation expired or is already in progress.", 409)
            self.claim = (exchange_id, claim_id)
            try:
                saved = json.loads(raw)
                original = saved["context"]
                if not _authority_matches(saved, self.context):
                    raise ContinuationError("Tool continuation authority does not match this request.", 409)
                suffix = _continuation_suffix(incoming, saved, api_format, idless=not explicit_identity)
                if suffix is None:
                    raise ContinuationError("Tool continuation transcript does not match the original request.", 409)
                suffix_body = { {"openai_responses": "input", "gemini": "contents"}.get(api_format, "messages"): suffix }
                actual = result_items(suffix_body, api_format)
                # Reject appended prompts, duplicate results, partial batches and
                # extra tool data. The client supplies exactly its obligations.
                if api_format in ("anthropic", "gemini"):
                    field = "content" if api_format == "anthropic" else "parts"
                    valid_shape = all(msg.get("role") == "user" and isinstance(msg.get(field), list) and len(msg[field]) == len(result_items({"messages" if api_format == "anthropic" else "contents": [msg]}, api_format)) for msg in suffix)
                else:
                    valid_shape = len(actual) == len(suffix)
                if not valid_shape or len(actual) != len(saved["mapping"]):
                    raise ContinuationError("Provide exactly one result for each pending client tool.", 409)
                remapped = []
                remaining = list(actual)
                for mapping in saved["mapping"]:
                    matches = [item for item in remaining if result_id(item, api_format) == mapping["token"] or (api_format == "gemini" and not result_id(item, api_format) and item["functionResponse"].get("name") == mapping["name"])]
                    if len(matches) != 1:
                        raise ContinuationError("Tool results do not match the pending client calls.", 409)
                    item = copy.deepcopy(matches[0])
                    remaining.remove(matches[0])
                    if api_format == "gemini":
                        item["functionResponse"].pop("id", None)
                        if mapping["id"]:
                            item["functionResponse"]["id"] = mapping["id"]
                    else:
                        item[{"anthropic": "tool_use_id", "openai": "tool_call_id", "openai_responses": "call_id"}[api_format]] = mapping["id"]
                    remapped.append(item)
                self.context = _restore_context(original, self.context)
                self.round = saved["round"]
                self.presented_refs = set(saved["presented_refs"])
                self.presented_facts = set(saved["presented_facts"])
                self.completed_prefix = saved.get("completed_prefix", [])
                restored = append_exchange(saved["body"], saved["response"], saved["results"] + remapped, api_format)
                # The next mixed batch must bind the client-visible transcript,
                # never the reconstructed hidden tool exchange.
                self.context = replace(self.context, source_body_json=json.dumps(incoming, ensure_ascii=False))
                if api_format != "gemini":
                    restored["stream"] = bool(incoming.get("stream", False))
                self.body = restored
                self.renewal = asyncio.create_task(self._renew())
                self.assert_live()
                return restored
            except BaseException:
                await self.close()
                raise
        return None

    async def advance(self, body, response):
        """Execute one round; return (next provider body, visible final response)."""
        self.assert_live()
        self.body = body
        usage = self.adapter.extract_usage(response)
        self.usage = [self.usage[i] + (usage[i] or 0) for i in range(2)]
        if self.context.api_format == "anthropic":
            for key in self.cache_usage:
                count = (response.get("usage") or {}).get(key, 0) or 0
                self.cache_usage[key] += count
                self.usage[0] += count
        calls = self.adapter.extract_tool_calls(response)
        internal = [call for call in calls if is_vc_tool(call["name"])]
        if not internal:
            visible = _prepend_public_text(response, self.context.api_format, self.pending_text, native_message_groups=self.pending_response_message_groups)
            self.pending_text = []
            self.pending_response_message_groups = []
            return None, visible
        if self.round >= MAX_ROUNDS:
            raise ContinuationError("The provider exceeded the memory tool round limit.")
        self.round += 1
        mixed = len(internal) != len(calls)
        if mixed:
            self._store()  # Require durable support before changing any working set.
        runtime = self.runtime_factory(engine=self.state.engine, api_format=self.context.api_format, conversation_id=self.context.conversation_id, get_target_body=lambda: self.body, speaker_context=self.context.speaker_context)
        results = []
        for call in internal:
            self.assert_live()
            started = time.monotonic()
            work = asyncio.create_task(asyncio.to_thread(self.execute_tool, self.state.engine, call["name"], call["input"], intent_context=_extract_last_user_intent_text(json.loads(self.context.source_body_json)), presented_segment_refs=self.presented_refs, presented_fact_ids=self.presented_facts, tool_runtime=runtime, speaker_context=self.context.speaker_context, roster_snapshot=self.context.roster_snapshot))
            try:
                result = await asyncio.shield(work)
            except asyncio.CancelledError:
                # A Python thread cannot be cancelled. Keep lease ownership
                # until the synchronous mutation stops, then release on unwind.
                try:
                    await work
                finally:
                    raise
            self.assert_live()
            elapsed = round((time.monotonic() - started) * 1000, 1)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            event = {"conversation_id": self.context.conversation_id, "request_turn": self.context.request_turn or self.context.turn, "round": self.round, "tool_name": call["name"], "tool_input": call["input"], "tool_result": result, "result_length": len(result), "duration_ms": elapsed, "timestamp": datetime.now(timezone.utc).isoformat()}
            if self.context.metrics:
                self.context.metrics.record({"type": "tool_intercept", "turn": self.context.turn, "conversation_id": self.context.conversation_id, "tool_name": call["name"], "tool_input": call["input"], "result": result[:200], "duration_ms": elapsed, "continuation_count": self.round})
            try:
                await asyncio.to_thread(self.state.engine._store.save_tool_call, event)
            except Exception:
                logger.debug("Tool telemetry persistence failed", exc_info=True)
            results.append(self.adapter.build_tool_result(call["id"], call["name"], result))
        if mixed:
            exchange_id = uuid.uuid4().hex
            visible, mapping = externalize(response, self.context.api_format, exchange_id)
            visible = _prepend_public_text(visible, self.context.api_format, self.pending_text, native_message_groups=self.pending_response_message_groups)
            self.completed_prefix.extend(self.pending_text)
            current_text = self.public_text(response)
            if current_text:
                self.completed_prefix.append(current_text)
            self.pending_text = []
            self.pending_response_message_groups = []
            saved = {"context": _context_checkpoint(self.context), "body": self.body, "response": response, "results": results, "mapping": mapping, "response_id": response.get("id"), "visible_prefix": messages(json.loads(self.context.source_body_json), self.context.api_format) + assistant_items(visible, self.context.api_format), "round": self.round, "presented_refs": sorted(self.presented_refs), "presented_facts": sorted(self.presented_facts), "completed_prefix": self.completed_prefix}
            ok = await asyncio.to_thread(self._store().put_pending_exchange, self.context.conversation_id, exchange_id, json.dumps(saved, ensure_ascii=False), expires_at=time.time() + EXCHANGE_TTL, max_entries=4, max_bytes=2 * 1024 * 1024)
            if not ok:
                raise ContinuationError("The conversation's pending tool budget is full; finish an earlier exchange.", 503)
            self.deferred = True
            return None, visible
        current_text = self.public_text(response)
        if current_text:
            self.pending_text.append(current_text)
            if self.context.api_format == "openai_responses":
                self.pending_response_message_groups.append(copy.deepcopy([
                    item for item in response.get("output", []) if item.get("type") == "message"
                    and any(part.get("type") == "output_text" for part in item.get("content", []))
                ]))
        continuation = append_exchange(self.body, response, results, self.context.api_format)
        new_context = await asyncio.to_thread(self.state.engine.reassemble_context, speaker_context=self.context.speaker_context)
        if new_context:
            self.adapter.inject_context(continuation, new_context)
        self.adapter.relax_tool_choice(continuation)
        return continuation, None

    async def admit(self, body):
        self.assert_live()
        return (await asyncio.to_thread(admit_provider_payload, body, self.context.upstream_limit, get_format(self.context.api_format)))[0]

    def persist_completed(self, text, raw_content, *, passthrough=False):
        """One completion owner for both transports; incomplete streams never enter."""
        if self.completed:
            return
        self.assert_live()
        self.completed = True
        self.state.engine._engine_state.last_request_time = time.time()
        if self.deferred or (not text and not self.completed_prefix):
            return
        if self.completed_prefix:
            raw_prefixes = _round_prefix_texts(self.completed_prefix, bool(text))
            text = "\n\n".join([*self.completed_prefix, *([text] if text else [])])
            if self.context.api_format == "anthropic" and isinstance(raw_content, list):
                raw_content = copy.deepcopy(raw_content)
                _insert_after_reasoning(raw_content, [{"type": "text", "text": prefix} for prefix in raw_prefixes], lambda block: block.get("type") in ("thinking", "redacted_thinking"))
        history = []
        for data in json.loads(self.context.history_json):
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            history.append(Message(**data))
        answer = Message(role="assistant", content=text, timestamp=datetime.now(timezone.utc), raw_content=raw_content)
        snapshot = history + [answer]
        # This list is a dashboard/cache view. A later concurrent request may
        # have replaced it; never append this answer under that request's user.
        current = self.state.conversation_history
        if [(m.role, m.content) for m in current] == [(m.role, m.content) for m in history]:
            self.state.conversation_history = snapshot
        if not passthrough:
            self.state.fire_turn_complete(snapshot, payload_tokens=self.context.payload_tokens or None, turn_id=self.context.turn_id)
