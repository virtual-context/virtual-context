"""Immutable identity, authority and limits for one proxy request.

Transport state lives in a continuation session, never in this value. In
particular, retrieving a deferred exchange replaces authority from its trusted
checkpoint but retains the newly authenticated request's metrics destination.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from ..types import SpeakerRetrievalContext, SpeakerRosterSnapshot


def provider_identity(url: str) -> str:
    """Bind the route without persisting query-string API credentials."""
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.replace(':streamGenerateContent', ':generateContent')}"


def history_checkpoint(messages) -> str:
    return json.dumps([asdict(message) for message in messages], ensure_ascii=False,
                      default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value))


@dataclass(frozen=True)
class RequestContext:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    operation_id: str = ""
    tenant_id: str = ""
    conversation_id: str = ""
    audience_route: str = ""
    provider: str = ""
    api_format: str = "anthropic"
    model: str = ""
    turn: int = 0
    request_turn: int = 0
    turn_id: str = ""
    lifecycle_epoch: int = 0
    upstream_limit: int = 0
    output_allowance: int = 0
    payload_tokens: int = 0
    speaker_context: SpeakerRetrievalContext | None = field(default=None, repr=False)
    roster_snapshot: SpeakerRosterSnapshot | None = field(default=None, repr=False)
    metrics: Any = field(default=None, repr=False, compare=False)
    source_body_json: str = field(default="{}", repr=False)
    history_json: str = field(default="[]", repr=False)

    @property
    def input_allowance(self) -> int:
        return max(0, self.upstream_limit - self.output_allowance)

    @classmethod
    def create(cls, *, body: dict, state=None, **values) -> RequestContext:
        epoch = getattr(getattr(getattr(state, "engine", None), "_engine_state", None), "lifecycle_epoch", 0)
        values.setdefault("lifecycle_epoch", epoch if type(epoch) is int else 0)
        values.setdefault("model", body.get("model", ""))
        history = getattr(state, "conversation_history", [])
        values.setdefault("history_json", history_checkpoint(history) if isinstance(history, list) else "[]")
        value = cls(source_body_json=json.dumps(body, ensure_ascii=False), **values)
        return replace(value, operation_id=value.operation_id or value.request_id)
