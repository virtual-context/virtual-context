"""Grok conversation export adapter."""

from datetime import datetime

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.types import Message


class GrokAdapter(ExportAdapter):
    """Adapter for Grok conversation exports.

    Expected schema (unique two-key envelope):
    {
        "conversation": {
            "id": "UUID",
            "title": "string",
            ...
        },
        "responses": [
            {
                "response": {
                    "_id": "UUID",
                    "sender": "human" | "ASSISTANT",  # Mixed case!
                    "message": "content",  # Note: "message" not "text"
                    "create_time": { "$date": { "$numberLong": "1753841416257" } },  # MongoDB ms
                    "thinking_trace": "...",  # Discarded per design decision
                    "steps": [...]  # Discarded per design decision
                }
            }
        ]
    }
    """

    @property
    def name(self) -> str:
        return "grok"

    def extract_messages(self, data: dict) -> list[Message]:
        messages: list[Message] = []
        for resp in data.get("responses", []):
            response = resp.get("response", {})
            sender = response.get("sender", "user")
            role = self._normalize_role(sender)
            content = response.get("message", "")
            timestamp = None
            if create_time := response.get("create_time"):
                if isinstance(create_time, dict):
                    date_obj = create_time.get("$date", {})
                    if isinstance(date_obj, dict):
                        ms_str = date_obj.get("$numberLong", "0")
                        timestamp = datetime.fromtimestamp(int(ms_str) / 1000)
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
            ))
        return messages

    def extract_conversation_id(self, data: dict) -> str:
        conversation = data.get("conversation", {})
        return conversation.get("id", "")
