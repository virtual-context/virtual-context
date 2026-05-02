"""Devin conversation export adapter."""

from datetime import datetime

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.types import Message


class DevinAdapter(ExportAdapter):
    """Adapter for Devin conversation exports.

    Expected schema:
    {
        "title": "string",
        "conversation_id": "hex-hash-no-hyphens",
        "create_time": 1754256914,  # Unix integer (seconds)
        "messages": [
            {
                "role": "user" | "assistant",
                "create_time": 1754256914,
                "text": "message content with [PLAN]...[/PLAN] blocks"
            }
        ]
    }
    """

    @property
    def name(self) -> str:
        return "devin"

    def extract_messages(self, data: dict) -> list[Message]:
        messages: list[Message] = []
        for msg in data.get("messages", []):
            role = self._normalize_role(msg.get("role", "user"))
            content = msg.get("text", "")
            timestamp = None
            if create_time := msg.get("create_time"):
                timestamp = datetime.fromtimestamp(int(create_time))
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
            ))
        return messages

    def extract_conversation_id(self, data: dict) -> str:
        return data.get("conversation_id", "")
