"""Claude conversation export adapter."""

from datetime import datetime

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.types import Message


class ClaudeAdapter(ExportAdapter):
    """Adapter for Claude conversation exports.

    Expected schema:
    {
        "name": "string",
        "uuid": "UUID",
        "created_at": "2025-09-28T16:41:18.433422Z",  # ISO 8601
        "messages": [
            {
                "role": "human" | "assistant",  # Note: "human" not "user"
                "created_at": "2025-09-28T16:41:19.245108Z",
                "text": "message content"
            }
        ]
    }
    """

    @property
    def name(self) -> str:
        return "claude"

    def extract_messages(self, data: dict) -> list[Message]:
        messages: list[Message] = []
        for msg in data.get("messages", []):
            role = self._normalize_role(msg.get("role", "user"))
            content = msg.get("text", "")
            timestamp = None
            if created_at := msg.get("created_at"):
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
            ))
        return messages

    def extract_conversation_id(self, data: dict) -> str:
        return data.get("uuid", "")
