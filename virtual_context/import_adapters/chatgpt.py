"""ChatGPT conversation export adapter."""

from datetime import datetime

from virtual_context.import_adapters.base import ExportAdapter
from virtual_context.types import Message


class ChatGPTAdapter(ExportAdapter):
    """Adapter for ChatGPT conversation exports.

    Expected schema:
    {
        "title": "string",
        "conversation_id": "UUID",
        "create_time": 1715848606.903149,  # Unix float (seconds)
        "messages": [
            {
                "role": "user" | "assistant",
                "create_time": 1715848606.917559,
                "text": "message content"
            }
        ]
    }
    """

    @property
    def name(self) -> str:
        return "chatgpt"

    def extract_messages(self, data: dict) -> list[Message]:
        """Parse ChatGPT export JSON and return normalized Message objects.

        Args:
            data: Parsed JSON from ChatGPT export file.

        Returns:
            List of Message objects with normalized roles and timestamps.
        """
        messages: list[Message] = []
        for msg in data.get("messages", []):
            role = self._normalize_role(msg.get("role", "user"))
            content = msg.get("text", "")
            timestamp = None
            if create_time := msg.get("create_time"):
                timestamp = datetime.fromtimestamp(float(create_time))
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
            ))
        return messages

    def extract_conversation_id(self, data: dict) -> str:
        """Extract conversation ID from ChatGPT export data.

        Args:
            data: Parsed JSON from ChatGPT export file.

        Returns:
            Unique identifier for this conversation.
        """
        return data.get("conversation_id", "")
