"""Base class for conversation export adapters."""

from abc import ABC, abstractmethod

from virtual_context.types import Message


class ExportAdapter(ABC):
    """Strategy interface for provider-specific export file parsing.

    Follows the PayloadFormat pattern from proxy/formats.py.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: "chatgpt", "claude", "devin", "grok"."""

    @abstractmethod
    def extract_messages(self, data: dict) -> list[Message]:
        """Parse export JSON and return normalized Message objects.

        Args:
            data: Parsed JSON from export file.

        Returns:
            List of Message objects with normalized roles and timestamps.
        """

    @abstractmethod
    def extract_conversation_id(self, data: dict) -> str:
        """Extract conversation identifier from export data.

        Args:
            data: Parsed JSON from export file.

        Returns:
            Unique identifier for this conversation.
        """

    def _normalize_role(self, role: str) -> str:
        """Normalize role to standard values: user, assistant, system, tool."""
        role_lower = role.lower()
        if role_lower in ("human",):
            return "user"
        if role_lower in ("assistant",):
            return "assistant"
        return role_lower
