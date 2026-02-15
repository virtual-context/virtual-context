"""Multi-line text input with Enter to submit."""

from __future__ import annotations

from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widgets import TextArea


class InputBox(TextArea):
    """Multi-line input. Enter submits the message."""

    class MessageSubmitted(Message):
        """Posted when the user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    BINDINGS = [
        Binding("ctrl+n", "newline", "New Line"),
    ]

    def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.MessageSubmitted(text))
                self.clear()

    def action_newline(self) -> None:
        """Insert a newline at the cursor."""
        self.insert("\n")
