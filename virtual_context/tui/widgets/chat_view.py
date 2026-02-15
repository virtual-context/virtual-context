"""Scrollable chat message display."""

from __future__ import annotations

import re

from textual.widgets import RichLog

# Regex matching most emoji codepoint ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0001f900-\U0001f9ff"  # supplemental
    "\U0001fa00-\U0001fa6f"  # chess/extended-A
    "\U0001fa70-\U0001faff"  # extended-B
    "\U00002600-\U000026ff"  # misc symbols
    "\U0000fe0f"             # variation selector
    "]+",
    flags=re.UNICODE,
)


class ChatView(RichLog):
    """Displays chat messages with Rich markup, auto-scrolling."""

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=True, wrap=True, auto_scroll=True, **kwargs)
        self._streaming_chunks: list[str] = []
        self._message_log: list[str] = []

    def _write_line(self, markup: str) -> None:
        """Write a line and track it for replay."""
        self._message_log.append(markup)
        self.write(markup)

    def add_user_message(self, text: str) -> None:
        self._write_line(f"[bold cyan]You:[/bold cyan] {text}")
        self._write_line("")

    def add_system_message(self, text: str) -> None:
        self._write_line(f"[dim italic]{text}[/dim italic]")
        self._write_line("")

    def begin_assistant_message(self) -> None:
        self._streaming_chunks = []
        self.write("[dim]Assistant is typing...[/dim]")

    def append_assistant_chunk(self, chunk: str) -> None:
        self._streaming_chunks.append(chunk)

    @staticmethod
    def _strip_emoji(text: str) -> str:
        return _EMOJI_RE.sub("", text)

    def end_assistant_message(self) -> None:
        full_text = self._strip_emoji("".join(self._streaming_chunks))
        self._streaming_chunks = []
        # Clear and replay all prior messages + the new one
        self.clear()
        for line in self._message_log:
            self.write(line)
        self._write_line(f"[bold green]Assistant:[/bold green] {full_text}")
        self._write_line("")
