"""Modal for inspecting the full API payload sent for a turn."""

from __future__ import annotations

import json
import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static

from ..state import TurnRecord, save_turn

# Reuse the same emoji regex from chat_view
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002600-\U000026ff"
    "\U0000fe0f"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


class TurnInspector(ModalScreen[None]):
    """Shows the full API payload (system + messages) sent for a turn.

    p/n keys navigate between turns without closing the modal.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("p", "prev_turn", "Prev Turn", priority=True),
        Binding("n", "next_turn", "Next Turn", priority=True),
        Binding("left", "prev_turn", "Prev Turn", show=False, priority=True),
        Binding("right", "next_turn", "Next Turn", show=False, priority=True),
        Binding("s", "save_turn", "Save Turn", priority=True),
    ]

    DEFAULT_CSS = """
    TurnInspector {
        align: center middle;
    }
    TurnInspector > Vertical {
        width: 90%;
        height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    TurnInspector #inspector-header {
        height: auto;
        margin-bottom: 1;
    }
    TurnInspector #inspector-footer {
        height: auto;
        margin-top: 1;
    }
    TurnInspector #inspector-log {
        height: 1fr;
    }
    """

    def __init__(self, turn: TurnRecord, all_turns: list[TurnRecord] | None = None) -> None:
        super().__init__()
        self._turn = turn
        self._all_turns = all_turns or []
        self._current_idx = 0
        for i, t in enumerate(self._all_turns):
            if t.turn_number == turn.turn_number:
                self._current_idx = i
                break

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("", id="inspector-header")
            yield RichLog(id="inspector-log", wrap=True, markup=False)
            yield Static("", id="inspector-footer")

    def on_mount(self) -> None:
        self._render_turn()

    def _render_turn(self) -> None:
        """Render the current turn's payload into the log."""
        t = self._turn
        header = self.query_one("#inspector-header", Static)
        header.update(
            f"[bold]Turn {t.turn_number}[/bold]  "
            f"Tag: {t.primary_tag}  "
            f"Tags: {', '.join(t.tags)}  "
            f"Broad: {t.broad}  "
            f"Temporal: {t.temporal}  "
            f"Tokens: {t.input_tokens:,}  "
            f"Bundled: {t.turns_in_payload}"
        )

        footer = self.query_one("#inspector-footer", Static)
        nav_parts = []
        if self._all_turns:
            pos = f"{self._current_idx + 1}/{len(self._all_turns)}"
            nav_parts.append(f"[dim]{pos}[/dim]")
        nav_parts.append("[dim]p/n or \u2190/\u2192 navigate, s save, Esc close[/dim]")
        footer.update("  ".join(nav_parts))

        log = self.query_one("#inspector-log", RichLog)
        log.clear()

        payload = t.api_payload
        if not payload:
            log.write("(No API payload captured for this turn)")
            return

        # Filtering stats
        total = payload.get("total_history", 0)
        filtered = payload.get("filtered_history", 0)
        if total > 0:
            dropped = total - filtered
            log.write(f"History: {filtered}/{total} messages sent ({dropped} filtered out by tag relevance)")
            log.write("")

        # System prompt
        system_text = payload.get("system", "")
        log.write("=" * 60)
        log.write("SYSTEM PROMPT")
        log.write("=" * 60)
        if system_text:
            log.write(_strip_emoji(system_text))
        else:
            log.write("(empty)")
        log.write("")

        # Messages array
        messages = payload.get("messages", [])
        log.write("=" * 60)
        log.write(f"MESSAGES ({len(messages)})")
        log.write("=" * 60)

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = _strip_emoji(msg.get("content", ""))
            log.write(f"--- [{i}] {role.upper()} ---")

            if len(content) > 2000:
                log.write(content[:2000])
                log.write(f"... ({len(content):,} chars total, truncated)")
            else:
                log.write(content)
            log.write("")

        # Raw JSON (also strip emojis)
        log.write("=" * 60)
        log.write("RAW JSON")
        log.write("=" * 60)
        try:
            raw = json.dumps(payload, indent=2, ensure_ascii=False)
            log.write(_strip_emoji(raw))
        except (TypeError, ValueError):
            log.write(_strip_emoji(str(payload)))

    def action_prev_turn(self) -> None:
        """Navigate to previous turn."""
        if self._all_turns and self._current_idx > 0:
            self._current_idx -= 1
            self._turn = self._all_turns[self._current_idx]
            self._render_turn()

    def action_next_turn(self) -> None:
        """Navigate to next turn."""
        if self._all_turns and self._current_idx < len(self._all_turns) - 1:
            self._current_idx += 1
            self._turn = self._all_turns[self._current_idx]
            self._render_turn()

    def action_save_turn(self) -> None:
        """Save current turn's data to vc-turn-{N}.json."""
        path = save_turn(self._turn)
        footer = self.query_one("#inspector-footer", Static)
        footer.update(f"[bold green]Saved to {path.resolve()}[/bold green]")
