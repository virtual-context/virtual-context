"""Turn history list with keyboard navigation and auto-scroll."""

from __future__ import annotations

from textual.widgets import RichLog

from ..state import TurnRecord


class TurnList(RichLog):
    """Lists turns with primary tag and token count. Ctrl+B/F to navigate.

    Uses RichLog for native scrolling — no manual windowing needed.
    """

    DEFAULT_CSS = """
    TurnList {
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(wrap=True, markup=True, **kwargs)
        self._turns: list[TurnRecord] = []
        self._selected: int = -1

    def add_turn(self, turn: TurnRecord) -> None:
        self._turns.append(turn)
        self._selected = len(self._turns) - 1
        self._refresh_display()

    @property
    def selected_turn(self) -> TurnRecord | None:
        if not self._turns:
            return None
        idx = self._selected if 0 <= self._selected < len(self._turns) else len(self._turns) - 1
        return self._turns[idx]

    def select_prev(self) -> None:
        """Move selection to previous turn."""
        if self._turns and self._selected > 0:
            self._selected -= 1
            self._refresh_display()

    def select_next(self) -> None:
        """Move selection to next turn."""
        if self._turns and self._selected < len(self._turns) - 1:
            self._selected += 1
            self._refresh_display()

    def _refresh_display(self) -> None:
        self.clear()

        if not self._turns:
            self.write("[bold]TURN HISTORY[/bold]")
            self.write("[dim]No turns yet[/dim]")
            return

        latest = self._turns[-1]
        total = len(self._turns)
        bundled = latest.turns_in_payload

        self.write(
            f"[bold]TURN HISTORY[/bold]  "
            f"[dim]{total} total, {bundled} in last payload[/dim]"
        )

        for i, t in enumerate(self._turns):
            tag = t.primary_tag or "_general"
            tokens = t.input_tokens
            packed = t.turns_in_payload
            if i == self._selected:
                self.write(
                    f"  [bold reverse]> Turn {t.turn_number}: {tag} "
                    f"({tokens:,}t, {packed} bundled)[/bold reverse]"
                )
            else:
                self.write(
                    f"    Turn {t.turn_number}: {tag} "
                    f"({tokens:,}t, {packed} bundled)"
                )

        self.write("")
        self.write("[dim]Ctrl+B/F nav, Ctrl+I inspect, Ctrl+S save[/dim]")

        # Scroll to keep selected turn visible
        # Header is line 0, turns start at line 1, selected is at line 1 + _selected
        target_line = 1 + self._selected
        self.scroll_to_line(target_line)

    def scroll_to_line(self, line: int) -> None:
        """Scroll so that the given line index is visible."""
        # RichLog scrolls via scroll_y; each write() adds one line
        # Approximate: set scroll so target line is in the middle of view
        try:
            height = self.size.height
            target_y = max(0, line - height // 2)
            self.scroll_y = target_y
        except Exception:
            pass
