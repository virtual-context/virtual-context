"""Active tags display panel."""

from __future__ import annotations

from textual.widgets import Static


class TagPanel(Static):
    """Shows active tags with recency-based activity bars."""

    DEFAULT_CSS = """
    TagPanel {
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._tags: list[tuple[str, float]] = []

    def update_tags(self, tags: list[tuple[str, float]]) -> None:
        self._tags = tags
        self._render_tags()
        self.refresh()

    def _render_tags(self) -> None:
        if not self._tags:
            self.update("[bold]ACTIVE TAGS[/bold]\n[dim]No tags yet[/dim]")
            return

        lines = ["[bold]ACTIVE TAGS[/bold]"]
        for tag, activity in self._tags:
            if activity >= 0.7:
                color = "green"
            elif activity >= 0.4:
                color = "yellow"
            else:
                color = "red"
            bar = "█" * int(activity * 10) + "░" * (10 - int(activity * 10))
            lines.append(f"  [{color}]{bar}[/{color}] {tag}")

        self.update("\n".join(lines))
