"""Token budget progress bar with breakdown."""

from __future__ import annotations

from textual.widgets import Static


class BudgetBar(Static):
    """Shows token budget usage as a visual bar with breakdown."""

    DEFAULT_CSS = """
    BudgetBar {
        padding: 0 1;
    }
    """

    def __init__(self, budget: int = 120_000, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._budget = budget
        self._breakdown: dict[str, int] = {}
        self._total = 0

    def on_mount(self) -> None:
        self._refresh_display()

    def update_budget(
        self, total: int, breakdown: dict[str, int], budget: int | None = None
    ) -> None:
        if budget is not None:
            self._budget = budget
        self._total = total
        self._breakdown = breakdown
        self._refresh_display()

    def _refresh_display(self) -> None:
        fraction = min(self._total / self._budget, 1.0) if self._budget > 0 else 0
        bar_width = 20
        filled = int(fraction * bar_width)

        if fraction >= 0.85:
            color = "red"
        elif fraction >= 0.70:
            color = "yellow"
        else:
            color = "green"

        bar = f"[{color}]{'█' * filled}{'░' * (bar_width - filled)}[/{color}]"

        lines = [
            "[bold]TOKEN BUDGET[/bold]",
            f"  {bar} {self._total:,}/{self._budget:,}",
        ]

        for key, val in self._breakdown.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label + ':':<16} {val:>8,}")

        self.update("\n".join(lines))
