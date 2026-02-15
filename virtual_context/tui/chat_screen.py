"""Main chat screen composing all widgets."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from .widgets.budget_bar import BudgetBar
from .widgets.chat_view import ChatView
from .widgets.input_box import InputBox
from .widgets.tag_panel import TagPanel
from .widgets.turn_list import TurnList


class ChatScreen(Screen):
    """Main screen: chat area (left) + context panel (right)."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-area"):
                yield ChatView(id="chat-view")
                yield InputBox(id="input-box")
            with Vertical(id="context-panel"):
                yield TagPanel(id="tag-panel")
                yield BudgetBar(id="budget-bar")
                yield TurnList(id="turn-list")
                yield Static(
                    "[bold]COMPACTION LOG[/bold]\n[dim]No compactions yet[/dim]",
                    id="compaction-log",
                )
