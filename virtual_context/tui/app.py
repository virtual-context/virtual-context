"""VChatApp: Textual application wiring engine, provider, and TUI together."""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from ..engine import VirtualContextEngine
from ..types import AssembledContext, Message
from .chat_provider import ChatProvider
from .modals.turn_inspector import TurnInspector
from .state import TurnRecord, save_session
from .widgets.budget_bar import BudgetBar
from .widgets.chat_view import ChatView
from .widgets.input_box import InputBox
from .widgets.tag_panel import TagPanel
from .widgets.turn_list import TurnList


class VChatApp(App):
    """Interactive chat with virtual-context side panel."""

    CSS_PATH = "chat.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+i", "inspect_turn", "Inspect Turn", priority=True),
        Binding("ctrl+b", "prev_turn", "Prev Turn", priority=True),
        Binding("ctrl+f", "next_turn", "Next Turn", priority=True),
        Binding("ctrl+s", "save_session", "Save Log", priority=True),
        Binding("ctrl+t", "toggle_brief", "Brief Mode", priority=True),
        Binding("ctrl+k", "compact", "Compact", priority=True),
    ]

    def __init__(
        self,
        config_path: str | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        replay_prompts: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._config_path = config_path
        self._api_key = api_key
        self._model = model
        self._conversation_history: list[Message] = []
        self._turns: list[TurnRecord] = []
        self._turn_counter = 0
        self._streaming = False
        self._brief_mode = True
        self._replay_prompts: list[str] = replay_prompts or []
        self._replaying = False

    def on_mount(self) -> None:
        # Initialize engine
        try:
            self.engine = VirtualContextEngine(config_path=self._config_path)
        except Exception as e:
            self.engine = None
            self._chat_view.add_system_message(f"Engine init failed: {e}")
            self._chat_view.add_system_message(
                "Chatting without virtual-context enrichment."
            )

        # Initialize provider
        try:
            self.provider = ChatProvider(api_key=self._api_key, model=self._model)
        except Exception as e:
            self._chat_view.add_system_message(f"Error: {e}")
            self._chat_view.add_system_message(
                "Set ANTHROPIC_API_KEY or restart with --api-key."
            )
            self.provider = None
            return

        self._chat_view.add_system_message(
            f"Connected to {self._model}. Type a message and press Enter to send."
        )
        if self.engine:
            cw = self.engine.config.context_window
            self._chat_view.add_system_message(
                f"Virtual-context active. Window: {cw:,} tokens."
            )
            self._budget_bar.update_budget(0, {}, budget=cw)

        if self._replay_prompts:
            n = len(self._replay_prompts)
            self._chat_view.add_system_message(
                f"Replay mode: {n} prompt{'s' if n != 1 else ''} queued."
            )
            self._run_replay()
        else:
            self.query_one("#input-box", InputBox).focus()

    @work(thread=True)
    def _run_replay(self) -> None:
        """Send queued replay prompts sequentially in a single worker thread."""
        self._replaying = True

        for i, prompt in enumerate(self._replay_prompts, 1):
            self.call_from_thread(
                self._chat_view.add_system_message,
                f"Replay [{i}/{len(self._replay_prompts)}]",
            )
            self.call_from_thread(self._chat_view.add_user_message, prompt)
            if prompt.strip() != "/compact":
                self._conversation_history.append(Message(role="user", content=prompt))

            # Run the turn directly in this thread (no separate worker)
            self._execute_turn(prompt)

        self._replaying = False

        self.call_from_thread(
            self._chat_view.add_system_message,
            f"Replay complete. {len(self._replay_prompts)} turns sent.",
        )
        # Auto-save session after replay
        self.call_from_thread(self.action_save_session)
        self.call_from_thread(
            lambda: self.query_one("#input-box", InputBox).focus()
        )

    @property
    def _chat_view(self) -> ChatView:
        return self.query_one("#chat-view", ChatView)

    @property
    def _tag_panel(self) -> TagPanel:
        return self.query_one("#tag-panel", TagPanel)

    @property
    def _budget_bar(self) -> BudgetBar:
        return self.query_one("#budget-bar", BudgetBar)

    @property
    def _turn_list(self) -> TurnList:
        return self.query_one("#turn-list", TurnList)

    def on_input_box_message_submitted(self, event: InputBox.MessageSubmitted) -> None:
        if self._streaming:
            return
        if self.provider is None:
            self._chat_view.add_system_message("No API key configured. Cannot send.")
            return
        self._send_message(event.text)

    def _send_message(self, text: str) -> None:
        self._streaming = True
        self._chat_view.add_user_message(text)

        if text.strip() == "/compact":
            self._do_compact_turn()
            return

        # Add to conversation history
        self._conversation_history.append(Message(role="user", content=text))

        # Run engine inbound + streaming in background worker
        self._do_turn(text)

    @work(thread=True)
    def _do_turn(self, user_message: str) -> None:
        """Background worker wrapper for interactive (non-replay) turns."""
        try:
            self._execute_turn(user_message)
        finally:
            self._streaming = False

    @work(thread=True)
    def _do_compact_turn(self) -> None:
        """Background worker for /compact slash command."""
        try:
            self._execute_compact()
        finally:
            self._streaming = False

    def _execute_compact(self) -> None:
        """Handle /compact slash command."""
        if not self.engine:
            self.call_from_thread(
                self._chat_view.add_system_message,
                "No engine configured. Cannot compact.",
            )
            return

        compaction = None
        try:
            compaction = self.engine.compact_manual(self._conversation_history)
        except Exception as e:
            self.call_from_thread(
                self._chat_view.add_system_message,
                f"Compact error: {e}",
            )

        if compaction is None:
            summary = "Nothing to compact."
        else:
            summary = (
                f"Compacted {compaction.segments_compacted} segments, "
                f"freed {compaction.tokens_freed:,} tokens. "
                f"Tags: {', '.join(compaction.tags)}"
            )
            if compaction.tag_summaries_built > 0:
                summary += f"\nBuilt {compaction.tag_summaries_built} tag summaries."

        self.call_from_thread(self._chat_view.add_system_message, summary)

        self._turn_counter += 1
        turn = TurnRecord(
            turn_number=self._turn_counter,
            user_message="/compact",
            assistant_message=summary,
            assembled=AssembledContext(),
            compaction=compaction,
            tags=compaction.tags if compaction else [],
            primary_tag="_system",
            input_tokens=0,
            turns_in_payload=0,
            api_payload={},
        )
        self._turns.append(turn)
        self.call_from_thread(self._update_side_panel_complete, turn, compaction)

    def _execute_turn(self, user_message: str) -> None:
        """Run the full turn: engine inbound -> stream -> engine turn complete.

        Called from either ``_do_turn`` (interactive) or ``_run_replay``.
        Must be called from a background thread (uses ``call_from_thread``).
        """
        if user_message.strip() == "/compact":
            self._execute_compact()
            return

        assembled = AssembledContext()
        system_text = ""

        # 1. Engine: tag + retrieve + assemble
        if self.engine:
            try:
                assembled = self.engine.on_message_inbound(
                    user_message, self._conversation_history
                )
                system_text = assembled.prepend_text

                # Update side panel on UI thread
                self.call_from_thread(self._update_side_panel_inbound, assembled)
            except Exception as e:
                self.call_from_thread(
                    self._chat_view.add_system_message,
                    f"Engine error: {e}",
                )

        # 2. Filter history by tag relevance, then build messages for Anthropic
        if self.engine:
            filtered = self.engine.filter_history(
                self._conversation_history,
                current_tags=assembled.matched_tags,
            )
        else:
            filtered = self._conversation_history

        api_messages = [
            {"role": m.role, "content": m.content}
            for m in filtered
        ]

        # Brief mode: silently append instruction
        if self._brief_mode and api_messages and api_messages[-1]["role"] == "user":
            api_messages[-1] = dict(api_messages[-1])
            api_messages[-1]["content"] += "\n\n(Answer in 2 lines.)"

        # Snapshot the full API payload for the inspector
        api_payload = {
            "system": system_text,
            "messages": list(api_messages),
            "total_history": len(self._conversation_history),
            "filtered_history": len(filtered),
        }

        # 3. Stream response
        self.call_from_thread(self._chat_view.begin_assistant_message)
        full_response = []

        try:
            for chunk in self.provider.stream_message(
                system=system_text, messages=api_messages
            ):
                full_response.append(chunk)
                self.call_from_thread(self._chat_view.append_assistant_chunk, chunk)
        except Exception as e:
            self.call_from_thread(
                self._chat_view.add_system_message,
                f"Stream error: {e}",
            )
            return

        self.call_from_thread(self._chat_view.end_assistant_message)

        assistant_text = "".join(full_response)
        self._conversation_history.append(
            Message(role="assistant", content=assistant_text)
        )

        # 4. Engine: turn complete
        compaction = None
        # Start with inbound tags as fallback
        tags: list[str] = assembled.matched_tags or []
        primary_tag = tags[0] if tags else "_general"

        if self.engine:
            try:
                compaction = self.engine.on_turn_complete(self._conversation_history)

                # Get tags from latest turn tag index entry (richer: user+assistant pair)
                entries = self.engine._turn_tag_index.entries
                if entries:
                    latest = entries[-1]
                    # Merge: turn-complete tags + any inbound tags not already covered
                    tc_tags = list(latest.tags)
                    inbound_set = set(assembled.matched_tags or [])
                    extra = [t for t in inbound_set if t not in set(tc_tags)]
                    tags = tc_tags + extra
                    primary_tag = latest.primary_tag
            except Exception as e:
                self.call_from_thread(
                    self._chat_view.add_system_message,
                    f"Turn-complete error: {e}",
                )

        # 5. Compute actual tokens sent (system + all messages)
        payload_chars = len(system_text)
        for m in api_messages:
            payload_chars += len(m.get("content", ""))
        estimated_tokens = payload_chars // 4  # rough estimate

        # Count how many turns were packaged into this payload
        # Each turn = 2 messages (user + assistant), except the current
        # user message which is unpaired. So: (msg_count - 1) // 2 + 1
        msg_count = len(api_messages)
        turns_in_payload = (msg_count + 1) // 2  # round up for unpaired user msg

        # 6. Record turn
        self._turn_counter += 1
        turn = TurnRecord(
            turn_number=self._turn_counter,
            user_message=user_message,
            assistant_message=assistant_text,
            assembled=assembled,
            compaction=compaction,
            tags=tags,
            primary_tag=primary_tag,
            temporal=False,
            input_tokens=estimated_tokens,
            turns_in_payload=turns_in_payload,
            api_payload=api_payload,
        )
        self._turns.append(turn)

        # Update side panel
        self.call_from_thread(self._update_side_panel_complete, turn, compaction)

    def _update_side_panel_inbound(self, assembled: AssembledContext) -> None:
        """Update side panel after engine.on_message_inbound."""
        budget = self.engine.config.context_window if self.engine else 120_000
        self._budget_bar.update_budget(
            assembled.total_tokens, assembled.budget_breakdown, budget=budget
        )

        # Show matched tags immediately (before turn completes)
        if assembled.matched_tags:
            tag_scores = [(tag, 0.5) for tag in assembled.matched_tags]
            self._tag_panel.update_tags(tag_scores)

    def _update_side_panel_complete(
        self, turn: TurnRecord, compaction
    ) -> None:
        """Update side panel after engine.on_turn_complete."""
        # Update tag panel with recent tags (recency-weighted)
        tag_recency: dict[str, float] = {}
        recent = self._turns[-8:]
        for i, t in enumerate(recent):
            weight = (i + 1) / len(recent)  # 0.125 → 1.0
            for tag in t.tags:
                tag_recency[tag] = max(tag_recency.get(tag, 0.0), weight)
        sorted_tags = sorted(tag_recency.items(), key=lambda x: x[1], reverse=True)
        self._tag_panel.update_tags(sorted_tags[:10])

        # Add turn to list
        self._turn_list.add_turn(turn)

        # Update compaction log
        if compaction:
            log_widget = self.query_one("#compaction-log", Static)
            ts = compaction.timestamp.strftime("%H:%M")
            log_widget.update(
                f"[bold]COMPACTION LOG[/bold]\n"
                f"  [{ts}] {compaction.segments_compacted} segments, "
                f"-{compaction.tokens_freed:,}t\n"
                f"  Tags: {', '.join(compaction.tags)}"
            )

    def action_inspect_turn(self) -> None:
        """Open modal for the currently selected turn."""
        turn = self._turn_list.selected_turn
        if turn:
            self.push_screen(TurnInspector(turn, all_turns=self._turns))

    def action_prev_turn(self) -> None:
        """Select previous turn in the turn list."""
        self._turn_list.select_prev()

    def action_next_turn(self) -> None:
        """Select next turn in the turn list."""
        self._turn_list.select_next()

    def action_toggle_brief(self) -> None:
        """Toggle brief mode — silently appends 'answer in 2 lines' to prompts."""
        self._brief_mode = not self._brief_mode
        state = "ON" if self._brief_mode else "OFF"
        self._chat_view.add_system_message(f"Brief mode {state}")

    def action_compact(self) -> None:
        """Trigger manual compaction via Ctrl+K."""
        if self._streaming:
            return
        self._send_message("/compact")

    def action_save_session(self) -> None:
        """Save full session log to vc-session.json."""
        if not self._turns:
            self._chat_view.add_system_message("Nothing to save yet.")
            return
        path = save_session(self._turns)
        self._chat_view.add_system_message(f"Session saved to {path.resolve()}")

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
        yield Footer()


def run_chat(
    config_path: str | None = None,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    replay_prompts: list[str] | None = None,
) -> None:
    """Entry point for the TUI chat."""
    app = VChatApp(
        config_path=config_path,
        api_key=api_key,
        model=model,
        replay_prompts=replay_prompts,
    )
    app.run()
