"""Headless replay runner — no TUI, same engine + provider pipeline."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..engine import VirtualContextEngine
from ..types import AssembledContext, CompactionReport, Message
from .chat_provider import ChatProvider
from .state import TurnRecord, save_session


class HeadlessRunner:
    """Run prompts through engine + Anthropic without a terminal UI.

    Produces the same ``TurnRecord`` / ``vc-session.json`` output as the
    interactive TUI, but prints progress to stderr instead of rendering
    widgets.

    ``on_turn_complete`` runs asynchronously in a background thread.
    The next turn's ``on_message_inbound`` blocks until it finishes,
    ensuring the TurnTagIndex is consistent before retrieval.
    """

    def __init__(
        self,
        config_path: str | None = None,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        brief_mode: bool = True,
    ) -> None:
        self._config_path = config_path
        self._brief_mode = brief_mode
        self._conversation_history: list[Message] = []
        self._turns: list[TurnRecord] = []
        self._turn_counter = 0
        self._pending_complete: Future | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)

        # Engine (optional — works without config)
        try:
            self.engine: VirtualContextEngine | None = VirtualContextEngine(
                config_path=config_path
            )
        except Exception as e:
            print(f"Engine init failed: {e}", file=sys.stderr)
            self.engine = None

        # Provider
        self.provider = ChatProvider(api_key=api_key, model=model)

    def run(
        self,
        prompts: list[str],
        output: Path | str | None = None,
    ) -> list[TurnRecord]:
        """Execute prompts sequentially, return TurnRecords.

        Args:
            prompts: User messages to send in order.
            output: Directory to write ``vc-session.json`` into.
                    Defaults to current working directory.
        """
        total = len(prompts)
        prev_turn: TurnRecord | None = None
        prev_index = 0
        for i, prompt in enumerate(prompts, 1):
            if prompt.strip() != "/compact":
                self._conversation_history.append(
                    Message(role="user", content=prompt)
                )
            # _execute_turn waits for previous on_turn_complete before proceeding,
            # so prev_turn's tags are resolved by the time this returns
            turn = self._execute_turn(prompt)
            if prev_turn is not None:
                self._print_turn(prev_index, total, prev_turn)
            prev_turn = turn
            prev_index = i

        # Wait for final on_turn_complete to resolve tags
        self._wait_for_complete()
        if prev_turn is not None:
            self._print_turn(prev_index, total, prev_turn)

        # Save session
        out_dir = str(output) if output is not None else "."
        path = save_session(self._turns, directory=out_dir)
        print(f"\nSession saved to {path.resolve()}", file=sys.stderr)

        return self._turns

    def _print_turn(self, index: int, total: int, turn: TurnRecord) -> None:
        """Print a turn's stats to stderr (called after tags are resolved)."""
        t = turn.timing
        timing_str = (
            f"wait={t.get('wait_ms', 0):.0f} "
            f"inbound={t.get('inbound_ms', 0):.0f} "
            f"filter={t.get('filter_ms', 0):.0f} "
            f"stream={t.get('llm_stream_ms', 0):.0f}"
        )
        print(
            f"Turn {index}/{total}: {turn.primary_tag} "
            f"({turn.input_tokens:,}t, {turn.turns_in_payload} bundled) "
            f"[{timing_str}]ms",
            file=sys.stderr,
        )

    def _wait_for_complete(self) -> None:
        """Block until the pending on_turn_complete finishes."""
        if self._pending_complete is not None:
            self._pending_complete.result()  # raises if the thread raised
            self._pending_complete = None

    def _execute_turn(self, user_message: str) -> TurnRecord:
        """Run one turn: engine inbound → stream → engine complete (async)."""
        if user_message.strip() == "/compact":
            return self._execute_compact()

        timing: dict[str, float] = {}

        # Block until previous on_turn_complete is done before retrieving
        t0 = time.perf_counter()
        self._wait_for_complete()
        timing["wait_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        assembled = AssembledContext()
        system_text = ""

        # 1. Engine: tag + retrieve + assemble
        if self.engine:
            try:
                t0 = time.perf_counter()
                assembled = self.engine.on_message_inbound(
                    user_message, self._conversation_history
                )
                timing["inbound_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                system_text = assembled.prepend_text
            except Exception as e:
                print(f"Engine error: {e}", file=sys.stderr)

        # 2. Filter history by tag relevance
        if self.engine:
            t0 = time.perf_counter()
            filtered = self.engine.filter_history(
                self._conversation_history,
                current_tags=assembled.matched_tags,
                broad=assembled.broad,
                temporal=assembled.temporal,
            )
            timing["filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            filtered = self._conversation_history

        api_messages = [
            {"role": m.role, "content": m.content} for m in filtered
        ]

        # Brief mode
        if (
            self._brief_mode
            and api_messages
            and api_messages[-1]["role"] == "user"
        ):
            api_messages[-1] = dict(api_messages[-1])
            api_messages[-1]["content"] += "\n\n(Answer in 2 lines.)"

        # Snapshot API payload
        api_payload = {
            "system": system_text,
            "messages": list(api_messages),
            "total_history": len(self._conversation_history),
            "filtered_history": len(filtered),
        }

        # 3. Stream response (collect all chunks)
        full_response: list[str] = []
        t0 = time.perf_counter()
        try:
            for chunk in self.provider.stream_message(
                system=system_text, messages=api_messages
            ):
                full_response.append(chunk)
        except Exception as e:
            print(f"Stream error: {e}", file=sys.stderr)
            pass
        timing["llm_stream_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        assistant_text = "".join(full_response)
        self._conversation_history.append(
            Message(role="assistant", content=assistant_text)
        )

        # 4. Compute token estimate (doesn't need on_turn_complete)
        payload_chars = len(system_text)
        for m in api_messages:
            payload_chars += len(m.get("content", ""))
        estimated_tokens = payload_chars // 4

        msg_count = len(api_messages)
        turns_in_payload = (msg_count + 1) // 2

        # 5. Record turn (tags filled in after on_turn_complete)
        self._turn_counter += 1
        turn = TurnRecord(
            turn_number=self._turn_counter,
            user_message=user_message,
            assistant_message=assistant_text,
            assembled=assembled,
            broad=assembled.broad,
            temporal=assembled.temporal,
            input_tokens=estimated_tokens,
            turns_in_payload=turns_in_payload,
            api_payload=api_payload,
            timing=timing,
        )
        self._turns.append(turn)

        # 6. Fire on_turn_complete in background
        if self.engine:
            history_snapshot = list(self._conversation_history)
            self._pending_complete = self._pool.submit(
                self._run_turn_complete, turn, history_snapshot
            )

        return turn

    def _run_turn_complete(
        self, turn: TurnRecord, history: list[Message]
    ) -> None:
        """Background: tag the turn, check compaction, update TurnRecord."""
        try:
            t0 = time.perf_counter()
            compaction = self.engine.on_turn_complete(history)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            turn.timing["turn_complete_ms"] = elapsed
            entries = self.engine._turn_tag_index.entries
            if entries:
                latest = entries[-1]
                turn.tags = latest.tags
                turn.primary_tag = latest.primary_tag
            turn.compaction = compaction
        except Exception as e:
            print(f"Turn-complete error: {e}", file=sys.stderr)

    def _execute_compact(self) -> TurnRecord:
        """Handle /compact slash command."""
        # Must wait for previous turn to finish before compacting
        self._wait_for_complete()

        compaction = None
        if self.engine:
            try:
                compaction = self.engine.compact_manual(self._conversation_history)
            except Exception as e:
                print(f"Compact error: {e}", file=sys.stderr)

        if compaction is None:
            summary = "Nothing to compact."
        else:
            summary = (
                f"Compacted {compaction.segments_compacted} segments, "
                f"freed {compaction.tokens_freed:,} tokens. "
                f"Tags: {', '.join(compaction.tags)}"
            )
            if compaction.tag_summaries_built > 0:
                summary += f" Built {compaction.tag_summaries_built} tag summaries."

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
        return turn
