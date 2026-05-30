"""Regression test for the retriever's ``post_compaction`` gate.

The gate must consult ``_engine_state.compacted_prefix_messages``
(durable, re-derived from canonical_turns on every hydration) rather
than ``_engine_state.flushed_prefix_messages`` (session-scoped,
designed to lag compacted under deferred-payload-mutation
deployments).

Concretely: when a freshly-hydrated engine carries the
post-compaction values that ``hydrate_from_session_state``'s
canonical-turns derivation produces (``compacted_prefix_messages > 0``
but ``flushed_prefix_messages == 0`` because the loaded SessionState
predates the next ``prepare_payload`` flush gate write), the first
inbound message of the request MUST still see ``post_compaction =
True`` so the retriever's ``summary_floor`` prefetch fires. Gating on
``flushed_prefix_messages`` made that first request fail closed.

These tests also assert the converse — pre-compaction state stays
gated False — and pin that the gate change does NOT mutate
``flushed_prefix_messages`` (preserving the deferred-payload
invariant ``flushed <= compacted`` and the session-scoped semantics
of the flushed watermark).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.retrieval_assembler import RetrievalAssembler
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.types import (
    AssembledContext,
    AssemblerConfig,
    EngineState,
    RetrievalResult,
    RetrieverConfig,
    TagGeneratorConfig,
    TurnTagEntry,
    VirtualContextConfig,
)


def _make_assembler(
    *,
    compacted_prefix_messages: int,
    flushed_prefix_messages: int,
) -> tuple[RetrievalAssembler, MagicMock]:
    """Build a RetrievalAssembler with stubbed delegates, captured retriever.

    Returns ``(assembler, retriever_mock)`` so tests can inspect the
    keyword arguments the assembler passes when delegating to
    ``retriever.retrieve``.
    """
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        tags_matched=[],
        summaries=[],
        total_tokens=0,
        facts=[],
        overflow_summaries=[],
        retrieval_metadata={},
    )

    assembler_delegate = MagicMock()
    assembler_delegate.assemble.return_value = AssembledContext()

    monitor = MagicMock()
    monitor.build_snapshot.return_value = SimpleNamespace(
        total_tokens=0,
        budget_tokens=10_000,
    )

    paging = MagicMock()
    paging.working_set = {}

    store = MagicMock()
    turn_tag_index = MagicMock()
    turn_tag_index.entries = []
    fact_curator = None

    engine_state = EngineState()
    engine_state.compacted_prefix_messages = compacted_prefix_messages
    engine_state.flushed_prefix_messages = flushed_prefix_messages
    engine_state.last_completed_turn = max(
        0, compacted_prefix_messages // 2 - 1,
    )

    config = VirtualContextConfig(
        context_window=10_000,
        tag_generator=TagGeneratorConfig(type="keyword"),
        retriever=RetrieverConfig(),
        assembler=AssemblerConfig(),
    )

    assembler = RetrievalAssembler(
        retriever=retriever,
        assembler=assembler_delegate,
        monitor=monitor,
        paging=paging,
        store=store,
        turn_tag_index=turn_tag_index,
        engine_state=engine_state,
        fact_curator=fact_curator,
        config=config,
        token_counter=lambda text: len(text) // 4,
        session_state_provider=None,
    )
    # ``_set_semantic`` is required to wire the optional semantic-search
    # dependency. Pass a mock so ``_get_recent_context`` doesn't blow up
    # on the bleed-threshold path.
    assembler._set_semantic(MagicMock())
    return assembler, retriever


class _AppendOnFlushEngineState:
    """Append one index entry after assembler snapshot capture."""

    compacted_prefix_messages = 0
    last_compacted_turn = -1
    conversation_generation = 0

    def __init__(
        self,
        turn_tag_index: TurnTagIndex,
        appended_entry: TurnTagEntry,
    ) -> None:
        self._turn_tag_index = turn_tag_index
        self._appended_entry = appended_entry
        self._flushed_read = False
        self.history_offset_total_turns: int | None = None

    @property
    def flushed_prefix_messages(self) -> int:
        if not self._flushed_read:
            self._flushed_read = True
            self._turn_tag_index.append(self._appended_entry)
        return 0

    def history_offset(
        self,
        history_len: int,
        *,
        total_turns_indexed: int | None = None,
        watermark: int | None = None,
    ) -> int:
        self.history_offset_total_turns = total_turns_indexed
        return 0


def _make_snapshot_assembler(
    *,
    turn_tag_index,
    engine_state,
    retrieval_result: RetrievalResult,
) -> tuple[RetrievalAssembler, MagicMock]:
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieval_result

    assembler_delegate = MagicMock()
    assembler_delegate.load_core_context.return_value = ""
    assembler_delegate.assemble.side_effect = lambda **kwargs: AssembledContext(
        core_context=kwargs.get("core_context", ""),
        conversation_history=kwargs.get("conversation_history", []),
    )

    monitor = MagicMock()
    monitor.build_snapshot.return_value = SimpleNamespace(
        total_tokens=0,
        budget_tokens=10_000,
    )

    paging = MagicMock()
    paging.working_set = {}

    config = VirtualContextConfig(
        context_window=10_000,
        tag_generator=TagGeneratorConfig(type="keyword"),
        retriever=RetrieverConfig(),
        assembler=AssemblerConfig(context_hint_enabled=False),
    )

    assembler = RetrievalAssembler(
        retriever=retriever,
        assembler=assembler_delegate,
        monitor=monitor,
        paging=paging,
        store=MagicMock(),
        turn_tag_index=turn_tag_index,
        engine_state=engine_state,
        fact_curator=None,
        config=config,
        token_counter=lambda text: len(text) // 4,
        session_state_provider=None,
    )
    assembler._set_semantic(MagicMock())
    return assembler, retriever


def test_on_message_inbound_uses_method_entry_turn_tag_snapshot():
    turn_tag_index = TurnTagIndex()
    stable_entry = TurnTagEntry(
        turn_number=0,
        message_hash="h0",
        tags=["stable"],
        primary_tag="stable",
    )
    late_entry = TurnTagEntry(
        turn_number=1,
        message_hash="h1",
        tags=["late"],
        primary_tag="late",
    )
    turn_tag_index.append(stable_entry)
    engine_state = _AppendOnFlushEngineState(turn_tag_index, late_entry)
    assembler, retriever = _make_snapshot_assembler(
        turn_tag_index=turn_tag_index,
        engine_state=engine_state,
        retrieval_result=RetrievalResult(
            tags_matched=["_general"],
            retrieval_metadata={"tags_from_message": ["_general"]},
        ),
    )

    result = assembler.on_message_inbound(
        message="what were we discussing?",
        conversation_history=[],
    )

    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["entries_snapshot"] == [stable_entry]
    assert set(call_kwargs["current_active_tags"]) == {"stable"}
    assert engine_state.history_offset_total_turns == 1
    assert late_entry in turn_tag_index.entries
    assert result.matched_tags == ["stable"]


def test_on_message_inbound_handles_missing_turn_tag_index():
    assembler, retriever = _make_snapshot_assembler(
        turn_tag_index=None,
        engine_state=EngineState(),
        retrieval_result=RetrievalResult(
            tags_matched=["_general"],
            retrieval_metadata={"tags_from_message": ["_general"]},
        ),
    )

    result = assembler.on_message_inbound(
        message="what were we discussing?",
        conversation_history=[],
    )

    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["entries_snapshot"] is None
    assert call_kwargs["current_active_tags"] == []
    assert result.matched_tags == ["_general"]


def test_post_compaction_gate_true_when_compacted_positive_and_flushed_zero():
    """The bug shape: SessionState reports zero-flushed (because the
    loaded blob predates the prepare_payload flush gate) but
    canonical_turns derivation has bumped compacted to a positive
    value during ``hydrate_from_session_state``.

    Before the fix, the retriever's ``post_compaction`` argument was
    computed from ``flushed_prefix_messages > 0`` and stayed False,
    causing the ``summary_floor`` prefetch path to be skipped on the
    first request of a freshly-hydrated engine — exactly the
    user-visible VCATTACH recall failure shape.

    The fix moves the gate onto ``compacted_prefix_messages > 0``,
    which is the durable signal — re-derived from canonical_turns on
    every hydration, never zero when compaction has actually happened.
    """
    assembler, retriever = _make_assembler(
        compacted_prefix_messages=958,
        flushed_prefix_messages=0,
    )

    assembler.on_message_inbound(message="what topics?", conversation_history=[])

    assert retriever.retrieve.call_count >= 1
    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["post_compaction"] is True, (
        "Gate must consult compacted_prefix_messages — the durable "
        "watermark — not flushed_prefix_messages, which can be 0 on a "
        "freshly-hydrated engine before prepare_payload's flush gate "
        "writes."
    )


def test_post_compaction_gate_false_when_compacted_zero():
    """Pre-compaction state (no segments yet) must stay gated False
    regardless of any flushed value. ``compacted_prefix_messages == 0``
    is the binary "compaction has not happened" signal."""
    assembler, retriever = _make_assembler(
        compacted_prefix_messages=0,
        flushed_prefix_messages=0,
    )

    assembler.on_message_inbound(message="what topics?", conversation_history=[])

    assert retriever.retrieve.call_count >= 1
    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["post_compaction"] is False, (
        "Pre-compaction state must NOT trigger summary_floor; the gate "
        "requires compacted_prefix_messages > 0."
    )


def test_post_compaction_gate_true_when_both_positive():
    """Normal post-flush steady state: both watermarks positive, gate
    True. Ensures the fix didn't break the happy path."""
    assembler, retriever = _make_assembler(
        compacted_prefix_messages=958,
        flushed_prefix_messages=958,
    )

    assembler.on_message_inbound(message="what topics?", conversation_history=[])

    assert retriever.retrieve.call_count >= 1
    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["post_compaction"] is True


def test_active_tags_cleared_under_bug_shape_compacted_positive_flushed_zero():
    """Companion to the primary post_compaction gate test: the
    active-tag clearing at the head of ``on_message_inbound``
    (``retrieval_assembler.py:157-160``) MUST also route through the
    new compacted-driven gate, not the legacy flushed-driven gate.

    Before the fix the assembler ran ``active_tags =
    self._get_active_tags(...)`` for the bug-shape input (compacted
    positive, flushed zero), which suppressed cross-tag retrieval on
    the first request of a freshly-hydrated post-compaction engine
    — exactly the comment's "don't suppress retrieval -- stored
    summaries are needed since raw turns have been compacted away"
    intent, inverted. The fix routes the same gate through
    ``_post_compaction``, so the assembler now passes ``active_tags
    = []`` to the retriever for the bug shape.
    """
    assembler, retriever = _make_assembler(
        compacted_prefix_messages=958,
        flushed_prefix_messages=0,
    )

    assembler.on_message_inbound(message="what topics?", conversation_history=[])

    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["current_active_tags"] == [], (
        "Bug shape (compacted > 0 with flushed == 0): active_tags "
        "must clear to []. Routing this gate through the flushed "
        "watermark suppressed retrieval on the first request of a "
        "freshly-hydrated post-compaction engine — the same failure "
        "shape as the primary summary_floor gate."
    )


def test_retrieval_does_not_mutate_flushed_prefix_messages():
    """Preserve the deferred-payload-mutation invariant that
    ``flushed_prefix_messages`` is session-scoped and lags
    ``compacted_prefix_messages``. The gate fix changes which field
    drives ``post_compaction``; it must NOT have a side effect on
    the flushed watermark.

    Regression contract:
        * ``flushed_prefix_messages`` value at request entry equals
          value at request exit — retrieval is pure-read for this
          field.
        * The retriever was invoked with ``post_compaction=True``
          despite ``flushed == 0`` (the bug-shape input).
    """
    assembler, retriever = _make_assembler(
        compacted_prefix_messages=958,
        flushed_prefix_messages=0,
    )
    before = assembler._engine_state.flushed_prefix_messages

    assembler.on_message_inbound(message="what topics?", conversation_history=[])

    after = assembler._engine_state.flushed_prefix_messages
    assert after == before == 0, (
        "Retrieval must not mutate flushed_prefix_messages; that field "
        "is the proxy's prepare_payload watermark and a session-scoped "
        "deferred-payload signal owned by a different code path."
    )

    call_kwargs = retriever.retrieve.call_args.kwargs
    assert call_kwargs["post_compaction"] is True
