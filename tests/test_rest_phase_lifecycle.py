"""Interactive prepare-then-ingest conversations never leave phase='init'.

On the single-turn completion flow every prepare persists a fresh
untagged user half BEFORE the phase decision evaluates, so the
``total == done`` branch that flips ``'init'`` to ``'active'`` is
unreachable on that lane: the decision always sees ``total > done``
with an incomplete physical group and returns without transitioning.
Tagging then completes in the post-ingest path, whose self-heal only
covers ``phase == 'ingesting'``. Net effect: a conversation on this
lane reports ``phase='init'`` forever no matter how many fully tagged
turns it holds, which misclassifies it to every phase-keyed consumer
(idle cleanup, compaction backlog detection, reattribution).

Fix (BUG-057): phase is derived from retrievable content. A
conversation in ``'init'`` that holds at least one fully tagged group
is established and flips to ``'active'`` — at tagging completion, and
as a self-heal on the next prepare.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.types import Message


def _turn(user_text: str, assistant_text: str) -> list[Message]:
    return [
        Message(role="user", content=user_text),
        Message(role="assistant", content=assistant_text),
    ]


def _prepare(state, history: list[Message], user_text: str):
    body = {
        "messages": [
            {"role": m.role, "content": m.content} for m in history
        ] + [{"role": "user", "content": user_text}],
    }
    return state.handle_prepare_payload(
        body=body,
        payload_accounting={
            "raw_payload_entry_count": len(body["messages"]),
            "ingestible_entry_count": len(body["messages"]),
        },
    )


@pytest.mark.regression("BUG-057")
def test_one_turn_conversation_activates_after_prepare_ingest_tag(tmp_path: Path):
    """The lane's normal shape: prepare persists the user half, the
    completion persists the assistant half, tagging completes in the
    post-ingest path. The conversation must come out ``'active'``."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    try:
        conv = state.engine.config.conversation_id
        inner = _inner_store(state.engine)

        decision = _prepare(state, [], "what weight should I squat this week")
        assert decision.phase == "init"

        history = _turn(
            "what weight should I squat this week",
            "start at 185 for your working sets",
        )
        state.engine.persist_completed_turn(list(history))
        state._run_tag_turn(list(history))

        assert inner.get_conversation_phase(conv) == "active", (
            "a fully tagged one-turn conversation must not stay 'init'"
        )
    finally:
        state.engine.close()


@pytest.mark.regression("BUG-057")
def test_next_prepare_activates_stuck_init_conversation(tmp_path: Path):
    """Self-heal ordering: a conversation left at ``'init'`` with tagged
    content (lost flip, historical steady state) must activate on the
    next prepare even though that prepare persists a fresh untagged
    user half first."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    try:
        conv = state.engine.config.conversation_id
        inner = _inner_store(state.engine)

        history = _turn("log my deadlift session", "logged: 3x5 at 315")
        _prepare(state, [], "log my deadlift session")
        state.engine.persist_completed_turn(list(history))
        state._run_tag_turn(list(history))

        # Simulate the flip never having happened for this conversation.
        inner.set_phase(
            conversation_id=conv,
            lifecycle_epoch=int(state.engine._engine_state.lifecycle_epoch),
            phase="init",
        )
        assert inner.get_conversation_phase(conv) == "init"

        decision = _prepare(state, history, "how about my squat")
        assert decision.phase in ("active", "ingesting"), decision.phase
        assert inner.get_conversation_phase(conv) != "init", (
            "a prepare against tagged content must lift the conversation "
            "out of 'init'"
        )
    finally:
        state.engine.close()


@pytest.mark.regression("BUG-057")
def test_first_prepare_alone_keeps_init(tmp_path: Path):
    """Guard: a conversation whose only content is the freshly persisted
    user half has no tagged group and must stay ``'init'``."""
    from tests.test_handle_prepare_payload import _make_proxy_state, _inner_store

    state = _make_proxy_state(tmp_path)
    try:
        conv = state.engine.config.conversation_id
        inner = _inner_store(state.engine)
        _prepare(state, [], "first message, nothing answered yet")
        assert inner.get_conversation_phase(conv) == "init"
    finally:
        state.engine.close()
