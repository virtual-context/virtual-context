"""The widening reset never destroys more than the payload can rebuild.

``_detect_and_reset_widened_history`` purges every table for the
conversation and re-ingests from the incoming payload, on the premise
that the payload IS the client's authoritative full history. On the
single-turn completion lane that premise is false: clients send only
the newest turns, the durable store is the transcript, and the
worker-local baseline counters undercount it. The first-turn hash then
trivially differs and the growth ratio is trivially satisfied, so the
reset destroyed history the payload could never rebuild (BUG-061).

Fix: the reset compares the payload's ingestible entry count against
the DURABLE canonical count — never the worker-local counters. A
payload smaller than the durable record is suppressed with a
``HISTORY_WIDENING_SUPPRESSED`` log; a payload at least as large as
the durable record keeps the existing reset behavior.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from pathlib import Path

import pytest

from virtual_context.types import Message


def _pairs(*texts: str) -> list[Message]:
    """Alternate user/assistant messages from flat texts."""
    out: list[Message] = []
    for i, text in enumerate(texts):
        out.append(Message(
            role="user" if i % 2 == 0 else "assistant", content=text,
        ))
    return out


def _seed_turns(state, flat_texts: list[str]) -> None:
    """Persist completed turns durably through the completion path."""
    history: list[Message] = []
    for i in range(0, len(flat_texts), 2):
        history.append(Message(role="user", content=flat_texts[i]))
        history.append(Message(role="assistant", content=flat_texts[i + 1]))
        state.engine.persist_completed_turn(list(history))


def _durable_rows(state) -> int:
    conv = state.engine.config.conversation_id
    try:
        return int(
            state.engine._store.read_progress_snapshot(conv).total_ingestible
        )
    except KeyError:
        # The conversation row itself was purged.
        return 0


@pytest.mark.regression("BUG-061")
def test_smaller_payload_never_resets_durable_history(tmp_path: Path, caplog):
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)
    try:
        conv = state.engine.config.conversation_id
        _seed_turns(state, [
            "first question", "first answer",
            "second question", "second answer",
            "third question", "third answer",
        ])
        assert _durable_rows(state) == 6

        # The worker-local baseline saw only one turn — the incident's
        # undercount shape.
        state._ingested_conversations.add(conv)
        state._record_ingestion_watermark(
            _pairs("first question", "first answer"), conv,
        )

        # A newest-turns-only window: different first turn, two pairs.
        window = _pairs(
            "third question", "third answer",
            "fourth question", "fourth answer",
        )
        with caplog.at_level(logging.WARNING):
            widened = state._detect_and_reset_widened_history(window, conv)

        assert widened is False, (
            "a payload smaller than the durable record must never reset"
        )
        assert _durable_rows(state) == 6, "durable history must survive"
        assert any(
            "HISTORY_WIDENING_SUPPRESSED" in rec.message for rec in caplog.records
        ), caplog.records
    finally:
        state.engine.close()


@pytest.mark.regression("BUG-061")
def test_genuine_widening_still_resets(tmp_path: Path):
    from tests.test_handle_prepare_payload import _make_proxy_state

    state = _make_proxy_state(tmp_path)
    try:
        conv = state.engine.config.conversation_id
        _seed_turns(state, ["first question", "first answer"])
        assert _durable_rows(state) == 2

        state._ingested_conversations.add(conv)
        state._record_ingestion_watermark(
            _pairs("first question", "first answer"), conv,
        )

        # The client's buffer grew backward: a different first turn and
        # MORE history than the durable record — the payload can rebuild
        # everything the reset destroys.
        wider = _pairs(
            "older prelude question", "older prelude answer",
            "first question", "first answer",
            "second question", "second answer",
        )
        widened = state._detect_and_reset_widened_history(wider, conv)

        assert widened is True, "authoritative wider history must still reset"
        assert _durable_rows(state) == 0, "the reset must have purged the rows"
    finally:
        state.engine.close()
