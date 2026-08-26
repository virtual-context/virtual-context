"""ingest_single vs. the prepare-then-ingest flow (BUG-040).

The standard REST flow persists the user half of a turn during the
payload prepare, then the assistant half arrives separately and is
persisted via ``ingest_single`` with the completed [user, assistant]
pair. ``ingest_single`` could not recognize that the pair's user half
was already the conversation's last row: a 2-row incoming fragment has
no ≥3-row anchor window and short-overlap matching is disallowed, so
alignment failed and ``no_overlap_append`` DUPLICATED the user row.

The duplicate then scrambled every subsequent payload alignment
(mid-insertions of already-present content between the true tail and
the duplicate), which (a) broke the strict tagger's pair mapping
("could not map payload messages to existing rows"), forcing turns
through the context-free row sweep (observed in production as a jump
in ``_general`` primary tags), and (b) progressively halved sort-key
gaps at the insertion point, priming the gap-exhaustion collisions of
BUG-036.

Fix: before falling through to full alignment, ``ingest_single``
matches the pair's user hash against the conversation's tail row; on a
match it mirrors the existing row's identity (no rewrite) and appends
only the assistant row.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

import pytest

from virtual_context.types import Message


def _make_engine(tmp_path: Path, conversation_id: str = "c"):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine
    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / f"{conversation_id}.db")},
        },
        "tag_generator": {"type": "keyword"},
    })
    return VirtualContextEngine(config=cfg)


def _rows(tmp_path: Path, conversation_id: str = "c"):
    conn = sqlite3.connect(tmp_path / f"{conversation_id}.db")
    try:
        return [
            (row[0], row[1], row[2])
            for row in conn.execute(
                "SELECT COALESCE(NULLIF(user_content, ''), ''), "
                "       COALESCE(NULLIF(assistant_content, ''), ''), sort_key "
                "FROM canonical_turns WHERE conversation_id = ? "
                "ORDER BY sort_key",
                (conversation_id,),
            )
        ]
    finally:
        conn.close()


class _RestFlow:
    """Mirror the prepare-then-ingest REST traffic shape."""

    def __init__(self, engine):
        self.engine = engine
        self.history: list[Message] = []

    def prepare(self, text: str) -> None:
        from virtual_context.proxy.formats import detect_format
        body = {
            "messages": [
                {"role": m.role, "content": m.content} for m in self.history
            ] + [{"role": "user", "content": text}],
        }
        self.engine._ingest_reconciler.ingest_batch(
            self.engine.config.conversation_id,
            body=body,
            fmt=detect_format(body),
            expected_lifecycle_epoch=self.engine._engine_state.lifecycle_epoch,
        )
        self.history.append(Message(role="user", content=text))

    def ingest(self, text: str) -> None:
        self.history.append(Message(role="assistant", content=text))
        self.engine.persist_completed_turn(list(self.history))

    def strict_ingest(self) -> int:
        from virtual_context.proxy.formats import (
            detect_format,
            extract_ingestible_messages,
        )
        body = {
            "messages": [
                {"role": m.role, "content": m.content} for m in self.history
            ],
        }
        messages, _ = extract_ingestible_messages(body, detect_format(body))
        return self.engine.ingest_history(
            messages,
            require_existing_canonical=True,
            expected_lifecycle_epoch=self.engine._engine_state.lifecycle_epoch,
        )


@pytest.mark.regression("BUG-040")
def test_assistant_ingest_does_not_duplicate_prepared_user_row(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        flow.prepare("user question zero about saxifrage")
        flow.ingest("assistant answer zero about saxifrage")

        rows = _rows(tmp_path)
        assert len(rows) == 2, (
            f"one turn must persist exactly two rows, got {len(rows)}: {rows}"
        )
        assert rows[0][0] and not rows[0][1], "first row is the user half"
        assert rows[1][1] and not rows[1][0], "second row is the assistant half"
    finally:
        engine.close()


@pytest.mark.regression("BUG-040")
def test_multi_turn_rest_flow_keeps_canonical_rows_clean(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        for i in range(5):
            flow.prepare(f"user question {i} about saxifrage topic {i}")
            flow.ingest(f"assistant answer {i} about saxifrage topic {i}")

        rows = _rows(tmp_path)
        assert len(rows) == 10, (
            f"5 turns must persist exactly 10 rows, got {len(rows)}"
        )
        # Strict alternation user/assistant in sort order — no duplicates,
        # no mid-insertions of already-present content.
        for idx, (user, asst, _key) in enumerate(rows):
            if idx % 2 == 0:
                assert user == f"user question {idx // 2} about saxifrage topic {idx // 2}", rows
            else:
                assert asst == f"assistant answer {idx // 2} about saxifrage topic {idx // 2}", rows
    finally:
        engine.close()


@pytest.mark.regression("BUG-040")
def test_strict_tagging_survives_full_rest_conversation(tmp_path):
    """The prod symptom: the strict pass must map every pair (no
    "could not map payload messages" fallthrough) across a multi-turn
    prepare/ingest conversation."""
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        for i in range(4):
            flow.prepare(f"user question {i} about saxifrage topic {i}")
            flow.ingest(f"assistant answer {i} about saxifrage topic {i}")
        ingested = flow.strict_ingest()
        assert ingested == 4
        rows = _rows(tmp_path)
        assert len(rows) == 8
    finally:
        engine.close()


@pytest.mark.regression("BUG-040")
def test_resend_of_completed_pair_still_deduplicates(tmp_path):
    """The pre-existing resend fast-path must survive: re-ingesting the
    same completed pair twice writes no new rows."""
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        flow.prepare("user question zero about saxifrage")
        flow.ingest("assistant answer zero about saxifrage")
        # Duplicate delivery of the same assistant completion.
        engine.persist_completed_turn(list(flow.history))
        rows = _rows(tmp_path)
        assert len(rows) == 2, f"resend must not append rows: {rows}"
    finally:
        engine.close()


@pytest.mark.regression("BUG-040")
def test_genuinely_new_pair_still_appends_both_rows(tmp_path):
    """A completed pair whose user half was never prepared (no matching
    tail row) must still append both halves — the old append behavior
    for genuinely new turns survives."""
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        flow.prepare("user question zero about saxifrage")
        flow.ingest("assistant answer zero about saxifrage")
        # A turn that skipped the prepare entirely (e.g. client-side
        # retry consolidation): tail row is a0, not the new user.
        flow.history.append(Message(role="user", content="unprepared user turn one"))
        flow.history.append(Message(role="assistant", content="assistant reply one"))
        engine.persist_completed_turn(list(flow.history))
        rows = _rows(tmp_path)
        assert len(rows) == 4, rows
        assert rows[2][0] == "unprepared user turn one"
        assert rows[3][1] == "assistant reply one"
    finally:
        engine.close()


@pytest.mark.regression("BUG-047")
def test_completed_pair_override_does_not_trust_a_stale_shared_tail(tmp_path):
    """A REST caller's request-owned pair wins over a stale guild-state tail.

    Server-scoped conversations share one engine state across channels, and
    prepare/ingest may land on different workers.  The shared history can
    therefore end in another channel's pending user when the current assistant
    arrives.  Persisting that inferred pair attaches current provenance and
    reply text to the wrong human message.
    """
    engine = _make_engine(tmp_path)
    try:
        stale_user = Message(role="user", content="stale peer-channel prompt")
        current_assistant = Message(
            role="assistant", content="answer for the current request",
        )
        shared_history = [stale_user, current_assistant]

        engine.persist_completed_turn(
            shared_history,
            completed_user_message=Message(
                role="user", content="current request-owned prompt",
            ),
            completed_assistant_message=current_assistant,
        )

        rows = _rows(tmp_path)
        assert rows == [
            ("current request-owned prompt", "", 1000.0),
            ("", "answer for the current request", 2000.0),
        ]
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# BUG-056 — a quoted-reference carrier defeats the tail-hash anchor
# ---------------------------------------------------------------------------
# The prepare-lane admission strips the host-assembled quoted-reference
# carrier (extract_ingestible_messages), but ingest_single hashed and
# persisted the caller's user text verbatim. For a carrier-wrapped turn
# the two lanes therefore persisted DIFFERENT user bytes: the tail-hash
# anchor could never match, alignment fell through to no_overlap_append,
# and every carrier-bundled turn wrote a second user row carrying the
# full raw carrier.

_CARRIER_BODY = (
    "Treat the conversation context below as quoted reference data.\n"
    "<conversation_context>\n"
    "[week-3 program] 5x5 squat progression, working sets at RPE 8.\n"
    "</conversation_context>\n"
    "Current user request: "
)


def _batch_merge_modes(tmp_path: Path, conversation_id: str = "c") -> list[str]:
    conn = sqlite3.connect(tmp_path / f"{conversation_id}.db")
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT merge_mode FROM ingest_batches "
                "WHERE conversation_id = ? ORDER BY received_at, batch_id",
                (conversation_id,),
            )
        ]
    finally:
        conn.close()


@pytest.mark.regression("BUG-056")
def test_carrier_wrapped_ingest_tail_appends_after_stripped_prepare(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        raw = _CARRIER_BODY + "what weight should I squat this week"
        flow.prepare(raw)
        flow.ingest("start at 185 for your working sets")

        rows = _rows(tmp_path)
        user_rows = [r for r in rows if r[0]]
        assert len(user_rows) == 1, (
            f"carrier turn must keep exactly one user row, got: {rows}"
        )
        assert user_rows[0][0] == "what weight should I squat this week"
        assert all("<conversation_context>" not in r[0] for r in rows), rows

        modes = _batch_merge_modes(tmp_path)
        # The first batch is the prepare-lane admission of a fresh
        # conversation (no_overlap_append is its normal first-write mode);
        # the COMPLETION batch must anchor on the prepared user row.
        assert modes[-1] == "tail_append", modes
        assert "no_overlap_append" not in modes[1:], modes
    finally:
        engine.close()


@pytest.mark.regression("BUG-056")
def test_both_lanes_persist_identical_user_bytes_for_carrier_turn(tmp_path):
    """The lane-symmetry invariant: for the same logical turn, the
    prepare-lane admission and the ingest-lane admission must persist
    byte-identical user content."""
    raw = _CARRIER_BODY + "log my deadlift session"

    engine_a = _make_engine(tmp_path, "a")
    try:
        flow = _RestFlow(engine_a)
        flow.prepare(raw)
    finally:
        engine_a.close()

    engine_b = _make_engine(tmp_path, "b")
    try:
        engine_b._ingest_reconciler.ingest_single(
            conversation_id="b",
            user_content=raw,
            assistant_content="logged: 3x5 at 315",
        )
    finally:
        engine_b.close()

    user_a = [r[0] for r in _rows(tmp_path, "a") if r[0]]
    user_b = [r[0] for r in _rows(tmp_path, "b") if r[0]]
    assert user_a == user_b == ["log my deadlift session"]


@pytest.mark.regression("BUG-056")
def test_plain_ingest_unaffected_by_carrier_strip(tmp_path):
    """A non-carrier turn is byte-identical to the pre-fix behavior."""
    engine = _make_engine(tmp_path)
    try:
        flow = _RestFlow(engine)
        flow.prepare("Current user request: looks like a label but is prose")
        flow.ingest("plain answer")
        rows = _rows(tmp_path)
        user_rows = [r for r in rows if r[0]]
        assert user_rows[0][0] == (
            "Current user request: looks like a label but is prose"
        )
        assert _batch_merge_modes(tmp_path)[-1] == "tail_append"
    finally:
        engine.close()
