"""The ingestion watermark's hash and count must move together.

``_detect_and_reset_widened_history`` purges a conversation — canonical turns
included — when the first turn group's hash differs from a recorded baseline AND
the completed-turn count grew past a threshold. Both come from a pair of
in-memory dicts. When only the count is rewritten, the pair describes two
different observations: a zeroed count beside a retained hash makes the growth
guard vacuous (``new_turns <= 0`` is false for any count), leaving one hash
comparison between an ordinary request and the purge.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from unittest.mock import MagicMock

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.state import ProxyState
from virtual_context.types import EngineState, Message

CONV = "watermark-conv"


def _state() -> ProxyState:
    engine = MagicMock()
    engine.config.conversation_id = CONV
    engine._turn_tag_index = TurnTagIndex()
    engine._engine_state = EngineState()
    engine._engine_state.compacted_prefix_messages = 0
    engine._store = MagicMock()
    engine._store.get_all_tags.return_value = []
    engine._store.iter_untagged_canonical_rows.return_value = []
    engine.config.proxy.history_widening_threshold = 0.10
    engine.config.monitor.context_window = 200000
    engine.config.monitor.protected_recent_turns = 6
    engine.config.tag_generator.context_lookback_pairs = 3
    engine.config.tag_generator.context_bleed_threshold = 0
    return ProxyState(engine, metrics=ProxyMetrics())


def _turn(first: str) -> list[Message]:
    """A buffer whose first completed group is identifiable by *first*."""
    return [
        Message(role="user", content=first),
        Message(role="assistant", content="A0"),
        Message(role="user", content="in-flight"),
    ]


def _no_completed_group() -> list[Message]:
    """Only the in-flight user turn: no completed group, so no head exists."""
    return [Message(role="user", content="in-flight")]


@pytest.mark.regression("BUG-053")
def test_group_less_buffer_does_not_zero_the_count() -> None:
    """Row 1 — a buffer with no completed group must update neither field."""
    state = _state()
    state._record_ingestion_watermark(_turn("Original first"), CONV)
    hash_before = state._ingested_first_hash[CONV]
    count_before = state._ingested_turn_count[CONV]
    assert count_before >= 1

    state._record_ingestion_watermark(_no_completed_group(), CONV)

    assert state._ingested_first_hash.get(CONV) == hash_before
    assert state._ingested_turn_count.get(CONV) == count_before, (
        "the count was rewritten from a buffer that carried no head, leaving "
        "it describing a different observation than the hash beside it"
    )


@pytest.mark.regression("BUG-053")
def test_empty_history_path_does_not_zero_an_existing_count() -> None:
    """Row 2 — THE DEFECT. The empty-history path must not zero the count
    while a hash from a real observation is still recorded."""
    state = _state()
    state._record_ingestion_watermark(_turn("Original first"), CONV)
    hash_before = state._ingested_first_hash[CONV]
    count_before = state._ingested_turn_count[CONV]

    state.resolve_prepare_state([])

    assert state._ingested_first_hash.get(CONV) == hash_before
    assert state._ingested_turn_count.get(CONV) == count_before, (
        "the empty-history path zeroed the count beside a retained hash; the "
        "growth guard is now vacuous for this conversation"
    )


@pytest.mark.regression("BUG-053")
def test_desynced_pair_does_not_make_the_growth_guard_vacuous() -> None:
    """Row 3 — after the sequence above, a differing head with no real growth
    must NOT purge. Today old_turns is 0, so any count counts as growth."""
    state = _state()
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_turn("Original first"), CONV)
    state.resolve_prepare_state([])
    state.engine._store.delete_conversation.reset_mock()

    widened = state._detect_and_reset_widened_history(_turn("Different first"), CONV)

    assert widened is False, (
        "a buffer with the same number of completed groups was treated as "
        "growth because the recorded count had been zeroed"
    )
    state.engine._store.delete_conversation.assert_not_called()


@pytest.mark.regression("BUG-053")
def test_no_baseline_is_created_from_a_group_less_buffer() -> None:
    """Row 5 — with no prior baseline, a group-less buffer writes neither key."""
    state = _state()
    state._record_ingestion_watermark(_no_completed_group(), CONV)

    assert CONV not in state._ingested_first_hash
    assert CONV not in state._ingested_turn_count, (
        "an orphan count was created with no hash beside it"
    )


@pytest.mark.regression("BUG-053")
def test_hash_is_written_before_the_count() -> None:
    """Row 6 — ordering. Gating both writes on one predicate is not atomicity:
    two assignments remain, and a reader between them sees a mixed pair.
    Hash-first fails closed, because the fresh hash matches the current head
    and the detector returns at its equality check.
    """
    state = _state()
    order: list[str] = []

    class _Recording(dict):
        def __init__(self, label: str) -> None:
            super().__init__()
            self._label = label

        def __setitem__(self, key, value):
            order.append(self._label)
            super().__setitem__(key, value)

    state._ingested_first_hash = _Recording("hash")
    state._ingested_turn_count = _Recording("count")

    state._record_ingestion_watermark(_turn("Original first"), CONV)

    assert order[:2] == ["hash", "count"], (
        f"writes landed in order {order}; a reader interleaving between them "
        "must never observe a new hash beside a stale count"
    )


def _multi_turn(first: str, groups: int) -> list[Message]:
    """A buffer with *groups* completed turn groups, headed by *first*."""
    messages: list[Message] = []
    for i in range(groups):
        messages.append(Message(role="user", content=first if i == 0 else f"U{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))
    messages.append(Message(role="user", content="in-flight"))
    return messages


@pytest.mark.regression("BUG-053")
def test_genuine_widening_still_purges_canonical_turns(tmp_path) -> None:
    """Row 4 — detection must still fire. Asserted through the effect on a
    real store: the canonical rows are gone, not that a mock was called.
    ``delete_conversation`` failures are swallowed and the detector returns
    True regardless, so the return value proves nothing about the purge.
    """
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore

    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id=CONV)
    now = utcnow_iso()
    for i in range(3):
        with store._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO canonical_turns (
                    canonical_turn_id, conversation_id, turn_hash, hash_version,
                    normalized_user_text, normalized_assistant_text,
                    user_content, assistant_content,
                    sort_key, source_batch_id, first_seen_at, last_seen_at,
                    covered_ingestible_entries, tagged_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 'u','a','u_raw','a_raw', ?, 'b', ?, ?, 1, NULL, ?, ?)
                """,
                (f"t{i}", CONV, f"h_t{i}", 1000.0 * (i + 1), now, now, now, now),
            )
    assert store.count_canonical_turns(CONV) == 3

    state = _state()
    state.engine._store = store
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_multi_turn("Original first", 2), CONV)

    widened = state._detect_and_reset_widened_history(_multi_turn("Different first", 4), CONV)

    assert widened is True
    assert store.count_canonical_turns(CONV) == 0, (
        "a genuinely widened history left the canonical rows in place; the "
        "detector has been disabled rather than made atomic"
    )


@pytest.mark.regression("BUG-053")
def test_widening_still_fires_after_the_empty_history_path(tmp_path) -> None:
    """Preserving the baseline across a group-less observation must not turn
    the detector off for that conversation. The path that used to zero the
    count is traversed first; a genuinely widened buffer must still purge.

    The narrower reading is the point: after this change a SMALLER buffer with
    a shifted head no longer purges, because the count it is compared against
    is a real observation rather than a zero. That is the defect being closed,
    not lost coverage. Real growth past the threshold still fires.
    """
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore

    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id=CONV)
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at, created_at, updated_at
            ) VALUES ('t0', ?, 'h_t0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b',
                      ?, ?, 1, NULL, ?, ?)
            """,
            (CONV, now, now, now, now),
        )

    state = _state()
    state.engine._store = store
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_multi_turn("Original first", 2), CONV)

    state.resolve_prepare_state([])

    widened = state._detect_and_reset_widened_history(_multi_turn("Different first", 4), CONV)

    assert widened is True, (
        "preserving the baseline across the empty-history path left the "
        "detector unable to fire on genuine widening"
    )
    assert store.count_canonical_turns(CONV) == 0


def _assistant_headed(groups: int) -> list[Message]:
    """A buffer whose first completed group is a lone assistant turn with empty
    content. ``pair_messages_into_turns`` closes a group on every assistant
    message, so a leading assistant message pairs into a completed group by
    itself, and joining that group's contents yields empty text.
    """
    messages: list[Message] = [Message(role="assistant", content="")]
    for i in range(groups - 1):
        messages.append(Message(role="user", content=f"U{i}"))
        messages.append(Message(role="assistant", content=f"A{i}"))
    messages.append(Message(role="user", content="in-flight"))
    return messages


@pytest.mark.regression("BUG-053")
def test_a_completed_group_can_have_empty_head_text() -> None:
    """The recorder's gate must be the completed-group count, not the head
    text, because those two predicates diverge.

    This is the fixture the other rows depend on. If it ever stops diverging
    the gate choice stops mattering, but while it does diverge, gating on the
    head text means "positive count, no write".
    """
    state = _state()
    buffer = _assistant_headed(2)

    assert state._history_turn_count(buffer) == 2
    assert state._combined_turn_text(buffer, 0) == "", (
        "the buffer no longer produces a completed group with empty head text"
    )


@pytest.mark.regression("BUG-053")
def test_empty_head_buffer_records_its_own_count() -> None:
    """A buffer with completed groups but empty head text is an observation.
    Both fields must be written from it, not skipped."""
    state = _state()
    state._record_ingestion_watermark(_turn("Original first"), CONV)

    state._record_ingestion_watermark(_assistant_headed(100), CONV)

    assert state._ingested_turn_count.get(CONV) == 100, (
        "the recorder skipped a buffer carrying 100 completed groups because "
        "its head text was empty, leaving a smaller count from an older buffer"
    )
    assert state._ingested_first_hash.get(CONV) == (
        hashlib.sha256(b"").hexdigest()[:16]
    ), "the hash was not rewritten from the same buffer that set the count"


@pytest.mark.regression("BUG-053")
def test_stale_small_count_does_not_purge_after_an_empty_head_buffer(tmp_path) -> None:
    """Skipping an empty-head buffer leaves the previous, SMALLER count in
    place, which makes ``new_turns > old_turns * 1.1`` easier to satisfy than
    the count that buffer would have recorded. The purge must not fire here.

    Sequence: a one-group baseline, then a hundred-group buffer whose head is
    empty, then a two-group buffer with a different head. Against the real
    count of 100 there is no growth. Against a retained count of 1 there is.
    """
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore

    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id=CONV)
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at, created_at, updated_at
            ) VALUES ('t0', ?, 'h_t0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b',
                      ?, ?, 1, NULL, ?, ?)
            """,
            (CONV, now, now, now, now),
        )

    state = _state()
    state.engine._store = store
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_turn("Original first"), CONV)
    state._record_ingestion_watermark(_assistant_headed(100), CONV)

    widened = state._detect_and_reset_widened_history(_multi_turn("Different first", 2), CONV)

    assert widened is False, (
        "two completed groups were treated as growth past a hundred, because "
        "the empty-head buffer's count was never recorded"
    )
    assert store.count_canonical_turns(CONV) == 1


@pytest.mark.regression("BUG-053")
def test_a_raising_count_helper_writes_neither_field() -> None:
    """Both values are computed before either is stored. If the count helper
    raises after the hash has already been stored, the conversation is left in
    exactly the hash-without-count state this fix exists to prevent."""
    state = _state()

    def _boom(_messages=None):
        raise RuntimeError("count helper failed")

    state._history_turn_count = _boom

    with pytest.raises(RuntimeError):
        state._record_ingestion_watermark(_turn("Original first"), CONV)

    assert CONV not in state._ingested_first_hash, (
        "a hash was stored before the count was computed; the count then "
        "failed, leaving the vacuous-growth-guard state"
    )
    assert CONV not in state._ingested_turn_count


@pytest.mark.regression("BUG-053")
def test_an_empty_head_baseline_still_arms_detection(tmp_path) -> None:
    """Recording a baseline from an empty-head buffer arms detection for a
    later buffer whose head is not empty. This is deliberate, and it is the
    one place the fix makes the purge reachable where it previously was not.

    The old writer stored no hash for such a buffer, so the detector's
    key-presence guard turned it off for the whole conversation. Storing the
    hash restores the detector's stated condition: the head prefix changed and
    the completed-group count grew past the threshold. An empty head can only
    ever be differed from, never matched, because the detector returns early on
    any incoming buffer whose own head is empty.
    """
    from virtual_context.core.canonical_turns import utcnow_iso
    from virtual_context.storage.sqlite import SQLiteStore

    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id=CONV)
    now = utcnow_iso()
    with store._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO canonical_turns (
                canonical_turn_id, conversation_id, turn_hash, hash_version,
                normalized_user_text, normalized_assistant_text,
                user_content, assistant_content,
                sort_key, source_batch_id, first_seen_at, last_seen_at,
                covered_ingestible_entries, tagged_at, created_at, updated_at
            ) VALUES ('t0', ?, 'h_t0', 1, 'u','a','u_raw','a_raw', 1000.0, 'b',
                      ?, ?, 1, NULL, ?, ?)
            """,
            (CONV, now, now, now, now),
        )

    state = _state()
    state.engine._store = store
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_assistant_headed(1), CONV)

    assert state._ingested_turn_count[CONV] == 1
    assert state._ingested_first_hash[CONV] == hashlib.sha256(b"").hexdigest()[:16]

    widened = state._detect_and_reset_widened_history(_multi_turn("Real head", 4), CONV)

    assert widened is True
    assert store.count_canonical_turns(CONV) == 0


@pytest.mark.regression("BUG-053")
def test_an_empty_head_baseline_is_never_matched() -> None:
    """The stored empty-head hash cannot suppress detection by matching, since
    the detector returns before the hash comparison whenever the incoming
    buffer's own head is empty."""
    state = _state()
    state._ingested_conversations.add(CONV)
    state._record_ingestion_watermark(_assistant_headed(1), CONV)

    widened = state._detect_and_reset_widened_history(_assistant_headed(100), CONV)

    assert widened is False
    state.engine._store.delete_conversation.assert_not_called()
