"""Unit tests for the pure protected-window merge helper.

Covers ``_merge_protected_window`` from
``virtual_context.core.protected_window``. The merge helper has no
engine, store, or proxy-state dependency, so these tests construct
synthetic ``Message`` lists and ``CanonicalTurnRow`` namespaces and
assert dedup / ordering / mode semantics directly.
"""

from __future__ import annotations

from types import SimpleNamespace

from virtual_context.core.protected_window import _merge_protected_window
from virtual_context.types import Message


def _row(
    *,
    canonical_turn_id: str = "",
    turn_number: int = 0,
    sort_key: float = 0.0,
    turn_hash: str = "",
    user_content: str = "",
    assistant_content: str = "",
    created_at: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        canonical_turn_id=canonical_turn_id,
        turn_number=turn_number,
        sort_key=sort_key,
        turn_hash=turn_hash,
        user_content=user_content,
        assistant_content=assistant_content,
        created_at=created_at,
        last_seen_at="",
        first_seen_at="",
        updated_at="",
    )


def _msg(role: str, content: str, **metadata) -> Message:
    return Message(role=role, content=content, metadata=metadata if metadata else None)


def test_off_mode_is_identity() -> None:
    payload = [_msg("user", "hi"), _msg("assistant", "hello")]
    rows = [_row(canonical_turn_id="db-1", user_content="db user", assistant_content="db asst")]
    merged = _merge_protected_window(payload, rows, mode="off")
    assert [m.content for m in merged] == ["hi", "hello"]
    # Input must not be mutated.
    assert merged is not payload
    assert payload == [_msg("user", "hi"), _msg("assistant", "hello")]


def test_merge_empty_rows_returns_payload_copy() -> None:
    payload = [_msg("user", "hi")]
    merged = _merge_protected_window(payload, [], mode="merge")
    assert [m.content for m in merged] == ["hi"]
    assert merged is not payload


def test_merge_appends_new_rows_chronologically() -> None:
    payload = [_msg("user", "active turn")]
    rows = [
        _row(canonical_turn_id="db-2", sort_key=2.0, user_content="b-user", assistant_content="b-asst"),
        _row(canonical_turn_id="db-1", sort_key=1.0, user_content="a-user", assistant_content="a-asst"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    # sort_key ASC ordering: a-user, a-asst, b-user, b-asst appended
    contents = [m.content for m in merged]
    assert contents == [
        "active turn",
        "a-user", "a-asst",
        "b-user", "b-asst",
    ]


def test_merge_dedups_by_canonical_turn_id() -> None:
    payload = [
        _msg("user", "stamped user", canonical_turn_id="db-1", turn_number=5),
        _msg("assistant", "stamped asst", canonical_turn_id="db-1", turn_number=5),
    ]
    rows = [
        _row(canonical_turn_id="db-1", sort_key=1.0, user_content="db user dup", assistant_content="db asst dup"),
        _row(canonical_turn_id="db-2", sort_key=2.0, user_content="new user", assistant_content="new asst"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    contents = [m.content for m in merged]
    # Payload entries win; db-1 row is dropped; db-2 appended.
    assert contents == ["stamped user", "stamped asst", "new user", "new asst"]


def test_merge_dedups_by_turn_hash_fallback() -> None:
    payload = [
        _msg("user", "u", turn_hash="hashA"),
        _msg("assistant", "a", turn_hash="hashA"),
    ]
    rows = [
        _row(canonical_turn_id="db-x", turn_hash="hashA", sort_key=1.0, user_content="dup u", assistant_content="dup a"),
        _row(canonical_turn_id="db-y", turn_hash="hashB", sort_key=2.0, user_content="new u", assistant_content="new a"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    contents = [m.content for m in merged]
    assert contents == ["u", "a", "new u", "new a"]


def test_merge_short_repeated_turns_with_distinct_ids_not_false_deduped() -> None:
    payload = [
        _msg("user", "ok", canonical_turn_id="db-1", turn_number=1),
        _msg("assistant", "k", canonical_turn_id="db-1", turn_number=1),
    ]
    rows = [
        _row(canonical_turn_id="db-2", sort_key=2.0, user_content="ok", assistant_content="k"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    contents = [m.content for m in merged]
    # Both "ok" / "k" pairs present; canonical_turn_id keeps them distinct.
    assert contents == ["ok", "k", "ok", "k"]


def test_merge_tool_only_user_row_emits_one_message() -> None:
    payload: list[Message] = []
    rows = [
        _row(canonical_turn_id="db-tool", sort_key=1.0, user_content="user only", assistant_content=""),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    assert len(merged) == 1
    assert merged[0].role == "user"
    assert merged[0].content == "user only"


def test_merge_tool_only_assistant_row_emits_one_message() -> None:
    payload: list[Message] = []
    rows = [
        _row(canonical_turn_id="db-tool", sort_key=1.0, user_content="", assistant_content="asst only"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    assert len(merged) == 1
    assert merged[0].role == "assistant"
    assert merged[0].content == "asst only"


def test_merge_tie_break_determinism() -> None:
    """Rows with identical sort_key sort by created_at ASC then canonical_turn_id ASC."""
    payload: list[Message] = []
    rows = [
        _row(canonical_turn_id="z", sort_key=1.0, created_at="2025-01-01T00:00:00Z",
             user_content="u-z", assistant_content="a-z"),
        _row(canonical_turn_id="a", sort_key=1.0, created_at="2025-01-01T00:00:00Z",
             user_content="u-a", assistant_content="a-a"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    contents = [m.content for m in merged]
    # "a" canonical_turn_id sorts before "z" deterministically.
    assert contents == ["u-a", "a-a", "u-z", "a-z"]


def test_merge_does_not_mutate_input() -> None:
    payload = [_msg("user", "p")]
    rows = [_row(canonical_turn_id="db-1", sort_key=1.0, user_content="u", assistant_content="a")]
    merged = _merge_protected_window(payload, rows, mode="merge")
    assert merged is not payload
    assert [m.content for m in payload] == ["p"]
