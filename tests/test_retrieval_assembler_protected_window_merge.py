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
from virtual_context.types import Message, build_user_turn_metadata


def _row(
    *,
    canonical_turn_id: str = "",
    turn_number: int = 0,
    sort_key: float = 0.0,
    turn_hash: str = "",
    user_content: str = "",
    assistant_content: str = "",
    created_at: str = "",
    turn_group_number: int = -1,
    source_message_id: str = "",
    origin_channel_id: str = "",
    origin_channel_label: str = "",
    audience_conversation_id: str = "",
    sender: str = "",
    sender_actor_id: str = "",
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
        turn_group_number=turn_group_number,
        source_message_id=source_message_id,
        origin_channel_id=origin_channel_id,
        origin_channel_label=origin_channel_label,
        audience_conversation_id=audience_conversation_id,
        sender=sender,
        sender_actor_id=sender_actor_id,
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


def test_merge_inserts_new_rows_before_trailing_active_user() -> None:
    payload = [_msg("user", "active turn")]
    rows = [
        _row(canonical_turn_id="db-2", sort_key=2.0, user_content="b-user", assistant_content="b-asst"),
        _row(canonical_turn_id="db-1", sort_key=1.0, user_content="a-user", assistant_content="a-asst"),
    ]
    merged = _merge_protected_window(payload, rows, mode="merge")
    # sort_key ASC ordering, with the unstamped active request remaining last.
    contents = [m.content for m in merged]
    assert contents == [
        "a-user", "a-asst",
        "b-user", "b-asst",
        "active turn",
    ]


def test_merge_inserts_before_every_trailing_unstamped_user() -> None:
    payload = [
        _msg("assistant", "completed reply", canonical_turn_id="old"),
        _msg("user", "queued one"),
        _msg("user", "queued two"),
    ]
    rows = [_row(
        canonical_turn_id="db-1",
        sort_key=1.0,
        user_content="cross-channel history",
    )]

    merged = _merge_protected_window(payload, rows, mode="merge")

    assert [m.content for m in merged] == [
        "completed reply",
        "cross-channel history",
        "queued one",
        "queued two",
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


def test_merge_dedups_same_request_race_by_source_message_id() -> None:
    payload = [Message(
        role="user",
        content="current request",
        metadata=build_user_turn_metadata(source_message_id="discord-123"),
    )]
    rows = [_row(
        canonical_turn_id="db-current",
        sort_key=9.0,
        source_message_id="discord-123",
        user_content="current request",
    )]
    merged = _merge_protected_window(payload, rows, mode="merge")
    assert [m.content for m in merged] == ["current request"]


def test_merge_propagates_group_channel_to_separate_assistant_row() -> None:
    rows = [
        _row(
            canonical_turn_id="u",
            turn_group_number=4,
            sort_key=1.0,
            user_content="preference",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="guild-owner",
            sender="optics",
            sender_actor_id="actor:discord:1",
        ),
        _row(
            canonical_turn_id="a",
            turn_group_number=4,
            sort_key=2.0,
            assistant_content="acknowledged",
        ),
    ]
    merged = _merge_protected_window([], rows, mode="merge")
    assert merged[0].metadata["origin_channel_id"] == "chan-a"
    assert merged[1].metadata["origin_channel_id"] == "chan-a"
    assert merged[1].metadata["audience_conversation_id"] == "guild-owner"
    assert merged[0].metadata["sender_actor_id"] == "actor:discord:1"
    assert "sender_actor_id" not in merged[1].metadata


def test_merge_never_propagates_group_channel_to_unproved_user_row() -> None:
    rows = [
        _row(
            canonical_turn_id="guild-user",
            turn_group_number=4,
            sort_key=1.0,
            user_content="proved guild message",
            origin_channel_id="chan-a",
            audience_conversation_id="guild-owner",
        ),
        _row(
            canonical_turn_id="unproved-user",
            turn_group_number=4,
            sort_key=2.0,
            user_content="must remain ineligible",
        ),
    ]

    merged = _merge_protected_window([], rows, mode="merge")

    assert merged[0].metadata["origin_channel_id"] == "chan-a"
    assert "origin_channel_id" not in merged[1].metadata
    assert "audience_conversation_id" not in merged[1].metadata


def test_merge_disables_assistant_inheritance_for_colliding_user_group() -> None:
    rows = [
        _row(
            canonical_turn_id="guild-user",
            turn_group_number=4,
            sort_key=1.0,
            user_content="proved guild message",
            origin_channel_id="chan-a",
            audience_conversation_id="guild-owner",
        ),
        _row(
            canonical_turn_id="unproved-user",
            turn_group_number=4,
            sort_key=2.0,
            user_content="unproved second user",
        ),
        _row(
            canonical_turn_id="ambiguous-assistant",
            turn_group_number=4,
            sort_key=3.0,
            assistant_content="cannot safely choose an audience",
        ),
    ]

    merged = _merge_protected_window([], rows, mode="merge")

    assert "origin_channel_id" not in merged[2].metadata
    assert "audience_conversation_id" not in merged[2].metadata
