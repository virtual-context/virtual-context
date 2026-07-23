"""Unit tests for the pure protected-window merge helper.

Covers ``_merge_protected_window`` from
``virtual_context.core.protected_window``. The merge helper has no
engine, store, or proxy-state dependency, so these tests construct
synthetic ``Message`` lists and ``CanonicalTurnRow`` namespaces and
assert dedup / ordering / mode semantics directly.
"""

from __future__ import annotations

from types import SimpleNamespace

from virtual_context.core.protected_window import (
    _merge_protected_window,
    _slice_payload_prefix_preserving_db_recent,
)
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
        _row(
            canonical_turn_id="db-2",
            turn_number=6,
            sort_key=2.0,
            user_content="new user",
            assistant_content="new asst",
        ),
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


def test_payload_offset_preserves_db_group_and_active_user() -> None:
    history = [
        _msg("user", f"old-{index}", canonical_turn_id=f"old-{index}")
        for index in range(7)
    ]
    history.extend([
        _msg(
            "user",
            "cross-channel instruction",
            source="db_recent",
            turn_group_number=4,
        ),
        _msg(
            "assistant",
            "cross-channel acknowledgement",
            source="db_recent",
            turn_group_number=4,
        ),
        _msg("user", "active request"),
    ])

    view = _slice_payload_prefix_preserving_db_recent(history, payload_offset=8)

    assert [message.content for message in view] == [
        "cross-channel instruction",
        "cross-channel acknowledgement",
        "active request",
    ]


def test_payload_offset_matches_plain_slice_without_db_or_active_tail() -> None:
    history = [
        _msg("user", "old-0", canonical_turn_id="old-0"),
        _msg("assistant", "old-1", canonical_turn_id="old-1"),
        _msg("user", "keep-2", canonical_turn_id="keep-2"),
        _msg("assistant", "keep-3", canonical_turn_id="keep-3"),
    ]

    view = _slice_payload_prefix_preserving_db_recent(history, payload_offset=2)

    assert [message.content for message in view] == ["keep-2", "keep-3"]


def test_payload_offset_preserves_every_trailing_unstamped_user() -> None:
    history = [
        _msg("assistant", "old", canonical_turn_id="old"),
        _msg("user", "queued one"),
        _msg("user", "queued two"),
    ]

    view = _slice_payload_prefix_preserving_db_recent(history, payload_offset=99)

    assert [message.content for message in view] == ["queued one", "queued two"]


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


def test_merge_source_match_suppresses_entire_split_db_group() -> None:
    payload = [
        Message(
            role="user",
            content="remember this",
            metadata=build_user_turn_metadata(
                source_message_id="discord-123",
            ),
        ),
        Message(role="assistant", content="acknowledged"),
        Message(role="user", content="active request"),
    ]
    rows = [
        _row(
            canonical_turn_id="db-user",
            turn_number=8,
            turn_group_number=4,
            sort_key=8.0,
            source_message_id="discord-123",
            user_content="remember this",
        ),
        _row(
            canonical_turn_id="db-assistant",
            turn_number=9,
            turn_group_number=4,
            sort_key=9.0,
            assistant_content="acknowledged",
        ),
    ]

    merged = _merge_protected_window(payload, rows, mode="merge")

    assert [(message.role, message.content) for message in merged] == [
        ("user", "remember this"),
        ("assistant", "acknowledged"),
        ("user", "active request"),
    ]
    assert all(
        (message.metadata or {}).get("source") != "db_recent"
        for message in merged
    )


def test_merge_replaces_incomplete_payload_fragment_with_canonical_pair() -> None:
    payload = [
        _msg(
            "user",
            "remember this",
            canonical_turn_id="db-user",
            turn_number=8,
        ),
        _msg("user", "active request"),
    ]
    rows = [
        _row(
            canonical_turn_id="db-user",
            turn_number=8,
            turn_group_number=4,
            sort_key=8.0,
            user_content="remember this",
        ),
        _row(
            canonical_turn_id="db-assistant",
            turn_number=9,
            turn_group_number=4,
            sort_key=9.0,
            assistant_content="acknowledged",
        ),
    ]

    merged = _merge_protected_window(payload, rows, mode="merge")

    assert [(message.role, message.content) for message in merged] == [
        ("user", "remember this"),
        ("assistant", "acknowledged"),
        ("user", "active request"),
    ]
    assert all(
        (message.metadata or {}).get("source") == "db_recent"
        for message in merged[:2]
    )


def test_merge_replacement_keeps_canonical_group_before_newer_payload() -> None:
    payload = [
        _msg(
            "assistant",
            "old acknowledgement fragment",
            canonical_turn_id="old-assistant",
            turn_number=1,
        ),
        _msg(
            "user",
            "newer user",
            canonical_turn_id="new-user",
            turn_number=2,
        ),
        _msg(
            "assistant",
            "newer assistant",
            canonical_turn_id="new-assistant",
            turn_number=3,
        ),
        _msg("user", "active request"),
    ]
    rows = [
        _row(
            canonical_turn_id="old-user",
            turn_number=0,
            turn_group_number=4,
            sort_key=1.0,
            user_content="old user",
        ),
        _row(
            canonical_turn_id="old-assistant",
            turn_number=1,
            turn_group_number=4,
            sort_key=2.0,
            assistant_content="old acknowledgement fragment",
        ),
    ]

    merged = _merge_protected_window(payload, rows, mode="merge")

    assert [(message.role, message.content) for message in merged] == [
        ("user", "old user"),
        ("assistant", "old acknowledgement fragment"),
        ("user", "newer user"),
        ("assistant", "newer assistant"),
        ("user", "active request"),
    ]


def test_merge_complete_payload_pair_can_span_tool_scaffolding() -> None:
    payload = [
        Message(
            role="user",
            content="remember this",
            metadata=build_user_turn_metadata(
                source_message_id="discord-123",
            ),
        ),
        _msg("tool", "tool result"),
        _msg("assistant", "acknowledged"),
        _msg("user", "active request"),
    ]
    rows = [
        _row(
            canonical_turn_id="db-user",
            turn_group_number=4,
            sort_key=1.0,
            source_message_id="discord-123",
            user_content="remember this",
        ),
        _row(
            canonical_turn_id="db-assistant",
            turn_group_number=4,
            sort_key=2.0,
            assistant_content="acknowledged",
        ),
    ]

    merged = _merge_protected_window(payload, rows, mode="merge")

    assert [(message.role, message.content) for message in merged] == [
        ("user", "remember this"),
        ("tool", "tool result"),
        ("assistant", "acknowledged"),
        ("user", "active request"),
    ]
    assert all(
        (message.metadata or {}).get("source") != "db_recent"
        for message in merged
    )


def test_merge_dedup_does_not_conflate_reused_group_numbers() -> None:
    payload = [
        Message(
            role="user",
            content="old instruction",
            metadata=build_user_turn_metadata(
                source_message_id="discord-old",
            ),
        ),
        Message(role="assistant", content="old acknowledgement"),
        Message(role="user", content="active request"),
    ]
    rows = [
        _row(
            canonical_turn_id="old-user",
            turn_group_number=7,
            sort_key=1.0,
            source_message_id="discord-old",
            user_content="old instruction",
        ),
        _row(
            canonical_turn_id="old-assistant",
            turn_group_number=7,
            sort_key=2.0,
            assistant_content="old acknowledgement",
        ),
        _row(
            canonical_turn_id="middle-user",
            turn_group_number=8,
            sort_key=3.0,
            user_content="middle instruction",
        ),
        _row(
            canonical_turn_id="middle-assistant",
            turn_group_number=8,
            sort_key=4.0,
            assistant_content="middle acknowledgement",
        ),
        _row(
            canonical_turn_id="fresh-user",
            turn_group_number=7,
            sort_key=5.0,
            user_content="fresh instruction",
        ),
        _row(
            canonical_turn_id="fresh-assistant",
            turn_group_number=7,
            sort_key=6.0,
            assistant_content="fresh acknowledgement",
        ),
    ]

    merged = _merge_protected_window(payload, rows, mode="merge")
    contents = [message.content for message in merged]

    assert contents.count("old instruction") == 1
    assert contents.count("old acknowledgement") == 1
    assert "fresh instruction" in contents
    assert "fresh acknowledgement" in contents
    assert contents[-1] == "active request"


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


def test_merge_legacy_adjacency_does_not_inherit_channel_provenance() -> None:
    rows = [
        _row(
            canonical_turn_id="legacy-user",
            turn_group_number=-1,
            sort_key=1.0,
            user_content="legacy guild message",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="guild-owner",
        ),
        _row(
            canonical_turn_id="legacy-assistant",
            turn_group_number=-1,
            sort_key=2.0,
            assistant_content="ambiguous legacy reply",
        ),
    ]

    merged = _merge_protected_window([], rows, mode="merge")

    assert merged[0].metadata["origin_channel_id"] == "chan-a"
    assert "origin_channel_id" not in merged[1].metadata
    assert "origin_channel_label" not in merged[1].metadata
    assert "audience_conversation_id" not in merged[1].metadata


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
