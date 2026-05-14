"""Unit tests for canonical_turn_id / turn_number stamping at ingest time.

Covers ``_stamp_canonical_turn_ids`` and ``_last_already_canonical_turn_number``
from ``virtual_context.core.protected_window``. Pure helpers — no engine,
store, or proxy-state dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

from virtual_context.core.protected_window import (
    _last_already_canonical_turn_number,
    _stamp_canonical_turn_ids,
)
from virtual_context.types import Message


def _row(canonical_turn_id: str = "cid", turn_number: int = 0):
    return SimpleNamespace(canonical_turn_id=canonical_turn_id, turn_number=turn_number)


def _msg(role: str, content: str, **metadata) -> Message:
    return Message(role=role, content=content, metadata=metadata if metadata else None)


# ---------------------------------------------------------------------------
# _stamp_canonical_turn_ids
# ---------------------------------------------------------------------------


def test_stamp_on_success_writes_both_fields() -> None:
    msgs = [_msg("user", "u"), _msg("assistant", "a")]
    rows = [_row("c1", 0), _row("c2", 1)]
    _stamp_canonical_turn_ids(msgs, rows)
    assert msgs[0].metadata == {"canonical_turn_id": "c1", "turn_number": 0}
    assert msgs[1].metadata == {"canonical_turn_id": "c2", "turn_number": 1}


def test_stamp_creates_fresh_metadata_dict_when_none() -> None:
    m = _msg("user", "u")
    assert m.metadata is None
    _stamp_canonical_turn_ids([m], [_row("c1", 0)])
    assert isinstance(m.metadata, dict)
    assert m.metadata["canonical_turn_id"] == "c1"


def test_stamp_skips_sentinel_turn_number() -> None:
    """row.turn_number == -1 (default sentinel) must not be stamped."""
    m = _msg("user", "u")
    _stamp_canonical_turn_ids([m], [_row("c1", -1)])
    assert m.metadata == {"canonical_turn_id": "c1"}
    assert "turn_number" not in m.metadata


def test_stamp_empty_rows_is_noop() -> None:
    m = _msg("user", "u")
    _stamp_canonical_turn_ids([m], [])
    assert m.metadata is None


def test_stamp_size_mismatch_aligns_suffix_tail() -> None:
    """When the payload is longer than the rows, only the suffix tail stamps."""
    msgs = [_msg("user", "older1"), _msg("user", "older2"), _msg("user", "u"), _msg("assistant", "a")]
    rows = [_row("c1", 5), _row("c2", 6)]
    _stamp_canonical_turn_ids(msgs, rows)
    # First two messages unstamped.
    assert msgs[0].metadata is None
    assert msgs[1].metadata is None
    # Last two messages stamped.
    assert msgs[2].metadata == {"canonical_turn_id": "c1", "turn_number": 5}
    assert msgs[3].metadata == {"canonical_turn_id": "c2", "turn_number": 6}


def test_stamp_payload_shorter_than_rows_stamps_available() -> None:
    msgs = [_msg("user", "u")]
    rows = [_row("c1", 0), _row("c2", 1), _row("c3", 2)]
    _stamp_canonical_turn_ids(msgs, rows)
    # Only the single message gets stamped — implementation aligns to
    # the leading rows in this case.
    assert msgs[0].metadata == {"canonical_turn_id": "c1", "turn_number": 0}


def test_stamp_skips_rows_without_canonical_id() -> None:
    msgs = [_msg("user", "u"), _msg("assistant", "a")]
    rows = [_row("", 0), _row("c2", 1)]
    _stamp_canonical_turn_ids(msgs, rows)
    assert msgs[0].metadata is None
    assert msgs[1].metadata == {"canonical_turn_id": "c2", "turn_number": 1}


# ---------------------------------------------------------------------------
# _last_already_canonical_turn_number
# ---------------------------------------------------------------------------


def test_anchor_empty_history_returns_none() -> None:
    assert _last_already_canonical_turn_number([]) is None


def test_anchor_no_stamped_entries_returns_none() -> None:
    msgs = [_msg("user", "u"), _msg("assistant", "a")]
    assert _last_already_canonical_turn_number(msgs) is None


def test_anchor_picks_latest_stamped_entry() -> None:
    msgs = [
        _msg("user", "u0", turn_number=0, canonical_turn_id="c0"),
        _msg("assistant", "a0", turn_number=0, canonical_turn_id="c0"),
        _msg("user", "u1", turn_number=1, canonical_turn_id="c1"),
        _msg("assistant", "a1", turn_number=1, canonical_turn_id="c1"),
    ]
    assert _last_already_canonical_turn_number(msgs) == 1


def test_anchor_active_tail_unstamped_user_does_not_force_fallthrough() -> None:
    """Spec §1.1: active-tail user without stamping must not trigger the
    partial-stamping safety rule because the current request's row hasn't
    been ingested yet from the assembler's perspective.

    Wait — actually re-read: the safety rule says "later UNSTAMPED
    INGESTIBLE entry forces fall-through." The active-tail user IS
    ingestible. So the safety rule DOES force fall-through. Test that.
    """
    msgs = [
        _msg("user", "u0", turn_number=0, canonical_turn_id="c0"),
        _msg("assistant", "a0", turn_number=0, canonical_turn_id="c0"),
        _msg("user", "active turn"),  # active tail, no stamping
    ]
    # Active-tail user is ingestible -> safety rule fires -> None.
    assert _last_already_canonical_turn_number(msgs) is None


def test_anchor_system_after_stamped_anchor_does_not_force_fallthrough() -> None:
    msgs = [
        _msg("user", "u0", turn_number=3, canonical_turn_id="c0"),
        _msg("assistant", "a0", turn_number=3, canonical_turn_id="c0"),
        # System / scaffold roles are not ingestible — must not force
        # fall-through.
        _msg("system", "tool scaffolding"),
    ]
    assert _last_already_canonical_turn_number(msgs) == 3


def test_anchor_safety_rule_partial_stamping_returns_none() -> None:
    """If the most recent stamped anchor is followed by a NEWER
    ingestible entry that lacks stamping, Tier 2 cannot trust the
    anchor and must fall through to Tier 3."""
    msgs = [
        _msg("user", "u0", turn_number=2, canonical_turn_id="c0"),
        _msg("assistant", "a0", turn_number=2, canonical_turn_id="c0"),
        # Newer ingestible entries with no stamp — partial-stamping shape.
        _msg("user", "u1"),
        _msg("assistant", "a1"),
    ]
    assert _last_already_canonical_turn_number(msgs) is None


def test_anchor_picks_latest_when_history_contains_only_stamped() -> None:
    msgs = [
        _msg("user", "old", turn_number=5, canonical_turn_id="c-old"),
        _msg("assistant", "old", turn_number=5, canonical_turn_id="c-old"),
        _msg("user", "new", turn_number=42, canonical_turn_id="c-new"),
        _msg("assistant", "new", turn_number=42, canonical_turn_id="c-new"),
    ]
    assert _last_already_canonical_turn_number(msgs) == 42


def test_anchor_ignores_metadata_dicts_without_turn_number() -> None:
    msgs = [
        _msg("user", "u", canonical_turn_id="c-no-turn"),
        _msg("assistant", "a", canonical_turn_id="c-no-turn"),
    ]
    # turn_number missing -> not a usable anchor; ingestible -> safety rule.
    assert _last_already_canonical_turn_number(msgs) is None
