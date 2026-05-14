"""Three-tier gate tests for the cross-channel-mirror protected window.

Exercises Tier 0 (config flag), Tier 1 (``_is_merge_participant``),
Tier 2 (Redis marker vs payload anchor INT comparison), and Tier 3
(DB read + merge) via the actual ``RetrievalAssembler`` instance on a
constructed engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message


def _make_config(tmp_path, *, mode: str = "merge", conversation_id: str | None = None):
    cfg_dict = {
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
        "tag_generator": {"type": "keyword"},
        "retrieval": {"inbound_tagger_type": "keyword"},
        "assembly": {"protected_window_db_source": mode},
    }
    if conversation_id is not None:
        cfg_dict["conversation_id"] = conversation_id
    return load_config(config_dict=cfg_dict)


def _underlying_store(engine):
    store = engine._store
    return getattr(store, "_store", store)


def _seed_attachable_target(raw_store, target_id: str, *, tenant_id: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    sqlite = raw_store._segments
    conn = sqlite._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO conversations
           (conversation_id, tenant_id, phase, deleted_at,
            created_at, updated_at)
           VALUES (?, ?, 'active', NULL, ?, ?)""",
        (target_id, tenant_id, now, now),
    )
    conn.commit()


def _stamped(role: str, content: str, *, canonical_turn_id: str, turn_number: int) -> Message:
    return Message(
        role=role,
        content=content,
        metadata={"canonical_turn_id": canonical_turn_id, "turn_number": turn_number},
    )


def _unstamped(role: str, content: str) -> Message:
    return Message(role=role, content=content)


# ---------------------------------------------------------------------------
# Tier 0 off — full no-op
# ---------------------------------------------------------------------------


def test_tier0_off_makes_zero_redis_or_db_calls(tmp_path):
    cfg = _make_config(tmp_path, mode="off", conversation_id="conv-1")
    eng = VirtualContextEngine(config=cfg)
    assembler = eng._retrieval
    fake_store = MagicMock(wraps=eng._store)
    fake_provider = MagicMock()
    assembler._store = fake_store
    assembler._session_state_provider = fake_provider
    history = [_unstamped("user", "hi"), _unstamped("assistant", "ack")]
    assembled = eng.on_message_inbound("next", history)
    # Tier 0 off: no get_marker, no get_recent_canonical_turns.
    fake_provider.get_marker.assert_not_called()
    fake_store.get_recent_canonical_turns.assert_not_called()
    # Pipeline ran end-to-end and produced an assembled context.
    assert assembled is not None


# ---------------------------------------------------------------------------
# Tier 1 unattached short-circuit
# ---------------------------------------------------------------------------


def test_tier1_unattached_skips_redis_and_db(tmp_path):
    cfg = _make_config(tmp_path, mode="merge", conversation_id="conv-unattached")
    eng = VirtualContextEngine(config=cfg)
    assert eng._is_merge_participant is False
    assembler = eng._retrieval
    fake_store = MagicMock(wraps=eng._store)
    fake_provider = MagicMock()
    assembler._store = fake_store
    assembler._session_state_provider = fake_provider
    history = [_unstamped("user", "hi"), _unstamped("assistant", "ack")]
    eng.on_message_inbound("next", history)
    # Tier 1 short-circuit: no Redis read, no DB read.
    fake_provider.get_marker.assert_not_called()
    fake_store.get_recent_canonical_turns.assert_not_called()


# ---------------------------------------------------------------------------
# Tier 2 equality skip + divergence + legacy fall-through
# ---------------------------------------------------------------------------


def _make_participant_engine(tmp_path) -> VirtualContextEngine:
    cfg_seed = _make_config(tmp_path, mode="off", conversation_id="seed-conv")
    seeder = VirtualContextEngine(config=cfg_seed)
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-1")
    raw.save_conversation_alias("source-1", "target-1")
    seeder.close()

    cfg = _make_config(tmp_path, mode="merge", conversation_id="target-1")
    eng = VirtualContextEngine(config=cfg)
    assert eng._is_merge_participant is True
    return eng


def test_tier2_equality_skips_tier3(tmp_path):
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 7  # Redis last_completed_turn = 7
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    assembler._store = fake_store
    # Payload has a stamped anchor with turn_number = 7 -> equality.
    history = [
        _stamped("user", "u0", canonical_turn_id="c0", turn_number=7),
        _stamped("assistant", "a0", canonical_turn_id="c0", turn_number=7),
    ]
    eng.on_message_inbound("next", history)
    fake_provider.get_marker.assert_called_once_with("target-1", "last_completed_turn")
    # Tier 2 equal -> NO Tier 3 DB read.
    fake_store.get_recent_canonical_turns.assert_not_called()


def test_tier2_divergence_fires_tier3(tmp_path):
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 9  # Redis advanced
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    history = [
        _stamped("user", "u0", canonical_turn_id="c0", turn_number=7),
        _stamped("assistant", "a0", canonical_turn_id="c0", turn_number=7),
    ]
    eng.on_message_inbound("next", history)
    fake_provider.get_marker.assert_called_once_with("target-1", "last_completed_turn")
    fake_store.get_recent_canonical_turns.assert_called_once_with(
        "target-1", limit=eng.config.monitor.protected_recent_turns,
    )


def test_tier2_legacy_payload_fallthrough(tmp_path):
    """Payload without stamped turn_number cannot anchor Tier 2 -> Tier 3 fires."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 0
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    history = [_unstamped("user", "u0"), _unstamped("assistant", "a0")]
    eng.on_message_inbound("next", history)
    # Tier 3 fires unconditionally on legacy-payload paths.
    fake_store.get_recent_canonical_turns.assert_called_once()


def test_tier2_marker_missing_falls_through(tmp_path):
    """When Redis marker is None (degraded / no SessionState), fall
    through to Tier 3 rather than skipping incorrectly."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    history = [_stamped("user", "u", canonical_turn_id="c1", turn_number=3)]
    eng.on_message_inbound("next", history)
    fake_store.get_recent_canonical_turns.assert_called_once()


def test_tier2_int_coercion_on_string_marker(tmp_path):
    """Redis blob JSON-decoded markers may arrive as int OR (defensively)
    as the original JSON literal type. ``"7"`` should int-coerce and
    compare equal to anchor 7."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = "7"  # string-typed marker
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    history = [
        _stamped("user", "u0", canonical_turn_id="c0", turn_number=7),
    ]
    eng.on_message_inbound("next", history)
    # Equal after int-coercion -> Tier 3 skipped.
    fake_store.get_recent_canonical_turns.assert_not_called()


def test_tier2_non_coercible_marker_falls_through(tmp_path):
    """A marker that can't be int-coerced (e.g. nested dict) must fall
    through to Tier 3 rather than crashing or skipping incorrectly."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = {"unexpected": "shape"}
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    history = [_stamped("user", "u", canonical_turn_id="c1", turn_number=3)]
    eng.on_message_inbound("next", history)
    fake_store.get_recent_canonical_turns.assert_called_once()


# ---------------------------------------------------------------------------
# Tier 3 read failure must not crash
# ---------------------------------------------------------------------------


def test_tier3_db_read_exception_degrades_gracefully(tmp_path):
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 9
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.side_effect = RuntimeError("DB down")
    assembler._store = fake_store
    history = [_stamped("user", "u", canonical_turn_id="c1", turn_number=3)]
    # Must not raise — gate degrades to no-merge.
    assembled = eng.on_message_inbound("next", history)
    assert assembled is not None


# ---------------------------------------------------------------------------
# Race-window self-heal
# ---------------------------------------------------------------------------


def test_tier2_marker_lag_self_heals_on_next_request(tmp_path):
    """Two-request scenario: first request sees stale-equal marker and
    skips; the next request after marker catch-up sees divergence and
    fires Tier 3. Single-request lag, not permanent miss."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = []
    assembler._store = fake_store
    assembler._session_state_provider = fake_provider

    history = [_stamped("user", "u", canonical_turn_id="c1", turn_number=3)]

    # Request 1: marker stale-equal to anchor (sibling's Postgres
    # commit landed but Redis write hasn't fired yet).
    fake_provider.get_marker.return_value = 3
    eng.on_message_inbound("first", history)
    assert fake_store.get_recent_canonical_turns.call_count == 0

    # Request 2: marker has caught up to sibling's commit.
    fake_provider.get_marker.return_value = 9
    eng.on_message_inbound("second", history)
    assert fake_store.get_recent_canonical_turns.call_count == 1
