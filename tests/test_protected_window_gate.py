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
from virtual_context.types import (
    CanonicalTurnRow,
    Message,
    RequestRoles,
    build_user_turn_metadata,
)


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


def test_tier3_compass_tail_is_model_visible_but_absent_from_returned_roles(tmp_path):
    """Pin the real gate -> assembler seam behind the original regression."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-1",
            turn_number=8,
            turn_group_number=4,
            sort_key=8.0,
            user_content='For future replies, begin with "Compass:".',
            sender="optics",
            sender_actor_id="actor:discord:42",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="source-a",
        ),
    ]
    assembler._store = fake_store
    active = Message(role="user", content="Name one moon of Mars.")
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="source-b",
        origin_channel_id="chan-b",
    )

    assembled = eng.on_message_inbound(
        active.content,
        [active],
        request_roles=roles,
    )

    assert "<recent-conversation" in assembled.prepend_text
    assert "Compass:" in assembled.prepend_text
    assert '"authority":"current_requester_user"' in assembled.prepend_text
    assert [(m.role, m.content) for m in assembled.conversation_history] == [
        ("user", "Name one moon of Mars."),
    ]


@pytest.mark.regression("BUG-044")
def test_tier3_db_pair_survives_payload_compaction_offset(tmp_path):
    """BUG-044: payload watermarks must not split a recovered DB turn group.

    This pins the live Discord shape: seven stamped channel-local messages,
    one trailing active request, and a two-row cross-channel DB group. The
    flushed watermark equals the payload-owned message count. Applying that
    offset to the post-merge list discards the recovered user instruction at
    index 7 while leaving its assistant acknowledgement at index 8.
    """
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-user",
            turn_number=8,
            turn_group_number=4,
            sort_key=8.0,
            user_content='For future replies, begin with "test3:".',
            sender="optics",
            sender_actor_id="actor:discord:42",
            source_message_id="discord-pref",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="source-a",
        ),
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-assistant",
            turn_number=9,
            turn_group_number=4,
            sort_key=9.0,
            assistant_content="test3: Understood.",
        ),
    ]
    assembler._store = fake_store
    eng._engine_state.flushed_prefix_messages = 8

    history = [
        _stamped(
            "user" if index % 2 == 0 else "assistant",
            f"old local message {index}",
            canonical_turn_id=f"local-{index}",
            turn_number=index,
        )
        for index in range(7)
    ]
    active = Message(role="user", content="How are you?")
    history.append(active)
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="source-b",
        origin_channel_id="chan-b",
    )

    assembled = eng.on_message_inbound(
        active.content,
        history,
        request_roles=roles,
    )

    recent = assembled.recent_conversation_text
    assert 'For future replies, begin with \\"test3:\\".' in recent
    assert "test3: Understood." in recent
    assert '"authority":"current_requester_user"' in recent
    assert recent.index("For future replies") < recent.index("test3: Understood.")
    assert [(m.role, m.content) for m in assembled.conversation_history] == [
        ("user", "How are you?"),
    ]
    assert all(
        (m.metadata or {}).get("source") != "db_recent"
        for m in assembled.conversation_history
    )
    reassembled = eng._retrieval.reassemble_context()
    assert 'For future replies, begin with \\"test3:\\".' in reassembled
    assert reassembled.index("For future replies") < reassembled.index(
        "test3: Understood."
    )


@pytest.mark.regression("BUG-045")
def test_tier3_payload_duplicate_cannot_suppress_db_user_before_offset(tmp_path):
    """A completed payload duplicate must not strand the DB assistant half.

    This is the exact second live Discord shape behind BUG-045.  The previous
    guild turn is still present in the in-memory payload when Tier 3 reads the
    same split canonical group.  Source-message dedup suppresses the DB user
    half, then the payload watermark consumes the payload copy of that user
    instruction.  Without group-aware replacement, only the DB assistant
    acknowledgement reaches ``recent-conversation``.
    """
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-user",
            turn_number=1072,
            turn_group_number=566,
            sort_key=1073000.0,
            user_content='For future replies, begin with "GuildProof73:".',
            sender="optics",
            sender_actor_id="actor:discord:42",
            source_message_id="discord-pref",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="target-1",
        ),
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-assistant",
            turn_number=1073,
            turn_group_number=566,
            sort_key=1074000.0,
            assistant_content="GuildProof73: Understood.",
        ),
    ]
    assembler._store = fake_store
    eng._engine_state.flushed_prefix_messages = 2

    payload_user = Message(
        role="user",
        content='For future replies, begin with "GuildProof73:".',
        metadata=build_user_turn_metadata(
            sender_name="optics",
            sender_actor_id="actor:discord:42",
            source_message_id="discord-pref",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
        ),
    )
    payload_assistant = Message(
        role="assistant",
        content="GuildProof73: Understood.",
    )
    active = Message(role="user", content="In one sentence, how are you?")
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="target-1",
        origin_channel_id="chan-b",
    )

    assembled = eng.on_message_inbound(
        active.content,
        [payload_user, payload_assistant, active],
        request_roles=roles,
    )

    recent = assembled.recent_conversation_text
    assert recent.count(
        'For future replies, begin with \\"GuildProof73:\\".'
    ) == 1
    assert recent.count("GuildProof73: Understood.") == 1
    assert '"authority":"current_requester_user"' in recent
    assert recent.index("For future replies") < recent.index(
        "GuildProof73: Understood."
    )
    assert [(message.role, message.content) for message in assembled.conversation_history] == [
        ("user", "In one sentence, how are you?"),
    ]


@pytest.mark.regression("BUG-046")
def test_other_channel_engine_history_cannot_suppress_canonical_recent_pair(tmp_path):
    """Retained unified-engine history is not necessarily in the client body.

    Discord sessions send channel-local payloads, but every channel in a
    unified guild reuses one engine state. The immediately preceding turn
    from channel A can therefore be present in ``conversation_history`` while
    absent from the model-visible channel-B payload. It must not suppress the
    canonical copy that provides cross-channel continuity.
    """
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-user",
            turn_number=1080,
            turn_group_number=570,
            sort_key=1081000.0,
            user_content='Keep replies concise and begin with "LiveGuild83:".',
            sender="optics",
            sender_actor_id="actor:discord:42",
            source_message_id="discord-pref",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="target-1",
        ),
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-assistant",
            turn_number=1081,
            turn_group_number=570,
            sort_key=1082000.0,
            assistant_content="LiveGuild83: Understood.",
        ),
    ]
    assembler._store = fake_store

    retained_user = Message(
        role="user",
        content='Keep replies concise and begin with "LiveGuild83:".',
        metadata={
            **build_user_turn_metadata(
                sender_name="optics",
                sender_actor_id="actor:discord:42",
                source_message_id="discord-pref",
                origin_channel_id="chan-a",
                origin_channel_label="#alpha",
            ),
            "canonical_turn_id": "pref-user",
            "turn_number": 1080,
        },
    )
    retained_assistant = Message(
        role="assistant",
        content="LiveGuild83: Understood.",
        metadata={
            **build_user_turn_metadata(origin_channel_id="chan-a"),
            "canonical_turn_id": "pref-assistant",
            "turn_number": 1081,
        },
    )
    active = Message(
        role="user",
        content="Name one moon of Mars.",
        metadata=build_user_turn_metadata(
            source_message_id="discord-probe",
            origin_channel_id="chan-b",
        ),
    )
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="target-1",
        origin_channel_id="chan-b",
    )

    assembled = eng.on_message_inbound(
        active.content,
        [retained_user, retained_assistant, active],
        request_roles=roles,
    )

    recent = assembled.recent_conversation_text
    assert recent.count(
        'Keep replies concise and begin with \\"LiveGuild83:\\".'
    ) == 1
    assert recent.count("LiveGuild83: Understood.") == 1
    assert recent.index("Keep replies concise") < recent.index(
        "LiveGuild83: Understood."
    )
    assert '"authority":"current_requester_user"' in recent


@pytest.mark.regression("BUG-046")
def test_same_channel_payload_twin_still_suppresses_canonical_copy(tmp_path):
    """Production-shaped nested channel metadata preserves native dedup."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="same-user",
            turn_number=40,
            turn_group_number=20,
            sort_key=41000.0,
            user_content="Same-channel prior question.",
            source_message_id="same-source",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="target-1",
        ),
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="same-assistant",
            turn_number=41,
            turn_group_number=20,
            sort_key=42000.0,
            assistant_content="Same-channel prior answer.",
        ),
    ]
    assembler._store = fake_store

    retained_user = Message(
        role="user",
        content="Same-channel prior question.",
        metadata={
            **build_user_turn_metadata(
                source_message_id="same-source",
                origin_channel_id="chan-a",
            ),
            "canonical_turn_id": "same-user",
            "turn_number": 40,
        },
    )
    retained_assistant = Message(
        role="assistant",
        content="Same-channel prior answer.",
        metadata={
            **build_user_turn_metadata(origin_channel_id="chan-a"),
            "canonical_turn_id": "same-assistant",
            "turn_number": 41,
        },
    )
    active = Message(
        role="user",
        content="Current same-channel question.",
        metadata=build_user_turn_metadata(
            source_message_id="current-source",
            origin_channel_id="chan-a",
        ),
    )
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="target-1",
        origin_channel_id="chan-a",
    )

    assembled = eng.on_message_inbound(
        active.content,
        [retained_user, retained_assistant, active],
        request_roles=roles,
    )

    assert assembled.recent_conversation_text == ""


@pytest.mark.regression("BUG-046")
def test_same_channel_active_tail_race_still_suppresses_db_user(tmp_path):
    """The nested source id remains reachable after channel filtering."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = None
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="active-user",
            turn_number=42,
            turn_group_number=21,
            sort_key=43000.0,
            user_content="Current same-channel question.",
            source_message_id="current-source",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="target-1",
        ),
    ]
    assembler._store = fake_store
    active = Message(
        role="user",
        content="Current same-channel question.",
        metadata=build_user_turn_metadata(
            source_message_id="current-source",
            origin_channel_id="chan-a",
        ),
    )
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="target-1",
        origin_channel_id="chan-a",
    )

    assembled = eng.on_message_inbound(
        active.content,
        [active],
        request_roles=roles,
    )

    assert assembled.recent_conversation_text == ""
    assert [(message.role, message.content) for message in assembled.conversation_history] == [
        ("user", "Current same-channel question."),
    ]


@pytest.mark.regression("BUG-045")
def test_tier2_equal_with_payload_offset_forces_canonical_replacement(tmp_path):
    """Marker equality cannot skip Tier 3 when the payload will be sliced."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 3
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.return_value = [
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-user",
            turn_number=2,
            turn_group_number=1,
            sort_key=3000.0,
            user_content="Remember the cross-channel instruction.",
            sender="optics",
            sender_actor_id="actor:discord:42",
            origin_channel_id="chan-a",
            origin_channel_label="#alpha",
            audience_conversation_id="target-1",
        ),
        CanonicalTurnRow(
            conversation_id="target-1",
            canonical_turn_id="pref-assistant",
            turn_number=3,
            turn_group_number=1,
            sort_key=4000.0,
            assistant_content="Instruction acknowledged.",
        ),
    ]
    assembler._store = fake_store
    eng._engine_state.flushed_prefix_messages = 2

    active = Message(role="user", content="Use it now.")
    history = [
        _stamped(
            "user",
            "Remember the cross-channel instruction.",
            canonical_turn_id="pref-user",
            turn_number=2,
        ),
        _stamped(
            "assistant",
            "Instruction acknowledged.",
            canonical_turn_id="pref-assistant",
            turn_number=3,
        ),
        active,
    ]
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="target-1",
        audience_conversation_id="target-1",
        origin_channel_id="chan-b",
    )

    assembled = eng.on_message_inbound(
        active.content,
        history,
        request_roles=roles,
    )

    fake_store.get_recent_canonical_turns.assert_called_once_with(
        "target-1",
        limit=eng.config.monitor.protected_recent_turns,
    )
    recent = assembled.recent_conversation_text
    assert recent.count("Remember the cross-channel instruction.") == 1
    assert recent.count("Instruction acknowledged.") == 1
    assert '"authority":"current_requester_user"' in recent
    assert [(message.role, message.content) for message in assembled.conversation_history] == [
        ("user", "Use it now."),
    ]
    assert eng._retrieval.reassemble_context() == assembled.prepend_text


@pytest.mark.regression("BUG-045")
def test_tier3_read_failure_preserves_unsliced_payload(tmp_path):
    """A failed canonical replacement must not erase local payload history."""
    eng = _make_participant_engine(tmp_path)
    assembler = eng._retrieval
    fake_provider = MagicMock()
    fake_provider.get_marker.return_value = 3
    assembler._session_state_provider = fake_provider
    fake_store = MagicMock(wraps=eng._store)
    fake_store.get_recent_canonical_turns.side_effect = RuntimeError(
        "canonical store unavailable"
    )
    assembler._store = fake_store
    eng._engine_state.flushed_prefix_messages = 2

    active = Message(role="user", content="Use it now.")
    history = [
        _stamped(
            "user",
            "Remember the cross-channel instruction.",
            canonical_turn_id="pref-user",
            turn_number=2,
        ),
        _stamped(
            "assistant",
            "Instruction acknowledged.",
            canonical_turn_id="pref-assistant",
            turn_number=3,
        ),
        active,
    ]

    assembled = eng.on_message_inbound(active.content, history)

    assert [
        (message.role, message.content)
        for message in assembled.conversation_history
    ] == [
        ("user", "Remember the cross-channel instruction."),
        ("assistant", "Instruction acknowledged."),
        ("user", "Use it now."),
    ]


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
