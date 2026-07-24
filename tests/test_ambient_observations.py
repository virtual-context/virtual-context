"""Ambient guild messages are first-class, user-only canonical observations."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virtual_context.config import VirtualContextConfig
from virtual_context.core.canonical_turns import generate_canonical_turn_id
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.lifecycle_epoch import LifecycleEpochMismatch
from virtual_context.engine import VirtualContextEngine
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    StorageConfig,
    TagGeneratorConfig,
)


CONV = "sk:agent:vast:discord:guild:1524917037191925871"
ACTOR = "actor:discord:387316537012518913"


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "ambient.db")
    store.upsert_conversation(tenant_id="tenant-1", conversation_id=CONV)
    return store


def _observe(
    reconciler: IngestReconciler,
    *,
    content: str = "NuncaBob should read The Left Hand of Darkness.",
    source_message_id: str = "discord-message-100",
    channel_id: str = "channel-a",
    actor_id: str = ACTOR,
):
    return reconciler.ingest_observation(
        CONV,
        content=content,
        source_message_id=source_message_id,
        audience_conversation_id=CONV,
        origin_channel_id=channel_id,
        origin_channel_label="fitness",
        sender="Optics",
        sender_actor_id=actor_id,
        observed_at="2026-07-23T12:34:56Z",
        expected_lifecycle_epoch=1,
    )


def test_observation_is_user_only_tagged_embedded_and_profiles_actor(tmp_path: Path):
    store = _store(tmp_path)
    semantic = MagicMock()
    result = _observe(IngestReconciler(store, semantic))

    assert result.merge_mode == "observation_append"
    assert result.turns_written == 1
    rows = store.get_all_canonical_turns(CONV)
    assert len(rows) == 1
    row = rows[0]
    assert row.user_content == "NuncaBob should read The Left Hand of Darkness."
    assert row.assistant_content == ""
    assert row.primary_tag == "_general"
    assert row.tags == ["_general"]
    assert row.tagged_at
    assert row.turn_group_number == 0
    assert row.source_message_id == "discord-message-100"
    assert row.sender_actor_id == ACTOR
    assert row.sender == "Optics"
    assert row.origin_channel_id == "channel-a"
    assert row.audience_conversation_id == CONV
    assert row.audience_attribution_version == AUDIENCE_ATTRIBUTION_VERSION

    semantic.embed_and_store_turn.assert_called_once()
    assert semantic.embed_and_store_turn.call_args.kwargs["assistant_text"] == ""
    profile = store.get_actor_profile("tenant-1", ACTOR)
    assert profile is not None
    assert profile.display_name == "Optics"

    # Standalone rows remain valid inputs to the normal compaction pipeline.
    compactable = store.get_uncompacted_canonical_turns(
        CONV,
        protected_recent_turns=0,
    )
    assert len(compactable) == 1
    assert compactable[0].user_content == row.user_content
    assert compactable[0].assistant_content == ""


def test_observation_retry_is_idempotent_and_content_change_does_not_overwrite(
    tmp_path: Path,
):
    store = _store(tmp_path)
    reconciler = IngestReconciler(store, MagicMock())
    first = _observe(reconciler)
    duplicate = _observe(reconciler)
    changed = _observe(reconciler, content="edited transport replay")

    assert first.merge_mode == "observation_append"
    assert duplicate.merge_mode == "observation_duplicate"
    assert duplicate.turns_matched == 1
    assert changed.merge_mode == "observation_duplicate_content_changed"
    assert len(store.get_all_canonical_turns(CONV)) == 1
    assert (
        store.get_all_canonical_turns(CONV)[0].user_content
        == "NuncaBob should read The Left Hand of Darkness."
    )


@pytest.mark.parametrize(
    ("channel_id", "actor_id"),
    [
        ("channel-b", ACTOR),
        ("channel-a", "actor:discord:someone-else"),
    ],
)
def test_observation_source_id_provenance_conflict_fails_closed(
    tmp_path: Path,
    channel_id: str,
    actor_id: str,
):
    store = _store(tmp_path)
    reconciler = IngestReconciler(store, MagicMock())
    _observe(reconciler)
    conflict = _observe(
        reconciler,
        channel_id=channel_id,
        actor_id=actor_id,
    )

    assert conflict.merge_mode == "observation_provenance_conflict"
    assert conflict.rows == []
    assert len(store.get_all_canonical_turns(CONV)) == 1


def test_observation_rejects_stale_lifecycle_epoch(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_conversation_deleted(CONV)
    store.increment_lifecycle_epoch_on_resurrect(CONV)
    with pytest.raises(LifecycleEpochMismatch):
        _observe(IngestReconciler(store, MagicMock()))
    assert store.get_all_canonical_turns(CONV) == []


def test_normal_ingest_reuses_observed_user_by_source_identity(tmp_path: Path):
    """A slow invocation cannot duplicate a timer-delivered observation."""
    store = _store(tmp_path)
    reconciler = IngestReconciler(store, MagicMock())
    observed = _observe(
        reconciler,
        content="Vast what do you think about the amber heron?",
        source_message_id="discord-message-race",
    )
    observed_id = observed.rows[0].canonical_turn_id

    ingested = reconciler.ingest_single(
        CONV,
        user_content="<@1485681229608259666> what about the amber heron?",
        assistant_content="It is an intentionally distinctive calibration.",
        user_sender="Optics",
        assistant_sender="Vast",
        user_origin_channel_id="channel-a",
        user_origin_channel_label="fitness",
        assistant_origin_channel_id="channel-a",
        assistant_origin_channel_label="fitness",
        user_sender_actor_id=ACTOR,
        user_reply_edge={
            "source_message_id": "discord-message-race",
            "audience_conversation_id": CONV,
            "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
        },
        expected_lifecycle_epoch=1,
    )

    assert ingested.turns_matched == 1
    assert ingested.turns_written == 1
    rows = store.get_all_canonical_turns(CONV)
    assert len(rows) == 2
    user_rows = [row for row in rows if (row.user_content or "").strip()]
    assistant_rows = [
        row for row in rows if (row.assistant_content or "").strip()
    ]
    assert len(user_rows) == 1
    assert user_rows[0].canonical_turn_id == observed_id
    assert user_rows[0].source_message_id == "discord-message-race"
    assert len(assistant_rows) == 1
    assert (
        assistant_rows[0].assistant_content
        == "It is an intentionally distinctive calibration."
    )

    # A later observation retry still resolves to the same single user row.
    duplicate = _observe(
        reconciler,
        content="Vast what do you think about the amber heron?",
        source_message_id="discord-message-race",
    )
    assert duplicate.merge_mode == "observation_duplicate"
    assert len(store.get_all_canonical_turns(CONV)) == 2


def test_normal_ingest_same_source_with_conflicting_provenance_fails_closed(
    tmp_path: Path,
):
    store = _store(tmp_path)
    reconciler = IngestReconciler(store, MagicMock())
    _observe(
        reconciler,
        source_message_id="discord-message-conflict",
    )

    result = reconciler.ingest_single(
        CONV,
        user_content="same platform message",
        assistant_content="must not be attached to the wrong actor",
        user_origin_channel_id="channel-b",
        user_sender_actor_id="actor:discord:someone-else",
        user_reply_edge={
            "source_message_id": "discord-message-conflict",
            "audience_conversation_id": CONV,
            "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
        },
        expected_lifecycle_epoch=1,
    )

    assert result.merge_mode == "source_provenance_conflict"
    assert result.rows == []
    assert len(store.get_all_canonical_turns(CONV)) == 1


def test_normal_ingest_refuses_ambiguous_preexisting_source_identity(
    tmp_path: Path,
):
    """Upgrade safety: two pre-fix race winners never become three."""
    store = _store(tmp_path)
    reconciler = IngestReconciler(store, MagicMock())
    observed = _observe(
        reconciler,
        source_message_id="discord-message-pre-fix-duplicate",
    )
    duplicate = replace(
        observed.rows[0],
        canonical_turn_id=generate_canonical_turn_id(),
        sort_key=2000.0,
        turn_group_number=1,
    )
    reconciler._write_turn(duplicate, turn_number=1)
    assert len(store.get_all_canonical_turns(CONV)) == 2

    result = reconciler.ingest_single(
        CONV,
        user_content="normalized invocation copy",
        assistant_content="must not choose an ambiguous user row",
        user_origin_channel_id="channel-a",
        user_sender_actor_id=ACTOR,
        user_reply_edge={
            "source_message_id": "discord-message-pre-fix-duplicate",
            "audience_conversation_id": CONV,
            "audience_attribution_version": AUDIENCE_ATTRIBUTION_VERSION,
        },
        expected_lifecycle_epoch=1,
    )

    assert result.merge_mode == "source_identity_conflict"
    assert result.rows == []
    assert len(store.get_all_canonical_turns(CONV)) == 2


def test_engine_wrapper_indexes_group_zero_and_advances_markers(tmp_path: Path):
    config = VirtualContextConfig(
        conversation_id=CONV,
        storage=StorageConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "engine.db"),
        ),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    engine = VirtualContextEngine(config=config)
    result = engine.persist_observed_message(
        content="The calibration phrase is amber heron.",
        source_message_id="discord-message-200",
        audience_conversation_id=CONV,
        origin_channel_id="channel-a",
        origin_channel_label="fitness",
        sender="Optics",
        sender_actor_id=ACTOR,
        observed_at="2026-07-23T12:35:00Z",
    )

    assert result.merge_mode == "observation_append"
    row = result.rows[0]
    entry = engine._turn_tag_index.get_tags_for_canonical_turn(
        row.canonical_turn_id,
    )
    assert entry is not None
    assert entry.turn_number == 0
    assert entry.tags == ["_general"]
    assert engine._engine_state.last_completed_turn == 0
    assert engine._engine_state.last_indexed_turn == 0
