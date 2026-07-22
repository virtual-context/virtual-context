"""Canonical content and provenance must not depend on the REST write path."""

from __future__ import annotations

from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.proxy.formats import detect_format
from virtual_context.types import Message, build_user_turn_metadata


def _engine(tmp_path: Path, name: str) -> VirtualContextEngine:
    config = load_config(config_dict={
        "context_window": 10_000,
        "conversation_id": "c",
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / f"{name}.db")},
        },
        "tag_generator": {"type": "keyword"},
    })
    return VirtualContextEngine(config=config)


def _metadata() -> dict:
    return build_user_turn_metadata(
        sender_name="optics",
        sender_actor_id="actor:discord:387316537012518913",
        source_message_id="1527739552456773642",
        origin_channel_id="152491",
        origin_channel_label="#vasttest",
        reply_target_message_id="1527739528876654792",
        source_conversation_key="agent:vast:discord:guild:1524917037191925871",
    )


def _row_contract(row) -> tuple:
    return (
        row.user_content,
        row.assistant_content,
        row.turn_hash,
        row.sender,
        row.origin_channel_id,
        row.origin_channel_label,
        row.sender_actor_id,
        row.source_message_id,
        row.reply_target_message_id,
        row.reply_attribution_version,
        row.audience_conversation_id,
        row.audience_attribution_version,
    )


def test_prepare_and_completed_turn_paths_store_identical_rows(tmp_path: Path):
    prepare_engine = _engine(tmp_path, "prepare")
    completed_engine = _engine(tmp_path, "completed")
    body = {
        "messages": [
            {"role": "user", "content": "@Vast what did we decide?"},
            {"role": "assistant", "content": "We chose the smaller index."},
        ],
    }
    try:
        prepare_engine._ingest_reconciler.ingest_batch(
            "c",
            body=body,
            fmt=detect_format(body),
            expected_lifecycle_epoch=prepare_engine._engine_state.lifecycle_epoch,
            source_audience_conversation_id="c",
            current_user_metadata=_metadata(),
        )
        completed_engine.persist_completed_turn(
            [
                Message(role="user", content="@Vast what did we decide?"),
                Message(role="assistant", content="We chose the smaller index."),
            ],
            source_audience_conversation_id="c",
            user_turn_metadata=_metadata(),
        )

        prepare_rows = prepare_engine._store.get_all_canonical_turns("c")
        completed_rows = completed_engine._store.get_all_canonical_turns("c")
        assert [_row_contract(row) for row in prepare_rows] == [
            _row_contract(row) for row in completed_rows
        ]
        assert prepare_rows[0].user_content == "@Vast what did we decide?"
        assert prepare_rows[0].sender == "optics"
        assert prepare_rows[0].sender_actor_id == "actor:discord:387316537012518913"
        assert prepare_rows[0].source_message_id == "1527739552456773642"
    finally:
        prepare_engine.close()
        completed_engine.close()


def test_completed_turn_upgrades_all_provenance_on_prepared_tail(tmp_path: Path):
    engine = _engine(tmp_path, "tail")
    body = {"messages": [{"role": "user", "content": "clean question"}]}
    try:
        engine._ingest_reconciler.ingest_batch(
            "c",
            body=body,
            fmt=detect_format(body),
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
        )
        engine.persist_completed_turn(
            [
                Message(role="user", content="clean question"),
                Message(role="assistant", content="clean answer"),
            ],
            source_audience_conversation_id="c",
            user_turn_metadata=_metadata(),
        )

        rows = engine._store.get_all_canonical_turns("c")
        assert len(rows) == 2
        user = rows[0]
        assert user.user_content == "clean question"
        assert user.sender == "optics"
        assert user.origin_channel_id == "152491"
        assert user.origin_channel_label == "#vasttest"
        assert user.sender_actor_id == "actor:discord:387316537012518913"
        assert user.source_message_id == "1527739552456773642"
        assert user.reply_target_message_id == "1527739528876654792"
        assert user.reply_attribution_version == 1
        assert user.audience_conversation_id == "c"
        assert user.audience_attribution_version == 1
    finally:
        engine.close()
