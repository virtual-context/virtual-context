"""Sender carried through the canonical-row to ``Message`` boundary.

``DomainCompactor._format_conversation`` already labels a message with
``get_sender_name(m.metadata) or m.role.capitalize()``. The gap was upstream:
every row-to-message adapter built ``Message(role=..., content=...)`` with no
metadata, so a stored sender never reached the summarizer.

These tests pin the adapters:

* Canonical-row compaction builds user ``Message.metadata`` so the compactor
  emits ``"BigTex: ..."``; a row without a safe sender emits ``"Source: ..."``.
* A mid-message session-boundary split preserves metadata on both halves.
* History reconstruction groups physical user/assistant half-rows into logical
  turns, labels only the user half, and still drops an incomplete trailing
  group.
* Sender never enters ``Message.content``.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from types import SimpleNamespace

from virtual_context.types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    CanonicalTurnRow,
    Message,
    get_sender_name,
)
from virtual_context.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# _load_compactable_rows -> Message.metadata
# ---------------------------------------------------------------------------

class _RowStore:
    def __init__(self, rows):
        self._rows = rows

    def get_uncompacted_canonical_turns(self, conversation_id, *, protected_recent_turns=0):
        return list(self._rows)


class _PhysicalRowStore(_RowStore):
    def __init__(self, logical_rows, physical_rows):
        super().__init__(logical_rows)
        self._physical_rows = physical_rows

    def get_all_canonical_turns(self, conversation_id):
        raise AssertionError("compaction enumerated the full physical archive")

    def get_canonical_turn_rows_by_group(self, conversation_id, groups, *, internal_validation=False):
        assert internal_validation is True
        return [row for row in self._physical_rows
                if row.conversation_id == conversation_id and row.turn_group_number in groups]


def _pipeline(rows):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.core.compaction_pipeline import CompactionPipeline
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    pipeline = CompactionPipeline.__new__(CompactionPipeline)
    pipeline._store = _RowStore(rows)
    pipeline._config = config
    return pipeline


def _pipeline_with_physical(logical_rows, physical_rows):
    pipeline = _pipeline(logical_rows)
    pipeline._store = _PhysicalRowStore(logical_rows, physical_rows)
    return pipeline


def _row(**kw) -> CanonicalTurnRow:
    kw.setdefault("conversation_id", "c")
    return CanonicalTurnRow(**kw)


class TestLoadCompactableRows:
    def test_user_message_carries_sender_metadata(self):
        pipeline = _pipeline([
            _row(user_content="toes tingling", assistant_content="", sender="BigTex"),
        ])
        _rows, messages = pipeline._load_compactable_rows()
        user_msg = messages[0]
        assert user_msg.role == "user"
        assert get_sender_name(user_msg.metadata) == "BigTex"

    def test_assistant_message_has_no_sender_metadata(self):
        """Legacy rows carry the logical-turn sender on both halves."""
        pipeline = _pipeline([
            _row(user_content="u", assistant_content="a", sender="BigTex"),
        ])
        _rows, messages = pipeline._load_compactable_rows()
        assistant_msg = messages[1]
        assert assistant_msg.role == "assistant"
        assert get_sender_name(assistant_msg.metadata) is None

    def test_no_sender_leaves_metadata_unset(self):
        pipeline = _pipeline([_row(user_content="u", assistant_content="a")])
        _rows, messages = pipeline._load_compactable_rows()
        assert get_sender_name(messages[0].metadata) is None

    def test_sender_without_user_content_is_not_attached(self):
        pipeline = _pipeline([
            _row(user_content="", assistant_content="a", sender="BigTex"),
        ])
        _rows, messages = pipeline._load_compactable_rows()
        assert get_sender_name(messages[0].metadata) is None

    def test_sender_never_enters_message_content(self):
        pipeline = _pipeline([
            _row(user_content="toes tingling", assistant_content="a", sender="BigTex"),
        ])
        _rows, messages = pipeline._load_compactable_rows()
        assert messages[0].content == "toes tingling"
        assert "BigTex" not in messages[0].content

    def test_legacy_rows_without_turn_groups_are_not_collapsed(self):
        pipeline = _pipeline([
            _row(canonical_turn_id="ct-1", user_content="first"),
            _row(canonical_turn_id="ct-2", user_content="second"),
        ])

        _rows, messages = pipeline._load_compactable_rows()

        assert [message.content for message in messages] == ["first", "second"]

    def test_mixed_logical_user_row_is_reconstructed_as_exact_physical_messages(self):
        logical = _row(
            canonical_turn_id="logical",
            turn_group_number=7,
            user_content="BigTex words\nKuw words",
            assistant_content="answer",
            sender="BigTex",
            sender_actor_id="actor:discord:1",
        )
        physical = [
            _row(
                canonical_turn_id="ct-bigtex",
                turn_number=10,
                turn_group_number=7,
                sort_key=10.0,
                user_content="BigTex words",
                sender="BigTex",
                sender_actor_id="actor:discord:1",
                audience_conversation_id="audience:guild:1",
                audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
                origin_channel_id="channel:medical",
            ),
            _row(
                canonical_turn_id="ct-kuw",
                turn_number=11,
                turn_group_number=7,
                sort_key=11.0,
                user_content="Kuw words",
                sender="Kuw9239",
                sender_actor_id="actor:discord:2",
                audience_conversation_id="audience:guild:1",
                audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
                origin_channel_id="channel:medical",
            ),
            _row(
                canonical_turn_id="ct-assistant",
                turn_number=12,
                turn_group_number=7,
                sort_key=12.0,
                assistant_content="answer",
                audience_conversation_id="audience:guild:1",
                audience_attribution_version=AUDIENCE_ATTRIBUTION_VERSION,
                origin_channel_id="channel:medical",
            ),
        ]
        pipeline = _pipeline_with_physical([logical], physical)

        _rows, messages = pipeline._load_compactable_rows()

        assert [message.content for message in messages] == [
            "BigTex words", "Kuw words", "answer",
        ]
        assert [message.source_actor_id for message in messages[:2]] == [
            "actor:discord:1", "actor:discord:2",
        ]
        assert [
            message.metadata["_vc_source_canonical_turn_ids"]
            for message in messages
        ] == [["ct-bigtex"], ["ct-kuw"], ["ct-assistant"]]
        assert {
            message.source_audience_conversation_id for message in messages
        } == {"audience:guild:1"}
        assert {
            message.source_origin_channel_id for message in messages
        } == {"channel:medical"}
        assert {
            message.source_audience_attribution_version for message in messages
        } == {AUDIENCE_ATTRIBUTION_VERSION}


# ---------------------------------------------------------------------------
# Compactor rendering
# ---------------------------------------------------------------------------

class TestFormatConversation:
    def _compactor(self):
        from virtual_context.core.compactor import DomainCompactor

        return DomainCompactor.__new__(DomainCompactor)

    def test_sender_metadata_becomes_the_label(self):
        compactor = self._compactor()
        text = compactor._format_conversation([
            Message(role="user", content="toes tingling",
                    metadata={"sender": {"name": "BigTex"}},
                    source_actor_id="actor:discord:bigtex"),
            Message(role="assistant", content="hm"),
        ])
        assert "BigTex: toes tingling" in text
        assert "Assistant: hm" in text

    def test_no_sender_emits_neutral_source_label(self):
        compactor = self._compactor()
        text = compactor._format_conversation([
            Message(role="user", content="toes tingling"),
        ])
        assert "Source: toes tingling" in text


# ---------------------------------------------------------------------------
# Session-boundary split
# ---------------------------------------------------------------------------

class TestSplitSessionBoundaryMessages:
    def test_split_preserves_metadata_on_both_halves(self):
        from virtual_context.core.segmenter import split_session_boundary_messages

        meta = {"sender": {"name": "BigTex"}}
        msg = Message(
            role="user",
            content="before text[Session from 2026-01-01]after text",
            metadata=meta,
        )
        out = split_session_boundary_messages([msg])
        assert len(out) == 2
        assert all(get_sender_name(m.metadata) == "BigTex" for m in out)

    def test_unsplit_message_is_passed_through(self):
        from virtual_context.core.segmenter import split_session_boundary_messages

        msg = Message(role="user", content="plain", metadata={"sender": {"name": "X"}})
        out = split_session_boundary_messages([msg])
        assert out == [msg]

    def test_split_without_metadata_still_works(self):
        from virtual_context.core.segmenter import split_session_boundary_messages

        msg = Message(role="user", content="a[Session from 2026-01-01]b")
        out = split_session_boundary_messages([msg])
        assert len(out) == 2
        assert all(m.metadata is None for m in out)

    def test_compactor_labels_survive_a_boundary_split(self):
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.core.segmenter import split_session_boundary_messages

        compactor = DomainCompactor.__new__(DomainCompactor)
        out = split_session_boundary_messages([
            Message(role="user", content="first[Session from 2026-01-01]second"),
        ])
        text = compactor._format_conversation(out)
        assert text.count("Source: ") == 2


# ---------------------------------------------------------------------------
# reconstruct_history_for_conv — logical-turn grouping
# ---------------------------------------------------------------------------

def _seed(store: SQLiteStore, conv: str) -> None:
    store.upsert_conversation(tenant_id="t", conversation_id=conv)


def _save(
    store: SQLiteStore,
    conv: str,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    sender: str = "",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
    )


class TestReconstructHistoryGrouping:
    def test_physical_half_rows_become_one_logical_pair(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "c")
        _save(store, "c", ct_id="ct-1", sort_key=1000.0,
              user_content="toes tingling", sender="BigTex")
        _save(store, "c", ct_id="ct-2", sort_key=2000.0,
              assistant_content="hm")

        history = store.reconstruct_history_for_conv("c")
        assert [m.role for m in history] == ["user", "assistant"]
        assert [m.content for m in history] == ["toes tingling", "hm"]
        assert get_sender_name(history[0].metadata) == "BigTex"
        assert get_sender_name(history[1].metadata) is None

    def test_incomplete_trailing_group_is_excluded(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "c")
        _save(store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u1")
        _save(store, "c", ct_id="ct-2", sort_key=2000.0, assistant_content="a1")
        _save(store, "c", ct_id="ct-3", sort_key=3000.0, user_content="u2-trailing")

        history = store.reconstruct_history_for_conv("c")
        assert [m.content for m in history] == ["u1", "a1"]

    def test_combined_legacy_rows_still_work(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "c")
        _save(store, "c", ct_id="ct-1", sort_key=1000.0,
              user_content="  u1\n", assistant_content="\n  a1  ", sender="BigTex")
        history = store.reconstruct_history_for_conv("c")
        assert [m.content for m in history] == ["  u1\n", "\n  a1  "]
        assert get_sender_name(history[0].metadata) == "BigTex"
        assert get_sender_name(history[1].metadata) is None

    def test_assistant_only_row_sender_is_not_propagated(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "c")
        _save(store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u1")
        _save(store, "c", ct_id="ct-2", sort_key=2000.0,
              assistant_content="a1", sender="Legacy")
        history = store.reconstruct_history_for_conv("c")
        assert get_sender_name(history[0].metadata) is None
        assert get_sender_name(history[1].metadata) is None


class TestCompositeReconstructFallbackGrouping:
    def _composite(self, rows):
        from virtual_context.core.composite_store import CompositeStore

        class LegacySegments:
            def get_all_canonical_turns(self, conversation_id: str):
                return rows

        seg = LegacySegments()
        return CompositeStore(
            segments=seg, facts=seg, fact_links=seg, state=seg, search=seg,
        )

    def test_fallback_groups_physical_half_rows(self):
        rows = [
            SimpleNamespace(user_content="u1", assistant_content="",
                            sender="BigTex", turn_group_number=0),
            SimpleNamespace(user_content="", assistant_content="a1",
                            sender="", turn_group_number=0),
        ]
        history = self._composite(rows).reconstruct_history_for_conv("c")
        assert [m.role for m in history] == ["user", "assistant"]
        assert get_sender_name(history[0].metadata) == "BigTex"
        assert get_sender_name(history[1].metadata) is None

    def test_fallback_keeps_legacy_combined_row_shape(self):
        rows = [
            SimpleNamespace(user_content=" u1 ", assistant_content=" a1 "),
            SimpleNamespace(user_content="u2", assistant_content=""),
        ]
        history = self._composite(rows).reconstruct_history_for_conv("c")
        assert [m.content for m in history] == [" u1 ", " a1 "]


# ---------------------------------------------------------------------------
# Protected-window DB merge
# ---------------------------------------------------------------------------

class TestProtectedWindowSenderMetadata:
    def test_user_row_gains_sender_and_assistant_does_not(self):
        from virtual_context.core.protected_window import _merge_protected_window

        rows = [
            SimpleNamespace(
                canonical_turn_id="ct-1",
                turn_number=0,
                user_content="toes tingling",
                assistant_content="hm",
                sender="BigTex",
                turn_hash="h1",
                sort_key=1000.0,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            ),
        ]
        merged = _merge_protected_window([], rows, mode="merge")
        user_msgs = [m for m in merged if m.role == "user"]
        asst_msgs = [m for m in merged if m.role == "assistant"]
        assert get_sender_name(user_msgs[0].metadata) == "BigTex"
        assert get_sender_name(asst_msgs[0].metadata) is None
        # Non-sender diagnostic metadata is preserved.
        assert user_msgs[0].metadata["source"] == "db_recent"
