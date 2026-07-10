"""Sender attribution across the canonical-turn write path.

``canonical_turns.sender`` exists and threads through the tagger, but the
batch ingest path never populated it: every turn reconciled through
``IngestReconciler.ingest_batch`` stored ``sender=''``. In multi-user group
channels the member name lives only inside the leading labeled-JSON envelope
that ``_extract_envelope_metadata`` strips before hashing, so the name ended
up in no normalized text, no summary, and no column.

These tests pin the write half:

* ``ingest_batch`` derives sender from the already-parsed ``Message.metadata``
  for user entries, across the shared envelope path (Anthropic + OpenAI).
* Assistant entries are never newly labeled with a human sender.
* Enrichment preservation is one-way: a stored non-empty sender survives a
  resend that carries no sender.
* The batch-alignment fast-skip durably upgrades an empty stored sender when
  the incoming payload finally carries one, without a full-row rewrite.
* Direct tagger rewrites are role-aware.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnRow


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _reconciler(store) -> IngestReconciler:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _anthropic_fmt():
    from virtual_context.proxy.formats import AnthropicFormat
    return AnthropicFormat()


def _openai_fmt():
    from virtual_context.proxy.formats import OpenAIFormat
    return OpenAIFormat()


def _sender_block(name: str, body: str) -> str:
    """A leading labeled-JSON sender block followed by real content."""
    return (
        "Sender (member):\n"
        "```json\n"
        '{"name": "%s"}\n' % name
        + "```\n"
        + body
    )


def _store(tmp_path: Path, name: str = "vc.db") -> SQLiteStore:
    store = SQLiteStore(tmp_path / name)
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _rows(store) -> list[CanonicalTurnRow]:
    return store.get_all_canonical_turns("c")


# ---------------------------------------------------------------------------
# W1 — ingest_batch derives sender from Message.metadata
# ---------------------------------------------------------------------------

class TestIngestBatchSenderThreading:
    def test_anthropic_string_content_user_row_carries_sender(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": _sender_block("BigTex", "toes tingling")},
                {"role": "assistant", "content": "that sounds neurological"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert len(rows) == 2
        assert rows[0].user_content == "toes tingling"
        assert rows[0].sender == "BigTex"
        # I5: an assistant-only row is never newly labeled with a human sender.
        assert rows[1].sender == ""

    def test_anthropic_list_content_user_row_carries_sender(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _sender_block("BigTex", "hello guild")},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].user_content == "hello guild"
        assert rows[0].sender == "BigTex"
        assert rows[1].sender == ""

    def test_openai_format_user_row_carries_sender(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": _sender_block("Marla", "ping")},
                {"role": "assistant", "content": "pong"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_openai_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].sender == "Marla"
        assert rows[1].sender == ""

    def test_conversation_info_sender_fallback(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        block = (
            "Conversation info (guild):\n"
            "```json\n"
            '{"sender": "bigtex"}\n'
            "```\n"
            "toe tingle"
        )
        body = {
            "messages": [
                {"role": "user", "content": block},
                {"role": "assistant", "content": "ok"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].sender == "bigtex"

    def test_no_metadata_leaves_sender_empty(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": "plain message"},
                {"role": "assistant", "content": "plain reply"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert [r.sender for r in rows] == ["", ""]

    def test_trailing_reply_target_block_is_not_a_sender(self, tmp_path: Path):
        """Reply-target blocks name the message being replied to, not the
        current speaker, and they are trailing rather than leading — the
        envelope parser must not surface them as sender.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        trailing = (
            "what do you think?\n"
            "Reply target of current user message (context):\n"
            "```json\n"
            '{"name": "SomeoneElse"}\n'
            "```\n"
        )
        body = {
            "messages": [
                {"role": "user", "content": trailing},
                {"role": "assistant", "content": "ok"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].sender == ""

    def test_sender_is_not_part_of_turn_hash(self, tmp_path: Path):
        """I1: sender is enrichment, never identity."""
        store_a = _store(tmp_path, "a.db")
        store_b = _store(tmp_path, "b.db")
        rec_a, rec_b = _reconciler(store_a), _reconciler(store_b)
        rec_a.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _sender_block("BigTex", "same text")},
                {"role": "assistant", "content": "reply"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rec_b.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "same text"},
                {"role": "assistant", "content": "reply"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        assert [r.turn_hash for r in _rows(store_a)] == [
            r.turn_hash for r in _rows(store_b)
        ]


# ---------------------------------------------------------------------------
# W2 — preservation across rewrite paths
# ---------------------------------------------------------------------------

class TestPreserveExistingEnrichment:
    def test_empty_incoming_sender_inherits_stored_sender(self):
        row = CanonicalTurnRow(
            conversation_id="c", turn_number=-1, sort_key=0.0,
            turn_hash="h", user_content="u", assistant_content="",
            sender="",
        )
        existing = CanonicalTurnRow(
            conversation_id="c", turn_number=0, sort_key=1000.0,
            turn_hash="h", user_content="u", assistant_content="",
            sender="BigTex",
        )
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert row.sender == "BigTex"

    def test_non_empty_incoming_sender_is_not_clobbered(self):
        """I2 in the other direction: a fresh derivation wins over a stale
        stored value only when the incoming value is non-empty.
        """
        row = CanonicalTurnRow(
            conversation_id="c", turn_number=-1, sort_key=0.0,
            turn_hash="h", user_content="u", assistant_content="",
            sender="NewName",
        )
        existing = CanonicalTurnRow(
            conversation_id="c", turn_number=0, sort_key=1000.0,
            turn_hash="h", user_content="u", assistant_content="",
            sender="OldName",
        )
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert row.sender == "NewName"

    def test_whitespace_only_stored_sender_is_not_inherited(self):
        row = CanonicalTurnRow(
            conversation_id="c", turn_number=-1, sort_key=0.0,
            turn_hash="h", user_content="u", assistant_content="", sender="",
        )
        existing = CanonicalTurnRow(
            conversation_id="c", turn_number=0, sort_key=1000.0,
            turn_hash="h", user_content="u", assistant_content="", sender="   ",
        )
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert row.sender == ""


class TestIngestSingleResendPreservesSender:
    def test_exact_resend_keeps_stored_sender(self, tmp_path: Path):
        """Row exists with sender, then the same pair is re-sent with no
        sender: the exact-resend rewrite must not clobber it.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
            sender="BigTex",
        )
        assert _rows(store)[0].sender == "BigTex"

        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
            sender="",
        )
        rows = _rows(store)
        assert len(rows) == 2
        assert rows[0].sender == "BigTex"


class TestFastSkipSenderUpgrade:
    def test_overlap_upgrades_empty_stored_sender(self, tmp_path: Path):
        """Resend-then-row ordering: the row landed with an empty sender
        (pre-fix ingest), and a later payload finally carries the sender
        block. The overlap row is fast-skipped, so a targeted CAS must make
        the sender durable.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "toe tingle"},
                {"role": "assistant", "content": "hm"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        assert _rows(store)[0].sender == ""

        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _sender_block("BigTex", "toe tingle")},
                {"role": "assistant", "content": "hm"},
                {"role": "user", "content": _sender_block("BigTex", "still tingling")},
                {"role": "assistant", "content": "see a doctor"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert len(rows) == 4
        # Overlap row (fast-skipped) got its sender durably upgraded.
        assert rows[0].sender == "BigTex"
        # Assistant overlap row is never labeled.
        assert rows[1].sender == ""
        # Newly appended rows carry sender through the normal write path.
        assert rows[2].sender == "BigTex"
        assert rows[3].sender == ""

    def test_overlap_never_clobbers_stored_sender_with_empty(self, tmp_path: Path):
        """I2: row exists WITH sender, resend carries none — no downgrade."""
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _sender_block("BigTex", "hello")},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "ok"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].sender == "BigTex"

    def test_upgrade_does_not_rewrite_the_full_row(self, tmp_path: Path):
        """The CAS must not go through ``save_canonical_turn`` — the whole
        point of the overlap fast-skip is that unchanged rows are not
        rewritten.
        """
        store = _store(tmp_path)

        class _Counter:
            def __init__(self, inner):
                self._inner = inner
                self.saves = 0
                self.cas_calls = 0

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def save_canonical_turn(self, *a, **kw):
                self.saves += 1
                return self._inner.save_canonical_turn(*a, **kw)

            def update_canonical_turn_senders_if_empty(self, *a, **kw):
                self.cas_calls += 1
                return self._inner.update_canonical_turn_senders_if_empty(*a, **kw)

        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        wrapped = _Counter(store)
        rec._store = wrapped
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _sender_block("BigTex", "hello")},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        assert wrapped.saves == 0, "overlap rows must stay fast-skipped"
        assert wrapped.cas_calls == 1
        assert _rows(store)[0].sender == "BigTex"

    def test_upgrade_is_epoch_guarded(self, tmp_path: Path):
        """A CAS issued against a stale lifecycle epoch must not land."""
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        ct_id = _rows(store)[0].canonical_turn_id
        updated = store.update_canonical_turn_senders_if_empty(
            "c", {ct_id: "BigTex"}, expected_lifecycle_epoch=99,
        )
        assert updated == 0
        assert _rows(store)[0].sender == ""

        updated = store.update_canonical_turn_senders_if_empty(
            "c", {ct_id: "BigTex"}, expected_lifecycle_epoch=1,
        )
        assert updated == 1
        assert _rows(store)[0].sender == "BigTex"

    def test_cas_is_idempotent_and_never_overwrites(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _sender_block("BigTex", "hello")},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        ct_id = _rows(store)[0].canonical_turn_id
        assert store.update_canonical_turn_senders_if_empty(
            "c", {ct_id: "Impostor"}, expected_lifecycle_epoch=1,
        ) == 0
        assert _rows(store)[0].sender == "BigTex"


# ---------------------------------------------------------------------------
# W2 — role-aware tagger direct updates
# ---------------------------------------------------------------------------

class TestTaggerRoleAwareSenderRewrite:
    def _pipeline(self, store):
        from virtual_context.config import VirtualContextConfig
        from virtual_context.core.tagging_pipeline import TaggingPipeline
        from virtual_context.types import StorageConfig, TagGeneratorConfig

        config = VirtualContextConfig(
            conversation_id="c",
            storage=StorageConfig(backend="sqlite"),
            tag_generator=TagGeneratorConfig(type="keyword"),
        )
        semantic = SemanticSearchManager(store=store, config=config)
        semantic._embed_fn = None
        pipeline = TaggingPipeline.__new__(TaggingPipeline)
        pipeline._store = store
        pipeline._semantic = semantic
        pipeline.config = config
        return pipeline

    def test_existing_rows_update_labels_user_but_not_assistant(self, tmp_path: Path):
        from virtual_context.types import Message, TurnTagEntry

        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"],
            session_date="", sender="BigTex",
        )
        consumed = pipeline._persist_existing_canonical_rows(
            entry,
            [Message(role="user", content="u1"),
             Message(role="assistant", content="a1")],
            rows,
        )
        assert consumed == 2
        after = _rows(store)
        assert after[0].sender == "BigTex"
        assert after[1].sender == "", (
            "I5: a direct tagger rewrite must not create a human sender "
            "on an assistant-only row"
        )

    def test_existing_assistant_row_keeps_legacy_sender(self, tmp_path: Path):
        """Compatibility: a legacy/proxy assistant row that already carries a
        logical-turn sender keeps it; the rewrite must not blank it.
        """
        from virtual_context.types import Message, TurnTagEntry

        store = _store(tmp_path)
        rec = _reconciler(store)
        # ingest_single stamps the logical-turn sender on BOTH halves.
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1", sender="Legacy",
        )
        rows = _rows(store)
        assert rows[1].sender == "Legacy"

        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"],
            session_date="", sender="",
        )
        pipeline._persist_existing_canonical_rows(
            entry,
            [Message(role="user", content="u1"),
             Message(role="assistant", content="a1")],
            rows,
        )
        after = _rows(store)
        assert after[0].sender == "Legacy"
        assert after[1].sender == "Legacy"

    def test_legacy_combined_row_takes_entry_sender(self, tmp_path: Path):
        from virtual_context.types import Message, TurnTagEntry

        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
        )
        # Collapse to a single legacy combined row.
        store.delete_canonical_turns("c")
        store.save_canonical_turn(
            "c", 0, "u1", "a1",
            canonical_turn_id="legacy-1", sort_key=1000.0,
            turn_hash="lh", sender="",
        )
        rows = _rows(store)
        assert len(rows) == 1

        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"],
            session_date="", sender="BigTex",
        )
        consumed = pipeline._persist_existing_canonical_rows(
            entry,
            [Message(role="user", content="u1"),
             Message(role="assistant", content="a1")],
            rows,
        )
        assert consumed == 1
        assert _rows(store)[0].sender == "BigTex"
