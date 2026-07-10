"""Channel attribution across the canonical-turn write path.

Multiple source channels can share one conversation id. The canonical-turn
ledger keeps one sort order and one retrieval surface, but rows recorded no
source channel: the provenance lived only in the leading labeled-JSON
envelope, which ``_extract_envelope_metadata`` strips before hashing.

These tests pin the write half:

* ``get_origin_channel`` reads only ``conversation info``, accepts only the
  explicit group kinds, and fills the two fields independently.
* ``ingest_batch`` derives per physical entry, for both roles.
* Preservation is one-way and per column, in both temporal orders.
* Fast-skipped overlap rows and the ``ingest_single`` tail get an
  epoch-guarded two-column compare-and-set instead of a full-row rewrite.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import CanonicalTurnRow, get_origin_channel


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


def _gemini_fmt():
    from virtual_context.proxy.formats import GeminiFormat
    return GeminiFormat()


def _responses_fmt():
    from virtual_context.proxy.formats import OpenAIResponsesFormat
    return OpenAIResponsesFormat()


def _channel_block(body: str, *, chat_id: str = "", group_channel: str = "") -> str:
    """A leading labeled-JSON conversation-info block followed by real content."""
    import json as _json

    payload: dict[str, str] = {}
    if chat_id:
        payload["chat_id"] = chat_id
    if group_channel:
        payload["group_channel"] = group_channel
    return (
        "Conversation info (guild):\n"
        "```json\n"
        + _json.dumps(payload) + "\n"
        + "```\n"
        + body
    )


def _store(tmp_path: Path, name: str = "vc.db") -> SQLiteStore:
    store = SQLiteStore(tmp_path / name)
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _rows(store, conv: str = "c") -> list[CanonicalTurnRow]:
    return store.get_all_canonical_turns(conv)


# ---------------------------------------------------------------------------
# Metadata contract — the shared derivation helper
# ---------------------------------------------------------------------------

class TestGetOriginChannel:
    def test_channel_kind_yields_id_without_prefix(self):
        meta = {"conversation info": {"chat_id": "channel:1524974537458974851"}}
        assert get_origin_channel(meta) == ("1524974537458974851", "")

    def test_group_kind_yields_id_without_prefix(self):
        meta = {"conversation info": {"chat_id": "group:99"}}
        assert get_origin_channel(meta) == ("99", "")

    @pytest.mark.parametrize("chat_id", ["direct:42", "dm:42", "user:42", "42"])
    def test_unsupported_chat_id_kinds_yield_no_id(self, chat_id: str):
        """I6: a DM id and an unprefixed id are never channel identity."""
        meta = {"conversation info": {"chat_id": chat_id}}
        assert get_origin_channel(meta) == ("", "")

    def test_dm_chat_id_still_allows_an_independent_label(self):
        """I6: the label is valid on its own; the id is not guessed from it."""
        meta = {"conversation info": {"chat_id": "dm:42", "group_channel": "#vasttest"}}
        assert get_origin_channel(meta) == ("", "#vasttest")

    def test_label_retains_leading_hash_and_is_trimmed(self):
        meta = {"conversation info": {"group_channel": "  #vasttest2  "}}
        assert get_origin_channel(meta) == ("", "#vasttest2")

    def test_id_without_label_and_label_without_id_are_independent(self):
        assert get_origin_channel(
            {"conversation info": {"chat_id": "channel:7"}}
        ) == ("7", "")
        assert get_origin_channel(
            {"conversation info": {"group_channel": "#x"}}
        ) == ("", "#x")

    def test_both_fields_together(self):
        meta = {"conversation info": {"chat_id": "channel:7", "group_channel": "#x"}}
        assert get_origin_channel(meta) == ("7", "#x")

    def test_non_string_values_are_ignored(self):
        meta = {"conversation info": {"chat_id": 5, "group_channel": ["#x"]}}
        assert get_origin_channel(meta) == ("", "")

    def test_non_dict_conversation_info_is_ignored(self):
        assert get_origin_channel({"conversation info": ["channel:7"]}) == ("", "")

    def test_empty_and_missing_metadata(self):
        assert get_origin_channel(None) == ("", "")
        assert get_origin_channel({}) == ("", "")
        assert get_origin_channel({"sender": {"name": "BigTex"}}) == ("", "")

    def test_group_space_is_not_used(self):
        meta = {"conversation info": {"group_space": "guild-1"}}
        assert get_origin_channel(meta) == ("", "")


# ---------------------------------------------------------------------------
# W1 — ingest_batch derives per physical entry, both roles
# ---------------------------------------------------------------------------

class TestIngestBatchChannelThreading:
    def test_anthropic_string_content_user_row_carries_channel(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": _channel_block(
                        "toes tingling",
                        chat_id="channel:152497",
                        group_channel="#vasttest",
                    ),
                },
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
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == (
            "152497", "#vasttest",
        )
        # No pairing, no inheritance: the assistant entry had no envelope.
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")

    def test_anthropic_list_content_both_roles_populate_independently(self, tmp_path: Path):
        """An assistant entry whose own selected text carries the envelope
        legitimately owns channel provenance. Unlike sender, this is allowed.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _channel_block(
                            "hello guild", chat_id="channel:7", group_channel="#a",
                        )},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": _channel_block(
                            "hi", chat_id="channel:7", group_channel="#a",
                        )},
                    ],
                },
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert rows[0].user_content == "hello guild"
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("7", "#a")
        assert rows[1].assistant_content == "hi"
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("7", "#a")

    def test_openai_format_user_row_carries_channel(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": _channel_block(
                    "ping", chat_id="channel:9", group_channel="#ops",
                )},
                {"role": "assistant", "content": "pong"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_openai_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("9", "#ops")
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")

    def test_gemini_format_user_row_carries_channel(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": _channel_block(
                        "gem ping", chat_id="channel:11", group_channel="#g",
                    )}],
                },
                {"role": "model", "parts": [{"text": "gem pong"}]},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_gemini_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("11", "#g")
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")

    def test_openai_responses_format_user_row_carries_channel(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": _channel_block(
                        "resp ping", chat_id="channel:13", group_channel="#r",
                    )}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "resp pong"}],
                },
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_responses_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("13", "#r")
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")

    def test_dm_shaped_metadata_without_label_stays_empty(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": _channel_block("private", chat_id="dm:42")},
                {"role": "assistant", "content": "ok"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert [(r.origin_channel_id, r.origin_channel_label) for r in rows] == [
            ("", ""), ("", ""),
        ]

    def test_label_only_row_has_no_id(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        body = {
            "messages": [
                {"role": "user", "content": _channel_block("x", group_channel="#only")},
                {"role": "assistant", "content": "y"},
            ]
        }
        rec.ingest_batch(
            conversation_id="c", body=body, fmt=_anthropic_fmt(),
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("", "#only")

    def test_channel_is_not_part_of_turn_hash(self, tmp_path: Path):
        """I1: channel is enrichment, never identity."""
        store_a = _store(tmp_path, "a.db")
        store_b = _store(tmp_path, "b.db")
        rec_a, rec_b = _reconciler(store_a), _reconciler(store_b)
        rec_a.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "same text", chat_id="channel:1", group_channel="#a",
                )},
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
        assert [r.sort_key for r in _rows(store_a)] == [
            r.sort_key for r in _rows(store_b)
        ]


# ---------------------------------------------------------------------------
# W2 — preservation, per column, both temporal orders
# ---------------------------------------------------------------------------

def _row(**kwargs) -> CanonicalTurnRow:
    base = dict(
        conversation_id="c", turn_number=-1, sort_key=0.0,
        turn_hash="h", user_content="u", assistant_content="",
    )
    base.update(kwargs)
    return CanonicalTurnRow(**base)


class TestPreserveExistingEnrichment:
    def test_empty_incoming_inherits_both_stored_fields(self):
        row = _row()
        existing = _row(origin_channel_id="7", origin_channel_label="#a")
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#a")

    def test_non_empty_incoming_is_not_clobbered(self):
        row = _row(origin_channel_id="9", origin_channel_label="#new")
        existing = _row(origin_channel_id="7", origin_channel_label="#old")
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert (row.origin_channel_id, row.origin_channel_label) == ("9", "#new")

    def test_fields_merge_independently_id_from_stored_label_from_incoming(self):
        """The pair is not atomic: a stored id survives an incoming label."""
        row = _row(origin_channel_id="", origin_channel_label="#fresh")
        existing = _row(origin_channel_id="7", origin_channel_label="")
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#fresh")

    def test_fields_merge_independently_label_from_stored_id_from_incoming(self):
        row = _row(origin_channel_id="9", origin_channel_label="")
        existing = _row(origin_channel_id="", origin_channel_label="#stored")
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert (row.origin_channel_id, row.origin_channel_label) == ("9", "#stored")

    def test_whitespace_only_stored_values_are_not_inherited(self):
        row = _row()
        existing = _row(origin_channel_id="  ", origin_channel_label="  ")
        IngestReconciler._preserve_existing_enrichment(row, existing)
        assert (row.origin_channel_id, row.origin_channel_label) == ("", "")


class TestIngestSingleResendPreservesChannel:
    def test_exact_resend_keeps_stored_channel(self, tmp_path: Path):
        """Channel-then-no-channel ordering: the resend must not blank it."""
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
            user_origin_channel_id="7", user_origin_channel_label="#a",
        )
        assert _rows(store)[0].origin_channel_id == "7"

        rec.ingest_single(conversation_id="c", user_content="u1", assistant_content="a1")
        rows = _rows(store)
        assert len(rows) == 2
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("7", "#a")

    def test_assistant_half_keeps_its_own_channel(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
            user_origin_channel_id="7", user_origin_channel_label="#a",
            assistant_origin_channel_id="7", assistant_origin_channel_label="#a",
        )
        rows = _rows(store)
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("7", "#a")

    def test_assistant_half_stays_empty_without_its_own_derivation(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c",
            user_content="u1", assistant_content="a1",
            user_origin_channel_id="7", user_origin_channel_label="#a",
        )
        rows = _rows(store)
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")


class TestFastSkipChannelUpgrade:
    def test_overlap_upgrades_empty_stored_channel(self, tmp_path: Path):
        """Row-then-channel ordering: the overlap row is fast-skipped, so a
        targeted CAS must make the late derivation durable.
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
        assert _rows(store)[0].origin_channel_id == ""

        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "toe tingle", chat_id="channel:7", group_channel="#vasttest",
                )},
                {"role": "assistant", "content": "hm"},
                {"role": "user", "content": _channel_block(
                    "still tingling", chat_id="channel:7", group_channel="#vasttest",
                )},
                {"role": "assistant", "content": "see a doctor"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert len(rows) == 4
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == (
            "7", "#vasttest",
        )
        # The assistant overlap row had no envelope of its own; still empty.
        assert (rows[1].origin_channel_id, rows[1].origin_channel_label) == ("", "")

    def test_overlap_upgrade_fills_only_the_empty_column(self, tmp_path: Path):
        """An origin-derived id must be able to gain a raw-derived label
        later without the id being rewritten.
        """
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
        # Pre-seed only the id, as the origin fallback would.
        store.update_canonical_turn_channels_if_empty("c", {ct_id: ("999", "")})
        assert _rows(store)[0].origin_channel_id == "999"
        assert _rows(store)[0].origin_channel_label == ""

        # A later payload carries a DIFFERENT id plus a label.
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "hello", chat_id="channel:7", group_channel="#late",
                )},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "ok"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        row = _rows(store)[0]
        # I2: the stored id is never overwritten; only the empty label fills.
        assert (row.origin_channel_id, row.origin_channel_label) == ("999", "#late")

    def test_overlap_upgrade_does_not_full_row_rewrite(self, tmp_path: Path):
        """The fast-skip contract: the overlap row is upgraded through the CAS,
        never through ``save_canonical_turn``.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        overlap_id = _rows(store)[0].canonical_turn_id

        saved: list[str] = []
        original_save = store.save_canonical_turn

        def _spy(conversation_id, turn_number, *args, **kwargs):
            ct = kwargs.get("canonical_turn_id")
            if ct:
                saved.append(ct)
            return original_save(conversation_id, turn_number, *args, **kwargs)

        store.save_canonical_turn = _spy  # type: ignore[assignment]
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "a", chat_id="channel:7", group_channel="#z",
                )},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        assert overlap_id not in saved
        assert _rows(store)[0].origin_channel_id == "7"

    def test_stale_epoch_cas_writes_nothing(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        ct_id = _rows(store)[0].canonical_turn_id
        written = store.update_canonical_turn_channels_if_empty(
            "c", {ct_id: ("7", "#z")}, expected_lifecycle_epoch=999,
        )
        assert written == 0
        assert _rows(store)[0].origin_channel_id == ""

    def test_cas_is_idempotent(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(conversation_id="c", user_content="a", assistant_content="b")
        ct_id = _rows(store)[0].canonical_turn_id
        assert store.update_canonical_turn_channels_if_empty("c", {ct_id: ("7", "#z")}) == 1
        assert store.update_canonical_turn_channels_if_empty("c", {ct_id: ("7", "#z")}) == 0
        assert store.update_canonical_turn_channels_if_empty("c", {ct_id: ("8", "#q")}) == 0
        row = _rows(store)[0]
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#z")

    def test_cas_ignores_all_empty_candidates(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(conversation_id="c", user_content="a", assistant_content="b")
        ct_id = _rows(store)[0].canonical_turn_id
        assert store.update_canonical_turn_channels_if_empty("c", {ct_id: ("", "")}) == 0
        assert store.update_canonical_turn_channels_if_empty("c", {}) == 0

    def test_epoch_change_between_check_and_update_blocks_the_write(self, tmp_path: Path):
        """The epoch predicate must live inside the channel UPDATE itself.

        A preflight SELECT would already have accepted epoch 1 and would leak
        the provenance into epoch 2; a correlated UPDATE predicate sees the
        change and matches no row.
        """
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
        inner = store._get_conn()

        class _EpochRaceConnection:
            def __init__(self):
                self.raced = False

            def execute(self, sql, params=()):
                if not self.raced and "UPDATE canonical_turns" in sql:
                    self.raced = True
                    inner.execute(
                        "UPDATE conversations SET lifecycle_epoch = 2 "
                        "WHERE conversation_id = ?",
                        ("c",),
                    )
                return inner.execute(sql, params)

            def commit(self):
                return inner.commit()

        racing_conn = _EpochRaceConnection()
        store._get_conn = lambda: racing_conn

        updated = store.update_canonical_turn_channels_if_empty(
            "c", {ct_id: ("7", "#z")}, expected_lifecycle_epoch=1,
        )
        assert racing_conn.raced is True
        assert updated == 0
        assert _rows(store)[0].origin_channel_id == ""

    def test_overlap_upgrade_uses_the_cas_not_a_save(self, tmp_path: Path):
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

            def update_canonical_turn_channels_if_empty(self, *a, **kw):
                self.cas_calls += 1
                return self._inner.update_canonical_turn_channels_if_empty(*a, **kw)

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
                {"role": "user", "content": _channel_block(
                    "hello", chat_id="channel:7", group_channel="#z",
                )},
                {"role": "assistant", "content": "hi"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        assert wrapped.saves == 0, "overlap rows must stay fast-skipped"
        assert wrapped.cas_calls == 1
        assert _rows(store)[0].origin_channel_id == "7"

    def test_overlap_never_clobbers_stored_channel_with_empty(self, tmp_path: Path):
        """I2: row exists WITH channel, resend carries none — no downgrade."""
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "hello", chat_id="channel:7", group_channel="#a",
                )},
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
        row = _rows(store)[0]
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#a")


# ---------------------------------------------------------------------------
# W2 — tagger rewrite paths
# ---------------------------------------------------------------------------

class TestTaggerChannelRewrite:
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

    def _seed(self, tmp_path: Path):
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "u1", chat_id="channel:7", group_channel="#a",
                )},
                {"role": "assistant", "content": "a1"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        return store

    def test_strict_role_window_rewrite_without_metadata_keeps_stored(self, tmp_path: Path):
        from virtual_context.types import Message, TurnTagEntry

        store = self._seed(tmp_path)
        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"], session_date="",
        )
        consumed = pipeline._persist_existing_canonical_rows(
            entry,
            [Message(role="user", content="u1"),
             Message(role="assistant", content="a1")],
            _rows(store),
        )
        assert consumed == 2
        after = _rows(store)
        assert (after[0].origin_channel_id, after[0].origin_channel_label) == ("7", "#a")
        assert (after[1].origin_channel_id, after[1].origin_channel_label) == ("", "")

    def test_strict_role_window_rewrite_derives_per_physical_message(self, tmp_path: Path):
        """An assistant message carrying its own envelope populates its own
        row; the user's values are never copied onto it.
        """
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
        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"], session_date="",
        )
        user_meta = {"conversation info": {"chat_id": "channel:7", "group_channel": "#a"}}
        pipeline._persist_existing_canonical_rows(
            entry,
            [Message(role="user", content="u1", metadata=user_meta),
             Message(role="assistant", content="a1")],
            _rows(store),
        )
        after = _rows(store)
        assert (after[0].origin_channel_id, after[0].origin_channel_label) == ("7", "#a")
        assert (after[1].origin_channel_id, after[1].origin_channel_label) == ("", "")

    def test_legacy_combined_row_rewrite_uses_user_first_precedence(self, tmp_path: Path):
        from virtual_context.types import Message, TurnTagEntry

        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c", user_content="u1", assistant_content="a1",
        )
        # Collapse to one legacy combined row.
        rows = _rows(store)
        store.save_canonical_turn(
            "c", 0, "u1", "a1",
            canonical_turn_id=rows[0].canonical_turn_id,
            sort_key=rows[0].sort_key,
        )
        conn = store._get_conn()
        conn.execute(
            "DELETE FROM canonical_turns WHERE canonical_turn_id = ?",
            (rows[1].canonical_turn_id,),
        )
        conn.commit()
        combined = _rows(store)
        assert len(combined) == 1

        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"], session_date="",
        )
        # User has the label only; assistant has the id only. User-first,
        # missing-field-only: each column fills from the first source that has it.
        consumed = pipeline._persist_existing_canonical_rows(
            entry,
            [
                Message(role="user", content="u1",
                        metadata={"conversation info": {"group_channel": "#a"}}),
                Message(role="assistant", content="a1",
                        metadata={"conversation info": {"chat_id": "channel:7"}}),
            ],
            combined,
        )
        assert consumed == 1
        row = _rows(store)[0]
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#a")

    def test_background_row_tagging_preserves_channel(self, tmp_path: Path):
        store = self._seed(tmp_path)
        pipeline = self._pipeline(store)

        class _Gen:
            def generate_tags(self, text, store_tags, context_turns=None):
                from virtual_context.types import TagResult
                return TagResult(
                    tags=["fresh"], primary="fresh", source="keyword",
                    fact_signals=[], code_refs=[],
                )

        pipeline._tag_generator = _Gen()
        row = _rows(store)[0]
        pipeline.tag_canonical_row(row)
        after = _rows(store)[0]
        assert after.primary_tag == "fresh"
        assert (after.origin_channel_id, after.origin_channel_label) == ("7", "#a")

    def test_hash_match_rewrite_preserves_channel(self, tmp_path: Path):
        from virtual_context.types import Message, TurnTagEntry

        store = self._seed(tmp_path)
        pipeline = self._pipeline(store)
        entry = TurnTagEntry(
            turn_number=0, primary_tag="topic", tags=["topic"], session_date="",
        )
        # No existing_rows -> hash-match direct rewrite path.
        pipeline._persist_canonical_turn(
            entry,
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
        )
        after = _rows(store)
        assert (after[0].origin_channel_id, after[0].origin_channel_label) == ("7", "#a")
        assert (after[1].origin_channel_id, after[1].origin_channel_label) == ("", "")


class TestIngestSingleTailFastSkipUpgrade:
    def test_tail_append_upgrades_the_mirrored_user_row(self, tmp_path: Path):
        """The prepare-then-ingest tail row is mirrored, not rewritten, so its
        newly derivable channel needs the CAS to become durable.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        # Prepare-phase: the user half lands alone, with no channel.
        rec.ingest_prepared_turns(
            "c",
            prepared_turns=[
                rec._prepare_message_row("c", role="user", content="u1"),
            ],
            raw_turn_count=1,
            expected_lifecycle_epoch=1,
        )
        assert len(_rows(store)) == 1

        rec.ingest_single(
            conversation_id="c",
            user_content="u1",
            assistant_content="a1",
            user_origin_channel_id="7",
            user_origin_channel_label="#late",
            expected_lifecycle_epoch=1,
        )
        rows = _rows(store)
        assert len(rows) == 2
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("7", "#late")

    def test_tail_fast_skip_honors_a_stale_caller_epoch(self, tmp_path: Path):
        """Obtaining the epoch after the fast-skip began would read the
        resurrected conversation's new epoch and write stale provenance.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_prepared_turns(
            "c",
            prepared_turns=[rec._prepare_message_row("c", role="user", content="u1")],
            raw_turn_count=1,
            expected_lifecycle_epoch=1,
        )
        rec.ingest_single(
            conversation_id="c",
            user_content="u1",
            assistant_content="a1",
            user_origin_channel_id="7",
            user_origin_channel_label="#late",
            expected_lifecycle_epoch=999,
        )
        rows = _rows(store)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("", "")


# ---------------------------------------------------------------------------
# W2 — full-row rewrite paths re-supply both columns
# ---------------------------------------------------------------------------

class TestFullRowRewritePreservation:
    def test_save_canonical_turn_defaults_would_erase_without_resupply(self, tmp_path: Path):
        """Pins the reason every full-row path must pass both values: the
        upsert overwrites omitted columns with their defaults.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c", user_content="u", assistant_content="a",
            user_origin_channel_id="7", user_origin_channel_label="#a",
        )
        row = _rows(store)[0]
        store.save_canonical_turn(
            "c", 0, row.user_content, row.assistant_content,
            canonical_turn_id=row.canonical_turn_id,
            sort_key=row.sort_key,
            turn_hash=row.turn_hash,
        )
        assert _rows(store)[0].origin_channel_id == ""

    def test_sort_key_rebalance_fallback_preserves_channel(self, tmp_path: Path):
        """The per-row rebalance save bypasses ``_write_turn``; it must still
        carry both columns. Exercised through ``_open_sort_key_gap`` with the
        bulk shifter hidden, which is the only path that reaches that save.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "u1", chat_id="channel:7", group_channel="#a",
                )},
                {"role": "assistant", "content": _channel_block(
                    "a1", chat_id="channel:7", group_channel="#a",
                )},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        existing = _rows(store)
        assert [r.origin_channel_id for r in existing] == ["7", "7"]

        # Force the per-row fallback: hide the bulk shifter.
        store.shift_canonical_turn_sort_keys = None  # type: ignore[assignment]
        rec._open_sort_key_gap(
            "c",
            existing=existing,
            rows_touched=[],
            right_key=existing[0].sort_key,
            count=2,
        )

        after = _rows(store)
        assert len(after) == 2
        assert [(r.origin_channel_id, r.origin_channel_label) for r in after] == [
            ("7", "#a"), ("7", "#a"),
        ]
        # Sanity: the rebalance really did shift the rows through that save.
        assert after[0].sort_key > 1000.0


# ---------------------------------------------------------------------------
# W4 — schema: legacy SQLite migration, logical merge does not synthesize
# ---------------------------------------------------------------------------

class TestSQLiteSchemaMigration:
    def test_legacy_db_without_channel_columns_migrates(self, tmp_path: Path):
        """A legacy DB whose canonical_turns predates the channel columns and
        still has NOT NULL lifecycle columns (forcing the table rebuild) must
        come out with both columns and its rows intact.
        """
        import sqlite3

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE canonical_turns (
                canonical_turn_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sort_key REAL NOT NULL,
                turn_hash TEXT NOT NULL,
                hash_version INTEGER NOT NULL DEFAULT 1,
                normalized_user_text TEXT NOT NULL DEFAULT '',
                normalized_assistant_text TEXT NOT NULL DEFAULT '',
                user_content TEXT NOT NULL DEFAULT '',
                assistant_content TEXT NOT NULL DEFAULT '',
                user_raw_content TEXT,
                assistant_raw_content TEXT,
                primary_tag TEXT NOT NULL DEFAULT '_general',
                tags_json TEXT NOT NULL DEFAULT '[]',
                session_date TEXT NOT NULL DEFAULT '',
                sender TEXT NOT NULL DEFAULT '',
                fact_signals_json TEXT NOT NULL DEFAULT '[]',
                code_refs_json TEXT NOT NULL DEFAULT '[]',
                tagged_at TEXT NOT NULL DEFAULT '',
                compacted_at TEXT NOT NULL DEFAULT '',
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                source_batch_id TEXT,
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT '',
                UNIQUE (conversation_id, sort_key)
            );
            INSERT INTO canonical_turns
                (canonical_turn_id, conversation_id, sort_key, turn_hash,
                 user_content, assistant_content, sender)
            VALUES ('ct-1', 'c', 1000.0, 'h1', 'legacy user', '', 'BigTex');
            """
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(db)
        store.upsert_conversation(tenant_id="t", conversation_id="c")
        rows = store.get_all_canonical_turns("c")
        assert len(rows) == 1
        assert rows[0].user_content == "legacy user"
        assert rows[0].sender == "BigTex"
        assert rows[0].origin_channel_id == ""
        assert rows[0].origin_channel_label == ""

        # The columns are real: the CAS can fill them.
        assert store.update_canonical_turn_channels_if_empty(
            "c", {"ct-1": ("7", "#a")},
        ) == 1
        row = store.get_all_canonical_turns("c")[0]
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#a")

    def test_reopening_a_migrated_db_keeps_channel_values(self, tmp_path: Path):
        """Startup must not discard the columns on a second open."""
        store = _store(tmp_path, "reopen.db")
        rec = _reconciler(store)
        rec.ingest_single(
            conversation_id="c", user_content="u", assistant_content="a",
            user_origin_channel_id="7", user_origin_channel_label="#a",
        )
        store.close()

        reopened = SQLiteStore(tmp_path / "reopen.db")
        row = reopened.get_all_canonical_turns("c")[0]
        assert (row.origin_channel_id, row.origin_channel_label) == ("7", "#a")


class TestLogicalMergeDoesNotSynthesizeChannel:
    def test_merged_logical_row_does_not_borrow_sibling_channel(self, tmp_path: Path):
        """A logical row folds two physical halves. Channel must not be
        synthesized across them: semantic filtering uses the physical row.
        """
        store = _store(tmp_path)
        rec = _reconciler(store)
        rec.ingest_batch(
            conversation_id="c",
            body={"messages": [
                {"role": "user", "content": _channel_block(
                    "u", chat_id="channel:7", group_channel="#a",
                )},
                {"role": "assistant", "content": "a"},
            ]},
            fmt=_anthropic_fmt(), expected_lifecycle_epoch=1,
        )
        physical = _rows(store)
        assert physical[0].origin_channel_id == "7"
        assert physical[1].origin_channel_id == ""

        from virtual_context.storage.sqlite import _merge_canonical_turn_rows

        merged = _merge_canonical_turn_rows(physical)
        # Whatever the merged row carries, the assistant half's own row is
        # still empty — the physical row is the filter boundary.
        assert physical[1].origin_channel_id == ""
        assert len(merged) == 1
