"""Admin backfill of ``canonical_turns.sender`` from stored raw content.

Recovery is narrower than it looks. ``extract_ingestible_messages`` retains
raw content only when the provider content is a list, and the reconciler
serializes that list to JSON before storing it. For string-content historical
rows ``user_raw_content`` is usually NULL and ``user_content`` is already
envelope-stripped, so sender cannot be recovered from the canonical row.

These tests pin the derivation and the update shape:

* Only rows with an empty ``sender`` and user-side raw content are inspected.
* A JSON-serialized list is decoded first and the envelope parser runs on the
  extracted text blocks, never on the JSON string itself.
* A plain string raw payload goes straight to the envelope parser.
* Malformed raw JSON is counted, not fatal.
* The update is a compare-and-set, so a re-run is a no-op.
* ``--dry-run`` reports without writing.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _sender_envelope(name: str, body: str) -> str:
    return (
        "Sender (member):\n"
        "```json\n"
        '{"name": "%s"}\n' % name
        + "```\n"
        + body
    )


def _engine(tmp_path: Path, conv: str = "c", tenant: str = "t"):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id=conv,
        tenant_id=tenant,
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "vc.db")),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    return VirtualContextEngine(config=config)


def _seed(store: SQLiteStore, conv: str, tenant: str = "t") -> None:
    store.upsert_conversation(tenant_id=tenant, conversation_id=conv)


def _row(
    store: SQLiteStore,
    conv: str,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    user_raw_content: str | None = None,
    sender: str = "",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        user_raw_content=user_raw_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
    )


def _senders(store: SQLiteStore, conv: str) -> list[str]:
    return [r.sender for r in store.get_all_canonical_turns(conv)]


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

class TestBackfillSenders:
    def test_json_list_raw_is_decoded_before_envelope_parse(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        raw = json.dumps([
            {"type": "text", "text": _sender_envelope("BigTex", "toes tingling")},
        ])
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="toes tingling", user_raw_content=raw)
        _row(store, "c", ct_id="ct-2", sort_key=2000.0,
             assistant_content="hm")

        report = engine.backfill_senders("c")
        assert report["eligible"] == 1
        assert report["updated"] == 1
        assert report["failed"] == 0
        assert _senders(store, "c") == ["BigTex", ""]
        engine.close()

    def test_plain_string_raw_goes_straight_to_envelope_parser(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="hello",
             user_raw_content=_sender_envelope("Marla", "hello"))
        report = engine.backfill_senders("c")
        assert report["updated"] == 1
        assert _senders(store, "c") == ["Marla"]
        engine.close()

    def test_row_without_raw_content_is_skipped(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="already stripped", user_raw_content=None)
        report = engine.backfill_senders("c")
        assert report["eligible"] == 0
        assert report["skipped_no_raw"] == 1
        assert report["updated"] == 0
        assert _senders(store, "c") == [""]
        engine.close()

    def test_populated_sender_is_skipped(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="hi",
             user_raw_content=json.dumps([
                 {"type": "text", "text": _sender_envelope("Impostor", "hi")},
             ]),
             sender="BigTex")
        report = engine.backfill_senders("c")
        assert report["skipped_existing"] == 1
        assert report["updated"] == 0
        assert _senders(store, "c") == ["BigTex"]
        engine.close()

    def test_malformed_raw_json_is_counted_not_fatal(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        # Looks like JSON, is not. Falls back to plain-text parsing, which
        # finds no leading sender block.
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="broken", user_raw_content='[{"type": "text",')
        _row(store, "c", ct_id="ct-2", sort_key=2000.0,
             user_content="fine",
             user_raw_content=json.dumps([
                 {"type": "text", "text": _sender_envelope("BigTex", "fine")},
             ]))
        report = engine.backfill_senders("c")
        assert report["updated"] == 1
        assert report["skipped_no_sender"] == 1
        assert _senders(store, "c") == ["", "BigTex"]
        engine.close()

    def test_assistant_only_row_is_never_labeled(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             assistant_content="reply",
             user_raw_content=json.dumps([
                 {"type": "text", "text": _sender_envelope("BigTex", "x")},
             ]))
        report = engine.backfill_senders("c")
        assert report["updated"] == 0
        assert _senders(store, "c") == [""]
        engine.close()

    def test_last_text_block_wins(self, tmp_path: Path):
        """Envelope extraction reads the LAST text block of a list payload."""
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        raw = json.dumps([
            {"type": "text", "text": _sender_envelope("Stale", "old")},
            {"type": "text", "text": _sender_envelope("BigTex", "current")},
        ])
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="current", user_raw_content=raw)
        engine.backfill_senders("c")
        assert _senders(store, "c") == ["BigTex"]
        engine.close()

    def test_dry_run_reports_without_writing(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="hi",
             user_raw_content=json.dumps([
                 {"type": "text", "text": _sender_envelope("BigTex", "hi")},
             ]))
        report = engine.backfill_senders("c", dry_run=True)
        assert report["eligible"] == 1
        assert report["updated"] == 1, "dry-run reports what it WOULD update"
        assert report["dry_run"] is True
        assert _senders(store, "c") == [""]
        engine.close()

    def test_rerun_is_a_noop(self, tmp_path: Path):
        """I4: idempotent."""
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0,
             user_content="hi",
             user_raw_content=json.dumps([
                 {"type": "text", "text": _sender_envelope("BigTex", "hi")},
             ]))
        first = engine.backfill_senders("c")
        assert first["updated"] == 1
        second = engine.backfill_senders("c")
        assert second["updated"] == 0
        assert second["skipped_existing"] == 1
        assert _senders(store, "c") == ["BigTex"]
        engine.close()

    def test_limit_bounds_the_scan(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        for idx in range(3):
            _row(store, "c", ct_id=f"ct-{idx}", sort_key=1000.0 * (idx + 1),
                 user_content=f"m{idx}",
                 user_raw_content=json.dumps([
                     {"type": "text", "text": _sender_envelope("BigTex", f"m{idx}")},
                 ]))
        report = engine.backfill_senders("c", limit=2)
        assert report["updated"] == 2
        assert _senders(store, "c") == ["BigTex", "BigTex", ""]
        engine.close()

    def test_empty_conversation_id_raises(self, tmp_path: Path):
        engine = _engine(tmp_path)
        with pytest.raises(ValueError):
            engine.backfill_senders("")
        engine.close()


# ---------------------------------------------------------------------------
# Canonical-row conversation enumeration
# ---------------------------------------------------------------------------

class TestListCanonicalConversationIds:
    def test_enumerates_canonical_only_conversations(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "conv-a", tenant="t1")
        _seed(store, "conv-b", tenant="t2")
        _row(store, "conv-a", ct_id="a1", sort_key=1000.0, user_content="u")
        _row(store, "conv-b", ct_id="b1", sort_key=1000.0, user_content="u")
        assert store.list_canonical_conversation_ids() == ["conv-a", "conv-b"]

    def test_tenant_scope_filters_through_conversations(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "conv-a", tenant="t1")
        _seed(store, "conv-b", tenant="t2")
        _row(store, "conv-a", ct_id="a1", sort_key=1000.0, user_content="u")
        _row(store, "conv-b", ct_id="b1", sort_key=1000.0, user_content="u")
        assert store.list_canonical_conversation_ids(tenant_id="t2") == ["conv-b"]

    def test_limit_applies(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "conv-a")
        _seed(store, "conv-b")
        _row(store, "conv-a", ct_id="a1", sort_key=1000.0, user_content="u")
        _row(store, "conv-b", ct_id="b1", sort_key=1000.0, user_content="u")
        assert store.list_canonical_conversation_ids(limit=1) == ["conv-a"]

    def test_conversation_without_canonical_rows_is_absent(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "conv-empty")
        assert store.list_canonical_conversation_ids() == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _parse_admin(monkeypatch, argv, handler_name):
    import sys

    import virtual_context.cli.main as cli_main

    captured = {}

    def _capture(args):
        captured["args"] = args

    monkeypatch.setattr(cli_main, handler_name, _capture)
    monkeypatch.setattr(sys, "argv", ["virtual-context", *argv])
    cli_main.main()
    return captured["args"]


class TestBackfillSendersCli:
    def test_command_is_registered_with_expected_flags(self, monkeypatch):
        args = _parse_admin(
            monkeypatch,
            [
                "admin", "backfill-senders", "conv-1",
                "--tenant-id", "t1", "--dry-run", "--limit", "5",
                "--storage-backend", "sqlite", "--sqlite-path", "/tmp/x.db",
            ],
            "cmd_admin_backfill_senders",
        )
        assert args.conversation_id == "conv-1"
        assert args.tenant_id == "t1"
        assert args.dry_run is True
        assert args.limit == 5
        assert args.storage_backend == "sqlite"
        assert args.sqlite_path == "/tmp/x.db"

    def test_all_convs_for_tenant_makes_conversation_id_optional(self, monkeypatch):
        args = _parse_admin(
            monkeypatch,
            ["admin", "backfill-senders", "--tenant-id", "t1", "--all-convs-for-tenant"],
            "cmd_admin_backfill_senders",
        )
        assert args.conversation_id is None
        assert args.all_convs_for_tenant is True

    def test_postgres_dsn_flag_is_accepted(self, monkeypatch):
        args = _parse_admin(
            monkeypatch,
            [
                "admin", "backfill-senders", "conv-1",
                "--storage-backend", "postgres",
                "--postgres-dsn", "postgresql://x/y",
            ],
            "cmd_admin_backfill_senders",
        )
        assert args.postgres_dsn == "postgresql://x/y"
