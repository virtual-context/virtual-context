"""Admin backfill of the ``canonical_turns`` channel columns.

Two independent sources combine per PHYSICAL row, neither replacing a
non-empty stored field:

1. Side-specific stored raw content. Raw is retained only when the provider
   content was a list, so string-content historical rows contribute nothing.
2. A stable-key fallback for a still-empty id, parsed from the row's
   ``origin_conversation_id`` (a moved row) or, only when that is empty, from
   its own ``conversation_id`` (a target-native row). A non-empty but
   unparseable UUID/hash origin must never fall through to the current id.

The update is a per-column compare-and-set, so a re-run is a no-op and a
partially-filled row can gain only its missing column.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import pytest

from virtual_context.storage.sqlite import SQLiteStore


def _channel_envelope(body: str, *, chat_id: str = "", group_channel: str = "") -> str:
    payload: dict[str, str] = {}
    if chat_id:
        payload["chat_id"] = chat_id
    if group_channel:
        payload["group_channel"] = group_channel
    return (
        "Conversation info (guild):\n"
        "```json\n"
        + json.dumps(payload) + "\n"
        + "```\n"
        + body
    )


def _engine(tmp_path: Path, conv: str = "c", tenant: str = "t"):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import RetrieverConfig, StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id=conv,
        tenant_id=tenant,
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "vc.db")),
        tag_generator=TagGeneratorConfig(type="keyword"),
        retriever=RetrieverConfig(inbound_tagger_type="llm"),
    )
    return VirtualContextEngine(config=config)


def _sqlite(store) -> SQLiteStore:
    """Unwrap the engine's conversation-scoped view / composite store."""
    seen = store
    for _ in range(5):
        if isinstance(seen, SQLiteStore):
            return seen
        nxt = getattr(seen, "_segments", None) or getattr(seen, "_store", None)
        if nxt is None:
            break
        seen = nxt
    raise AssertionError(f"no SQLiteStore behind {type(store).__name__}")


def _seed(store, conv: str, tenant: str = "t") -> None:
    store.upsert_conversation(tenant_id=tenant, conversation_id=conv)


def _row(
    store,
    conv: str,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    user_raw_content: str | None = None,
    assistant_raw_content: str | None = None,
    origin_channel_id: str = "",
    origin_channel_label: str = "",
    origin_conversation_id: str | None = None,
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        user_raw_content=user_raw_content,
        assistant_raw_content=assistant_raw_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        origin_channel_id=origin_channel_id,
        origin_channel_label=origin_channel_label,
    )
    if origin_conversation_id is not None:
        # VCMERGE owns this column; save_canonical_turn deliberately never
        # writes it, so set it directly for the fixture.
        conn = _sqlite(store)._get_conn()
        conn.execute(
            "UPDATE canonical_turns SET origin_conversation_id = ? "
            "WHERE canonical_turn_id = ?",
            (origin_conversation_id, ct_id),
        )
        conn.commit()


def _channels(store, conv: str) -> list[tuple[str, str]]:
    return [
        (r.origin_channel_id, r.origin_channel_label)
        for r in store.get_all_canonical_turns(conv)
    ]


# ---------------------------------------------------------------------------
# Stable-key parsing
# ---------------------------------------------------------------------------

class TestStableKeyParsing:
    def _parse(self, key: str) -> str:
        from virtual_context.engine import VirtualContextEngine
        return VirtualContextEngine._channel_id_from_provenance_key(key)

    @pytest.mark.parametrize("key,expected", [
        ("sk:agent:bast:discord:channel:1524974537458974851", "1524974537458974851"),
        ("agent:bast:discord:channel:99", "99"),
        ("sk:agent:bast:discord:group:77", "77"),
        ("agent:a:telegram:group:5", "5"),
    ])
    def test_parseable_keys(self, key: str, expected: str):
        assert self._parse(key) == expected

    @pytest.mark.parametrize("key", [
        "sk:agent:bast:discord:direct:42",
        "sk:agent:bast:discord:dm:42",
        "3fdac837-1234-5678-9abc-def012345678",
        "deadbeefcafe",
        "",
        "sk:agent:bast:channel:42",
        "prefix-sk:agent:bast:discord:channel:42",
        "sk:agent:bast:discord:channel:42:extra",
        "sk:agent:bast:discord:channel:",
    ])
    def test_unparseable_keys_yield_nothing(self, key: str):
        """I6: no attribution is guessed from a DM key or an opaque id."""
        assert self._parse(key) == ""


# ---------------------------------------------------------------------------
# Raw-content decoding
# ---------------------------------------------------------------------------

class TestChannelFromRawContent:
    def _derive(self, raw):
        from virtual_context.engine import VirtualContextEngine
        return VirtualContextEngine._channel_from_raw_content(raw)

    def test_json_list_of_text_blocks(self):
        raw = json.dumps([
            {"type": "text", "text": _channel_envelope(
                "hello", chat_id="channel:7", group_channel="#a",
            )},
        ])
        assert self._derive(raw) == ("7", "#a")

    def test_json_list_of_plain_strings(self):
        raw = json.dumps([_channel_envelope("hi", chat_id="channel:7")])
        assert self._derive(raw) == ("7", "")

    def test_json_object_with_text(self):
        raw = json.dumps({"type": "text", "text": _channel_envelope(
            "hi", group_channel="#only",
        )})
        assert self._derive(raw) == ("", "#only")

    def test_last_candidate_wins(self):
        """Mirrors the ingest path's ``_last_text_block`` selection rule."""
        raw = json.dumps([
            {"type": "text", "text": _channel_envelope("first", chat_id="channel:1")},
            {"type": "text", "text": _channel_envelope("second", chat_id="channel:2")},
        ])
        assert self._derive(raw) == ("2", "")

    def test_fields_fill_from_different_blocks(self):
        raw = json.dumps([
            {"type": "text", "text": _channel_envelope("first", group_channel="#a")},
            {"type": "text", "text": _channel_envelope("second", chat_id="channel:2")},
        ])
        # Scanned last-first: block 2 gives the id, block 1 gives the label.
        assert self._derive(raw) == ("2", "#a")

    def test_plain_string_raw_payload(self):
        raw = _channel_envelope("hi", chat_id="channel:7", group_channel="#a")
        assert self._derive(raw) == ("7", "#a")

    def test_malformed_json_degrades_to_direct_text_parse(self):
        raw = "[{not json" + _channel_envelope("x", chat_id="channel:7")
        # Not valid JSON -> parsed directly as text. The envelope is not
        # leading here, so nothing is derived, and nothing raises.
        assert self._derive(raw) == ("", "")

    def test_valid_list_with_no_supported_candidate_contributes_nothing(self):
        assert self._derive(json.dumps([{"type": "image", "source": {}}])) == ("", "")

    def test_none_and_empty(self):
        assert self._derive(None) == ("", "")
        assert self._derive("") == ("", "")

    def test_dm_envelope_yields_nothing(self):
        raw = json.dumps([{"type": "text", "text": _channel_envelope(
            "hi", chat_id="dm:42",
        )}])
        assert self._derive(raw) == ("", "")


# ---------------------------------------------------------------------------
# backfill_channels
# ---------------------------------------------------------------------------

class TestBackfillChannels:
    def test_user_row_derives_from_user_raw_content(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0,
            user_content="hello",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "hello", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        report = engine.backfill_channels("c")
        assert report["updated"] == 1
        assert report["derived_from_raw"] == 1
        assert _channels(store, "c") == [("7", "#a")]

    def test_assistant_row_derives_from_assistant_raw_content(self, tmp_path: Path):
        """Unlike sender, an assistant physical row may own channel."""
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0,
            assistant_content="reply",
            assistant_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "reply", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        report = engine.backfill_channels("c")
        assert report["updated"] == 1
        assert _channels(store, "c") == [("7", "#a")]

    def test_assistant_raw_is_not_read_for_a_user_only_row(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0,
            user_content="hello",
            assistant_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "x", chat_id="channel:9",
            )}]),
        )
        report = engine.backfill_channels("c")
        assert report["updated"] == 0
        assert _channels(store, "c") == [("", "")]

    def test_combined_row_prefers_user_raw_then_fills_from_assistant(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0,
            user_content="u", assistant_content="a",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", group_channel="#from-user",
            )}]),
            assistant_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "a", chat_id="channel:7", group_channel="#from-assistant",
            )}]),
        )
        engine.backfill_channels("c")
        # User-first, missing-field-only: label from the user, id from the
        # assistant because the user side had none.
        assert _channels(store, "c") == [("7", "#from-user")]

    def test_string_content_row_without_raw_is_not_guessed(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(store, "c", ct_id="ct-1", sort_key=1000.0, user_content="already stripped")
        report = engine.backfill_channels("c")
        assert report["updated"] == 0
        assert report["skipped_no_derivation"] == 1
        assert _channels(store, "c") == [("", "")]

    def test_already_populated_row_is_skipped(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u",
            origin_channel_id="7", origin_channel_label="#a",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:999", group_channel="#other",
            )}]),
        )
        report = engine.backfill_channels("c")
        assert report["skipped_existing"] == 1
        assert report["eligible"] == 0
        assert _channels(store, "c") == [("7", "#a")]

    def test_partially_populated_row_fills_only_the_empty_column(self, tmp_path: Path):
        """I5: the two fields fill independently."""
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u",
            origin_channel_id="999",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        report = engine.backfill_channels("c")
        assert report["updated"] == 1
        # Stored id survives; only the empty label fills.
        assert _channels(store, "c") == [("999", "#a")]

    def test_target_native_row_recovers_id_from_its_own_stable_key(self, tmp_path: Path):
        conv = "sk:agent:bast:discord:channel:1524974537458974851"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        _row(store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u")
        report = engine.backfill_channels(conv)
        assert report["updated"] == 1
        assert report["derived_from_origin"] == 1
        assert _channels(store, conv) == [("1524974537458974851", "")]

    def test_moved_row_recovers_id_from_its_parseable_origin(self, tmp_path: Path):
        engine = _engine(tmp_path, conv="target-uuid")
        store = engine._store
        _seed(store, "target-uuid")
        _row(
            store, "target-uuid", ct_id="ct-1", sort_key=1000.0, user_content="u",
            origin_conversation_id="sk:agent:bast:discord:channel:42",
        )
        report = engine.backfill_channels("target-uuid")
        assert report["derived_from_origin"] == 1
        assert _channels(store, "target-uuid") == [("42", "")]

    def test_unparseable_origin_never_falls_through_to_the_target_id(self, tmp_path: Path):
        """A moved row whose origin is a UUID must not be misattributed to the
        merge target's stable channel key.
        """
        conv = "sk:agent:bast:discord:channel:999"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        _row(
            store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u",
            origin_conversation_id="3fdac837-1234-5678-9abc-def012345678",
        )
        report = engine.backfill_channels(conv)
        assert report["derived_from_origin"] == 0
        assert report["updated"] == 0
        assert _channels(store, conv) == [("", "")]

    def test_dm_origin_key_yields_no_channel(self, tmp_path: Path):
        conv = "sk:agent:bast:discord:dm:42"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        _row(store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u")
        report = engine.backfill_channels(conv)
        assert report["updated"] == 0
        assert _channels(store, conv) == [("", "")]

    def test_raw_label_and_origin_id_combine_on_one_row(self, tmp_path: Path):
        """The two sources fill different columns of the same row."""
        conv = "sk:agent:bast:discord:channel:42"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        _row(
            store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", group_channel="#vasttest",
            )}]),
        )
        report = engine.backfill_channels(conv)
        assert report["derived_from_raw"] == 1
        assert report["derived_from_origin"] == 1
        assert _channels(store, conv) == [("42", "#vasttest")]

    def test_raw_id_wins_over_the_stable_key_fallback(self, tmp_path: Path):
        conv = "sk:agent:bast:discord:channel:42"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        _row(
            store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7",
            )}]),
        )
        report = engine.backfill_channels(conv)
        assert report["derived_from_origin"] == 0
        assert _channels(store, conv) == [("7", "")]

    def test_mixed_rows_in_one_conversation(self, tmp_path: Path):
        conv = "sk:agent:bast:discord:channel:42"
        engine = _engine(tmp_path, conv=conv)
        store = engine._store
        _seed(store, conv)
        # raw-derived
        _row(
            store, conv, ct_id="ct-1", sort_key=1000.0, user_content="u1",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u1", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        # origin-derived (target-native)
        _row(store, conv, ct_id="ct-2", sort_key=2000.0, user_content="u2")
        # already populated
        _row(
            store, conv, ct_id="ct-3", sort_key=3000.0, user_content="u3",
            origin_channel_id="55", origin_channel_label="#z",
        )
        # moved with an opaque origin -> unrecoverable
        _row(
            store, conv, ct_id="ct-4", sort_key=4000.0, user_content="u4",
            origin_conversation_id="3fdac837-1234-5678-9abc-def012345678",
        )
        report = engine.backfill_channels(conv)
        assert report["skipped_existing"] == 1
        assert report["derived_from_raw"] == 1
        assert report["derived_from_origin"] == 1
        assert report["updated"] == 2
        assert _channels(store, conv) == [
            ("7", "#a"), ("42", ""), ("55", "#z"), ("", ""),
        ]

    def test_dry_run_reports_without_writing(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        report = engine.backfill_channels("c", dry_run=True)
        assert report["dry_run"] is True
        assert report["updated"] == 1
        assert _channels(store, "c") == [("", "")]

    def test_rerun_is_a_no_op(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        assert engine.backfill_channels("c")["updated"] == 1
        second = engine.backfill_channels("c")
        assert second["updated"] == 0
        assert second["skipped_existing"] == 1
        assert _channels(store, "c") == [("7", "#a")]

    def test_limit_caps_row_upgrades(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        for idx in range(3):
            _row(
                store, "c", ct_id=f"ct-{idx}", sort_key=1000.0 * (idx + 1),
                user_content=f"u{idx}",
                user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                    f"u{idx}", chat_id="channel:7",
                )}]),
            )
        report = engine.backfill_channels("c", limit=2)
        assert report["updated"] == 2
        assert [c[0] for c in _channels(store, "c")] == ["7", "7", ""]

    def test_empty_conversation_id_raises(self, tmp_path: Path):
        engine = _engine(tmp_path)
        with pytest.raises(ValueError):
            engine.backfill_channels("")

    def test_writes_go_through_the_cas_not_a_full_row_save(self, tmp_path: Path):
        engine = _engine(tmp_path)
        store = engine._store
        _seed(store, "c")
        _row(
            store, "c", ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7",
            )}]),
        )
        saves: list[object] = []
        original = store.save_canonical_turn
        store.save_canonical_turn = lambda *a, **kw: (  # type: ignore[assignment]
            saves.append(kw.get("canonical_turn_id")), original(*a, **kw),
        )[1]
        engine.backfill_channels("c")
        assert saves == []
        assert _channels(store, "c") == [("7", "")]


# ---------------------------------------------------------------------------
# Batch enumeration
# ---------------------------------------------------------------------------

class TestBatchEnumeration:
    def test_list_canonical_conversation_ids_is_tenant_scoped(self, tmp_path: Path):
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "c1", tenant="t1")
        _seed(store, "c2", tenant="t2")
        _row(store, "c1", ct_id="a", sort_key=1000.0, user_content="u")
        _row(store, "c2", ct_id="b", sort_key=1000.0, user_content="u")

        assert store.list_canonical_conversation_ids() == ["c1", "c2"]
        assert store.list_canonical_conversation_ids(tenant_id="t1") == ["c1"]
        assert store.list_canonical_conversation_ids(limit=1) == ["c1"]

    def test_enumeration_finds_canonical_only_conversations(self, tmp_path: Path):
        """``get_conversation_stats`` enumerates segments and would miss a
        conversation that was ingested but never compacted.
        """
        store = SQLiteStore(tmp_path / "vc.db")
        _seed(store, "never-compacted")
        _row(store, "never-compacted", ct_id="a", sort_key=1000.0, user_content="u")
        assert "never-compacted" in store.list_canonical_conversation_ids()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _run_cli(monkeypatch, capsys, argv: list[str]) -> dict:
    import sys as _sys
    from virtual_context.cli import main as cli_main

    monkeypatch.setattr(_sys, "argv", ["virtual-context", *argv])
    cli_main.main()
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


class TestCliWiring:
    def _db(self, tmp_path: Path) -> tuple[SQLiteStore, str]:
        db = tmp_path / "cli.db"
        store = SQLiteStore(db)
        return store, str(db)

    def test_single_conversation_mode(self, tmp_path: Path, monkeypatch, capsys):
        store, db = self._db(tmp_path)
        _seed(store, "c1", tenant="t1")
        _row(
            store, "c1", ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7", group_channel="#a",
            )}]),
        )
        store.close()

        payload = _run_cli(monkeypatch, capsys, [
            "admin", "backfill-channels", "c1",
            "--tenant-id", "t1",
            "--storage-backend", "sqlite", "--sqlite-path", db,
        ])
        assert payload["status"] == "ok"
        assert payload["conversations"] == 1
        assert payload["updated"] == 1
        assert payload["derived_from_raw"] == 1

        reopened = SQLiteStore(Path(db))
        assert _channels(reopened, "c1") == [("7", "#a")]

    def test_dry_run_writes_nothing(self, tmp_path: Path, monkeypatch, capsys):
        store, db = self._db(tmp_path)
        _seed(store, "c1", tenant="t1")
        _row(
            store, "c1", ct_id="ct-1", sort_key=1000.0, user_content="u",
            user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                "u", chat_id="channel:7",
            )}]),
        )
        store.close()

        payload = _run_cli(monkeypatch, capsys, [
            "admin", "backfill-channels", "c1", "--dry-run",
            "--storage-backend", "sqlite", "--sqlite-path", db,
        ])
        assert payload["dry_run"] is True
        assert payload["updated"] == 1

        reopened = SQLiteStore(Path(db))
        assert _channels(reopened, "c1") == [("", "")]

    def test_batch_mode_limit_caps_conversations_not_rows(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        store, db = self._db(tmp_path)
        for conv in ("c1", "c2", "c3"):
            _seed(store, conv, tenant="t1")
            for idx in range(2):
                _row(
                    store, conv, ct_id=f"{conv}-{idx}", sort_key=1000.0 * (idx + 1),
                    user_content=f"u{idx}",
                    user_raw_content=json.dumps([
                        {"type": "text", "text": _channel_envelope(
                            f"u{idx}", chat_id="channel:7",
                        )},
                    ]),
                )
        store.close()

        payload = _run_cli(monkeypatch, capsys, [
            "admin", "backfill-channels",
            "--tenant-id", "t1", "--all-convs-for-tenant", "--limit", "2",
            "--storage-backend", "sqlite", "--sqlite-path", db,
        ])
        # --limit caps enumerated conversations; the per-conversation row
        # limit is unset, so both rows of each conversation are upgraded.
        assert payload["conversations"] == 2
        assert payload["updated"] == 4

        reopened = SQLiteStore(Path(db))
        assert _channels(reopened, "c3") == [("", ""), ("", "")]

    def test_batch_mode_is_tenant_scoped(self, tmp_path: Path, monkeypatch, capsys):
        store, db = self._db(tmp_path)
        _seed(store, "c1", tenant="t1")
        _seed(store, "c2", tenant="t2")
        for conv in ("c1", "c2"):
            _row(
                store, conv, ct_id=f"{conv}-0", sort_key=1000.0, user_content="u",
                user_raw_content=json.dumps([{"type": "text", "text": _channel_envelope(
                    "u", chat_id="channel:7",
                )}]),
            )
        store.close()

        payload = _run_cli(monkeypatch, capsys, [
            "admin", "backfill-channels",
            "--tenant-id", "t1", "--all-convs-for-tenant",
            "--storage-backend", "sqlite", "--sqlite-path", db,
        ])
        assert payload["conversations"] == 1
        assert payload["results"][0]["conversation_id"] == "c1"

        reopened = SQLiteStore(Path(db))
        assert _channels(reopened, "c2") == [("", "")]

    def test_missing_target_arguments_exit_2(self, tmp_path: Path, monkeypatch, capsys):
        _store_obj, db = self._db(tmp_path)
        _store_obj.close()
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, capsys, [
                "admin", "backfill-channels",
                "--storage-backend", "sqlite", "--sqlite-path", db,
            ])
        assert exc.value.code == 2
