"""Postgres mirror of the channel store contracts.

Skipped unless a Postgres DSN is configured. Keeps the two SQL backends in
lockstep on:

* the additive ``origin_channel_id`` / ``origin_channel_label`` columns on a
  fresh schema and on a table migrated by ``_ensure_canonical_turn_schema``,
  including their visibility through the ``canonical_turns_ordinal`` view;
* full-row read/write/preserve through ``save_canonical_turn``;
* ``update_canonical_turn_channels_if_empty`` — per-column compare-and-set,
  epoch-guarded, idempotent;
* ``search_canonical_turn_text`` — channel filtering before ORDER BY / LIMIT,
  scoped excerpt labels, and byte-identical unscoped output.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid

import pytest
from tests.pg_helpers import pg_dsn

PG_URL = pg_dsn()

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set"
)


@pytest.fixture()
def store():
    from virtual_context.storage.postgres import PostgresStore  # deferred

    return PostgresStore(PG_URL)


def _conv(prefix: str = "conv") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _row(
    store,
    conv: str,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    sender: str = "",
    origin_channel_id: str = "",
    origin_channel_label: str = "",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
        primary_tag="chat",
        tags=["chat"],
        origin_channel_id=origin_channel_id,
        origin_channel_label=origin_channel_label,
    )


def _channels(store, conv: str) -> list[tuple[str, str]]:
    return [
        (r.origin_channel_id, r.origin_channel_label)
        for r in store.get_all_canonical_turns(conv)
    ]


# ---------------------------------------------------------------------------
# Schema / catalog
# ---------------------------------------------------------------------------

class TestChannelSchemaPg:
    def test_columns_exist_with_text_not_null_default_empty(self, store):
        with store.pool.connection() as conn:
            rows = conn.execute(
                """SELECT column_name, data_type, is_nullable, column_default
                     FROM information_schema.columns
                    WHERE table_name = 'canonical_turns'
                      AND column_name IN
                          ('origin_channel_id', 'origin_channel_label')
                    ORDER BY column_name"""
            ).fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["data_type"] == "text"
            assert row["is_nullable"] == "NO"
            assert "''" in (row["column_default"] or "")

    def test_columns_are_visible_through_the_ordinal_view(self, store):
        with store.pool.connection() as conn:
            rows = conn.execute(
                """SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'canonical_turns_ordinal'
                      AND column_name IN
                          ('origin_channel_id', 'origin_channel_label',
                           'origin_conversation_id')"""
            ).fetchall()
        names = {r["column_name"] for r in rows}
        assert names == {
            "origin_channel_id", "origin_channel_label", "origin_conversation_id",
        }

    def test_migration_adds_the_columns_to_a_legacy_table(self, store):
        """Drop the columns, then re-run the migration path the way startup
        does, and confirm the rows and the view come back intact.
        """
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="legacy", origin_channel_id="7",
             origin_channel_label="#a")

        with store.pool.connection() as conn:
            conn.execute("DROP VIEW IF EXISTS canonical_turns_ordinal")
            conn.execute(
                "ALTER TABLE canonical_turns DROP COLUMN origin_channel_id"
            )
            conn.execute(
                "ALTER TABLE canonical_turns DROP COLUMN origin_channel_label"
            )

        store._ensure_canonical_turn_schema()
        store._ensure_canonical_turn_views()

        rows = store.get_all_canonical_turns(conv)
        assert len(rows) == 1
        assert rows[0].user_content == "legacy"
        # The values are gone with the dropped columns, but the columns are
        # back and writable.
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("", "")
        assert store.update_canonical_turn_channels_if_empty(
            conv, {rows[0].canonical_turn_id: ("7", "#a")},
        ) == 1
        assert _channels(store, conv) == [("7", "#a")]


# ---------------------------------------------------------------------------
# Read / write / preserve
# ---------------------------------------------------------------------------

class TestSaveAndLoadPg:
    def test_round_trip_both_columns(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="u", origin_channel_id="7", origin_channel_label="#a")
        assert _channels(store, conv) == [("7", "#a")]

    def test_assistant_row_may_carry_channel(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             assistant_content="a", origin_channel_id="7",
             origin_channel_label="#a")
        assert _channels(store, conv) == [("7", "#a")]

    def test_omitted_values_default_to_empty_on_rewrite(self, store):
        """Pins why every full-row path must re-supply both columns."""
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        ct_id = str(uuid.uuid4())
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="u",
             origin_channel_id="7", origin_channel_label="#a")
        store.save_canonical_turn(
            conv, -1, "u", "",
            canonical_turn_id=ct_id, sort_key=1000.0, turn_hash=f"h-{ct_id}",
        )
        assert _channels(store, conv) == [("", "")]

    def test_recent_rows_projection_carries_channel(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="u", origin_channel_id="7", origin_channel_label="#a")
        rows = store.get_recent_canonical_turns(conv, limit=5)
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("7", "#a")

    def test_untagged_rows_projection_carries_channel(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="u", origin_channel_id="7", origin_channel_label="#a")
        rows = store.iter_untagged_canonical_rows(
            conversation_id=conv, expected_lifecycle_epoch=1,
        )
        assert (rows[0].origin_channel_id, rows[0].origin_channel_label) == ("7", "#a")

    def test_raw_loader_exposes_origin_conversation_id(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        ct_id = str(uuid.uuid4())
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="u")
        with store.pool.connection() as conn:
            conn.execute(
                "UPDATE canonical_turns SET origin_conversation_id = %s "
                "WHERE canonical_turn_id = %s",
                ("sk:agent:bast:discord:channel:42", ct_id),
            )
        row = store.get_all_canonical_turns(conv)[0]
        assert row.origin_conversation_id == "sk:agent:bast:discord:channel:42"

    def test_ordinary_upsert_never_erases_origin_conversation_id(self, store):
        """VCMERGE owns that column; a default-empty rewrite must not clear it."""
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        ct_id = str(uuid.uuid4())
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="u")
        with store.pool.connection() as conn:
            conn.execute(
                "UPDATE canonical_turns SET origin_conversation_id = %s "
                "WHERE canonical_turn_id = %s",
                ("sk:agent:bast:discord:channel:42", ct_id),
            )
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="u rewritten")
        row = store.get_all_canonical_turns(conv)[0]
        assert row.origin_conversation_id == "sk:agent:bast:discord:channel:42"


# ---------------------------------------------------------------------------
# CAS
# ---------------------------------------------------------------------------

class TestUpdateCanonicalTurnChannelsIfEmptyPg:
    def _seed(self, store) -> tuple[str, str]:
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        ct_id = str(uuid.uuid4())
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="hello")
        return conv, ct_id

    def test_cas_sets_both_empty_columns(self, store):
        conv, ct_id = self._seed(store)
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#a")}, expected_lifecycle_epoch=1,
        ) == 1
        assert _channels(store, conv) == [("7", "#a")]

    def test_cas_fills_only_the_empty_column(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        ct_id = str(uuid.uuid4())
        _row(store, conv, ct_id=ct_id, sort_key=1000.0, user_content="u",
             origin_channel_id="999")
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#late")},
        ) == 1
        assert _channels(store, conv) == [("999", "#late")]

    def test_cas_never_overwrites_and_is_idempotent(self, store):
        conv, ct_id = self._seed(store)
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#a")},
        ) == 1
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#a")},
        ) == 0
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("8", "#b")},
        ) == 0
        assert _channels(store, conv) == [("7", "#a")]

    def test_cas_ignores_empty_candidates(self, store):
        conv, ct_id = self._seed(store)
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("", "")},
        ) == 0
        assert store.update_canonical_turn_channels_if_empty(conv, {}) == 0
        assert _channels(store, conv) == [("", "")]

    def test_cas_is_epoch_guarded(self, store):
        conv, ct_id = self._seed(store)
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#a")}, expected_lifecycle_epoch=99,
        ) == 0
        assert _channels(store, conv) == [("", "")]
        assert store.update_canonical_turn_channels_if_empty(
            conv, {ct_id: ("7", "#a")}, expected_lifecycle_epoch=1,
        ) == 1

    def test_cas_is_scoped_to_the_conversation(self, store):
        conv_a, ct_a = self._seed(store)
        conv_b, _ct_b = self._seed(store)
        assert store.update_canonical_turn_channels_if_empty(
            conv_b, {ct_a: ("7", "#a")},
        ) == 0
        assert _channels(store, conv_a) == [("", "")]


# ---------------------------------------------------------------------------
# Lexical channel filter
# ---------------------------------------------------------------------------

class TestSearchCanonicalTurnTextChannelPg:
    def _seed(self, store) -> str:
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="peptide dosing", sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=2000.0,
             user_content="peptide elsewhere",
             origin_channel_id="9", origin_channel_label="#other")
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=3000.0,
             user_content="peptide unattributed")
        return conv

    def test_filter_by_label_with_and_without_hash(self, store):
        conv = self._seed(store)
        for wanted in ("#vasttest", "vasttest", "VASTTEST"):
            results = store.search_canonical_turn_text(
                "peptide", conversation_id=conv, channel=wanted,
            )
            assert len(results) == 1, wanted
            assert "peptide dosing" in results[0].text

    def test_filter_by_id(self, store):
        conv = self._seed(store)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id=conv, channel="9",
        )
        assert len(results) == 1
        assert "peptide elsewhere" in results[0].text

    def test_filter_applies_before_the_limit(self, store):
        """The out-of-channel rows sort first (higher sort_key)."""
        conv = self._seed(store)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id=conv, channel="#vasttest", limit=1,
        )
        assert len(results) == 1
        assert "peptide dosing" in results[0].text

    def test_unattributed_rows_never_match(self, store):
        conv = self._seed(store)
        assert store.search_canonical_turn_text(
            "peptide", conversation_id=conv, channel="#nope",
        ) == []

    def test_scoped_excerpt_composes_channel_then_sender(self, store):
        conv = self._seed(store)
        results = store.search_canonical_turn_text(
            "peptide", conversation_id=conv, channel="#vasttest",
        )
        assert results[0].text.startswith("[#vasttest] BigTex: ")
        assert results[0].matched_side == "user"

    def test_scoped_assistant_excerpt_is_never_sender_labeled(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="q", assistant_content="the answer is 42",
             sender="BigTex", origin_channel_id="7",
             origin_channel_label="#vasttest")
        results = store.search_canonical_turn_text(
            "answer", conversation_id=conv, channel="#vasttest",
        )
        assert results[0].matched_side == "assistant"
        assert results[0].text.startswith("[#vasttest] Assistant: ")

    def test_id_only_row_is_labeled_with_its_id(self, store):
        conv = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=conv)
        _row(store, conv, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="toes tingle", origin_channel_id="1524974537458974851")
        results = store.search_canonical_turn_text(
            "toes", conversation_id=conv, channel="1524974537458974851",
        )
        assert results[0].text.startswith("[1524974537458974851] User: ")

    def test_a_channel_name_is_not_a_text_match(self, store):
        conv = self._seed(store)
        assert store.search_canonical_turn_text(
            "vasttest", conversation_id=conv,
        ) == []

    def test_unscoped_output_ignores_stored_channel_values(self, store):
        """I3: byte-identical output with and without stored provenance."""
        plain = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=plain)
        _row(store, plain, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="my toes tingle", sender="BigTex")

        tagged = _conv()
        store.upsert_conversation(tenant_id="t", conversation_id=tagged)
        _row(store, tagged, ct_id=str(uuid.uuid4()), sort_key=1000.0,
             user_content="my toes tingle", sender="BigTex",
             origin_channel_id="7", origin_channel_label="#vasttest")

        a = store.search_canonical_turn_text("toes", conversation_id=plain)
        b = store.search_canonical_turn_text("toes", conversation_id=tagged)
        assert [r.text for r in a] == [r.text for r in b]
        assert [r.matched_side for r in a] == [r.matched_side for r in b]
        assert a[0].text == "BigTex: my toes tingle"

    def test_percent_in_channel_request_is_not_a_wildcard(self, store):
        conv = self._seed(store)
        assert store.search_canonical_turn_text(
            "peptide", conversation_id=conv, channel="%",
        ) == []
