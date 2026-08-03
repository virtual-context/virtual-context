"""The attestation loader: the producer that did not exist.

``verify_candidates`` was written against a mapping nothing built outside a
test fixture. In production it would have received ``{}`` and rejected every
candidate as ``no_attested_source`` — a pipeline that is correct, runs, and
yields nothing forever. These tests exist so that gap cannot reopen silently.
"""

from __future__ import annotations

import uuid

import pytest

from tests.pg_helpers import pg_dsn

from virtual_context.core.engagement import (
    MessageSourceRecord, load_message_sources,
)

TURN = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


class _Store:
    def __init__(self, rows, *, record=None):
        self._rows = rows
        self.calls = [] if record is None else record

    def list_attested_message_sources(self, *, tenant_id, canonical_turn_ids):
        self.calls.append((tenant_id, list(canonical_turn_ids)))
        return [r for r in self._rows
                if r["canonical_turn_id"] in set(canonical_turn_ids)]


def _row(turn_id=TURN, **kw):
    base = dict(
        canonical_turn_id=turn_id, message_id="1524917968440524991",
        channel_id="1524917968440524990", guild_id="1524917037191925871",
        author_id="1338726888809697364",
        source_actor_id="actor:discord:1338726888809697364",
    )
    base.update(kw)
    return base


class TestTheLoaderProducesWhatTheVerifierConsumes:
    def test_a_ledger_row_becomes_a_record(self):
        store = _Store([_row()])
        out = load_message_sources(store, tenant_id="t", canonical_turn_ids=[TURN])
        assert isinstance(out[TURN], MessageSourceRecord)
        assert out[TURN].author_id == "1338726888809697364"
        assert out[TURN].message_id == "1524917968440524991"

    def test_the_result_is_keyed_by_canonical_turn_id(self):
        """verify_candidates looks up by turn id, not message id."""
        store = _Store([_row()])
        out = load_message_sources(store, tenant_id="t", canonical_turn_ids=[TURN])
        assert set(out) == {TURN}

    def test_a_turn_with_no_attestation_is_absent_not_blank(self):
        """A missing attestation and an empty one mean different things.

        A blank record would compare equal to nothing and be rejected for the
        wrong reason; absence makes the verifier say no_attested_source.
        """
        store = _Store([])
        out = load_message_sources(store, tenant_id="t", canonical_turn_ids=[TURN])
        assert out == {}

    def test_the_query_is_bounded_to_the_ids_asked_for(self):
        """Not a scan. The ledger grows with every ingested message."""
        store = _Store([_row(), _row(OTHER)])
        load_message_sources(store, tenant_id="t", canonical_turn_ids=[TURN])
        assert store.calls == [("t", [TURN])]

    def test_no_ids_does_not_touch_the_store(self):
        store = _Store([_row()])
        assert load_message_sources(store, tenant_id="t", canonical_turn_ids=[]) == {}
        assert store.calls == []

    def test_the_tenant_is_forwarded(self):
        store = _Store([_row()])
        load_message_sources(store, tenant_id="tenant-a", canonical_turn_ids=[TURN])
        assert store.calls[0][0] == "tenant-a"

    def test_a_store_that_cannot_attest_refuses_loudly(self):
        """Returning {} would look like 'nothing attested' and post nothing.

        Silence here is indistinguishable from a working pipeline with no
        candidates, which is the failure that hid this gap in the first
        place. It has to raise.
        """
        with pytest.raises(RuntimeError, match="attested message sources"):
            load_message_sources(object(), tenant_id="t", canonical_turn_ids=[TURN])


class TestItFeedsTheVerifierEndToEnd:
    def test_a_loaded_mapping_verifies_a_matching_candidate(self):
        from virtual_context.core.engagement import verify_candidates
        from virtual_context.core.engagement.candidates import Candidate

        store = _Store([_row()])
        sources = load_message_sources(
            store, tenant_id="t", canonical_turn_ids=[TURN],
        )
        candidate = Candidate(
            canonical_turn_id=TURN, actor_id="actor:discord:1338726888809697364",
            channel_id="1524917968440524990",
            source_message_id="1524917968440524991",
            text="Adding ss31 for four weeks.", sent_at=None,
        )
        verified, rejected = verify_candidates([candidate], sources)
        assert [c.canonical_turn_id for c in verified] == [TURN]
        assert rejected == []

    def test_an_empty_mapping_rejects_everything(self):
        """What production would have done before this module existed."""
        from virtual_context.core.engagement import verify_candidates
        from virtual_context.core.engagement.candidates import Candidate

        candidate = Candidate(
            canonical_turn_id=TURN, actor_id="actor:discord:1338726888809697364",
            channel_id="1524917968440524990",
            source_message_id="1524917968440524991",
            text="Adding ss31 for four weeks.", sent_at=None,
        )
        verified, rejected = verify_candidates([candidate], {})
        assert verified == []
        assert [r.reason for r in rejected] == ["no_attested_source"]


# ------------------------------------------------- against real backends


class TestAgainstRealBackends:
    """The fixture tests above cannot catch a wrong column name or cast.

    This gap existed precisely because every test built the mapping by hand,
    so a fixture-only test here would reproduce the exact blindness that hid
    it. These execute the shipped SQL against a real database.
    """

    TEN = "tenant-a"

    def _seed(self, store, execute, *, tenant=None, ordinal=100):
        """One attested user turn. Returns its canonical turn id.

        Every identity is unique per call. An earlier version pinned the
        conversation id and sort key, which passed once and then collided on
        the next run against the same database — the test assumed a fresh
        database without saying so, which is the same class of hidden ruler
        this file exists to catch.

        The ledger's pair-shape constraint requires an assistant turn and a
        non-empty assistant hash, so a lone user row cannot be inserted —
        worth knowing before anyone writes a repair script against it.
        """
        conv = f"conv-{uuid.uuid4().hex[:12]}"
        account = f"acct-{uuid.uuid4().hex[:12]}"
        user_turn, asst_turn = str(uuid.uuid4()), str(uuid.uuid4())
        for offset, turn in enumerate((user_turn, asst_turn)):
            execute(
                """INSERT INTO canonical_turns
                   (canonical_turn_id, conversation_id, turn_group_number,
                    sort_key, turn_hash, hash_version, normalized_user_text,
                    normalized_assistant_text, user_content, assistant_content,
                    primary_tag, tags_json)
                   VALUES (?,?,1,?,?,1,'u','a','u','a','t','[]')""",
                (turn, conv, ordinal + offset, uuid.uuid4().hex),
            )
        execute(
            """INSERT INTO canonical_message_sources
               (tenant_id, agent_scope_id, platform, account_id, message_id,
                canonical_turn_id, assistant_canonical_turn_id,
                assistant_turn_hash, turn_group_number, pair_version,
                audience_conversation_id, channel_id, guild_id, author_id,
                source_actor_id, transport_body_sha256, canonical_body_sha256,
                projection_version, canonical_turn_hash,
                reply_target_message_id, observed_at, created_at)
               VALUES (?,'vast','discord',?,'1524917968440524991',?,?,
                       'ah',-1,1,?,'1524917968440524990',
                       '1524917037191925871','1338726888809697364',
                       'actor:discord:1338726888809697364','a','b','v1','h',
                       '','2026-08-02','2026-08-02')""",
            (tenant or self.TEN, account, user_turn, asst_turn, conv),
        )
        return user_turn

    @pytest.mark.skipif(not pg_dsn(), reason="no Postgres DSN")
    def test_postgres_round_trip(self):
        from virtual_context.storage.postgres import PostgresStore

        store = PostgresStore(pg_dsn())

        def execute(sql, params):
            with store.pool.connection() as conn:
                conn.execute(sql.replace("?", "%s"), params)

        turn = self._seed(store, execute, ordinal=200)
        out = load_message_sources(
            store, tenant_id=self.TEN, canonical_turn_ids=[turn],
        )
        assert out[turn].author_id == "1338726888809697364"
        assert out[turn].message_id == "1524917968440524991"
        assert out[turn].channel_id == "1524917968440524990"

    @pytest.mark.skipif(not pg_dsn(), reason="no Postgres DSN")
    def test_postgres_will_not_cross_tenants(self):
        from virtual_context.storage.postgres import PostgresStore

        store = PostgresStore(pg_dsn())

        def execute(sql, params):
            with store.pool.connection() as conn:
                conn.execute(sql.replace("?", "%s"), params)

        turn = self._seed(store, execute, ordinal=300)
        assert load_message_sources(
            store, tenant_id="somebody-else", canonical_turn_ids=[turn],
        ) == {}

    @pytest.mark.skipif(not pg_dsn(), reason="no Postgres DSN")
    def test_postgres_unknown_id_is_absent(self):
        from virtual_context.storage.postgres import PostgresStore

        store = PostgresStore(pg_dsn())
        assert load_message_sources(
            store, tenant_id=self.TEN, canonical_turn_ids=[str(uuid.uuid4())],
        ) == {}

    def test_sqlite_round_trip(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path / "s.db"))
        conn = store._get_conn()

        def execute(sql, params):
            conn.execute(sql, params)
            conn.commit()

        turn = self._seed(store, execute, ordinal=400)
        out = load_message_sources(
            store, tenant_id=self.TEN, canonical_turn_ids=[turn],
        )
        assert out[turn].author_id == "1338726888809697364"
        assert out[turn].message_id == "1524917968440524991"
