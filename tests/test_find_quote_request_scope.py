"""Request-owned audience gates for ordinary ``vc_find_quote`` retrieval."""

from __future__ import annotations

from types import SimpleNamespace

from virtual_context.core.quote_search import find_quote
from virtual_context.storage.postgres import PostgresStore
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    QuoteResult,
    SourceProvenance,
    SpeakerRetrievalContext,
)


OWNER = "owner"
AUDIENCE = "aud-a"
CHANNEL = "chan-a"


def _ctx(**overrides) -> SpeakerRetrievalContext:
    values = {
        "tenant_id": "tenant",
        "owner_conversation_id": OWNER,
        "audience_conversation_id": AUDIENCE,
        "audience_channel_id": CHANNEL,
        "request_origin_channel_id": CHANNEL,
    }
    values.update(overrides)
    return SpeakerRetrievalContext(**values)


def _save(
    store: SQLiteStore,
    canonical_id: str,
    text: str,
    sort_key: float,
    *,
    audience: str = AUDIENCE,
    version: int = 1,
    channel: str = CHANNEL,
) -> None:
    store.save_canonical_turn(
        OWNER,
        -1,
        text,
        "",
        canonical_turn_id=canonical_id,
        turn_hash=f"hash-{canonical_id}",
        sort_key=sort_key,
        sender="Speaker",
        sender_actor_id="actor:discord:speaker",
        audience_conversation_id=audience,
        audience_attribution_version=version,
        origin_channel_id=channel,
    )


class _NoSemantic:
    def semantic_canonical_turn_search(self, *args, **kwargs):
        return []


def test_sqlite_ordinary_quote_scope_filters_before_limit(tmp_path):
    store = SQLiteStore(tmp_path / "scope.db")
    store.upsert_conversation(tenant_id="tenant", conversation_id=OWNER)
    # Every ineligible row sorts above the authorized row.  A post-LIMIT
    # filter would return nothing (or leak the private needle) at limit=1.
    _save(store, "safe", "private needle safe audience", 1.0)
    _save(
        store, "other-audience", "private needle other audience", 5.0,
        audience="aud-b",
    )
    _save(
        store, "wrong-channel", "private needle sibling channel", 4.0,
        channel="chan-b",
    )
    _save(
        store, "old-attribution", "private needle stale attribution", 3.0,
        version=0,
    )
    _save(
        store, "missing-attribution", "private needle missing audience", 2.0,
        audience="",
    )

    results = store.search_canonical_turn_text(
        "private needle",
        limit=1,
        conversation_id=OWNER,
        speaker_context=_ctx(),
    )

    assert len(results) == 1
    assert "safe audience" in results[0].text
    assert results[0].provenance is not None
    assert results[0].provenance.audience_conversation_id == AUDIENCE
    assert results[0].provenance.origin_channel_id == CHANNEL


def test_sqlite_conversation_policy_excludes_private_empty_channel(tmp_path):
    store = SQLiteStore(tmp_path / "conversation-scope.db")
    store.upsert_conversation(tenant_id="tenant", conversation_id=OWNER)
    _save(store, "same-channel", "needle same channel", 1.0)
    _save(store, "sibling-channel", "needle sibling channel", 2.0,
          channel="chan-b")
    _save(store, "private", "needle private row", 3.0, channel="")

    results = store.search_canonical_turn_text(
        "needle",
        limit=10,
        conversation_id=OWNER,
        speaker_context=_ctx(
            audience_channel_scope="conversation",
            audience_channel_id="",
        ),
    )

    assert {result.provenance.origin_channel_id for result in results} == {
        "chan-a", "chan-b",
    }
    assert all("private row" not in result.text for result in results)


def _quote(
    text: str,
    canonical_id: str,
    *,
    audience: str = AUDIENCE,
    version: int = 1,
    channel: str = CHANNEL,
) -> QuoteResult:
    return QuoteResult(
        text=text,
        tag="chat",
        segment_ref=f"canonical_turn_{canonical_id}",
        source_scope="turn",
        turn_number=1,
        matched_side="user",
        provenance=SourceProvenance(
            conversation_id=OWNER,
            canonical_turn_id=canonical_id,
            source_role="requester",
            actor_id="actor:discord:speaker",
            audience_conversation_id=audience,
            audience_attribution_version=version,
            origin_channel_id=channel,
        ),
    )


def test_quote_boundary_rejects_unscoped_adapter_candidates():
    class IgnoringStore:
        def search_canonical_turn_text(self, *args, **kwargs):
            return [
                _quote("private needle from aud-b", "bad", audience="aud-b"),
                _quote("private needle stale", "stale", version=0),
                _quote("private needle safe", "safe"),
            ]

    result = find_quote(
        IgnoringStore(),
        _NoSemantic(),
        "private needle",
        max_results=5,
        conversation_id=OWNER,
        speaker_context=_ctx(),
    )

    assert [entry["excerpt"] for entry in result["results"]] == [
        "private needle safe",
    ]


def test_explicit_ineligible_context_fails_before_candidate_access():
    store = SimpleNamespace(
        search_canonical_turn_text=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate source must not be called")
        ),
    )
    result = find_quote(
        store,
        _NoSemantic(),
        "needle",
        conversation_id=OWNER,
        speaker_context=SpeakerRetrievalContext.ineligible(),
    )
    assert result["found"] is False
    assert result["results"] == []
    assert "authority is unproved" in result["message"]


class _Rows:
    def fetchall(self):
        return []


class _PgConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object] | tuple[object, ...]]] = []

    def execute(self, sql, params=()):
        self.executed.append((str(sql), params))
        return _Rows()


class _Checkout:
    def __init__(self, connection: _PgConnection) -> None:
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Pool:
    def __init__(self) -> None:
        self.connection_object = _PgConnection()
        self.checkouts = 0

    def connection(self):
        self.checkouts += 1
        return _Checkout(self.connection_object)


def _postgres_store() -> tuple[PostgresStore, _Pool]:
    store = PostgresStore.__new__(PostgresStore)
    pool = _Pool()
    store.pool = pool
    store.search_config = None
    return store, pool


def test_postgres_lexical_sql_scopes_each_lane_before_order_and_limit():
    store, pool = _postgres_store()

    assert store.search_canonical_turn_text(
        "private needle",
        limit=2,
        conversation_id=OWNER,
        speaker_context=_ctx(),
    ) == []

    assert len(pool.connection_object.executed) == 2
    for sql, params in pool.connection_object.executed:
        normalized = " ".join(sql.split())
        assert "conversation_id = %s" in normalized
        assert "audience_conversation_id = %s" in normalized
        assert "audience_attribution_version = %s" in normalized
        assert "COALESCE(origin_channel_id, '') = %s" in normalized
        assert normalized.index("audience_conversation_id = %s") \
            < normalized.index("ORDER BY") < normalized.index("LIMIT %s")
        assert list(params)[-5:] == [OWNER, AUDIENCE, 1, CHANNEL, 2]


def test_postgres_ineligible_context_executes_no_candidate_sql():
    store, pool = _postgres_store()
    assert store.search_canonical_turn_text(
        "needle",
        conversation_id=OWNER,
        speaker_context=SpeakerRetrievalContext.ineligible(),
    ) == []
    assert pool.checkouts == 0
