"""Physical store seams for speaker-aware retrieval.

The SQL backends expose three physical seams for the speaker-aware branch:
context-bearing chunk enumeration, a batched physical row lookup by
``(conversation_id, canonical_turn_id)``, and a raw orphan-chunk inventory
for the admin reindex. All three admit rows only through the physical
``canonical_turns`` table — never the logical merge seam — and every
legacy call shape stays byte-identical.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from types import SimpleNamespace

from virtual_context.config import VirtualContextConfig
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CanonicalTurnChunkEmbedding,
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
)


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "vc.db")
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _row(
    store: SQLiteStore,
    *,
    ct_id: str,
    sort_key: float,
    user_content: str = "",
    assistant_content: str = "",
    reply_target_body: str = "",
    reply_subject_actor_id: str = "",
    conv: str = "c",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        primary_tag="chat",
        tags=["chat"],
        reply_target_body=reply_target_body,
        reply_subject_actor_id=reply_subject_actor_id,
    )


def _chunk(
    store: SQLiteStore,
    *,
    ct_id: str,
    side: str,
    text: str,
    turn_number: int = 0,
    conv: str = "c",
) -> None:
    store.store_canonical_turn_chunk_embeddings(
        conv, turn_number, side,
        [CanonicalTurnChunkEmbedding(
            conversation_id=conv,
            side=side,
            chunk_index=0,
            text=text,
            embedding=[1.0, 0.0],
            canonical_turn_id=ct_id,
            turn_number=turn_number,
        )],
        canonical_turn_id=ct_id,
    )


def _ctx(**kw) -> SpeakerRetrievalContext:
    base = dict(
        tenant_id="t",
        owner_conversation_id="c",
        audience_conversation_id="c",
    )
    base.update(kw)
    return SpeakerRetrievalContext(**base)


def _semantic(store) -> SemanticSearchManager:
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    manager = SemanticSearchManager(store=store, config=config)
    manager._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
    return manager


class TestOrphanChunkInventory:
    def test_returns_only_chunks_without_a_physical_row(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-live", sort_key=1000.0, user_content="hello")
        _chunk(store, ct_id="ct-live", side="user", text="hello")
        _chunk(store, ct_id="ct-gone", side="user", text="stranded")
        orphans = store.get_orphan_canonical_turn_chunk_embeddings("c")
        assert [chunk.canonical_turn_id for chunk in orphans] == ["ct-gone"]
        assert orphans[0].turn_number == -1
        assert orphans[0].text == "stranded"

    def test_conversation_filter_is_literal(self, tmp_path: Path):
        store = _store(tmp_path)
        store.upsert_conversation(tenant_id="t", conversation_id="other")
        _chunk(store, ct_id="ct-gone", side="user", text="stranded")
        _chunk(store, ct_id="ct-gone-2", side="user", text="stranded too",
               conv="other")
        assert [
            chunk.conversation_id
            for chunk in store.get_orphan_canonical_turn_chunk_embeddings("c")
        ] == ["c"]
        both = store.get_orphan_canonical_turn_chunk_embeddings()
        assert {chunk.conversation_id for chunk in both} == {"c", "other"}

    def test_reindex_uses_the_real_store_inventory(self, tmp_path: Path):
        from virtual_context.engine import VirtualContextEngine

        store = _store(tmp_path)
        _chunk(store, ct_id="ct-gone", side="subject", text="stranded")
        engine_self = SimpleNamespace(_store=store, _semantic=_semantic(store))
        report = VirtualContextEngine.reindex_canonical_turn_embeddings(
            engine_self, "c", dry_run=True,
        )
        assert report["orphan_chunks"] == 1
        assert report["orphan_rows"] == 1
        assert report["orphan_deleted"] == 0

        report = VirtualContextEngine.reindex_canonical_turn_embeddings(
            engine_self, "c", dry_run=False,
        )
        assert report["orphan_deleted"] == 1
        assert store.get_orphan_canonical_turn_chunk_embeddings("c") == []


class TestSpeakerChunkEnumeration:
    def test_orphan_chunks_never_reach_the_speaker_branch(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-live", sort_key=1000.0, user_content="hello")
        _chunk(store, ct_id="ct-live", side="user", text="hello")
        _chunk(store, ct_id="ct-gone", side="user", text="stranded")
        chunks = store.get_all_canonical_turn_chunk_embeddings(
            conversation_id="c", speaker_context=_ctx(),
        )
        assert [chunk.canonical_turn_id for chunk in chunks] == ["ct-live"]

    def test_subject_side_is_enumerated(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="q",
             reply_target_body="the trip was great")
        _chunk(store, ct_id="ct-1", side="subject", text="the trip was great")
        chunks = store.get_all_canonical_turn_chunk_embeddings(
            conversation_id="c", speaker_context=_ctx(),
        )
        assert [chunk.side for chunk in chunks] == ["subject"]

    def test_conflicting_conversation_returns_no_candidates(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="hello")
        _chunk(store, ct_id="ct-1", side="user", text="hello")
        assert store.get_all_canonical_turn_chunk_embeddings(
            conversation_id="elsewhere", speaker_context=_ctx(),
        ) == []

    def test_context_owner_scopes_an_unscoped_call(self, tmp_path: Path):
        store = _store(tmp_path)
        store.upsert_conversation(tenant_id="t", conversation_id="other")
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="hello")
        _row(store, ct_id="ct-2", sort_key=1000.0, user_content="hello",
             conv="other")
        _chunk(store, ct_id="ct-1", side="user", text="hello")
        _chunk(store, ct_id="ct-2", side="user", text="hello", conv="other")
        chunks = store.get_all_canonical_turn_chunk_embeddings(
            speaker_context=_ctx(),
        )
        assert {chunk.conversation_id for chunk in chunks} == {"c"}

    def test_legacy_call_shape_is_unchanged(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="hello")
        _chunk(store, ct_id="ct-1", side="user", text="hello")
        default = store.get_all_canonical_turn_chunk_embeddings(
            conversation_id="c",
        )
        explicit = store.get_all_canonical_turn_chunk_embeddings(
            conversation_id="c", speaker_context=None,
        )
        assert default == explicit
        assert default[0].turn_number == 0


class TestPhysicalRowLookup:
    def test_returns_physical_rows_keyed_by_canonical_id(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="first")
        _row(store, ct_id="ct-2", sort_key=2000.0, assistant_content="second")
        rows = store.get_canonical_turn_rows_by_id(
            [("c", "ct-1"), ("c", "ct-2"), ("c", "ct-missing")],
            speaker_context=_ctx(),
        )
        assert set(rows) == {("c", "ct-1"), ("c", "ct-2")}
        assert rows[("c", "ct-1")].user_content == "first"
        assert rows[("c", "ct-2")].assistant_content == "second"
        assert rows[("c", "ct-1")].turn_number == 0
        assert rows[("c", "ct-2")].turn_number == 1

    def test_sibling_rows_stay_separate_physical_rows(self, tmp_path: Path):
        # Two physical rows in one logical turn group: the lookup must not
        # merge them or let one half supply the other's text.
        store = _store(tmp_path)
        store.save_canonical_turn(
            "c", -1, "user half", "",
            canonical_turn_id="ct-u", sort_key=1000.0, turn_hash="h-u",
            turn_group_number=0,
        )
        store.save_canonical_turn(
            "c", -1, "", "assistant half",
            canonical_turn_id="ct-a", sort_key=1001.0, turn_hash="h-a",
            turn_group_number=0,
        )
        rows = store.get_canonical_turn_rows_by_id(
            [("c", "ct-u"), ("c", "ct-a")], speaker_context=_ctx(),
        )
        assert rows[("c", "ct-u")].user_content == "user half"
        assert rows[("c", "ct-u")].assistant_content == ""
        assert rows[("c", "ct-a")].assistant_content == "assistant half"
        assert rows[("c", "ct-a")].user_content == ""

    def test_keys_outside_the_proved_owner_are_rejected(self, tmp_path: Path):
        store = _store(tmp_path)
        store.upsert_conversation(tenant_id="t", conversation_id="other")
        _row(store, ct_id="ct-1", sort_key=1000.0, user_content="mine")
        _row(store, ct_id="ct-x", sort_key=1000.0, user_content="not mine",
             conv="other")
        rows = store.get_canonical_turn_rows_by_id(
            [("c", "ct-1"), ("other", "ct-x")], speaker_context=_ctx(),
        )
        assert set(rows) == {("c", "ct-1")}

    def test_empty_keys_return_empty(self, tmp_path: Path):
        store = _store(tmp_path)
        assert store.get_canonical_turn_rows_by_id(
            [], speaker_context=_ctx(),
        ) == {}


class TestSpeakerSemanticOnRealStore:
    def test_speaker_branch_runs_end_to_end_on_sqlite(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="what about it?",
             reply_target_body="the boston trip was amazing",
             reply_subject_actor_id="actor:tg:222")
        _chunk(store, ct_id="ct-1", side="subject",
               text="the boston trip was amazing")
        manager = _semantic(store)
        results = manager.semantic_canonical_turn_search(
            "boston trip", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        assert results[0].provenance is not None
        assert results[0].provenance.source_role == "subject"
        assert results[0].provenance.actor_id == "actor:tg:222"
        assert results[0].text == "the boston trip was amazing"

    def test_legacy_branch_still_ignores_subject_chunks(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="what about it?",
             reply_target_body="the boston trip was amazing")
        _chunk(store, ct_id="ct-1", side="subject",
               text="the boston trip was amazing")
        manager = _semantic(store)
        assert manager.semantic_canonical_turn_search(
            "boston trip", conversation_id="c",
        ) == []


