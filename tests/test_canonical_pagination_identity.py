"""Physical canonical identity survives shifted presentation ordinals."""

from types import SimpleNamespace
import uuid

import pytest

from tests.test_storage_domain_contracts import store as _shared_store
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.quote_search import _candidate_identity
from virtual_context.engine import VirtualContextEngine
from virtual_context.proxy.session_state import SessionState
from virtual_context.storage.postgres_vectors import VECTOR_MODEL
from virtual_context.types import CanonicalTurnChunkEmbedding, EngineState, QuoteResult, VirtualContextConfig


VECTOR = [1.0] + [0.0] * 383
store = _shared_store


def _source(store, name, ordinal, group, side, *, embed=True):
    key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"canonical-page-fixture/{name}"))
    store.save_canonical_turn(
        "owner", ordinal, f"exact user {name}" if side == "user" else "",
        f"exact assistant {name}" if side == "assistant" else "",
        canonical_turn_id=key, turn_group_number=group, sort_key=float(ordinal),
        primary_tag=f"tag-{name}", tags=[f"tag-{name}"],
    )
    if embed:
        store.store_canonical_turn_chunk_embeddings(
            "owner", ordinal, side, [CanonicalTurnChunkEmbedding(
                conversation_id="owner", canonical_turn_id=key, turn_number=ordinal,
                side=side, chunk_index=0, text=f"index only {name}", embedding=VECTOR,
            )], canonical_turn_id=key, embedding_model=VECTOR_MODEL,
        )
    return key


@pytest.mark.parametrize("native", [False, True])
def test_keyset_does_not_repeat_sources_when_earlier_insert_shifts_ordinals(store, native):
    if native:
        if store._relational_dialect != "postgres":
            pytest.skip("Native vector ranking is PostgreSQL-only")
        assert store.migrate_semantic_vectors(dry_run=False)["ready"]
    first_id = _source(store, "first", 10, 0, "user")
    second_id = _source(store, "second", 20, 0, "assistant")
    third_id = _source(store, "third", 30, 1, "user")
    def page(after=None):
        if native:
            return store.search_canonical_turn_chunks_by_embedding(
                VECTOR, conversation_id="owner", limit=1, after=after,
            )
        return store.get_canonical_turn_chunk_embedding_page(conversation_id="owner", limit=1, after=after)
    first, = page()
    assert first["canonical_turn_id"] == first_id and first["turn_number"] == 0
    _source(store, "inserted-earlier", 5, 2, "user", embed=False)
    remainder, cursor = [], first["cursor"]
    for _ in range(4):
        rows = page(cursor)
        if not rows:
            break
        row, = rows
        remainder.append(row["canonical_turn_id"])
        assert row["physical_row"].canonical_turn_id == row["canonical_turn_id"]
        cursor = row["cursor"]
    assert remainder == [second_id, third_id]


@pytest.mark.parametrize("native", [False, True])
def test_unscoped_legacy_search_renders_exact_split_source_not_logical_ordinal(store, native, monkeypatch):
    if native:
        if store._relational_dialect != "postgres":
            pytest.skip("Native vector ranking is PostgreSQL-only")
        assert store.migrate_semantic_vectors(dry_run=False)["ready"]
    _source(store, "first-user", 10, 0, "user", embed=False)
    _source(store, "first-assistant", 20, 0, "assistant")
    _source(store, "unrelated-group", 30, 1, "assistant", embed=False)
    monkeypatch.setattr(store, "get_canonical_turn_rows", lambda *a, **k: pytest.fail("physical ordinal used as logical group"))
    config = VirtualContextConfig(conversation_id="owner")
    config.retriever.vector_search_enabled = native
    manager = SemanticSearchManager(store, config)
    manager._embed_fn = lambda texts: [VECTOR for _ in texts]
    result, = manager.semantic_canonical_turn_search("q", conversation_id="owner")
    assert result.text == "Assistant: exact assistant first-assistant"
    assert result.tag == "tag-first-assistant"
    assert result.turn_number == 1


def test_watermark_and_healthy_hydration_do_not_load_archive_bodies(store, monkeypatch):
    for number in range(3):
        key = _source(store, f"paired-{number}", number * 10, number, "user", embed=False)
        _source(store, f"paired-answer-{number}", number * 10 + 1, number, "assistant", embed=False)
        if number < 2:
            store.mark_canonical_turns_compacted("owner", [key])
    monkeypatch.setattr(store, "get_all_canonical_turns", lambda *a, **k: pytest.fail("watermark archive body hydration"))
    monkeypatch.setattr(store, "_canonical_decoder", lambda: pytest.fail("watermark decoded source bodies"))
    engine = VirtualContextEngine.__new__(VirtualContextEngine)
    engine._store = store
    engine.config = VirtualContextConfig(conversation_id="owner")
    engine._engine_state = EngineState()
    engine._paging = SimpleNamespace(working_set={})
    assert engine._derive_compacted_prefix_messages_from_rows("owner") == (4, 1)
    engine.hydrate_from_session_state(SessionState(
        compacted_prefix_messages=0, last_completed_turn=2, last_indexed_turn=2,
        turn_tag_entries=[dict(turn_number=2, tags=["topic"], primary_tag="topic")],
    ))
    assert engine._engine_state.compacted_prefix_messages == 4
    assert engine._engine_state.last_compacted_turn == 1


def test_failed_watermark_preserves_cache_without_archive_fallback(store, monkeypatch):
    engine = VirtualContextEngine.__new__(VirtualContextEngine)
    engine._store = store
    def unavailable(_owner):
        raise OSError("temporary database read failure")
    monkeypatch.setattr(store, "get_compaction_watermark", unavailable)
    monkeypatch.setattr(store, "get_all_canonical_turns", lambda *a, **k: pytest.fail("archive fallback"))
    assert engine._derive_compacted_prefix_messages_from_rows("owner") is None


def test_legacy_quote_dedupe_uses_stable_source_ref_when_ordinals_shift():
    def hit(source, ordinal, side):
        return QuoteResult(text="source", tag="topic", segment_ref=f"canonical_turn_{source}",
                           source_scope="turn", turn_number=ordinal, matched_side=side)
    first = _candidate_identity(hit("first", 1, "user"), speaker_aware=False)
    assert first == _candidate_identity(hit("first", 2, "assistant"), speaker_aware=False)
    assert first != _candidate_identity(hit("second", 1, "user"), speaker_aware=False)
