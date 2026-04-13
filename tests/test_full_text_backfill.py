import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timedelta, timezone

from virtual_context.core.full_text_backfill import (
    backfill_full_text_rows,
    build_authoritative_session_dates_from_source_pairs,
    bootstrap_full_text_chunks,
    repair_session_date_lineage,
    repair_full_text_session_dates,
    rebuild_full_text_chunks,
    resolve_segment_turn_ranges,
)
from virtual_context.core.semantic_search import SemanticSearchManager, persist_turn_with_embeddings
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    ChunkEmbedding,
    EngineStateSnapshot,
    Fact,
    FactSignal,
    FullTextChunkEmbedding,
    SegmentMetadata,
    StorageConfig,
    StoredSegment,
    TurnTagEntry,
    VirtualContextConfig,
)


class _StubEmbeddingProvider:
    def get_embed_fn(self):
        def _embed(texts: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for idx, text in enumerate(texts):
                total = sum(ord(ch) for ch in text)
                vectors.append([float(len(text) or 1), float((total + idx) % 997)])
            return vectors
        return _embed


def _make_config(db_path) -> VirtualContextConfig:
    return VirtualContextConfig(
        conversation_id="conv-1",
        storage=StorageConfig(backend="sqlite", sqlite_path=str(db_path)),
    )


def _make_store(tmp_path):
    return SQLiteStore(tmp_path / "store.db")


def _make_segment(
    *,
    ref: str,
    conversation_id: str,
    start_turn: int,
    end_turn: int,
    created_at: datetime,
    messages: list[dict],
) -> StoredSegment:
    return StoredSegment(
        ref=ref,
        conversation_id=conversation_id,
        primary_tag="topic",
        tags=["topic"],
        summary=f"summary-{ref}",
        summary_tokens=10,
        full_text=f"full-text-{ref}",
        full_tokens=20,
        messages=messages,
        metadata=SegmentMetadata(
            turn_count=max(0, end_turn - start_turn + 1),
            start_turn_number=start_turn,
            end_turn_number=end_turn,
        ),
        created_at=created_at,
        start_timestamp=created_at,
        end_timestamp=created_at + timedelta(minutes=1),
    )


def test_persist_turn_with_embeddings_dual_writes_full_text_and_chunks(tmp_path):
    store = _make_store(tmp_path)
    config = _make_config(tmp_path / "store.db")
    semantic = SemanticSearchManager(
        store,
        config,
        embedding_provider=_StubEmbeddingProvider(),
    )

    persist_turn_with_embeddings(
        store,
        semantic,
        conversation_id="conv-1",
        turn_number=0,
        user_content="hello world",
        assistant_content="hi there",
        primary_tag="greeting-flow",
        tags=["greeting-flow", "intro"],
        session_date="2026-04-11",
        sender="user",
        fact_signals=[FactSignal(subject="hello", verb="started", object="conversation")],
        code_refs=[{"file": "app.py", "line": 10}],
    )

    assert store.get_turn_messages("conv-1", [0])[0][:2] == ("hello world", "hi there")
    archived = store.get_full_text_rows("conv-1", [0])[0]
    assert archived.user_content == "hello world"
    assert archived.assistant_content == "hi there"
    assert archived.primary_tag == "greeting-flow"
    assert archived.tags == ["greeting-flow", "intro"]
    assert archived.session_date == "2026-04-11"
    assert archived.sender == "user"
    assert archived.fact_signals[0].verb == "started"
    assert archived.code_refs == [{"file": "app.py", "line": 10}]

    chunks = store.get_all_full_text_chunk_embeddings(conversation_id="conv-1")
    assert {chunk.side for chunk in chunks} == {"user", "assistant"}

    results = store.search_canonical_full_text("hello", conversation_id="conv-1")
    assert len(results) >= 1
    assert results[0].turn_number == 0
    assert results[0].source_scope == "turn"
    assert results[0].tag == "greeting-flow"
    assert results[0].tags == ["greeting-flow", "intro"]
    assert results[0].session_date == "2026-04-11"

    deleted = store.prune_turn_messages("conv-1", keep_from_turn=1)
    assert deleted == 1
    assert store.get_turn_messages("conv-1", [0]) == {}
    assert store.get_full_text_rows("conv-1", [0])[0].user_content == "hello world"


def test_backfill_full_text_rows_uses_newest_covering_segment_and_skips_malformed(tmp_path):
    store = _make_store(tmp_path)
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    store.save_engine_state(
        EngineStateSnapshot(
            conversation_id="conv-1",
            compacted_through=0,
            turn_count=2,
            turn_tag_entries=[
                TurnTagEntry(
                    turn_number=0,
                    message_hash="hash-0",
                    tags=["search-infra"],
                    primary_tag="search-infra",
                    session_date="2024-07-01",
                ),
                TurnTagEntry(
                    turn_number=1,
                    message_hash="hash-1",
                    tags=["latency-optimization", "milvus"],
                    primary_tag="latency-optimization",
                    session_date="2024-07-02",
                    fact_signals=[FactSignal(subject="latency", verb="improved", object="p95")],
                    code_refs=[{"file": "search.py", "line": 42}],
                ),
            ],
        )
    )

    old_segment = _make_segment(
        ref="seg-old",
        conversation_id="conv-1",
        start_turn=0,
        end_turn=1,
        created_at=base,
        messages=[
            {"role": "user", "content": "turn0 user"},
            {"role": "assistant", "content": "turn0 assistant"},
            {"role": "user", "content": "turn1 user old"},
            {"role": "assistant", "content": "turn1 assistant old"},
        ],
    )
    newer_overlap = _make_segment(
        ref="seg-new",
        conversation_id="conv-1",
        start_turn=1,
        end_turn=1,
        created_at=base + timedelta(days=1),
        messages=[
            {"role": "user", "content": "turn1 user new"},
            {"role": "assistant", "content": "turn1 assistant new"},
        ],
    )
    malformed = _make_segment(
        ref="seg-bad",
        conversation_id="conv-1",
        start_turn=2,
        end_turn=3,
        created_at=base + timedelta(days=2),
        messages=[
            {"role": "user", "content": "only one side present"},
            {"role": "assistant", "content": "one reply only"},
        ],
    )

    store.store_segment(old_segment)
    store.store_segment(newer_overlap)
    store.store_segment(malformed)

    report = backfill_full_text_rows(store, "conv-1")

    rows = store.get_all_full_text_rows("conv-1")
    assert [row.turn_number for row in rows] == [0, 1]
    assert rows[0].user_content == "turn0 user"
    assert rows[1].user_content == "turn1 user new"
    assert rows[0].primary_tag == "search-infra"
    assert rows[1].primary_tag == "latency-optimization"
    assert rows[1].tags == ["latency-optimization", "milvus"]
    assert rows[1].session_date == "2024-07-02"
    assert rows[1].fact_signals[0].verb == "improved"
    assert rows[1].code_refs == [{"file": "search.py", "line": 42}]
    assert "seg-bad" in report["malformed_segments"]
    assert report["turns_written"] == 2


def test_bootstrap_then_rebuild_full_text_chunks(tmp_path):
    store = _make_store(tmp_path)
    config = _make_config(tmp_path / "store.db")
    semantic = SemanticSearchManager(
        store,
        config,
        embedding_provider=_StubEmbeddingProvider(),
    )
    created = datetime(2024, 8, 1, tzinfo=timezone.utc)

    segment = _make_segment(
        ref="seg-1",
        conversation_id="conv-1",
        start_turn=0,
        end_turn=0,
        created_at=created,
        messages=[
            {"role": "user", "content": "vector db question"},
            {"role": "assistant", "content": "vector db answer"},
        ],
    )
    store.store_segment(segment)
    store.store_chunk_embeddings(
        segment.ref,
        [
            ChunkEmbedding(
                segment_ref=segment.ref,
                chunk_index=0,
                text="vector db question answer",
                embedding=[1.0, 2.0],
            )
        ],
    )
    store.save_full_text(
        "conv-1",
        0,
        "vector db question",
        "vector db answer",
    )

    bootstrap = bootstrap_full_text_chunks(store, "conv-1")
    boot_rows = store.get_all_full_text_chunk_embeddings(conversation_id="conv-1")
    assert bootstrap["turns_written"] == 1
    assert len(boot_rows) == 1
    assert boot_rows[0].side == "combined"

    rebuild = rebuild_full_text_chunks(store, semantic, "conv-1")
    rebuilt_rows = store.get_all_full_text_chunk_embeddings(conversation_id="conv-1")
    assert rebuild["turns_embedded"] == 1
    assert {row.side for row in rebuilt_rows} == {"user", "assistant"}
    assert all(isinstance(row, FullTextChunkEmbedding) for row in rebuilt_rows)


def test_resolve_segment_turn_ranges_falls_back_to_turn_count_when_explicit_ranges_missing():
    created = datetime(2024, 9, 1, tzinfo=timezone.utc)
    inferred_a = StoredSegment(
        ref="seg-a",
        conversation_id="conv-1",
        primary_tag="topic",
        tags=["topic"],
        summary="summary-a",
        summary_tokens=10,
        full_text="text-a",
        full_tokens=20,
        messages=[],
        metadata=SegmentMetadata(turn_count=2),
        created_at=created,
        start_timestamp=created,
        end_timestamp=created + timedelta(minutes=1),
    )
    inferred_b = StoredSegment(
        ref="seg-b",
        conversation_id="conv-1",
        primary_tag="topic",
        tags=["topic"],
        summary="summary-b",
        summary_tokens=10,
        full_text="text-b",
        full_tokens=20,
        messages=[],
        metadata=SegmentMetadata(turn_count=1),
        created_at=created + timedelta(minutes=2),
        start_timestamp=created + timedelta(minutes=2),
        end_timestamp=created + timedelta(minutes=3),
    )

    resolved, inferred = resolve_segment_turn_ranges([inferred_b, inferred_a])

    assert resolved == {
        "seg-a": (0, 1),
        "seg-b": (2, 2),
    }
    assert inferred == ["seg-a", "seg-b"]


def test_repair_full_text_session_dates_uses_inline_headers(tmp_path):
    store = _make_store(tmp_path)
    store.save_full_text(
        "conv-1",
        0,
        "[Session from November-01-2024] user turn",
        "assistant turn",
        primary_tag="topic-a",
        tags=["topic-a"],
        session_date="November-05-2024",
    )
    store.save_full_text(
        "conv-1",
        1,
        "plain user turn",
        "plain assistant turn",
        primary_tag="topic-b",
        tags=["topic-b"],
        session_date="November-06-2024",
    )
    store.save_full_text(
        "conv-1",
        2,
        "[Session from December-16-2024] second user turn",
        "assistant turn",
        primary_tag="topic-c",
        tags=["topic-c"],
        session_date="December-16-2024",
    )

    report = repair_full_text_session_dates(store, "conv-1")

    rows = store.get_all_full_text_rows("conv-1")
    assert report["rows_scanned"] == 3
    assert report["rows_repaired"] == 1
    assert report["rows_already_correct"] == 1
    assert report["rows_without_explicit_header"] == 1
    assert report["repairs"] == [
        {
            "turn_number": 0,
            "from": "November-05-2024",
            "to": "November-01-2024",
        }
    ]
    assert rows[0].session_date == "November-01-2024"
    assert rows[1].session_date == "November-06-2024"
    assert rows[2].session_date == "December-16-2024"


def test_build_authoritative_session_dates_from_source_pairs_covers_full_text_and_tail(tmp_path):
    store = _make_store(tmp_path)
    store.save_engine_state(
        EngineStateSnapshot(
            conversation_id="conv-1",
            compacted_through=4,
            turn_count=3,
            turn_tag_entries=[
                TurnTagEntry(turn_number=0, message_hash="h0", session_date="wrong-0"),
                TurnTagEntry(turn_number=1, message_hash="h1", session_date="wrong-1"),
                TurnTagEntry(turn_number=2, message_hash="h2", session_date="wrong-2"),
            ],
        )
    )
    store.save_full_text("conv-1", 0, "u0", "a0", session_date="wrong-0")
    store.save_full_text("conv-1", 1, "u1", "a1", session_date="wrong-1")
    store.save_turn_message("conv-1", 2, "u2", "a2")

    report = build_authoritative_session_dates_from_source_pairs(
        store,
        "conv-1",
        source_turn_pairs=[("u0", "a0"), ("u1", "a1"), ("u2", "a2")],
        source_session_dates=["d0", "d1", "d2"],
    )

    assert report["turn_session_dates"] == {0: "d0", 1: "d1", 2: "d2"}
    assert report["turn_source_indexes"] == {0: 0, 1: 1, 2: 2}
    assert report["unmatched_count"] == 0
    assert report["ambiguous_match_count"] == 0


def test_repair_session_date_lineage_updates_engine_state_full_text_segments_and_facts(tmp_path):
    store = _make_store(tmp_path)
    now = datetime(2024, 7, 1, tzinfo=timezone.utc)
    store.save_engine_state(
        EngineStateSnapshot(
            conversation_id="conv-1",
            compacted_through=4,
            turn_count=2,
            turn_tag_entries=[
                TurnTagEntry(turn_number=0, message_hash="h0", session_date="wrong"),
                TurnTagEntry(turn_number=1, message_hash="h1", session_date="wrong"),
            ],
        )
    )
    store.save_full_text("conv-1", 0, "u0", "a0", session_date="wrong")
    store.save_full_text("conv-1", 1, "u1", "a1", session_date="wrong")
    segment = StoredSegment(
        ref="seg-1",
        conversation_id="conv-1",
        primary_tag="topic",
        tags=["topic"],
        summary="summary",
        summary_tokens=10,
        full_text="full text",
        full_tokens=20,
        messages=[
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        metadata=SegmentMetadata(
            turn_count=2,
            start_turn_number=0,
            end_turn_number=1,
            session_date="wrong",
        ),
        created_at=now,
        start_timestamp=now,
        end_timestamp=now,
    )
    store.store_segment(segment)
    store.store_facts(
        [
            Fact(
                id="fact-1",
                subject="user",
                verb="built",
                object="thing",
                tags=["topic"],
                segment_ref="seg-1",
                conversation_id="conv-1",
                turn_numbers=[0, 1],
                session_date="wrong",
                mentioned_at=now,
            )
        ]
    )

    report = repair_session_date_lineage(
        store,
        "conv-1",
        turn_session_dates={0: "2024-07-01", 1: "2024-07-01"},
        apply=True,
    )

    snapshot = store.load_engine_state("conv-1")
    assert snapshot is not None
    assert [entry.session_date for entry in snapshot.turn_tag_entries] == ["2024-07-01", "2024-07-01"]

    rows = store.get_all_full_text_rows("conv-1")
    assert [row.session_date for row in rows] == ["2024-07-01", "2024-07-01"]

    stored_segment = next(iter(store.get_all_segments(conversation_id="conv-1")))
    assert stored_segment.metadata.session_date == "2024-07-01"

    facts = store.get_facts_by_segment("seg-1")
    assert len(facts) == 1
    assert facts[0].session_date == "2024-07-01"

    assert report["engine_state_repair_count"] == 2
    assert report["full_text_repair_count"] == 2
    assert report["segment_repair_count"] == 1
    assert report["fact_repair_count"] == 1
