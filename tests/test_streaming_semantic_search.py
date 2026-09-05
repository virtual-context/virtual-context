"""Legacy cosine search bounds embeddings, source hydration, and result state."""


import pytest

from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.types import CanonicalTurnRow, ChunkEmbedding, SegmentMetadata, SpeakerRetrievalContext, StoredSegment, VirtualContextConfig


class _Pages:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.hydrations = []

    def _page(self, **kwargs):
        assert kwargs["limit"] == 200 and kwargs["conversation_id"] == "owner"
        self.calls.append(kwargs)
        after = kwargs["after"]
        return self.pages[0] if after is None else next(
            (self.pages[index + 1] for index, page in enumerate(self.pages[:-1])
             if page[-1]["cursor"] == after), [],
        )

    get_segment_chunk_embedding_page = _page
    get_canonical_turn_chunk_embedding_page = _page

    def get_all_chunk_embeddings(self, *args, **kwargs):
        raise AssertionError("archive embedding materialization")

    get_all_canonical_turn_chunk_embeddings = get_all_chunk_embeddings
    get_all_canonical_turns = get_all_chunk_embeddings

    def get_segment(self, ref, **kwargs):
        return None if ref == "missing" else StoredSegment(ref=ref, primary_tag="topic", tags=["topic"])

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context):
        self.hydrations.append((keys, speaker_context))
        return {(row.conversation_id, row.canonical_turn_id): row
                for page in self.pages for item in page if (row := item.get("physical_row")) is not None
                and (row.conversation_id, row.canonical_turn_id) in keys}

    def get_canonical_turn_rows(self, conversation_id, numbers):
        self.hydrations.append((numbers, None))
        return {}


def _row(index, score, *, ref=None, channel="wanted", side="user", present=True):
    ref = ref or f"source-{index}"
    return dict(
        cursor=(index,), segment_ref=ref, conversation_id="owner", canonical_turn_id=ref,
        turn_number=index, side=side, text=f"text {index}", chunk_index=index,
        embedding=[score, (1 - score ** 2) ** 0.5],
        physical_row=CanonicalTurnRow(
            conversation_id="owner", canonical_turn_id=ref, turn_number=index,
            user_content=f"human {index}", sender="Alice", sender_actor_id="actor:a",
            primary_tag="topic", tags=["topic"], origin_channel_id=channel,
        ) if present else None,
    )


def _manager(store):
    manager = SemanticSearchManager(store, VirtualContextConfig(conversation_id="owner"))
    manager._embed_fn = lambda values: [[1.0, 0.0] for _ in values]
    return manager


@pytest.mark.parametrize("speaker", [False, True])
def test_streams_past_rejected_and_duplicate_rows_with_bounded_rank_state(speaker, monkeypatch):
    rows = [_row(i, .6, ref=f"source-{i % 5}") for i in range(420)]
    rows += [_row(420, .99, ref="missing", present=False),
             _row(421, .98, channel="other"), _row(422, .97, ref="winner"),
             _row(423, .96, ref="winner"), _row(424, .95, ref="runner")]
    pages = [rows[i:i + 200] for i in range(0, len(rows), 200)]
    store = _Pages(pages)
    manager = _manager(store)
    counts = []
    keep = manager._keep_best_result
    def observed(top, *args):
        keep(top, *args)
        counts.append(len(top))
    monkeypatch.setattr(manager, "_keep_best_result", observed)
    context = SpeakerRetrievalContext(tenant_id="tenant", owner_conversation_id="owner",
                                      audience_conversation_id="owner") if speaker else None
    results = manager.semantic_canonical_turn_search(
        "q", max_results=2, conversation_id="owner", channel="wanted", speaker_context=context,
    )
    assert [result.similarity for result in results] == [.97, .95]
    assert max(counts) == 2
    assert len(store.calls) == 4
    assert all(len(keys) <= 200 and scope is context for keys, scope in store.hydrations)


def test_segment_stream_selects_late_winners_and_stable_equal_score_order():
    rows = [_row(i, .5) for i in range(210)] + [_row(210, .9), _row(211, .9), _row(212, .99, ref="missing")]
    store = _Pages([rows[:200], rows[200:]])
    results = _manager(store).semantic_search("q", max_results=2, conversation_id="owner")
    assert [result.segment_ref for result in results] == ["source-210", "source-211"]
    assert len(store.calls) == 3


def test_zero_query_skips_enumeration_and_bad_cursor_fails_visible():
    store = _Pages([[_row(0, .5)]])
    manager = _manager(store)
    manager._embed_fn = lambda values: [[0.0, 0.0]]
    assert manager.semantic_search("q", conversation_id="owner") == []
    assert store.calls == []
    manager._embed_fn = lambda values: [[1.0, 0.0]]
    store.get_segment_chunk_embedding_page = lambda **kwargs: store.pages[0]
    with pytest.raises(RuntimeError, match="did not advance"):
        manager.semantic_search("q", conversation_id="owner")


def test_sqlite_page_metadata_avoids_per_candidate_source_reads(tmp_path, monkeypatch):
    from virtual_context.storage.sqlite import SQLiteStore
    store = SQLiteStore(tmp_path / "metadata.db")
    try:
        for ref, owner in [("a", "owner"), ("z", "foreign")]:
            store.store_segment(StoredSegment(ref=ref, conversation_id=owner, primary_tag="topic",
                                              tags=["topic", "alpha"], metadata=SegmentMetadata(session_date="2026-09-05")))
            store.store_chunk_embeddings(ref, [ChunkEmbedding(segment_ref=ref, chunk_index=index, text=f"{ref}-{index}", embedding=[1.0, 0.0]) for index in range(230)])
        page = store.get_segment_chunk_embedding_page(conversation_id="owner", limit=3)
        assert len(page) == 3
        assert all(row["primary_tag"] == "topic" and row["tags"] == ["alpha", "topic"]
                   and row["session_date"] == "2026-09-05" and row["conversation_id"] == "owner" for row in page)
        monkeypatch.setattr(store, "get_segment", lambda *args, **kwargs: pytest.fail("per-candidate source read"))
        result, = _manager(store).semantic_search("q", conversation_id="owner")
        assert result.segment_ref == "a" and result.text == "a-0" and result.tags == ["alpha", "topic"]
        assert result.session_date == "2026-09-05"
        # Joining the live source prevents an orphan embedding from entering
        # the metadata fast path even when foreign-key enforcement is absent.
        store.delete_segment("a")
        assert store.get_segment_chunk_embedding_page(conversation_id="owner") == []
    finally:
        store.close()
