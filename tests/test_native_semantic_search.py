"""Native ranking never materializes stored vectors and continues past rejected pages."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    CanonicalTurnRow, ChunkEmbedding,
    SpeakerRetrievalContext, StoredSegment, VirtualContextConfig,
)


def _physical(identifier, **kwargs):
    fields = dict(
        conversation_id="owner", canonical_turn_id=identifier, turn_number=1,
        user_content=f"exact user {identifier}", assistant_content="exact assistant",
        reply_target_body="exact quoted subject", sender_actor_id="actor:discord:1",
        reply_subject_actor_id="actor:discord:2", reply_subject_label="Bob",
        audience_conversation_id="audience", audience_attribution_version=1,
        origin_channel_id="channel", origin_channel_label="#channel",
    )
    fields.update(kwargs)
    return CanonicalTurnRow(**fields)


def _candidate(identifier, index, *, side="user", turn_number=1, **kwargs):
    return dict(
        conversation_id="owner", canonical_turn_id=identifier,
        turn_number=turn_number, side=side, chunk_index=index,
        text=f"retrieval index {identifier}", similarity=0.9,
        cursor=(index,), **kwargs,
    )


class _NativeStore:
    def __init__(self, pages, *, physical=(), logical=None, segments=(), context=None):
        self.pages = pages
        self.physical = {(row.conversation_id, row.canonical_turn_id): row for row in physical}
        self.logical = logical or {}
        self.segments = {segment.ref: segment for segment in segments}
        self.context = context
        self.calls = []
        self.hydration = []
        self.logical_batches = []
        self.ready = True

    def vector_search_ready(self, model):
        assert model == "all-MiniLM-L6-v2"
        return self.ready

    def _page(self, method, vector, *, conversation_id, limit, after, min_similarity,
              speaker_context=None):
        assert vector == [1.0] + [0.0] * 383
        assert conversation_id == "owner"
        assert 0 < limit <= 200
        assert min_similarity == 0.25
        if method == "speaker":
            assert speaker_context is self.context
        self.calls.append((method, after))
        if after is None:
            return self.pages[0] if self.pages else []
        for index, page in enumerate(self.pages):
            if page[-1]["cursor"] == after:
                return self.pages[index + 1] if index + 1 < len(self.pages) else []
        raise AssertionError(f"Unexpected cursor {after}")

    def search_speaker_turn_chunks_by_embedding(self, vector, **kwargs):
        return self._page("speaker", vector, **kwargs)

    def search_canonical_turn_chunks_by_embedding(self, vector, **kwargs):
        return self._page("canonical", vector, **kwargs)

    def search_segment_chunks_by_embedding(self, vector, **kwargs):
        return self._page("segment", vector, **kwargs)

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context):
        assert speaker_context is self.context and speaker_context is not None
        self.hydration.append(list(keys))
        return {key: self.physical[key] for key in keys if key in self.physical}

    def get_canonical_turn_rows(self, conversation_id, turn_numbers):
        assert conversation_id == "owner"
        self.logical_batches.append(list(turn_numbers))
        return {number: self.logical[number] for number in turn_numbers if number in self.logical}

    def get_segment(self, ref, *, conversation_id):
        assert conversation_id == "owner"
        return self.segments.get(ref)

    def get_all_chunk_embeddings(self, *args, **kwargs):
        raise AssertionError("native search materialized segment vectors")

    def get_all_canonical_turn_chunk_embeddings(self, *args, **kwargs):
        raise AssertionError("native search materialized canonical vectors")

    def get_all_canonical_turns(self, *args, **kwargs):
        raise AssertionError("native search hydrated the entire canonical archive")


def _manager(store):
    config = VirtualContextConfig(conversation_id="owner")
    config.retriever.vector_search_enabled = True
    manager = SemanticSearchManager(store, config)
    manager._embed_fn = lambda texts: [[1.0] + [0.0] * 383 for text in texts]
    return manager


def _context():
    return SpeakerRetrievalContext(
        tenant_id="tenant", owner_conversation_id="owner",
        audience_conversation_id="audience", audience_channel_id="channel",
        requester_actor_id="actor:discord:1",
    )


def test_speaker_continuation_passes_missing_channel_and_duplicate_chunks(caplog):
    context = _context()
    rows = [_physical("outside", origin_channel_id="other", origin_channel_label="#other"),
            _physical("one"), _physical("two", turn_number=2)]
    pages = [
        [_candidate("missing", 1), _candidate("outside", 2)],
        [_candidate("one", 3), _candidate("one", 4)],
        [_candidate("one", 5), _candidate("two", 6, side="subject", turn_number=2)],
    ]
    store = _NativeStore(pages, physical=rows, context=context)
    results = _manager(store).semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=2,
        channel="channel", speaker_context=context,
    )
    assert [result.provenance.source_role for result in results] == ["requester", "subject"]
    assert [result.provenance.actor_id for result in results] == ["actor:discord:1", "actor:discord:2"]
    assert "exact user one" in results[0].text
    assert "exact quoted subject" in results[1].text
    assert all("retrieval index" not in result.text for result in results)
    assert store.calls == [("speaker", None), ("speaker", (2,)), ("speaker", (4,))]
    assert [len(batch) for batch in store.hydration] == [2, 1, 1]
    assert "SEMANTIC_CHUNK_NO_PHYSICAL_ROW" in caplog.text


def test_speaker_assistant_never_inherits_physical_human_actor():
    context = _context()
    store = _NativeStore(
        [[_candidate("one", 1, side="assistant")]],
        physical=[_physical("one")], context=context,
    )
    result = _manager(store).semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=1, speaker_context=context,
    )[0]
    assert result.text == "Assistant: exact assistant"
    assert result.provenance.source_role == "assistant"
    assert result.provenance.actor_id == ""


def test_legacy_channel_uses_native_physical_projection_and_continues():
    pages = [
        [_candidate("subject", 1, side="subject", physical_row=_physical("subject")),
         _candidate("missing", 2, physical_row=None)],
        [_candidate("outside", 3, physical_row=_physical("outside", origin_channel_id="other",
                                                        origin_channel_label="#other"))],
        [_candidate("one", 4, physical_row=_physical("one"))],
    ]
    store = _NativeStore(pages)
    result = _manager(store).semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=1, channel="channel",
    )[0]
    assert "exact user one" in result.text
    assert store.hydration == store.logical_batches == []
    assert len(store.calls) == 3


def test_legacy_projection_uses_physical_identity_and_rejects_missing_sources():
    store = _NativeStore([
        [_candidate("one", 1, physical_row=_physical("one")),
         _candidate("one", 2, physical_row=_physical("one"))],
        [_candidate("missing", 3, turn_number=2, physical_row=None)],
        [_candidate("two", 4, turn_number=1, side="assistant", physical_row=_physical("two"))],
    ], logical={1: _physical("unrelated-logical-group")})
    results = _manager(store).semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=2,
    )
    assert [result.text for result in results] == [
        "User: exact user one", "Assistant: exact assistant",
    ]
    assert store.logical_batches == []
    assert store.calls == [("canonical", None), ("canonical", (2,)), ("canonical", (3,))]


def test_segment_continuation_does_not_stop_on_empty_accepted_page():
    segments = [StoredSegment(ref=ref, conversation_id="owner", primary_tag="topic")
                for ref in ("one", "two")]
    pages = [
        [dict(segment_ref="missing", text="missing", similarity=0.9, cursor=(1,))],
        [dict(segment_ref="one", text="best one", similarity=0.8, cursor=(2,)),
         dict(segment_ref="one", text="duplicate one", similarity=0.7, cursor=(3,))],
        [dict(segment_ref="two", text="best two", similarity=0.6, cursor=(4,))],
    ]
    store = _NativeStore(pages, segments=segments)
    results = _manager(store).semantic_search("question", max_results=2, conversation_id="owner")
    assert [result.text for result in results] == ["best one", "best two"]
    assert store.calls == [("segment", None), ("segment", (1,)), ("segment", (3,))]


@pytest.mark.parametrize("method", ["semantic_search", "semantic_canonical_turn_search"])
def test_native_unavailable_and_query_errors_never_fall_back(method, monkeypatch):
    store = _NativeStore([])
    manager = _manager(store)
    store.ready = False
    with pytest.raises(RuntimeError, match="migrate-semantic-vectors"):
        getattr(manager, method)("question", conversation_id="owner")
    store.ready = True
    def fail(*args, **kwargs):
        raise OSError("database unavailable")
    monkeypatch.setattr(store, "_page", fail)
    with pytest.raises(RuntimeError, match="migrate-semantic-vectors"):
        getattr(manager, method)("question", conversation_id="owner")


def test_zero_vector_returns_no_matches_without_native_query():
    store = _NativeStore([])
    manager = _manager(store)
    manager._embed_fn = lambda texts: [[0.0] * 384 for text in texts]
    assert manager.semantic_search("zero", conversation_id="owner") == []
    assert manager.semantic_canonical_turn_search("zero", conversation_id="owner") == []
    assert manager.semantic_canonical_turn_search(
        "zero", conversation_id="owner", speaker_context=_context(),
    ) == []
    assert store.calls == []


def test_native_rejects_nonadvancing_cursor_instead_of_looping(monkeypatch):
    store = _NativeStore([])
    page = [dict(segment_ref="missing", text="missing", similarity=0.9, cursor=(1,))]
    monkeypatch.setattr(store, "_page", lambda *args, **kwargs: page)
    with pytest.raises(RuntimeError, match="non-advancing"):
        _manager(store).semantic_search("question", conversation_id="owner")


def test_embedding_writes_carry_each_managers_explicit_model():
    models = []
    store = SimpleNamespace(
        store_chunk_embeddings=lambda *args, **kwargs: models.append(kwargs["embedding_model"]),
        delete_canonical_turn_chunk_embeddings=lambda *args, **kwargs: None,
        store_canonical_turn_chunk_embeddings=lambda *args, **kwargs: models.append(kwargs["embedding_model"]),
    )
    first = _manager(store)
    second = _manager(store)
    second._config.retriever.embedding_model = "another-model"
    segment = StoredSegment(ref="segment", full_text="word " * 30)
    first.embed_and_store_chunks(segment)
    second.embed_and_store_chunks(segment)
    first.embed_and_store_turn("owner", 0, canonical_turn_id="turn", user_text="Exact user text")
    assert models == ["all-MiniLM-L6-v2", "another-model", "all-MiniLM-L6-v2"]


def test_scoped_legacy_segment_loader_does_not_return_other_conversation(tmp_path):
    store = SQLiteStore(tmp_path / "scoped.db")
    for owner in ("owner", "other"):
        store.store_segment(StoredSegment(ref=owner, conversation_id=owner, primary_tag="topic"))
        store.store_chunk_embeddings(owner, [ChunkEmbedding(
            segment_ref=owner, chunk_index=0, text=owner, embedding=[1.0, 0.0],
        )])
    assert [chunk.segment_ref for chunk in store.get_all_chunk_embeddings(
        conversation_id="owner",
    )] == ["owner"]
    manager = _manager(store)
    manager._config.retriever.vector_search_enabled = False
    manager._embed_fn = lambda texts: [[1.0, 0.0] for text in texts]
    assert [result.segment_ref for result in manager.semantic_search(
        "question", conversation_id="owner",
    )] == ["owner"]


def test_native_and_legacy_speaker_output_parity():
    context = _context()
    physical = [_physical("one"), _physical("two", turn_number=2)]
    page = [_candidate("one", 1), _candidate("two", 2, side="subject", turn_number=2)]
    for candidate in page:
        candidate["similarity"] = 1.0
    store = _NativeStore([page], physical=physical, context=context)
    native = _manager(store).semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=2, speaker_context=context,
    )
    store.get_canonical_turn_chunk_embedding_page = lambda **kwargs: [
        dict(row, embedding=[1.0] + [0.0] * 383) for row in page
    ] if kwargs["after"] is None else []
    legacy_manager = _manager(store)
    legacy_manager._config.retriever.vector_search_enabled = False
    legacy = legacy_manager.semantic_canonical_turn_search(
        "question", conversation_id="owner", max_results=2, speaker_context=context,
    )
    assert [asdict(result) for result in native] == [asdict(result) for result in legacy]
