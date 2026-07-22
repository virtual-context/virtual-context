"""Physical, role-local semantic retrieval over canonical turn chunks.

The semantic manager has two branches. ``speaker_context=None`` is the
shipped legacy branch: legacy chunk enumeration, logical hydration on the
unscoped path, and no ``subject``-side consumption. A non-None context is
the speaker-aware branch: candidate enumeration and ONE batched physical
hydration by ``(conversation_id, canonical_turn_id)`` both receive the same
immutable context, on the scoped and unscoped paths alike. A chunk whose
physical row is missing proves nothing — it is skipped and reported, and
the admin reindex owns the repair.
"""

from __future__ import annotations

import logging

from virtual_context.config import VirtualContextConfig
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.types import (
    CanonicalTurnChunkEmbedding,
    CanonicalTurnRow,
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
)

_NOT_PASSED = object()


def _vec(text: str) -> list[float]:
    """Similarity is 1.0 when the marker matches, 0.8 otherwise."""
    return [1.0, 0.0] if "MATCH" in text else [0.8, 0.6]


def _context(**kw) -> SpeakerRetrievalContext:
    base = dict(
        tenant_id="t1",
        owner_conversation_id="c",
        audience_conversation_id="c",
        audience_channel_id="",
        requester_actor_id="actor:discord:9",
    )
    base.update(kw)
    return SpeakerRetrievalContext(**base)


def _physical(ct_id, turn_number, *, conversation_id="c", **kw):
    base = dict(
        conversation_id=conversation_id,
        canonical_turn_id=ct_id,
        turn_number=turn_number,
        turn_group_number=turn_number // 2,
        primary_tag="chat",
        tags=["chat"],
        audience_conversation_id="c",
        audience_attribution_version=1,
    )
    base.update(kw)
    return CanonicalTurnRow(**base)


def _chunk(ct_id, turn_number, side, text, *, conversation_id="c"):
    return CanonicalTurnChunkEmbedding(
        conversation_id=conversation_id,
        side=side,
        chunk_index=0,
        text=text,
        embedding=_vec(text),
        canonical_turn_id=ct_id,
        turn_number=turn_number,
    )


class _SpeakerStore:
    """Strict double: records the exact context each store seam receives."""

    def __init__(self, rows, chunks):
        self._rows = {
            (row.conversation_id, row.canonical_turn_id): row for row in rows
        }
        self._chunks = list(chunks)
        self.logical_calls = 0
        self.enumeration_contexts: list[object] = []
        self.hydration_contexts: list[object] = []
        self.hydration_batches: list[list[tuple[str, str]]] = []

    def get_all_canonical_turn_chunk_embeddings(
        self, conversation_id=None, *, speaker_context=_NOT_PASSED,
    ):
        self.enumeration_contexts.append(speaker_context)
        return list(self._chunks)

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context):
        self.hydration_contexts.append(speaker_context)
        self.hydration_batches.append(list(keys))
        return {key: self._rows[key] for key in keys if key in self._rows}

    def get_canonical_turn_rows(self, conversation_id, turn_numbers):
        # The LOGICAL seam. The speaker-aware branch must never touch it.
        self.logical_calls += 1
        return {}

    def get_all_canonical_turns(self, conversation_id):
        return [
            row for row in self._rows.values()
            if row.conversation_id == conversation_id
        ]


def _semantic(store) -> SemanticSearchManager:
    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    manager = SemanticSearchManager(store=store, config=config)
    manager._embed_fn = lambda texts: [_vec(t) for t in texts]
    return manager


class TestSpeakerBranchPhysicalHydration:
    def test_unscoped_path_hydrates_physical_rows_batched(self):
        rows = [
            _physical("ct-1", 0, user_content="peptide MATCH",
                      sender_actor_id="actor:discord:1"),
            _physical("ct-2", 2, assistant_content="reply MATCH"),
        ]
        chunks = [
            _chunk("ct-1", 0, "user", "peptide MATCH"),
            _chunk("ct-2", 2, "assistant", "reply MATCH"),
        ]
        store = _SpeakerStore(rows, chunks)
        ctx = _context()
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", speaker_context=ctx,
        )
        assert {r.text for r in results} == {
            "User: peptide MATCH", "Assistant: reply MATCH",
        }
        # The logical seam is never used; hydration is one batched call.
        assert store.logical_calls == 0
        assert len(store.hydration_batches) == 1
        assert set(store.hydration_batches[0]) == {("c", "ct-1"), ("c", "ct-2")}
        # Both store seams received the exact same immutable context.
        assert store.enumeration_contexts == [ctx]
        assert store.enumeration_contexts[0] is ctx
        assert store.hydration_contexts == [ctx]
        assert store.hydration_contexts[0] is ctx

    def test_scoped_path_hydrates_physical_rows_batched(self):
        rows = [
            _physical("ct-1", 0, user_content="in channel MATCH",
                      origin_channel_label="#vasttest",
                      sender_actor_id="actor:discord:1"),
            _physical("ct-2", 1, user_content="out of channel MATCH",
                      origin_channel_label="#other"),
        ]
        chunks = [
            _chunk("ct-1", 0, "user", "in channel MATCH"),
            _chunk("ct-2", 1, "user", "out of channel MATCH"),
        ]
        store = _SpeakerStore(rows, chunks)
        ctx = _context()
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
            channel="#vasttest", speaker_context=ctx,
        )
        assert [r.text for r in results] == ["[#vasttest] User: in channel MATCH"]
        assert store.logical_calls == 0
        assert len(store.hydration_batches) == 1
        assert store.hydration_contexts[0] is ctx
        assert store.enumeration_contexts[0] is ctx

    def test_missing_physical_row_is_skipped_and_reported(self, caplog):
        rows = [_physical("ct-live", 1, user_content="weaker text")]
        chunks = [
            _chunk("ct-gone", 0, "user", "orphan MATCH"),
            _chunk("ct-live", 1, "user", "weaker text"),
        ]
        store = _SpeakerStore(rows, chunks)
        with caplog.at_level(logging.WARNING, "virtual_context.core.semantic_search"):
            results = _semantic(store).semantic_canonical_turn_search(
                "MATCH", max_results=1, conversation_id="c",
                speaker_context=_context(),
            )
        # The orphan chunk outranks the live one but proves nothing; the
        # scan continues to the hydratable candidate instead of guessing.
        assert [r.text for r in results] == ["User: weaker text"]
        assert any(
            "SEMANTIC_CHUNK_NO_PHYSICAL_ROW" in record.message
            for record in caplog.records
        )

    def test_chunks_dedupe_by_physical_identity_and_side(self):
        rows = [_physical("ct-1", 0, user_content="alpha MATCH",
                          assistant_content="beta MATCH")]
        chunks = [
            _chunk("ct-1", 0, "user", "alpha MATCH"),
            CanonicalTurnChunkEmbedding(
                conversation_id="c", side="user", chunk_index=1,
                text="alpha MATCH again", embedding=_vec("alpha MATCH again"),
                canonical_turn_id="ct-1", turn_number=0,
            ),
            _chunk("ct-1", 0, "assistant", "beta MATCH"),
        ]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
            speaker_context=_context(),
        )
        # Two sides survive; the duplicate user chunk collapses.
        assert sorted(r.matched_side for r in results) == ["assistant", "user"]

    def test_subject_lane_surfaces_with_role_local_provenance(self):
        rows = [
            _physical(
                "ct-s", 0,
                user_content="what do you think",
                sender_actor_id="actor:discord:1",
                reply_subject_actor_id="actor:telegram:42",
                reply_subject_label="Sania",
                reply_target_body="peptide MATCH data",
            ),
        ]
        chunks = [_chunk("ct-s", 0, "subject", "peptide MATCH data")]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
            speaker_context=_context(),
        )
        assert len(results) == 1
        result = results[0]
        # The excerpt is the copied reply text alone: no ``User:`` label
        # that would misassign the quote to the requester.
        assert result.text == "peptide MATCH data"
        assert result.matched_side == ""
        prov = result.provenance
        assert prov.source_role == "subject"
        assert prov.canonical_turn_id == "ct-s"
        # Role-local: the subject lane carries ONLY the reply subject, never
        # the containing requester row's actor. The raw stored label rides
        # along strictly as an unverified claim.
        assert prov.actor_id == "actor:telegram:42"
        assert prov.claimed_subject_label == "Sania"

    def test_requester_lane_carries_only_sender_actor(self):
        rows = [
            _physical(
                "ct-1", 0, user_content="peptide MATCH",
                sender_actor_id="actor:discord:1",
                reply_subject_actor_id="actor:telegram:42",
            ),
        ]
        chunks = [_chunk("ct-1", 0, "user", "peptide MATCH")]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
            speaker_context=_context(),
        )
        prov = results[0].provenance
        assert prov.source_role == "requester"
        assert prov.actor_id == "actor:discord:1"

    def test_actor_ids_never_appear_in_repr(self):
        rows = [
            _physical("ct-1", 0, user_content="peptide MATCH",
                      sender_actor_id="actor:discord:1"),
        ]
        chunks = [_chunk("ct-1", 0, "user", "peptide MATCH")]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
            speaker_context=_context(),
        )
        assert "actor:discord:1" not in repr(results[0])
        assert "actor:discord:1" not in repr(results[0].provenance)


class TestNoneContextLegacyBranch:
    def test_none_context_selects_only_legacy_semantic_branch(self):
        rows = [_physical("ct-1", 0, user_content="toes MATCH")]
        chunks = [_chunk("ct-1", 0, "user", "toes MATCH")]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
        )
        # Legacy unscoped hydration is the logical seam, exactly as shipped.
        assert store.logical_calls == 1
        assert results[0].text == "User: toes MATCH"
        # The physical batch lookup is never called, and the legacy
        # enumeration call carries no speaker kwarg at all.
        assert store.hydration_contexts == []
        assert store.enumeration_contexts == [_NOT_PASSED]
        # Legacy results carry no provenance.
        assert results[0].provenance is None

    def test_legacy_search_ignores_subject_chunks(self):
        row = _physical(
            "ct-s", 0, user_content="unrelated",
            reply_target_body="peptide MATCH data",
        )
        chunks = [_chunk("ct-s", 0, "subject", "peptide MATCH data")]
        store = _SpeakerStore([row], chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
        )
        assert results == []
        # It never even reached hydration: the subject side is filtered out
        # of the candidate set, not post-filtered from results.
        assert store.logical_calls == 0

    def test_legacy_results_are_identical_with_and_without_subject_chunks(self):
        rows = [_physical("ct-1", 0, user_content="toes MATCH")]
        user_chunk = _chunk("ct-1", 0, "user", "toes MATCH")
        subject_chunk = _chunk("ct-1", 0, "subject", "shadow MATCH body")
        with_subject = _semantic(
            _SpeakerStore(rows, [user_chunk, subject_chunk]),
        ).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
        )
        without_subject = _semantic(
            _SpeakerStore(rows, [user_chunk]),
        ).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c",
        )
        assert with_subject == without_subject

    def test_legacy_scoped_path_also_ignores_subject_chunks(self):
        rows = [
            _physical("ct-s", 0, user_content="unrelated",
                      origin_channel_label="#vasttest",
                      reply_target_body="peptide MATCH data"),
        ]
        chunks = [_chunk("ct-s", 0, "subject", "peptide MATCH data")]
        store = _SpeakerStore(rows, chunks)
        results = _semantic(store).semantic_canonical_turn_search(
            "MATCH", max_results=5, conversation_id="c", channel="#vasttest",
        )
        assert results == []


class _WriteStore:
    def __init__(self):
        self.deleted = []
        self.writes = []

    def delete_canonical_turn_chunk_embeddings(
        self, conversation_id, turn_number=None, canonical_turn_id=None,
    ):
        self.deleted.append((conversation_id, turn_number, canonical_turn_id))
        return 0

    def store_canonical_turn_chunk_embeddings(
        self, conversation_id, turn_number, side, chunks, canonical_turn_id=None,
    ):
        self.writes.append(
            (side, canonical_turn_id, [c.side for c in chunks],
             [c.text for c in chunks]),
        )


class TestSubjectChunkIngest:
    def test_user_embedding_prefers_admitted_text_over_polluted_raw_lane(self):
        store = _WriteStore()
        manager = _semantic(store)
        manager.embed_and_store_turn(
            "c", 0,
            canonical_turn_id="ct-1",
            user_text="the admitted request",
            user_raw_content=(
                "Conversation info (untrusted metadata):\n"
                "```json\n{\"message_id\":\"m1\"}\n```\n\n"
                "the admitted request"
            ),
        )
        assert store.writes[0][3] == ["the admitted request"]

    def test_user_embedding_rejects_scaffold_only_raw_fallback(self):
        store = _WriteStore()
        manager = _semantic(store)
        manager.embed_and_store_turn(
            "c", 0,
            canonical_turn_id="ct-1",
            user_text="",
            user_raw_content=(
                "OpenClaw assembled context for this turn:\n"
                "<conversation_context>quoted history</conversation_context>"
            ),
        )
        assert store.writes == []

    def test_user_embedding_keeps_non_scaffold_raw_fallback(self):
        store = _WriteStore()
        manager = _semantic(store)
        manager.embed_and_store_turn(
            "c", 0,
            canonical_turn_id="ct-1",
            user_text="",
            user_raw_content='[{"type":"image","source":"attachment"}]',
        )
        assert store.writes[0][3] == [
            '[{"type":"image","source":"attachment"}]',
        ]

    def test_reply_target_body_is_indexed_as_a_subject_side(self):
        store = _WriteStore()
        manager = _semantic(store)
        manager.embed_and_store_turn(
            "c", 0,
            canonical_turn_id="ct-1",
            user_text="hello there",
            assistant_text="hi",
            reply_target_body="the quoted claim",
        )
        by_side = {write[0]: write for write in store.writes}
        assert set(by_side) == {"user", "assistant", "subject"}
        side, ct_id, chunk_sides, texts = by_side["subject"]
        # Keyed to the SAME physical row as the other lanes.
        assert ct_id == "ct-1"
        assert chunk_sides == ["subject"]
        assert texts == ["the quoted claim"]
        # Never concatenated into the requester lane.
        assert by_side["user"][3] == ["hello there"]

    def test_empty_reply_target_body_writes_no_subject_side(self):
        store = _WriteStore()
        manager = _semantic(store)
        manager.embed_and_store_turn(
            "c", 0, canonical_turn_id="ct-1", user_text="hello there",
        )
        assert [write[0] for write in store.writes] == ["user"]
