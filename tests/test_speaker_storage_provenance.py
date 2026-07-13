"""Physical role-local provenance projection in canonical-turn text search.

``search_canonical_turn_text`` gains an opt-in ``speaker_context`` branch that
returns physical role-local candidates carrying ``SourceProvenance``:

* the turn lane projects ``sender_actor_id`` onto requester candidates and
  splits a both-side match into requester and assistant halves before any
  dedupe or ranking;
* ``reply_target_body`` is a distinct lexical lane whose candidates carry
  ONLY ``reply_subject_actor_id``, never the containing requester's actor or
  sender label, and whose text is excerpted from the reply body alone; and
* every candidate carries the exact physical row's conversation, canonical
  turn, audience, attribution version, and channel provenance.

The legacy path — every call without ``speaker_context`` — must stay
byte-identical: no provenance, no subject lane, no split, same excerpts.
Actor ids are internal provenance and must never surface through ``repr``.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SourceProvenance, SpeakerRetrievalContext


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
    sender: str = "",
    sender_actor_id: str = "",
    reply_subject_actor_id: str = "",
    reply_subject_label: str = "",
    reply_target_body: str = "",
    audience_conversation_id: str = "",
    audience_attribution_version: int = 0,
    origin_channel_id: str = "",
    origin_channel_label: str = "",
    conv: str = "c",
) -> None:
    store.save_canonical_turn(
        conv, -1, user_content, assistant_content,
        canonical_turn_id=ct_id,
        sort_key=sort_key,
        turn_hash=f"h-{ct_id}",
        sender=sender,
        primary_tag="chat",
        tags=["chat"],
        sender_actor_id=sender_actor_id,
        reply_subject_actor_id=reply_subject_actor_id,
        reply_subject_label=reply_subject_label,
        reply_target_body=reply_target_body,
        audience_conversation_id=audience_conversation_id,
        audience_attribution_version=audience_attribution_version,
        origin_channel_id=origin_channel_id,
        origin_channel_label=origin_channel_label,
    )


def _ctx() -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="t",
        owner_conversation_id="c",
        audience_conversation_id="c",
    )


class TestLegacyPathByteIdentical:
    def test_legacy_results_carry_no_provenance(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex",
             sender_actor_id="actor:tg:111")
        results = store.search_canonical_turn_text("tingling", conversation_id="c")
        assert len(results) == 1
        assert results[0].provenance is None

    def test_default_and_explicit_none_select_the_same_branch(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="tingling toes", assistant_content="tingling is odd",
             sender="BigTex", sender_actor_id="actor:tg:111",
             reply_target_body="tingling report")
        default = store.search_canonical_turn_text("tingling", conversation_id="c")
        explicit = store.search_canonical_turn_text(
            "tingling", conversation_id="c", speaker_context=None,
        )
        assert default == explicit

    def test_legacy_ignores_the_reply_body_lane(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="what do you think about this?",
             reply_target_body="the trip to boston was amazing",
             reply_subject_actor_id="actor:tg:222")
        assert store.search_canonical_turn_text("boston", conversation_id="c") == []

    def test_legacy_both_row_stays_one_combined_result(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="tingling toes", assistant_content="tingling is odd",
             sender="BigTex")
        results = store.search_canonical_turn_text("tingling", conversation_id="c")
        assert len(results) == 1
        qr = results[0]
        assert qr.matched_side == "both"
        assert qr.text == "BigTex: tingling toes\n\nAssistant: tingling is odd"
        assert qr.provenance is None

    def test_legacy_exact_render_snapshot(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex",
             sender_actor_id="actor:tg:111")
        qr = store.search_canonical_turn_text("tingling", conversation_id="c")[0]
        assert qr.text == "BigTex: my toes are tingling"
        assert qr.matched_side == "user"
        assert qr.match_type == "full_text_search"
        assert qr.source_scope == "turn"
        assert qr.segment_ref == "canonical_turn_ct-1"
        assert qr.tag == "chat"
        assert qr.tags == ["chat"]
        assert qr.turn_number == 0


class TestRequesterLaneProjection:
    def test_user_match_projects_the_physical_row(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex",
             sender_actor_id="actor:tg:111",
             audience_conversation_id="aud-1",
             audience_attribution_version=2,
             origin_channel_id="chan-9")
        results = store.search_canonical_turn_text(
            "tingling", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        qr = results[0]
        prov = qr.provenance
        assert isinstance(prov, SourceProvenance)
        assert prov.conversation_id == "c"
        assert prov.canonical_turn_id == "ct-1"
        assert prov.source_role == "requester"
        assert prov.actor_id == "actor:tg:111"
        assert prov.audience_conversation_id == "aud-1"
        assert prov.audience_attribution_version == 2
        assert prov.origin_channel_id == "chan-9"
        assert prov.claimed_subject_label == ""
        assert qr.matched_side == "user"

    def test_sender_only_match_is_a_requester_candidate(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="hello there", sender="BigTex",
             sender_actor_id="actor:tg:111")
        results = store.search_canonical_turn_text(
            "bigtex", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        assert results[0].provenance.source_role == "requester"
        assert results[0].provenance.actor_id == "actor:tg:111"

    def test_single_side_excerpts_match_the_legacy_bytes(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex")
        _row(store, ct_id="ct-2", sort_key=2000.0,
             assistant_content="that sounds neurological")
        for query in ("tingling", "neurological"):
            legacy = store.search_canonical_turn_text(query, conversation_id="c")
            speaker = store.search_canonical_turn_text(
                query, conversation_id="c", speaker_context=_ctx(),
            )
            assert [qr.text for qr in legacy] == [qr.text for qr in speaker]
            assert [qr.matched_side for qr in legacy] == [
                qr.matched_side for qr in speaker
            ]


class TestAssistantLaneProjection:
    def test_assistant_text_never_carries_a_human_actor(self, tmp_path: Path):
        # Legacy rows carry the logical-turn sender and actor on both halves.
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             assistant_content="that sounds neurological", sender="BigTex",
             sender_actor_id="actor:tg:111")
        results = store.search_canonical_turn_text(
            "neurological", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        qr = results[0]
        assert qr.text.startswith("Assistant: ")
        assert qr.provenance.source_role == "assistant"
        assert qr.provenance.actor_id == ""


class TestBothSideSplit:
    def test_both_row_splits_into_role_local_halves(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="tingling toes", assistant_content="tingling is odd",
             sender="BigTex", sender_actor_id="actor:tg:111")
        results = store.search_canonical_turn_text(
            "tingling", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 2
        by_role = {qr.provenance.source_role: qr for qr in results}
        assert set(by_role) == {"requester", "assistant"}

        requester = by_role["requester"]
        assert requester.text == "BigTex: tingling toes"
        assert "\n\nAssistant:" not in requester.text
        assert requester.matched_side == "user"
        assert requester.provenance.actor_id == "actor:tg:111"

        assistant = by_role["assistant"]
        assert assistant.text == "Assistant: tingling is odd"
        assert "BigTex" not in assistant.text
        assert assistant.matched_side == "assistant"
        assert assistant.provenance.actor_id == ""

        # The halves share the physical row identity for later display
        # dedupe, but the store itself never collapses them.
        assert requester.segment_ref == assistant.segment_ref
        assert (
            requester.provenance.canonical_turn_id
            == assistant.provenance.canonical_turn_id
            == "ct-1"
        )


class TestSubjectLane:
    def test_reply_body_is_its_own_lane_with_subject_actor_only(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="what do you think about this?",
             sender="Requester", sender_actor_id="actor:req",
             reply_subject_actor_id="actor:subj",
             reply_subject_label="Sania",
             reply_target_body="the trip to boston was amazing",
             audience_conversation_id="aud-1",
             audience_attribution_version=2)
        results = store.search_canonical_turn_text(
            "boston", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        qr = results[0]
        prov = qr.provenance
        assert prov.source_role == "subject"
        assert prov.actor_id == "actor:subj"
        assert prov.claimed_subject_label == "Sania"
        assert prov.canonical_turn_id == "ct-1"
        assert prov.audience_conversation_id == "aud-1"
        assert prov.audience_attribution_version == 2
        # Lane-local excerpt: reply text only — no requester text, no
        # sender label, and no asserted claim label in the model haystack.
        assert "trip to boston" in qr.text
        assert "what do you think" not in qr.text
        assert "Requester" not in qr.text
        assert "Sania" not in qr.text
        assert qr.matched_side == ""

    def test_reply_text_is_never_concatenated_into_requester_text(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="planning my own trip now",
             sender="Requester", sender_actor_id="actor:req",
             reply_subject_actor_id="actor:subj",
             reply_target_body="the trip to boston was amazing")
        results = store.search_canonical_turn_text(
            "trip", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 2
        by_role = {qr.provenance.source_role: qr for qr in results}
        assert set(by_role) == {"requester", "subject"}
        assert "planning my own trip" in by_role["requester"].text
        assert "boston" not in by_role["requester"].text
        assert "boston" in by_role["subject"].text
        assert "planning my own trip" not in by_role["subject"].text
        assert by_role["requester"].provenance.actor_id == "actor:req"
        assert by_role["subject"].provenance.actor_id == "actor:subj"

    def test_unresolved_subject_keeps_empty_actor_and_carries_the_claim(
        self, tmp_path: Path,
    ):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="look at this",
             sender_actor_id="actor:req",
             reply_subject_actor_id="",
             reply_subject_label="Spoofed Name",
             reply_target_body="the dosage was fifty milligrams")
        qr = store.search_canonical_turn_text(
            "dosage", conversation_id="c", speaker_context=_ctx(),
        )[0]
        assert qr.provenance.source_role == "subject"
        assert qr.provenance.actor_id == ""
        assert qr.provenance.claimed_subject_label == "Spoofed Name"

    def test_subject_lane_limit_is_lane_local(self, tmp_path: Path):
        store = _store(tmp_path)
        for i in range(3):
            _row(store, ct_id=f"ct-u{i}", sort_key=1000.0 + i,
                 user_content=f"peptide question {i}",
                 sender_actor_id="actor:req")
            _row(store, ct_id=f"ct-s{i}", sort_key=5000.0 + i,
                 user_content=f"unrelated {i}",
                 reply_subject_actor_id="actor:subj",
                 reply_target_body=f"peptide answer {i}")
        results = store.search_canonical_turn_text(
            "peptide", conversation_id="c", limit=2, speaker_context=_ctx(),
        )
        roles = [qr.provenance.source_role for qr in results]
        assert roles.count("requester") == 2
        assert roles.count("subject") == 2
        assert len(results) == 4


class TestUnlocatableMatchIsMixed:
    def test_wildcard_only_match_gets_no_singular_speaker(self, tmp_path: Path):
        # ``%`` in the query is a LIKE wildcard, so SQL matches while the
        # Python-side matcher cannot locate a side. The combined excerpt
        # spans both lanes and must not receive a human actor.
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling", sender="BigTex",
             sender_actor_id="actor:tg:111")
        results = store.search_canonical_turn_text(
            "toes%tingling", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 1
        assert results[0].matched_side == "unknown"
        assert results[0].provenance.source_role == "mixed"
        assert results[0].provenance.actor_id == ""


class TestChannelScopedSpeakerPath:
    def test_channel_filter_and_prefix_apply_to_both_lanes(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="vasttest in general", sender_actor_id="actor:a",
             origin_channel_id="chan-a", origin_channel_label="#general")
        _row(store, ct_id="ct-2", sort_key=2000.0,
             user_content="vasttest in random", sender_actor_id="actor:b",
             origin_channel_id="chan-b", origin_channel_label="#random")
        _row(store, ct_id="ct-3", sort_key=3000.0,
             user_content="asking",
             reply_subject_actor_id="actor:s",
             reply_target_body="vasttest reply",
             origin_channel_id="chan-a", origin_channel_label="#general")
        results = store.search_canonical_turn_text(
            "vasttest", conversation_id="c", channel="chan-a",
            speaker_context=_ctx(),
        )
        assert len(results) == 2
        for qr in results:
            assert qr.provenance.origin_channel_id == "chan-a"
            assert qr.text.startswith("[#general] ")
        assert {qr.provenance.source_role for qr in results} == {
            "requester", "subject",
        }


class TestActorIdContainment:
    def test_actor_ids_are_absent_from_reprs(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="canary text here", sender_actor_id="actor:canary-9f",
             reply_subject_actor_id="actor:canary-7c",
             reply_target_body="canary reply body")
        results = store.search_canonical_turn_text(
            "canary", conversation_id="c", speaker_context=_ctx(),
        )
        assert len(results) == 2
        for qr in results:
            assert "canary-9f" not in repr(qr)
            assert "canary-7c" not in repr(qr)
            assert "canary-9f" not in repr(qr.provenance)
            assert "canary-7c" not in repr(qr.provenance)


class TestCompositeForwarding:
    def _composite(self, store: SQLiteStore):
        from virtual_context.core.composite_store import CompositeStore
        return CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=store,
        )

    def test_composite_forwards_the_speaker_context(self, tmp_path: Path):
        store = _store(tmp_path)
        _row(store, ct_id="ct-1", sort_key=1000.0,
             user_content="my toes are tingling",
             sender_actor_id="actor:tg:111")
        composite = self._composite(store)
        results = composite.search_canonical_turn_text(
            "tingling", conversation_id="c", speaker_context=_ctx(),
        )
        assert results[0].provenance is not None
        assert results[0].provenance.source_role == "requester"

    def test_composite_drops_none_for_backends_predating_the_argument(
        self, tmp_path: Path,
    ):
        store = _store(tmp_path)

        class _LegacySearch:
            """A search backend whose signature predates ``speaker_context``."""

            def search_canonical_turn_text(
                self, query, limit=5, conversation_id=None, channel="",
            ):
                return []

        from virtual_context.core.composite_store import CompositeStore
        composite = CompositeStore(
            segments=store, facts=store, fact_links=store,
            state=store, search=_LegacySearch(),
        )
        assert composite.search_canonical_turn_text("q") == []
        assert composite.search_canonical_turn_text("q", speaker_context=None) == []
