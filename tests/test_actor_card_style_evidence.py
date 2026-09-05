"""BUG-069: one utterance cannot become high-confidence recurring style."""

import json

import pytest

from tests.test_actor_cards import (
    _AdmitAll,
    _admission,
    _card_pipeline,
    _conversation,
    _curation,
    _entry,
    _fact,
    _now,
    _segment,
    _source,
    _turn,
    _turn_source,
    store as store,
)
from virtual_context.types import (
    CARD_KIND_ACTIVE_GOAL,
    CARD_KIND_COMMUNICATION_PREF,
    CARD_KIND_INTERACTION_STYLE,
    CARD_KIND_RELEVANT_HISTORY,
    CARD_SCOPE_CROSS_CONTEXT,
)


ACTOR = "actor:test:member-a"


class _Curator:
    def __init__(self, entry):
        self.entry = entry
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(_curation([self.entry])), {}


def _message(store, canonical_id, message_id, content, *, actor=ACTOR,
             timestamp="2026-01-02T12:00:00+00:00"):
    _turn(store, canonical_id, "guild", actor, "guild", "chan-a", content=content)
    store._get_conn().execute(
        "UPDATE canonical_turns SET source_message_id = ?, created_at = ? "
        "WHERE canonical_turn_id = ?",
        (message_id, timestamp, canonical_id),
    )


def _exact_fact(store, fact_id, canonical_ids, source_message_id, *, actor=ACTOR):
    ref = f"seg-{fact_id}"
    _segment(store, ref, "guild", canonical_ids)
    _fact(store, fact_id, "guild", ref, actor, version=2)
    store._get_conn().execute(
        "UPDATE facts SET subject = 'MemberA', verb = 'said', "
        "object = 'I am a cosmic cucumber, pal.', what = 'I am a cosmic cucumber, pal.', "
        "author_source_message_id = ?, mentioned_at = '2026-01-02T12:00:00+00:00' "
        "WHERE id = ?",
        (source_message_id, fact_id),
    )


def _active_entries(store, actor=ACTOR):
    return {
        row["body"]: dict(row)
        for row in store._get_conn().execute(
            "SELECT * FROM actor_card_entries WHERE actor_id = ? "
            "AND superseded_by IS NULL", (actor,),
        )
    }


@pytest.mark.regression("BUG-069")
def test_single_quoted_phrase_cannot_inflate_existing_style_even_if_admission_allows(store):
    _conversation(store, "guild")
    # Synthetic earlier examples support an existing style entry; one later
    # utterance must not inflate a newly proposed habit claim.
    for index, content in enumerate(("Thanks, pal-o.", "Feeling like a cosmic cucumber, pal."), 1):
        _message(
            store, f"old-{index}", f"old-message-{index}",
            content, actor=ACTOR,
            timestamp=f"2026-01-01T12:0{index}:00+00:00",
        )
    store.upsert_actor_profile_from_turn("guild", ACTOR, "MemberA", seen_at=_now())
    old = _entry(
        "earlier-style", CARD_KIND_INTERACTION_STYLE,
        "Uses playful language such as 'pal-o' and 'cosmic cucumber'.",
        scope=CARD_SCOPE_CROSS_CONTEXT,
        confidence=0.9,
    )
    old.created_at = old.updated_at = "2026-01-01T12:02:00+00:00"
    assert store.replace_actor_card(
        "t1", ACTOR,
        [(old, [_turn_source(old.id, "guild", "guild", f"old-{index}", "chan-a")
                for index in (1, 2)])],
        input_hash="earlier-policy16", expected_source_epochs={"guild": 1},
    )
    _message(store, "utterance", "utterance-message", "I am a cosmic cucumber, pal.", actor=ACTOR)
    _exact_fact(store, "utterance-fact", ["utterance"], "utterance-message", actor=ACTOR)
    store._get_conn().execute(
        "UPDATE facts SET subject = 'MemberA', verb = 'is', object = 'a cosmic cucumber, pal', "
        "what = 'MemberA is a cosmic cucumber, pal.' WHERE id = 'utterance-fact'",
    )
    inflated_body = (
        "Frequently uses 'pal' and playful language in communication."
    )
    curator = _Curator({
        "kind": CARD_KIND_INTERACTION_STYLE, "body": inflated_body,
        "confidence": 1.0, "fact_ids": ["utterance-fact"], "turn_ids": [],
    })
    pipeline = _card_pipeline(store, curator, admission=_AdmitAll())

    assert pipeline._rebuild_actor_card(ACTOR) == 2
    active = _active_entries(store, ACTOR)
    assert active[old.body]["id"] == old.id
    assert active[old.body]["confidence"] == 0.9
    # The independent judge deliberately permits the bad generalization.
    # This assertion must be enforced by code, not a rejecting provider stub.
    assert active[inflated_body]["confidence"] <= 0.7
    assert active[inflated_body]["body"] == inflated_body


def _seed_evidence(store, case):
    _conversation(store, "guild")
    _message(store, "ct-1", "message-1", "Thanks pal-o, that explanation helped.")
    _message(store, "ct-2", "message-2", "Pal-o, the next explanation helped too.")
    _exact_fact(store, "f-1", ["ct-1", "ct-2"], "message-1")
    _exact_fact(store, "f-2", ["ct-1", "ct-2"], "message-2")
    store.upsert_actor_profile_from_turn("guild", ACTOR, "MemberA", seen_at=_now())
    fact_ids, turn_ids = [], []
    expected = 0.7
    if case == "single_turn":
        turn_ids = ["ct-1"]
    elif case == "single_fact_long_segment":
        fact_ids = ["f-1"]
    elif case == "fact_and_its_turn":
        fact_ids, turn_ids = ["f-1"], ["ct-1"]
    elif case == "two_facts_same_message":
        store._get_conn().execute(
            "UPDATE facts SET author_source_message_id = 'message-1' WHERE id = 'f-2'",
        )
        fact_ids = ["f-1", "f-2"]
    elif case == "legacy_facts_do_not_inherit_segment_neighbors":
        store._get_conn().execute(
            "UPDATE facts SET author_attribution_version = 1, author_source_message_id = ''",
        )
        fact_ids = ["f-1", "f-2"]
    elif case == "unknown_fact_source":
        store._get_conn().execute(
            "UPDATE facts SET author_source_message_id = 'missing' WHERE id = 'f-2'",
        )
        fact_ids = ["f-1", "f-2"]
    elif case == "fact_native_id_outside_its_segment":
        store._get_conn().execute(
            "UPDATE segments SET metadata_json = ? WHERE ref = 'seg-f-2'",
            (json.dumps({"canonical_turn_ids": ["ct-1"], "source_mapping_complete": True}),),
        )
        fact_ids = ["f-1", "f-2"]
    elif case == "duplicate_native_message":
        store._get_conn().execute(
            "UPDATE canonical_turns SET source_message_id = 'message-1' WHERE canonical_turn_id = 'ct-2'",
        )
        turn_ids = ["ct-1", "ct-2"]
    elif case == "distinct_turns":
        turn_ids, expected = ["ct-1", "ct-2"], 1.0
    elif case == "distinct_facts":
        fact_ids, expected = ["f-1", "f-2"], 1.0
    elif case == "distinct_turns_plus_one_fact":
        fact_ids, turn_ids, expected = ["f-1"], ["ct-1", "ct-2"], 1.0
    else:
        raise AssertionError(case)
    return fact_ids, turn_ids, expected


@pytest.mark.regression("BUG-069")
@pytest.mark.parametrize("kind", [CARD_KIND_COMMUNICATION_PREF, CARD_KIND_INTERACTION_STYLE])
@pytest.mark.parametrize("case", [
    "single_turn", "single_fact_long_segment", "fact_and_its_turn",
    "two_facts_same_message", "legacy_facts_do_not_inherit_segment_neighbors",
    "unknown_fact_source", "fact_native_id_outside_its_segment",
    "duplicate_native_message", "distinct_turns", "distinct_facts",
    "distinct_turns_plus_one_fact",
])
def test_style_confidence_counts_distinct_proven_messages(store, kind, case):
    fact_ids, turn_ids, expected = _seed_evidence(store, case)
    body = "Uses 'pal-o' as an informal style example."
    curator = _Curator({
        "kind": kind, "body": body, "confidence": 1.0,
        "fact_ids": fact_ids, "turn_ids": turn_ids,
    })
    assert _card_pipeline(store, curator, admission=_AdmitAll())._rebuild_actor_card(ACTOR) == 1
    assert _active_entries(store)[body]["confidence"] == expected


@pytest.mark.regression("BUG-069")
@pytest.mark.parametrize("kind", [CARD_KIND_COMMUNICATION_PREF, CARD_KIND_INTERACTION_STYLE])
@pytest.mark.parametrize("case", ["single_turn", "fact_and_its_turn", "two_facts_same_message", "distinct_turns"])
def test_carryover_style_is_recalibrated_without_rewriting(store, kind, case):
    fact_ids, turn_ids, expected = _seed_evidence(store, case)
    old = _entry("old-entry", kind, "Uses the exact term 'pal-o'.",
                 scope=CARD_SCOPE_CROSS_CONTEXT, confidence=1.0)
    citations = [_source(old.id, "guild", "guild", fid, "chan-a") for fid in fact_ids]
    citations += [_turn_source(old.id, "guild", "guild", tid, "chan-a") for tid in turn_ids]
    assert store.replace_actor_card(
        "t1", ACTOR, [(old, citations)], input_hash="policy16",
        expected_source_epochs={"guild": 1},
    )

    class OmitExisting:
        def complete(self, **kwargs):
            return json.dumps(_curation([])), {}

    assert _card_pipeline(store, OmitExisting(), admission=_AdmitAll())._rebuild_actor_card(ACTOR) == 1
    active = _active_entries(store)
    assert list(active) == [old.body]
    assert active[old.body]["id"] == old.id
    assert active[old.body]["confidence"] == expected


@pytest.mark.regression("BUG-069")
@pytest.mark.parametrize("kind", [CARD_KIND_ACTIVE_GOAL, CARD_KIND_RELEVANT_HISTORY])
def test_other_card_kinds_keep_existing_single_source_cap(store, kind):
    _seed_evidence(store, "single_turn")
    body = "Is working through a structured strength program."
    curator = _Curator({"kind": kind, "body": body, "confidence": 1.0,
                        "fact_ids": [], "turn_ids": ["ct-1"]})
    assert _card_pipeline(store, curator, admission=_AdmitAll())._rebuild_actor_card(ACTOR) == 1
    assert _active_entries(store)[body]["confidence"] == 0.8


@pytest.mark.regression("BUG-069")
def test_both_prompt_surfaces_forbid_single_utterance_habits_and_preserve_exact_terms(store):
    _seed_evidence(store, "distinct_turns")
    curator = _Curator({"kind": CARD_KIND_INTERACTION_STYLE,
                        "body": "Uses 'pal-o' as an informal style example.",
                        "confidence": 0.6, "fact_ids": [], "turn_ids": ["ct-1", "ct-2"]})

    class Admission(_AdmitAll):
        calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return super().complete(**kwargs)

    admission = Admission()
    assert _card_pipeline(store, curator, admission=admission)._rebuild_actor_card(ACTOR) == 1
    for kwargs in (curator.calls[0], admission.calls[0]):
        system = kwargs["system"]
        assert "a quoted phrase or single utterance never establishes frequency or habit" in system
        assert "'frequently', 'often', 'always'" in system
        assert "multiple distinct cited actor-authored messages" in system
        assert "quote it as an example instead" in system
        assert "keep 'pal-o' as 'pal-o'; do not normalize 'pal-o' to 'pal'" in system
        assert "confidence no higher than 0.7" in system
    assert _active_entries(store)[curator.entry["body"]]["confidence"] == 0.6


@pytest.mark.regression("BUG-069")
@pytest.mark.parametrize("identity_change", ["fact_source", "turn_source", "fact_segment"])
def test_source_identity_change_invalidates_confidence_input_hash(store, identity_change):
    _seed_evidence(store, "distinct_facts")
    if identity_change == "fact_segment":
        for index in (1, 2):
            store._get_conn().execute(
                "UPDATE segments SET metadata_json = ? WHERE ref = ?",
                (json.dumps({"canonical_turn_ids": [f"ct-{index}"],
                             "source_mapping_complete": True}), f"seg-f-{index}"),
            )
    body = "Uses 'pal-o' as an informal style example."
    curator = _Curator({"kind": CARD_KIND_INTERACTION_STYLE, "body": body,
                        "confidence": 1.0, "fact_ids": ["f-1", "f-2"], "turn_ids": []})
    pipeline = _card_pipeline(store, curator, admission=_AdmitAll())
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    # The newly stored entry becomes carryover input on the next rebuild.
    # Settle that transition first, then prove a clean unchanged input skips
    # the curator. Otherwise adding the carryover could mask a missing source
    # identity field in the fingerprint.
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    before = store.get_actor_profile("t1", ACTOR).card_input_hash
    calls_before = len(curator.calls)
    assert pipeline._rebuild_actor_card(ACTOR) == 0
    assert len(curator.calls) == calls_before
    assert store.get_actor_profile("t1", ACTOR).card_input_hash == before
    assert _active_entries(store)[body]["confidence"] == 1.0
    if identity_change == "fact_source":
        store._get_conn().execute(
            "UPDATE facts SET author_source_message_id = 'message-1' WHERE id = 'f-2'",
        )
    elif identity_change == "turn_source":
        store._get_conn().execute(
            "UPDATE canonical_turns SET source_message_id = 'message-1' WHERE canonical_turn_id = 'ct-2'",
        )
    else:
        # The set of segment manifests stays identical. Only each fact's
        # membership changes, so the fact-to-segment edge must be hashed.
        store._get_conn().execute(
            "UPDATE facts SET segment_ref = CASE id "
            "WHEN 'f-1' THEN 'seg-f-2' ELSE 'seg-f-1' END "
            "WHERE id IN ('f-1', 'f-2')",
        )
    # Prove the manifest itself changes; do not rely on the dirty-trigger hint.
    store._get_conn().execute(
        "UPDATE actor_profiles SET card_dirty = 0, card_invalid = 0 WHERE actor_id = ?",
        (ACTOR,),
    )
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    assert store.get_actor_profile("t1", ACTOR).card_input_hash != before
    assert _active_entries(store)[body]["confidence"] == 0.7


@pytest.mark.regression("BUG-069")
@pytest.mark.parametrize("kind", [CARD_KIND_COMMUNICATION_PREF, CARD_KIND_INTERACTION_STYLE])
def test_carryover_is_calibrated_before_confidence_sensitive_admission(store, kind):
    _seed_evidence(store, "single_turn")
    old = _entry("old-style", kind, "Uses the exact term 'pal-o'.",
                 scope=CARD_SCOPE_CROSS_CONTEXT, confidence=1.0)
    assert store.replace_actor_card(
        "t1", ACTOR,
        [(old, [_turn_source(old.id, "guild", "guild", "ct-1", "chan-a")])],
        input_hash="policy16", expected_source_epochs={"guild": 1},
    )

    class OmitExisting:
        def complete(self, **kwargs):
            return json.dumps(_curation([])), {}

    class ConfidenceSensitiveAdmission:
        proposed = None

        def complete(self, **kwargs):
            candidates = json.loads(kwargs["user"])["candidates"]
            self.proposed = {item["candidate_id"]: item["proposed_confidence"] for item in candidates}
            return json.dumps(_admission([
                {"candidate_id": item["candidate_id"],
                 "admit": item["proposed_confidence"] <= 0.7,
                 "reason": "durable" if item["proposed_confidence"] <= 0.7 else "insufficient_evidence"}
                for item in candidates
            ])), {}

    admission = ConfidenceSensitiveAdmission()
    pipeline = _card_pipeline(store, OmitExisting(), admission=admission)
    assert pipeline._rebuild_actor_card(ACTOR) == 1
    assert admission.proposed == {old.id: 0.7}
    active = _active_entries(store)
    assert list(active) == [old.body]
    assert active[old.body]["id"] == old.id
    assert active[old.body]["confidence"] == 0.7
