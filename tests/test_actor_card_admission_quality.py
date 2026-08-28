"""Card curation/admission judgment rules and confidence calibration (BUG-063).

Three defect classes were found in live guild cards: banter admitted as
fact at confidence 1.00 (a joking "going to jump on the Tren" became
"Plans to start using Tren."), single questions and one-shot service
requests minted as standing active_goals ("give me a leg workout for
today" -> a durable goal at 0.90), and a question about a THIRD PARTY
filed as the asker's own goal. No prompt surface carried a sincerity or
register rule, active_goal had neither a first-person-intent nor a
recurrence requirement, and confidence was the curator model's own
uncalibrated assertion passed through verbatim.

These tests pin the fixed contract: the judgment rules present on both
prompt surfaces, the confidence scale defined for the curator, and the
single-source confidence cap enforced in code.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import inspect

import pytest

from virtual_context.core import compaction_pipeline as cp
from tests.test_actor_cards import (
    OPTICS,
    _now,
    _AdmitAll,
    _card_pipeline,
    _conversation,
    _curation,
    _turn,
    store,  # fixture
)


# ---------------------------------------------------------------------------
# Prompt-contract pins
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-063")
def test_judgment_rules_cover_banter_requests_and_subject():
    rules = cp._ACTOR_CARD_JUDGMENT_RULES.lower()
    # Sincerity/banter with a corroboration escape hatch.
    assert "banter" in rules and "sarcasm" in rules
    assert "non-joking register" in rules or "corroborat" in rules
    # A request or question is not a goal; transience markers decisive.
    assert "never an active_goal" in rules
    assert "for today" in rules
    # First-person subject attribution.
    assert "first-person" in rules
    assert "third party" in rules
    # Durability bar extended beyond pref/style.
    assert "relevant_history" in rules


@pytest.mark.regression("BUG-063")
def test_confidence_scale_is_defined_for_the_curator():
    scale = cp._ACTOR_CARD_CONFIDENCE_SCALE
    assert "1.0" in scale
    assert "0.7" in scale
    assert "0.4" in scale


@pytest.mark.regression("BUG-063")
def test_both_prompt_surfaces_reference_the_judgment_rules():
    curation_src = inspect.getsource(
        cp.CompactionPipeline._curate_actor_card_partition
    )
    admission_src = inspect.getsource(
        cp.CompactionPipeline._admit_actor_card_entries
    )
    assert "_ACTOR_CARD_JUDGMENT_RULES" in curation_src
    assert "_ACTOR_CARD_JUDGMENT_RULES" in admission_src
    assert "_ACTOR_CARD_CONFIDENCE_SCALE" in curation_src


# ---------------------------------------------------------------------------
# Single-source confidence cap (code-enforced)
# ---------------------------------------------------------------------------

def _goal_curator(confidence: float, turn_ids: list[str]):
    import json

    class Curator:
        def complete(self, **kwargs):
            return json.dumps(_curation([{
                "kind": "active_goal",
                "body": "Is working through a structured strength program.",
                "confidence": confidence,
                "fact_ids": [],
                "turn_ids": list(turn_ids),
            }])), {}

    return Curator()


def _stored_confidences(store):
    return [
        float(r["confidence"])
        for r in store._get_conn().execute(
            "SELECT confidence FROM actor_card_entries "
            "WHERE tenant_id = 't1' AND actor_id = ? "
            "AND superseded_by IS NULL",
            (OPTICS,),
        ).fetchall()
    ]


@pytest.mark.regression("BUG-063")
def test_single_source_entry_confidence_is_capped(store):
    _conversation(store, "guild")
    _turn(store, "ct-1", "guild", OPTICS, "guild", "chan-a",
          content="I am working through a structured strength program.")
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    pipeline = _card_pipeline(
        store, _goal_curator(1.0, ["ct-1"]), admission=_AdmitAll(),
    )
    assert pipeline._rebuild_actor_card(OPTICS) == 1
    assert _stored_confidences(store) == [0.8], (
        "a single-citation entry must not exceed the 0.8 cap"
    )


@pytest.mark.regression("BUG-063")
def test_multi_source_entry_confidence_is_uncapped(store):
    _conversation(store, "guild")
    _turn(store, "ct-1", "guild", OPTICS, "guild", "chan-a",
          content="I am working through a structured strength program.")
    _turn(store, "ct-2", "guild", OPTICS, "guild", "chan-a",
          content="Week four of the strength program, still on plan.")
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    pipeline = _card_pipeline(
        store, _goal_curator(1.0, ["ct-1", "ct-2"]), admission=_AdmitAll(),
    )
    assert pipeline._rebuild_actor_card(OPTICS) == 1
    assert _stored_confidences(store) == [1.0]


@pytest.mark.regression("BUG-063")
def test_carryover_single_source_confidence_is_clamped(store):
    """A carried-over cross-context entry keeps its immutable body but not
    an over-cap confidence: the evidence invariant (one cited message can
    never support more than the cap) applies to every admitted entry,
    whichever path produced it."""
    from virtual_context.types import CARD_KIND_COMMUNICATION_PREF
    from tests.test_actor_cards import (
        CARD_SCOPE_CROSS_CONTEXT,
        _entry,
        _turn_source,
    )

    _conversation(store, "guild")
    _turn(store, "ct-1", "guild", OPTICS, "guild", "chan-a",
          content="I am working through a structured strength program.")
    store.upsert_actor_profile_from_turn(
        "guild", OPTICS, "Optics", seen_at=_now(),
    )

    legacy = _entry(
        "e-legacy", CARD_KIND_COMMUNICATION_PREF,
        "Prefers rapid replies from the agent.",
        scope=CARD_SCOPE_CROSS_CONTEXT, confidence=0.98,
    )
    assert store.replace_actor_card(
        "t1", OPTICS,
        [(legacy, [_turn_source("e-legacy", "guild", "guild", "ct-1", "chan-a")])],
        input_hash="legacy-hash",
        expected_source_epochs={"guild": 1},
    )

    pipeline = _card_pipeline(
        store, _goal_curator(0.5, ["ct-1"]), admission=_AdmitAll(),
    )
    assert pipeline._rebuild_actor_card(OPTICS) >= 1
    confs = _stored_confidences(store)
    assert confs and max(confs) <= 0.8, confs
