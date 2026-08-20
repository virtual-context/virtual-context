"""A finished action and a stopped state must not share one token.

`completed` was asked to carry both "an action concluded" and "a state ended".
Only the second answers "is this person still doing X?" with no, so a person's
record of stopping arrived wearing the same token as an inventory tally. Even
with perfect authorship labelling, the fact that would correct a false ongoing
claim was dressed identically to a stock count.

Beside it, an unrecognised status was coerced to `active` — turning "the parser
could not classify this" into the strongest assertion the vocabulary has that
the state continues. That is the claim a cessation exists to contradict.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.types import (
    Fact,
    TemporalStatus,
    normalize_temporal_status,
)


def _status_prompts() -> list[str]:
    """Every prompt string in the compactor that names the status vocabulary.

    Found by scanning the module rather than naming constants, so a new prompt
    that carries the tokens cannot quietly escape these assertions.
    """
    from virtual_context.core import compactor

    found = [
        value for value in vars(compactor).values()
        if isinstance(value, str) and '"status": one of' in value
    ]
    return found


def test_ceased_is_a_distinct_token_from_completed():
    """Row 1 and 8. Without this the two meanings cannot be told apart at all."""
    assert TemporalStatus.CEASED.value == "ceased"
    assert TemporalStatus.COMPLETED.value == "completed"
    assert normalize_temporal_status("ceased") == ("ceased", "")


def test_completed_is_unchanged():
    """Row 2. The token is ADDED beside completed, never redefined. Existing
    rows keep the meaning they were written with, ambiguous as that is, rather
    than being silently reclassified by a vocabulary change."""
    assert normalize_temporal_status("completed") == ("completed", "")


def test_an_unmappable_status_is_unset_and_never_active():
    """Rows 3, 4, 5. The defect: unknown became a positive ongoing claim."""
    for raw in ("frobnicated", "", "   ", None, 42, [], {"a": 1}):
        status, reason = normalize_temporal_status(raw)
        assert status == "", f"{raw!r} resolved to {status!r}"
        assert reason == "unmapped"
        assert status != "active"


def test_a_real_active_status_still_resolves_to_active():
    """Row 6. Fixing the fallback must not disturb a genuine active."""
    assert normalize_temporal_status("active") == ("active", "")


def test_synonyms_for_an_ongoing_fact_still_resolve_to_active():
    """The refutation that changed this design.

    A model emitting `ongoing` for a genuinely current fact was previously
    normalized to `active` by the very fallback being removed. Dropping to
    unset would have LOST a correct classification, so a bare fallback is not
    strictly safer. Mapping the synonym keeps the correct answer, and the
    evidence for it is the word the model actually chose.
    """
    for raw in ("ongoing", "current", "in_progress", "continuing", " Ongoing "):
        status, reason = normalize_temporal_status(raw)
        assert status == "active", f"{raw!r} lost its classification"
        assert reason == "synonym"


def test_synonyms_for_stopping_resolve_to_ceased():
    for raw in ("stopped", "ended", "discontinued", "quit", "no longer"):
        status, reason = normalize_temporal_status(raw)
        assert status == "ceased", f"{raw!r} did not resolve to ceased"
        assert reason == "synonym"


def test_normalization_is_reported_so_it_can_be_measured():
    """How often the fallback fires is unmeasurable from stored rows, because
    the old fallback wrote a valid-looking `active`. The reason field is what
    makes the rate observable at all."""
    assert normalize_temporal_status("active")[1] == ""
    assert normalize_temporal_status("ongoing")[1] == "synonym"
    assert normalize_temporal_status("frobnicated")[1] == "unmapped"


def test_a_course_run_to_its_end_is_completed_not_ceased():
    """The counterexample that sharpened the prompt.

    A prescribed course taken for its full length IS a formerly ongoing state
    that ended, so a naive reading of `ceased` would claim it. Rendering that
    as ceased tells a reader the person discontinued treatment, which is more
    wrong than the ambiguity it replaced. Both prompts must say so explicitly.
    """
    for prompt in _status_prompts():
        assert "ran to its intended end" in prompt
        assert "completed, NOT ceased" in prompt


def test_ceased_does_not_claim_the_stopping_was_deliberate():
    """A treatment that ended because a clinic closed is no longer ongoing but
    was not deliberately stopped. The definition must not make the model assert
    a reason the text does not support."""
    for prompt in _status_prompts():
        assert "does not claim the stopping was deliberate" in prompt


def test_every_token_is_defined_in_every_prompt_that_names_them():
    """Row 7. Five bare tokens with no definitions was the original state, and
    a vocabulary the model is not told the meaning of is not a vocabulary.

    Discovered by source, not by a hand-listed constant name: a second prompt
    that names the tokens and is missed here would leave that path with the
    undefined vocabulary while this test stayed green.
    """
    prompts = _status_prompts()
    assert len(prompts) >= 2, (
        f"expected every status-bearing prompt to be discovered, found {len(prompts)}"
    )
    for prompt in prompts:
        for token in ("active", "completed", "ceased", "planned", "abandoned", "recurring"):
            assert f"{token} " in prompt or f"{token}=" in prompt, (
                f"{token} is named but never defined"
            )


def test_an_unset_status_renders_no_marker_like_active():
    """A stated limitation, pinned so it is not mistaken for a fix.

    `active` and unset render identically, so this change corrects the STORED
    record and what any status-aware consumer sees. It does not change what the
    model is shown. A hedged marker was rejected deliberately: hedged
    provenance markers were measured to be ignored while definite ones are
    obeyed, so it would cost tokens and buy nothing.
    """
    assert "[status:" not in Fact(subject="s", verb="v", object="o", status="").format_for_prompt()
    assert "[status:" not in Fact(subject="s", verb="v", object="o", status="active").format_for_prompt()
    assert "[status: ceased]" in Fact(
        subject="s", verb="v", object="o", status="ceased",
    ).format_for_prompt()
