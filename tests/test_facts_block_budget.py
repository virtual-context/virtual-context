"""The Fact allocation is hard-zero at the model boundary.

Fact rows remain available as retrieval indexes, but their generated prose is
not source evidence.  A configured fact budget must therefore never cause a
``<facts>`` block to be rendered or charged to model context.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.types import AssemblerConfig, Fact


def _assembler() -> ContextAssembler:
    return ContextAssembler(config=AssemblerConfig())


def _facts(n: int) -> list[Fact]:
    """Lines of VARIED length.

    A fixture whose rendered lines are all the same length does not
    discriminate: when every line sits at the same remainder modulo the
    estimator's divisor, charging one extra character per line crosses a token
    boundary on every line and masks the per-line truncation exactly. Vary the
    length so all residues are represented.
    """
    return [
        Fact(
            subject=f"member{i:04d}",
            verb="reported",
            object="x" * (i % 17) + f" observation {i:04d}",
            what=f"Member {i:04d} reported " + "detail " * (i % 11) + "in full.",
        )
        for i in range(n)
    ]


def test_fact_formatter_withholds_prose_even_with_large_budget() -> None:
    assembler = _assembler()
    budget = 20_000
    rendered = assembler._format_facts(_facts(2_000), max_tokens=budget)
    assert rendered == ""
    assert assembler.token_counter(rendered) == 0


def test_formatter_never_emits_a_facts_wrapper() -> None:
    assembler = _assembler()
    rendered = assembler._format_facts(_facts(500), max_tokens=5_000)
    assert rendered == ""
    assert "<facts>" not in rendered


def _rendered_lines(facts_text: str) -> list[str]:
    """The fact lines the model actually receives."""
    if not facts_text:
        return []
    body = facts_text[len("<facts>\n"):-len("\n</facts>")]
    return body.split("\n")


def test_no_fact_is_recorded_as_selected_when_prose_is_withheld() -> None:
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import RetrievalResult

    facts = _facts(400)
    rr = RetrievalResult(facts=facts, retrieval_metadata={})
    cap = 2_000
    asm = ContextAssembler(config=AssemblerConfig(facts_max_tokens=cap))

    out = asm.assemble("", rr, [], token_budget=100_000)

    lines = _rendered_lines(out.facts_text)
    assert lines == []
    assert out.selected_facts == []
    assert rr.retrieval_metadata["facts_block"]["withheld"] == len(facts)


@pytest.mark.regression("BUG-052")
def test_assembled_facts_block_respects_the_facts_cap() -> None:
    """The block the model receives must fit the cap the fill enforced, so the
    overrun is not silently taken from whatever is budgeted after it."""
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import RetrievalResult

    cap = 2_000
    rr = RetrievalResult(facts=_facts(400), retrieval_metadata={})
    asm = ContextAssembler(config=AssemblerConfig(facts_max_tokens=cap))

    out = asm.assemble("", rr, [], token_budget=100_000)

    shipped = asm.token_counter(out.facts_text)
    assert shipped <= cap, f"facts block shipped {shipped}t against a {cap}t cap"
    assert out.budget_breakdown.get("facts") == shipped, (
        "the reported facts cost is not the cost of the block that shipped"
    )


def test_fact_candidates_do_not_consume_the_model_context_pool() -> None:
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import RetrievalResult

    rr = RetrievalResult(facts=_facts(400), retrieval_metadata={})
    asm = ContextAssembler(config=AssemblerConfig(facts_max_tokens=2_000))

    asm.assemble("", rr, [], token_budget=100_000)

    block = rr.retrieval_metadata["facts_block"]
    assert block["selected"] == 0
    assert block["rendered"] == 0
    assert block["tokens"] == 0
    assert block["withheld"] == 400


@pytest.mark.regression("BUG-052")
def test_every_budget_in_range_reports_a_count_that_survives_a_recount() -> None:
    """Exhaustive rather than sampled, across every budget in the range.

    ``_format_facts_admitted`` returns the token count measured for the prefix
    it accepted instead of measuring the rendered block a second time. That
    saves a call and closes the only window where a counter returning different
    values for the same input could put an over-cap total into the budget, but
    it means the returned count is trusted rather than checked. This is what
    keeps it honest: for every budget, the reported count must equal an
    independent recount of what was actually returned, and must never exceed
    the budget.
    """
    assembler = _assembler()
    facts = _facts(10)
    violations = []
    for budget in range(0, 200):
        text, admitted, reported = assembler._format_facts_admitted(facts, budget)
        recounted = assembler.token_counter(text) if text else 0
        if reported != recounted or reported > budget:
            violations.append((budget, admitted, reported, recounted))
    assert not violations, f"budget/count disagreements: {violations[:5]}"


def test_exact_former_block_budget_does_not_override_withholding() -> None:
    assembler = _assembler()
    facts = _facts(10)
    exact = assembler.token_counter(assembler._facts_block(
        [fact.format_for_prompt() for fact in facts]
    ))

    text, admitted, reported = assembler._format_facts_admitted(facts, exact)

    assert (text, admitted, reported) == ("", 0, 0)

    tighter, fewer, _ = assembler._format_facts_admitted(facts, exact - 1)
    assert (tighter, fewer) == ("", 0)


@pytest.mark.regression("BUG-052")
def test_an_empty_selection_renders_nothing_rather_than_empty_tags() -> None:
    """A bare wrapper around no lines is not an empty block, it is a block
    claiming there are facts and showing none."""
    assembler = _assembler()

    assert assembler._format_facts_admitted([], 1000) == ("", 0, 0)
    assert assembler._format_facts_admitted(_facts(5), 0) == ("", 0, 0)


def test_every_former_prefix_budget_still_withholds_fact_prose() -> None:
    assembler = _assembler()
    facts = _facts(10)
    lines = [fact.format_for_prompt() for fact in facts]
    whole = assembler.token_counter(assembler._facts_block(lines))

    for keep in range(1, len(lines)):
        budget = assembler.token_counter(assembler._facts_block(lines[:keep]))
        if budget >= whole:
            continue
        text, admitted, reported = assembler._format_facts_admitted(facts, budget)
        assert (text, admitted, reported) == ("", 0, 0), keep
