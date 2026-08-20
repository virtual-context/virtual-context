"""The rendered facts block must fit the budget it was built against.

``_format_facts`` stops admitting lines when a running total would exceed
``max_tokens``, but that total is a sum of per-line counts while the value
delivered is a single joined string. Two undercounts accumulate: the ``"\\n"``
separators between lines are never charged, and the deployed estimator
truncates each line's remainder independently. The block therefore ships larger
than the budget the loop enforced.
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


@pytest.mark.regression("BUG-052")
def test_rendered_facts_block_fits_its_budget() -> None:
    assembler = _assembler()
    budget = 20_000
    rendered = assembler._format_facts(_facts(2_000), max_tokens=budget)
    assert rendered
    actual = assembler.token_counter(rendered)
    assert actual <= budget, (
        f"facts block shipped {actual} tokens against a {budget} budget "
        f"({actual - budget} over)"
    )


@pytest.mark.regression("BUG-052")
def test_line_separators_are_charged() -> None:
    """The admitted lines plus their separators must not exceed the budget."""
    assembler = _assembler()
    budget = 5_000
    rendered = assembler._format_facts(_facts(500), max_tokens=budget)
    assert rendered
    body = rendered[len("<facts>\n"):-len("\n</facts>")]
    lines = body.split("\n")
    additive = sum(assembler.token_counter(line) for line in lines)
    whole = assembler.token_counter(body)
    assert whole <= budget, (
        f"joined body is {whole} tokens; the loop counted {additive} "
        f"({whole - additive} unaccounted) against a {budget} budget"
    )


def _rendered_lines(facts_text: str) -> list[str]:
    """The fact lines the model actually receives."""
    if not facts_text:
        return []
    body = facts_text[len("<facts>\n"):-len("\n</facts>")]
    return body.split("\n")


@pytest.mark.regression("BUG-052")
def test_selected_facts_and_the_rendered_block_cannot_disagree() -> None:
    """The acceptance property. ``selected_facts`` is the record of what the
    model was shown; the block is what it was shown. A fact recorded as
    selected but absent from the block is a false claim about the payload, and
    it is what a renderer-side budget fix produces.

    The budget here is tight enough that selection cannot admit everything, so
    the trimming path is exercised rather than skipped.
    """
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import RetrievalResult

    facts = _facts(400)
    rr = RetrievalResult(facts=facts, retrieval_metadata={})
    cap = 2_000
    asm = ContextAssembler(config=AssemblerConfig(facts_max_tokens=cap))

    out = asm.assemble("", rr, [], token_budget=100_000)

    lines = _rendered_lines(out.facts_text)
    assert lines, "no facts rendered; the fixture does not exercise the path"
    assert len(lines) < len(facts), (
        "the budget admitted every fact; this fixture does not exercise "
        "trimming and cannot discriminate"
    )
    assert len(out.selected_facts) == len(lines), (
        f"{len(out.selected_facts)} facts recorded as selected but "
        f"{len(lines)} lines rendered"
    )
    for fact in out.selected_facts:
        assert fact.format_for_prompt() in lines, (
            "a fact recorded as selected is absent from the rendered block"
        )


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


@pytest.mark.regression("BUG-052")
def test_selection_charges_the_block_so_rendering_never_trims() -> None:
    """The fill must charge what the block costs, not a sum of per-line counts.

    Without this the renderer is the only thing keeping the block inside its
    cap, and it does so by discarding lines the fill had already selected and
    already charged against the pool. The block would still fit and the record
    would still match, so every other assertion here passes while the selection
    stage is wrong. The trim count is the only outward sign.
    """
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import RetrievalResult

    rr = RetrievalResult(facts=_facts(400), retrieval_metadata={})
    asm = ContextAssembler(config=AssemblerConfig(facts_max_tokens=2_000))

    asm.assemble("", rr, [], token_budget=100_000)

    block = rr.retrieval_metadata["facts_block"]
    assert block["rendered"] > 0
    assert block["trimmed"] == 0, (
        f"the renderer discarded {block['trimmed']} of {block['selected']} "
        "selected lines, so the fill charged less than the block costs"
    )


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


@pytest.mark.regression("BUG-052")
def test_a_block_costing_exactly_the_budget_is_admitted() -> None:
    """The comparison is ``<=``, not ``<``. An off-by-one here silently drops a
    fact from every block that happens to land on its budget."""
    assembler = _assembler()
    facts = _facts(10)
    exact = assembler.token_counter(assembler._facts_block(
        [fact.format_for_prompt() for fact in facts]
    ))

    text, admitted, reported = assembler._format_facts_admitted(facts, exact)

    assert admitted == len(facts), "a block costing exactly its budget was trimmed"
    assert reported == exact

    tighter, fewer, _ = assembler._format_facts_admitted(facts, exact - 1)
    assert fewer < len(facts)
    assert assembler.token_counter(tighter) <= exact - 1


@pytest.mark.regression("BUG-052")
def test_an_empty_selection_renders_nothing_rather_than_empty_tags() -> None:
    """A bare wrapper around no lines is not an empty block, it is a block
    claiming there are facts and showing none."""
    assembler = _assembler()

    assert assembler._format_facts_admitted([], 1000) == ("", 0, 0)
    assert assembler._format_facts_admitted(_facts(5), 0) == ("", 0, 0)


@pytest.mark.regression("BUG-052")
def test_a_prefix_costing_exactly_the_budget_is_admitted_by_the_bisection() -> None:
    """The same boundary, but on the path that actually decides it.

    A budget equal to the WHOLE block is answered by the fast path and never
    reaches the search, so it cannot discriminate the comparison inside the
    bisection. This picks a budget equal to the cost of a proper prefix, which
    forces the search to run and makes an off-by-one there drop a line that
    fits exactly.
    """
    assembler = _assembler()
    facts = _facts(10)
    lines = [fact.format_for_prompt() for fact in facts]
    whole = assembler.token_counter(assembler._facts_block(lines))

    for keep in range(1, len(lines)):
        budget = assembler.token_counter(assembler._facts_block(lines[:keep]))
        if budget >= whole:
            continue
        text, admitted, reported = assembler._format_facts_admitted(facts, budget)
        assert admitted >= keep, (
            f"a prefix of {keep} costing exactly {budget} was not admitted "
            f"at that budget; only {admitted} lines were"
        )
        assert reported <= budget
        assert assembler.token_counter(text) <= budget
