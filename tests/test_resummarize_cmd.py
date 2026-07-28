"""The repair command's acceptance gate and selection SQL.

The gate delegates to the compactor's own validator (one ruler for
repair and compaction) and adds the two repair-only postconditions:
a repair must destroy the selection predicate, and must fit the summary
token bound. The selection SQL's strict-prefix clause is load-bearing:
equality rows are intentional passthrough stubs.
"""

import pytest

from virtual_context.cli.resummarize_cmd import (
    _DAMAGE_PREDICATE,
    _selection_sql,
    classify_generated,
)

_COUNT = lambda text: len(text) // 4  # noqa: E731 - the compactor default


LONG_SOURCE = "Filing detail: the deadline moved to March. " * 60


def test_faithful_summary_is_accepted():
    assert classify_generated(
        "They discussed moving the filing deadline to March.",
        LONG_SOURCE, _COUNT, 500,
    ) is None


def test_validator_rejection_carries_the_criterion():
    # A fence fragment is degenerate under the compactor's validator.
    assert classify_generated(
        "```json", LONG_SOURCE, _COUNT, 500,
    ) == "validator_degenerate"
    # Overshoot: summary longer than its source.
    assert classify_generated(
        "x" * 200, "short source", _COUNT, 500,
    ) == "validator_overshoot"


def test_prefix_repair_is_rejected_as_still_prefix():
    """A 'repair' that is still a prefix of full_text would be selected
    again on every future run; idempotency requires refusing it."""
    assert classify_generated(
        LONG_SOURCE[:120], LONG_SOURCE, _COUNT, 500,
    ) == "still_prefix"


def test_full_text_itself_is_rejected_not_written():
    assert classify_generated(
        LONG_SOURCE, LONG_SOURCE, _COUNT, 500,
    ) == "still_prefix"


def test_overlong_summary_is_rejected():
    summary = "A distinct wording of the filing story. " * 60
    assert not LONG_SOURCE.startswith(summary)
    assert classify_generated(
        summary, LONG_SOURCE, _COUNT, 100,
    ) == "overlong"


def test_selection_sql_is_strict_prefix():
    sql = _selection_sql(False, None, None, None)
    assert "length(summary) < length(full_text)" in sql
    assert "left(full_text, length(summary)) = summary" in sql
    assert "ORDER BY ref ASC" in sql


def test_strict_clause_lives_in_the_shared_predicate():
    """The equality-overlap probe reuses _DAMAGE_PREDICATE; strictness
    must live there, not in per-call clauses, or the probe and the
    selection could diverge."""
    assert "length(summary) < length(full_text)" in _DAMAGE_PREDICATE


def test_short_split_toggles_with_include_short():
    gated = _selection_sql(False, None, None, None)
    opted = _selection_sql(True, None, None, None)
    assert "btrim(full_text" in gated
    assert ">= 256" in gated
    assert "btrim" not in opted


def test_range_and_resume_clauses_appear_only_when_set():
    plain = _selection_sql(False, None, None, None)
    assert "created_at::timestamptz" not in plain
    assert "ref >" not in plain
    ranged = _selection_sql(False, "2026-07-23", "2026-07-28", "abc")
    assert "created_at::timestamptz >= %(since)s::timestamptz" in ranged
    assert "created_at::timestamptz < %(until)s::timestamptz" in ranged
    assert "ref > %(after_ref)s" in ranged


def test_stripped_length_uses_bound_parameter_not_literal():
    """The whitespace set must be a bound parameter: the escaped-literal
    spelling of this set is the known footgun that dropped the vertical
    tab and matched the letter v."""
    sql = _selection_sql(False, None, None, None)
    assert "%(strip_ws)s" in sql
    assert "E'" not in sql
