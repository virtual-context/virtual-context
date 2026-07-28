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


def test_cascade_runbook_neutralizes_hostile_conversation_id(capsys):
    """Every interpolation site in the printed runbook treats the
    conversation id as data: SQL literals cannot be terminated, shell
    arguments are quoted, and Redis globs match only themselves."""
    from virtual_context.cli.resummarize_cmd import _print_cascade_runbook

    hostile = "conv'; SELECT pg_sleep(9); --"
    _print_cascade_runbook(hostile, ["legal", "court"])
    out = capsys.readouterr().out

    # The SQL literal doubles the quote, so the payload stays inside it.
    assert "conv''; SELECT pg_sleep(9); --" in out
    # No line contains the raw quote-terminated payload.
    assert "= 'conv';" not in out
    # Shell arguments are quoted (shlex wraps the whole id).
    assert "'conv'\"'\"'; SELECT pg_sleep(9); --'" in out


def test_cascade_runbook_escapes_redis_glob_metacharacters(capsys):
    from virtual_context.cli.resummarize_cmd import _print_cascade_runbook

    _print_cascade_runbook("conv*with?glob[chars]", ["legal"])
    out = capsys.readouterr().out
    eval_lines = [l for l in out.splitlines() if "redis-cli EVAL" in l]
    assert eval_lines
    for line in eval_lines:
        assert "conv\\*with\\?glob\\[chars\\]" in line


def test_resume_cursor_freezes_at_first_lost_row():
    """failure(A), success(B), failure(C), breaker-trip(D): the cursor
    must still point BEFORE A. A later success must not advance it past
    a row that received no response and was never repaired."""
    from virtual_context.cli.resummarize_cmd import _ResumeCursor

    cursor = _ResumeCursor(None)
    cursor.on_provider_failure()          # A: lost
    cursor.on_response("B")               # B: success must NOT advance
    cursor.on_provider_failure()          # C: lost
    # D trips the breaker; no cursor call happens for it.
    assert cursor.ref is None
    assert cursor.frozen


def test_resume_cursor_advances_past_responses_until_first_failure():
    from virtual_context.cli.resummarize_cmd import _ResumeCursor

    cursor = _ResumeCursor("start")
    cursor.on_response("A")
    cursor.on_response("B")
    assert cursor.ref == "B"
    cursor.on_provider_failure()
    cursor.on_response("D")
    assert cursor.ref == "B"
    assert cursor.frozen


def test_cascade_runbook_never_reparses_redis_keys_in_the_shell(capsys):
    """Hint-key deletion must be a server-side script with the pattern
    as ARGV: piping scan output through xargs re-parses raw key text,
    where a quote aborts the pipeline and a space splits one key into
    several DEL arguments."""
    from virtual_context.cli.resummarize_cmd import _print_cascade_runbook

    _print_cascade_runbook("conv with spaces' and quote", ["legal"])
    out = capsys.readouterr().out
    assert "xargs" not in out
    assert "--scan" not in out
    eval_lines = [l for l in out.splitlines() if "redis-cli EVAL" in l]
    assert len(eval_lines) == 2  # delete script + count verify script
    for line in eval_lines:
        assert "ARGV[1]" in line
