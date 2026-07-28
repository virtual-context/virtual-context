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
    eval_lines = [l for l in out.splitlines() if "EVAL" in l]
    assert eval_lines
    for line in eval_lines:
        assert "conv\\*with\\?glob\\[chars\\]" in line


def test_resume_cursor_freezes_at_first_lost_row():
    """failure(A), decided(B), failure(C), breaker-trip(D): the cursor
    must still point BEFORE A. Calls mirror the apply loop exactly:
    every provider failure calls on_provider_failure, INCLUDING the row
    that then trips the breaker."""
    from virtual_context.cli.resummarize_cmd import _ResumeCursor

    cursor = _ResumeCursor(None)
    cursor.on_provider_failure()          # A: lost
    cursor.on_decided("B")                # B: decided, must NOT advance
    cursor.on_provider_failure()          # C: lost
    cursor.on_provider_failure()          # D: lost, trips the breaker
    assert cursor.ref is None
    assert cursor.frozen


def test_resume_cursor_advances_past_decided_rows_until_first_freeze():
    """Accepted, malformed, and rejected rows are DECIDED and advance
    the cursor (a block of persistent rejectors must not starve later
    damage on resumed runs); provider failures and CAS skips are
    UNDECIDED and freeze it permanently."""
    from virtual_context.cli.resummarize_cmd import _ResumeCursor

    cursor = _ResumeCursor("start")
    cursor.on_decided("A")                # accepted
    cursor.on_decided("B")                # rejected: decided, advances
    assert cursor.ref == "B"
    cursor.on_provider_failure()
    cursor.on_decided("D")
    assert cursor.ref == "B"
    assert cursor.frozen


def test_resume_cursor_freezes_on_concurrent_cas_skip():
    """A CAS skip means no decision landed: the concurrent writer may
    have left the row damaged, so the cursor must not pass it."""
    from virtual_context.cli.resummarize_cmd import _ResumeCursor

    cursor = _ResumeCursor(None)
    cursor.on_decided("A")
    cursor.freeze()                       # B: skipped_concurrent
    cursor.on_decided("C")
    assert cursor.ref == "A"
    assert cursor.frozen


def test_cascade_runbook_never_reparses_redis_keys_in_the_shell(capsys):
    """Hint-key deletion must be server-side, paged, with the pattern
    as ARGV: piping scan output through xargs re-parses raw key text
    (a quote aborts the pipeline, a space splits one key into several
    DEL arguments), and looping SCAN to completion inside one EVAL
    blocks the server for the whole keyspace."""
    from virtual_context.cli.resummarize_cmd import _print_cascade_runbook

    _print_cascade_runbook("conv with spaces' and quote", ["legal"])
    out = capsys.readouterr().out
    assert "xargs" not in out
    assert "--scan" not in out
    eval_lines = [l for l in out.splitlines() if "EVAL" in l]
    assert len(eval_lines) == 2  # delete page + count page
    for line in eval_lines:
        assert "ARGV[1]" in line
        assert "ARGV[2]" in line          # cursor is a parameter...
        assert "repeat" not in line       # ...not an in-script loop
    # The cursor loop lives client-side in the printed shell.
    assert out.count('[ "$c" = "0" ] && break') == 2


def test_cascade_runbook_shell_sections_parse_as_shell(capsys):
    """Every non-comment, non-SQL line of the runbook must be valid
    shell, verified by bash -n with a hostile conversation id embedded.
    The runbook's [shell] labels are a promise; this test enforces it,
    including that the loops stop instead of spinning when redis-cli
    fails (the guard lines are part of the parsed script)."""
    import shutil
    import subprocess

    from virtual_context.cli.resummarize_cmd import _print_cascade_runbook

    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    _print_cascade_runbook("conv with spaces' and quote", ["legal", "court"])
    out = capsys.readouterr().out
    shell_lines = [
        l for l in out.splitlines()
        if l.strip()
        and not l.lstrip().startswith("#")
        and not l.startswith(("DELETE", "SELECT"))
    ]
    script = "\n".join(shell_lines) + "\n"
    proc = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"bash -n rejected:\n{proc.stderr}\n{script}"
    # The failure guards are present in both loops.
    assert script.count("redis-cli failed") == 2
    assert script.count("unexpected reply") == 2


def test_report_note_names_the_completion_path():
    """The operator-facing note must state the completion path in full:
    malformed/rejected rows processed before a cursor freeze MAY be
    behind the resume cursor (rows classified after a freeze sit ahead
    of it), and a final fresh run without --after-ref retries every
    still-damaged row. The class docstring alone is not
    operator-visible, and fragment assertions are not a pin — a
    mutation removing half the sentence passed a fragment check."""
    import inspect

    from virtual_context.cli.resummarize_cmd import (
        _REPORT_NOTE,
        cmd_admin_resummarize_segments,
    )

    assert (
        "COMPLETION PATH: "
        "malformed/rejected rows processed before any cursor freeze "
        "may be behind resume_after_ref; finish with a fresh "
        "invocation WITHOUT --after-ref to retry all still-damaged "
        "malformed, rejected, and skipped-concurrent rows"
    ) in _REPORT_NOTE
    assert "BEHIND resume_after_ref" not in _REPORT_NOTE
    # The constant is what the report actually emits.
    src = inspect.getsource(cmd_admin_resummarize_segments)
    assert '"note": _REPORT_NOTE' in src
