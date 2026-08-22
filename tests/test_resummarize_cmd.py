"""The repair command's acceptance gate and selection SQL.

The gate delegates to the compactor's own validator (one ruler for
repair and compaction) and adds the two repair-only postconditions:
a repair must destroy the selection predicate, and must fit the summary
token bound. The selection SQL's strict-prefix clause is load-bearing:
equality rows are intentional passthrough stubs.
"""

import json
from types import SimpleNamespace

import pytest

from virtual_context.cli.resummarize_cmd import (
    _DAMAGE_PREDICATE,
    _actor_fingerprint,
    _audience_fingerprint,
    _identity_selection_sql,
    _selection_sql,
    classify_identity_violation,
    classify_tag_summary_identity_violation,
    classify_generated,
)

_COUNT = lambda text: len(text) // 4  # noqa: E731 - the compactor default


LONG_SOURCE = "Filing detail: the deadline moved to March. " * 60
def _proved_metadata(**updates):
    metadata = {
        "canonical_turn_ids": ["ct-1"],
        "source_mapping_complete": True,
        "source_speaker_labels": ["BigTex"],
        "source_speaker_identity_count": 1,
        "source_speaker_identity_fingerprint": _actor_fingerprint({"actor-a"}),
        "source_audience_fingerprint": _audience_fingerprint({
            ("guild-1", "channel-1"),
        }),
    }
    metadata.update(updates)
    return metadata


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


def test_identity_selection_scans_conversation_without_prefix_predicate():
    sql = _identity_selection_sql(None, None, None)
    assert "conversation_id = %(conversation_id)s" in sql
    assert _DAMAGE_PREDICATE not in sql
    assert "left(full_text" not in sql
    assert "ORDER BY ref ASC" in sql


def test_identity_selection_honors_range_and_resume_bounds():
    sql = _identity_selection_sql("2026-08-01", "2026-08-22", "seg-9")
    assert "created_at::timestamptz >= %(since)s::timestamptz" in sql
    assert "created_at::timestamptz < %(until)s::timestamptz" in sql
    assert "ref > %(after_ref)s" in sql


def test_identity_classifier_requires_positive_proof_even_for_subjectless_prose():
    reasons = classify_identity_violation({
        "summary": "Stopped MOTS-c after the appointment.",
        "metadata_json": "{}",
    })
    # This passive fragment has no lexical referent; provenance, not wording,
    # is what makes it an identity violation.
    assert "summary_ambiguous_human_referent" not in reasons
    assert "source_mapping_incomplete" in reasons
    assert "speaker_identity_count_unproved" in reasons
    assert "audience_fingerprint_missing" in reasons


def test_identity_classifier_accepts_current_single_actor_audience_proof():
    assert classify_identity_violation({
        "summary": "BigTex stopped MOTS-c after the appointment.",
        "metadata_json": _proved_metadata(),
    }, {
        "ct-1": {
            "canonical_turn_id": "ct-1",
            "sender_actor_id": "actor-a",
            "sender": "BigTex",
            "audience_conversation_id": "guild-1",
            "origin_channel_id": "channel-1",
            "audience_attribution_version": 1,
            "reply_subject_actor_id": "",
            "reply_subject_label": "",
            "has_user_content": True,
            "has_assistant_content": True,
            "has_reply_target_body": False,
        },
    }) == ()


def test_identity_classifier_reports_lexical_and_multi_actor_damage():
    reasons = classify_identity_violation({
        "summary": "The user tolerates Tesamorelin.",
        "metadata_json": _proved_metadata(
            source_speaker_identity_count=2,
            source_speaker_labels=["BigTex", "Kuw9239"],
        ),
    })
    assert "summary_ambiguous_human_referent" in reasons
    assert "speaker_identity_not_single" in reasons


def test_tag_summary_classifier_names_only_violating_prose_fields():
    assert classify_tag_summary_identity_violation({
        "summary": "The user changed medication.",
        "description": "Medication timeline for BigTex.",
    }) == ("summary", "description")


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


class _Rows:
    def __init__(self, *, one=None, many=None):
        self._one = one
        self._many = many

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _IdentityAuditConnection:
    def __init__(self):
        self.statements = []
        self.segment_rows = [
            {
                "ref": "safe", "summary": "BigTex changed medication.",
                "metadata_json": _proved_metadata(), "tags": ["health"],
            },
            {
                "ref": "ambiguous", "summary": "The user changed medication.",
                "metadata_json": _proved_metadata(), "tags": ["health"],
            },
            {
                "ref": "unproved", "summary": "Stopped after the visit.",
                "metadata_json": {}, "tags": ["appointments"],
            },
        ]
        self.tag_rows = [
            {
                "tag": "health", "summary": "The user changed medication.",
                "description": "BigTex health history",
                "source_segment_refs": '["ambiguous"]',
            },
            {
                "tag": "unlinked", "summary": "Kuw9239 discussed sleep.",
                "description": "She later changed the dose.",
                "source_segment_refs": '["safe"]',
            },
        ]
        self.source_rows = [{
            "canonical_turn_id": "ct-1",
            "sender_actor_id": "actor-a",
            "sender": "BigTex",
            "audience_conversation_id": "guild-1",
            "origin_channel_id": "channel-1",
            "audience_attribution_version": 1,
            "reply_subject_actor_id": "",
            "reply_subject_label": "",
            "has_user_content": True,
            "has_assistant_content": True,
            "has_reply_target_body": False,
        }]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if sql.startswith("SELECT count(*) AS n, md5") and "FROM segments" in sql:
            return _Rows(one={"n": 3, "digest": "segments"})
        if sql.startswith("SELECT count(*) AS n, max(updated_at)"):
            return _Rows(one={"n": 1, "max_updated": "now"})
        if sql.startswith("SELECT count(*) AS n, md5") and "FROM tag_summaries" in sql:
            return _Rows(one={"n": 2, "digest": "tags"})
        if sql.startswith("SELECT ref, conversation_id"):
            return _Rows(many=self.segment_rows)
        if sql.startswith("SELECT canonical_turn_id, sender_actor_id"):
            return _Rows(many=self.source_rows)
        if sql.startswith("SELECT tag, summary, description"):
            return _Rows(many=self.tag_rows)
        raise AssertionError(f"unexpected SQL: {sql}")


def _identity_args(**updates):
    values = {
        "conversation_id": "guild-1",
        "tenant_id": "tenant-1",
        "identity_violations": True,
        "apply": False,
        "postgres_dsn": "postgresql://unused",
        "since": None,
        "until": None,
        "after_ref": None,
        "include_short": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_identity_dry_run_is_read_only_and_reports_segments_and_tag_fields(
    monkeypatch, capsys,
):
    from virtual_context.cli import resummarize_cmd

    connection = _IdentityAuditConnection()
    connection_modes = []

    def fake_connect(_dsn, *, read_only):
        connection_modes.append(read_only)
        return connection

    monkeypatch.setattr(resummarize_cmd, "_connect", fake_connect)
    resummarize_cmd.cmd_admin_resummarize_segments(_identity_args())
    report = json.loads(capsys.readouterr().out)

    assert connection_modes == [True]
    assert report["status"] == "dry_run"
    assert report["mode"] == "identity_violations"
    assert report["server_enforced_read_only"] is True
    assert report["scanned_segments"] == 3
    assert report["selected"] == 2
    assert [row["ref"] for row in report["segments"]] == [
        "ambiguous", "unproved",
    ]
    assert report["reason_counts"]["summary_ambiguous_human_referent"] == 1
    assert report["reason_counts"]["source_mapping_incomplete"] == 1
    assert report["affected_tags"] == ["appointments", "health"]
    assert report["tag_summaries"]["rows_with_field_violations"] == 2
    assert report["tag_summaries"]["field_violation_counts"] == {
        "description": 2, "summary": 2,
    }
    assert report["tag_summaries"]["affected_prose_field_counts"] == {
        "description": 1, "summary": 1,
    }
    assert report["checksums_stable"] is True
    assert not any(
        sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE"))
        for sql, _params in connection.statements
    )


def test_identity_apply_is_refused_before_database_or_engine_access(
    monkeypatch, capsys,
):
    from virtual_context.cli import resummarize_cmd

    monkeypatch.setattr(
        resummarize_cmd, "_connect",
        lambda *_args, **_kwargs: pytest.fail("database must not be opened"),
    )
    with pytest.raises(SystemExit) as exc:
        resummarize_cmd.cmd_admin_resummarize_segments(
            _identity_args(apply=True),
        )
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "identity_violations_apply"
    assert "exact audience-scoped canonical source rows" in report["error"]
    assert "rerun without --apply" in report["error"]


def test_default_mode_still_dispatches_unchanged_prefix_dry_run(monkeypatch):
    from virtual_context.cli import resummarize_cmd

    calls = []
    monkeypatch.setattr(resummarize_cmd, "_dry_run", lambda args: calls.append(args))
    monkeypatch.setattr(
        resummarize_cmd, "_identity_dry_run",
        lambda _args: pytest.fail("identity mode must be explicit"),
    )
    args = _identity_args(identity_violations=False)
    resummarize_cmd.cmd_admin_resummarize_segments(args)
    assert calls == [args]
