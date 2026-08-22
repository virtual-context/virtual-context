"""Both fact-write log branches must name the segment ref.

`primary_tag` is not 1:1 with a segment ref -- one tag spans many segments and
a ref's primary tag can change between compactions -- so a log line naming
only the tag cannot be counted by segment. When `ref=` was added it went to
the replace branch and not to the store branch one line below, which left half
the fact-write log uncountable.

This is a structural assertion over the module's own logging calls rather than
a captured-log test: driving the pipeline far enough to emit these lines needs
a live store and a compaction lease, and the property worth pinning is that
NO branch of this decision logs without the ref -- including one added later.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "virtual_context" / "core" / "compaction_pipeline.py"


def _fact_write_log_calls() -> list[ast.Call]:
    """Every logger.* call whose format string reports a per-segment fact write."""
    tree = ast.parse(SRC.read_text())
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fmt = node.args[0] if node.args else None
        if not isinstance(fmt, ast.Constant) or not isinstance(fmt.value, str):
            continue
        text = fmt.value
        if "for segment" in text and ("Stored" in text or "Replaced" in text):
            found.append(node)
    return found


def test_both_branches_are_present():
    """If this drops to one, the other branch was deleted or reworded."""
    calls = _fact_write_log_calls()
    assert len(calls) == 2, [c.args[0].value for c in calls]


def test_every_fact_write_log_line_names_the_ref():
    """The property: no branch of this decision logs without the ref."""
    for call in _fact_write_log_calls():
        fmt = call.args[0].value
        assert "ref=%s" in fmt, f"log line does not name the segment ref: {fmt!r}"


def test_the_ref_placeholder_is_actually_supplied_an_argument():
    """A format string with ref=%s and no matching arg raises at log time."""
    for call in _fact_write_log_calls():
        fmt = call.args[0].value
        assert fmt.count("%s") + fmt.count("%d") == len(call.args) - 1, (
            f"placeholder/argument count mismatch in {fmt!r}"
        )


def test_the_ref_argument_is_the_segment_ref_not_the_tag():
    """Passing primary_tag twice would satisfy the count test and be wrong."""
    for call in _fact_write_log_calls():
        last = call.args[-1]
        name = getattr(last, "id", None) or getattr(last, "attr", None)
        assert name and "ref" in name.lower(), (
            f"last argument is {name!r}, expected the segment ref"
        )
