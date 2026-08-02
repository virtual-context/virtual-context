"""``vc_find_quote`` bounded by when a message was sent.

The window is proved from ``source_message_id``, so a result that cannot
prove its send time is excluded rather than assumed to be inside. A
malformed bound is refused rather than dropped: silently ignoring an
unparseable date would answer a different question than the one asked, which
is the same failure shape as a silently ignored speaker selection.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from virtual_context.core.discord_snowflake import datetime_to_snowflake_floor
from virtual_context.core.tool_loop import execute_vc_tool, vc_tool_definitions
from virtual_context.types import QuoteResult, SearchConfig, SourceProvenance

CONV = "conv-window"
NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def _id_at(moment: datetime) -> str:
    return str(datetime_to_snowflake_floor(moment) + 1)


OLD_ID = _id_at(NOW - timedelta(days=10))
RECENT_ID = _id_at(NOW - timedelta(days=1))


def _result(text: str, message_id: str) -> QuoteResult:
    return QuoteResult(
        text=text, tag="t", segment_ref=f"ref-{message_id or 'none'}",
        match_type="fts", source_scope="turn", matched_side="user",
        provenance=SourceProvenance(
            conversation_id=CONV,
            canonical_turn_id=f"ct-{message_id or 'none'}",
            source_role="requester",
            actor_id="actor:discord:1",
            audience_conversation_id=CONV,
            audience_attribution_version=1,
            source_message_id=message_id,
        ),
    )


class _Engine:
    """Records what find_quote was handed, and returns fixed candidates."""

    def __init__(self):
        self.seen: dict = {}
        self.config = SimpleNamespace(
            search=SearchConfig(tool_guard_enabled=False),
            conversation_id=CONV,
        )
        self._store = SimpleNamespace()
        self._semantic = SimpleNamespace()

    def find_quote(self, query, **kwargs):
        self.seen = dict(kwargs)
        return {
            "found": True,
            "results": [
                _result("ten days ago", OLD_ID),
                _result("yesterday", RECENT_ID),
                _result("no provable time", ""),
            ],
        }


def _call(engine, **args):
    return json.loads(execute_vc_tool(
        engine, "vc_find_quote", {"query": "dose", **args},
    ))


def test_schema_advertises_the_window():
    definition = next(
        d for d in vc_tool_definitions() if d["name"] == "vc_find_quote"
    )
    props = definition["input_schema"]["properties"]
    assert "after" in props and "before" in props
    for key in ("after", "before"):
        assert props[key]["type"] == "string"
        assert "YYYY-MM-DD" in props[key]["description"]


def test_malformed_after_is_refused_not_ignored():
    engine = _Engine()
    got = _call(engine, after="last tuesday")
    assert "error" in got
    assert "after" in got["error"]
    assert engine.seen == {}, "the search ran despite an unparseable bound"


def test_malformed_before_is_refused_not_ignored():
    engine = _Engine()
    got = _call(engine, before="2026-13-45")
    assert "error" in got
    assert "before" in got["error"]
    assert engine.seen == {}


def test_inverted_window_is_refused():
    engine = _Engine()
    got = _call(engine, after="2026-08-01", before="2026-07-01")
    assert "error" in got
    assert engine.seen == {}


def test_valid_bounds_reach_the_search():
    engine = _Engine()
    _call(engine, after="2026-07-26", before="2026-08-02")
    assert engine.seen["after"] == datetime(
        2026, 7, 26, tzinfo=timezone.utc,
    )
    assert engine.seen["before"] == datetime(
        2026, 8, 2, tzinfo=timezone.utc,
    )


def test_no_bounds_leaves_the_call_unchanged():
    engine = _Engine()
    _call(engine)
    assert "after" not in engine.seen
    assert "before" not in engine.seen


def test_full_iso_timestamps_are_accepted():
    engine = _Engine()
    _call(engine, after="2026-07-26T13:45:00+00:00")
    assert engine.seen["after"] == datetime(
        2026, 7, 26, 13, 45, tzinfo=timezone.utc,
    )


def test_naive_iso_is_read_as_utc():
    engine = _Engine()
    _call(engine, after="2026-07-26T00:00:00")
    assert engine.seen["after"].tzinfo is timezone.utc


@pytest.mark.parametrize("bad", ["", "   ", "2026/07/26", "26-07-2026", "x"])
def test_every_unparseable_form_is_refused(bad):
    engine = _Engine()
    got = _call(engine, after=bad)
    assert "error" in got
    assert engine.seen == {}


def _window(results, *, after=None, before=None):
    from virtual_context.core.quote_search import _within_send_window

    return _within_send_window(results, after=after, before=before)


def test_filter_drops_results_outside_the_window():
    results = [_result("old", OLD_ID), _result("recent", RECENT_ID)]
    kept = _window(results, after=NOW - timedelta(days=7))
    assert [r.text for r in kept] == ["recent"]


def test_filter_drops_results_that_cannot_prove_a_send_time():
    """A segment-backed hit has no message id, so it cannot be placed."""
    results = [_result("recent", RECENT_ID), _result("no id", "")]
    kept = _window(results, after=NOW - timedelta(days=30))
    assert [r.text for r in kept] == ["recent"]

    no_prov = QuoteResult(text="segment hit", tag="t", segment_ref="seg-1")
    assert _window([no_prov], before=NOW) == []


def test_filter_is_inclusive_at_both_bounds():
    from virtual_context.core.discord_snowflake import snowflake_to_datetime

    sent = snowflake_to_datetime(RECENT_ID)
    results = [_result("recent", RECENT_ID)]
    assert _window(results, after=sent) == results
    assert _window(results, before=sent) == results
    assert _window(results, after=sent, before=sent) == results


def test_filter_is_a_no_op_shape_when_only_one_bound_is_given():
    results = [_result("old", OLD_ID), _result("recent", RECENT_ID)]
    assert len(_window(results, before=NOW)) == 2
    assert len(_window(results, after=NOW - timedelta(days=365))) == 2
