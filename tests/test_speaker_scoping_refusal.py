"""A tool that cannot scope by speaker must refuse the argument, not drop it.

Only ``vc_find_quote`` and ``vc_query_facts`` implement speaker scoping.
Every other tool used to accept ``speaker`` / ``speaker_only``, return 200,
apply no filtering and say nothing about it — so a caller asking "what did X
say" received every participant's material under X's question and attributed
it to X. An argument that errors is safe because the caller learns; an
argument that is silently dropped is a within-conversation attribution
failure. These tests pin the refusal and the routing advice that comes with
it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from virtual_context.core.tool_loop import (
    SPEAKER_SCOPING_TOOLS,
    VC_TOOL_NAMES,
    execute_vc_tool,
    vc_tool_definitions,
)
from virtual_context.types import SearchConfig

OWNER = "conv-refusal"

# Tools that take no speaker argument and must therefore refuse one.
NON_SCOPING_TOOLS = sorted(VC_TOOL_NAMES - SPEAKER_SCOPING_TOOLS)


class _Recorder:
    """Engine stub that records any tool body reached past the refusal."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.config = SimpleNamespace(
            search=SearchConfig(
                tool_guard_enabled=False,
                speaker_annotations_enabled=False,
                speaker_selection_enabled=True,
            ),
            conversation_id=OWNER,
        )
        self._store = SimpleNamespace()
        self._semantic = SimpleNamespace()

    def _record(self, name, *args, **kwargs):
        self.calls.append(name)
        return {"reached": name}

    def remember_when(self, **kwargs):
        return self._record("remember_when")

    def search_summaries(self, **kwargs):
        return self._record("search_summaries")

    def recall_all(self, **kwargs):
        return self._record("recall_all")

    def expand_topic(self, **kwargs):
        return self._record("expand_topic")

    def find_session(self, **kwargs):
        return self._record("find_session")

    def query_facts(self, **kwargs):
        return self._record("query_facts")

    def find_quote(self, *args, **kwargs):
        return self._record("find_quote")


@pytest.mark.regression("BUG-049")
@pytest.mark.parametrize("name", NON_SCOPING_TOOLS)
@pytest.mark.parametrize(
    "arguments",
    [
        {"speaker": "roo", "speaker_only": True},
        {"speaker": "roo"},
        {"speaker_only": True},
    ],
    ids=["both", "speaker-only-arg", "speaker_only-alone"],
)
def test_non_scoping_tool_refuses_speaker_arguments(name, arguments):
    engine = _Recorder()
    payload = {"query": "peptides", "tag": "t", "ref": "r", **arguments}

    got = json.loads(execute_vc_tool(engine, name, payload))

    assert "error" in got, f"{name} did not refuse {sorted(arguments)}"
    assert name in got["error"]
    # The refusal must route the caller somewhere that actually scopes.
    for alternative in sorted(SPEAKER_SCOPING_TOOLS):
        assert alternative in got["error"]
    # Nothing ran: a refusal that still returns unscoped content is the bug.
    assert engine.calls == []


@pytest.mark.regression("BUG-049")
def test_remember_when_refusal_names_the_supported_tools():
    engine = _Recorder()
    got = json.loads(execute_vc_tool(
        engine,
        "vc_remember_when",
        {
            "query": "peptides",
            "time_range": {"kind": "relative", "preset": "last_30_days"},
            "speaker": "roo",
            "speaker_only": True,
        },
    ))
    assert "vc_remember_when" in got["error"]
    assert "vc_find_quote" in got["error"]
    assert "vc_query_facts" in got["error"]
    assert engine.calls == []


@pytest.mark.regression("BUG-049")
def test_non_scoping_tool_still_runs_without_speaker_arguments():
    """The refusal is scoped to the argument, not to the tool."""
    engine = _Recorder()
    got = json.loads(execute_vc_tool(
        engine,
        "vc_remember_when",
        {
            "query": "peptides",
            "time_range": {"kind": "relative", "preset": "last_30_days"},
        },
    ))
    assert "error" not in got
    assert engine.calls == ["remember_when"]


@pytest.mark.regression("BUG-049")
def test_explicit_null_speaker_is_not_a_request_to_scope():
    """A serializer that emits ``speaker: null`` has asked for nothing."""
    engine = _Recorder()
    got = json.loads(execute_vc_tool(
        engine,
        "vc_remember_when",
        {
            "query": "peptides",
            "time_range": {"kind": "relative", "preset": "last_30_days"},
            "speaker": None,
            "speaker_only": False,
        },
    ))
    assert "error" not in got
    assert engine.calls == ["remember_when"]


@pytest.mark.regression("BUG-049")
def test_scoping_tools_are_exactly_the_ones_that_read_the_arguments():
    """The refusal set is derived from execution, not maintained by hand."""
    assert SPEAKER_SCOPING_TOOLS == frozenset({
        "vc_find_quote", "vc_query_facts",
    })
    assert SPEAKER_SCOPING_TOOLS <= VC_TOOL_NAMES


@pytest.mark.regression("BUG-049")
def test_remember_when_description_states_it_cannot_scope_by_speaker():
    """The model can only route correctly if the schema tells the truth."""
    definition = next(
        d for d in vc_tool_definitions() if d["name"] == "vc_remember_when"
    )
    description = definition["description"]
    assert "speaker" in description.lower()
    assert "vc_find_quote" in description
    assert "vc_query_facts" in description
    # And it still advertises no speaker properties.
    assert "speaker" not in definition["input_schema"]["properties"]
    assert "speaker_only" not in definition["input_schema"]["properties"]
