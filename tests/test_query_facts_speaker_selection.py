"""Speaker selection on the fact-retrieval tool.

``vc_query_facts`` advertises a ``speaker`` input bound to the request's
roster, but its execution read only the structured filters and dropped the
selection. A reader that asked for one participant's facts therefore
received every participant's facts, with no speaker fields and no signal
that the selection had been ignored — the reader then attributed all of
them to the participant it had asked about.

Selection is now resolved with the same validated helper the quote path
uses. A fact exposes an author only when its attribution is role-local, so
model-assisted facts cannot be filtered on; that case fails open on the
results but never silently — the response says the facts are NOT
attributable to the selected participant.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from types import SimpleNamespace

import pytest

from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import (
    SpeakerRetrievalContext,
    SpeakerRosterEntry,
    SpeakerRosterSnapshot,
)


ROO = "actor:discord:111"
OTHER = "actor:discord:222"
TENANT = "t1"
AUDIENCE = "sk:agent:a:discord:guild:9"


def _fact(fact_id: str, subject: str, actor: str, *, version: int, role: str):
    return SimpleNamespace(
        id=fact_id, subject=subject, verb="says", object="x", status="active",
        fact_type="personal", what=f"{subject} fact", who="", when_date="",
        where="", why="", conversation_id=AUDIENCE, tags=[],
        author_actor_id=actor,
        author_attribution_version=version,
        author_source_role=role,
    )


def _snapshot():
    return SpeakerRosterSnapshot(
        snapshot_id="snap-1",
        entries=(
            SpeakerRosterEntry(handle="roo", name="Roo", actor_id=ROO),
            SpeakerRosterEntry(handle="bigtex", name="BigTex", actor_id=OTHER),
        ),
        truncated=False,
        tenant_id=TENANT,
        audience_conversation_id=AUDIENCE,
        lifecycle_epoch=1,
    )


def _context():
    return SpeakerRetrievalContext(
        tenant_id=TENANT,
        owner_conversation_id=AUDIENCE,
        audience_conversation_id=AUDIENCE,
        audience_channel_id="",
        requester_actor_id=OTHER,
        original_active_user_text="what did roo say",
        roster_snapshot_id="snap-1",
    )


class _Engine:
    def __init__(self, facts):
        self._facts_list = facts
        self.config = SimpleNamespace(
            search=SimpleNamespace(
                speaker_selection_enabled=True,
                speaker_annotations_enabled=True,
                find_quote_max_results=10,
            ),
            facts=SimpleNamespace(graph_links=False),
            conversation_id=AUDIENCE,
            tenant_id=TENANT,
        )
        self._store = SimpleNamespace(
            get_lifecycle_epoch=lambda _c: 1,
            get_actor_profile=lambda _t, actor: SimpleNamespace(
                display_name={ROO: "Roo", OTHER: "BigTex"}.get(actor, ""),
            ),
        )

    def query_facts(self, **kwargs):
        return {"facts": list(self._facts_list)}


def _run(engine, tool_input):
    return execute_vc_tool(
        engine,
        "vc_query_facts",
        tool_input,
        speaker_context=_context(),
        roster_snapshot=_snapshot(),
    )


def _payload(result):
    return json.loads(result) if isinstance(result, str) else result


def test_unattributed_facts_disclose_that_no_filter_applied():
    """The regression: a selection over model-assisted facts must NOT return
    them as if they were the selected participant's."""
    engine = _Engine([
        _fact("f1", "retatrutide", "", version=1, role=""),
        _fact("f2", "MOTS-c", "", version=1, role=""),
    ])
    payload = _payload(_run(engine, {"speaker": "roo"}))
    assert payload["count"] == 2, payload
    assert payload["filter_applied"] is False
    assert "speaker_selection_note" in payload
    assert "NOT attributable" in payload["speaker_selection_note"]


def test_speaker_only_filters_role_local_facts():
    engine = _Engine([
        _fact("f1", "roo topic", ROO, version=2, role="requester"),
        _fact("f2", "other topic", OTHER, version=2, role="requester"),
    ])
    payload = _payload(_run(engine, {"speaker": "roo", "speaker_only": True}))
    assert payload["count"] == 1, payload
    assert payload["facts"][0]["subject"] == "roo topic"
    assert payload["filter_applied"] is True
    assert payload["excluded_other_speakers"] == 1
    assert "speaker_selection_note" not in payload


def test_speaker_hint_ranks_without_dropping_other_facts():
    engine = _Engine([
        _fact("f1", "other topic", OTHER, version=2, role="requester"),
        _fact("f2", "roo topic", ROO, version=2, role="requester"),
    ])
    payload = _payload(_run(engine, {"speaker": "roo"}))
    assert payload["count"] == 2, payload
    assert payload["facts"][0]["subject"] == "roo topic"
    assert payload["filter_applied"] is False


def test_no_speaker_selection_leaves_response_unchanged():
    engine = _Engine([
        _fact("f1", "retatrutide", "", version=1, role=""),
    ])
    payload = _payload(_run(engine, {"subject": "retatrutide"}))
    assert payload["count"] == 1
    assert "conditioning_source" not in payload
    assert "speaker_selection_note" not in payload
