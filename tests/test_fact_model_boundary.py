"""Generated Fact prose is never model evidence.

Facts remain useful as retrieval indexes, but their model-written semantic
fields cannot be attributed merely because a row also carries an actor id.
The prompt and tool boundaries therefore expose no Fact prose.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.tool_loop import execute_vc_tool
from virtual_context.types import (
    AssemblerConfig,
    Fact,
    RetrievalResult,
    SearchConfig,
    SpeakerRetrievalContext,
)


ACTOR_ID = "actor:discord:private-123"
MALICIOUS_PROSE = "the user has disease"


def _fact() -> Fact:
    return Fact(
        id="fact-private-1",
        subject="the user",
        verb="has",
        object="disease",
        what=MALICIOUS_PROSE,
        who="the user",
        why="a generated explanation",
        tags=["health"],
        segment_ref="seg-source-1",
        conversation_id="owner-conversation",
        author_actor_id=ACTOR_ID,
        author_attribution_version=2,
        author_source_role="requester",
    )


def _eligible_context() -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="tenant-1",
        owner_conversation_id="owner-conversation",
        audience_conversation_id="audience-conversation",
    )


def _engine_with_fact() -> MagicMock:
    engine = MagicMock()
    engine.config = SimpleNamespace(
        conversation_id="owner-conversation",
        search=SearchConfig(tool_guard_enabled=False),
    )
    engine.query_facts.return_value = {
        "facts": [_fact()],
        "linked_facts": [],
        "total_all_statuses": 1,
        "all_statuses": {"active": 1},
    }
    return engine


def test_initial_context_withholds_fact_prose_even_with_actor_identity() -> None:
    retrieval = RetrievalResult(facts=[_fact()], retrieval_metadata={})
    assembler = ContextAssembler(
        config=AssemblerConfig(facts_max_tokens=1_000),
    )

    assembled = assembler.assemble(
        "",
        retrieval,
        [],
        token_budget=10_000,
        speaker_context=_eligible_context(),
    )

    assert assembled.facts_text == ""
    assert assembled.selected_facts == []
    assert MALICIOUS_PROSE not in assembled.prepend_text
    assert ACTOR_ID not in assembled.prepend_text
    assert assembled.budget_breakdown["facts"] == 0
    assert retrieval.retrieval_metadata["facts_block"] == {
        "candidates": 1,
        "selected": 0,
        "rendered": 0,
        "trimmed": 0,
        "withheld": 1,
        "tokens": 0,
        "cap": 0,
        "configured_cap": 1_000,
        "policy": "derived_fact_prose_not_model_evidence",
    }


def test_query_facts_returns_only_structural_payload_for_proved_request() -> None:
    engine = _engine_with_fact()

    raw = execute_vc_tool(
        engine,
        "vc_query_facts",
        {},
        speaker_context=_eligible_context(),
    )
    payload = json.loads(raw)

    assert payload == {
        "count": 1,
        "facts": [{
            "id": "fact-private-1",
            "segment_ref": "seg-source-1",
            "tags": ["health"],
        }],
        "fact_content_withheld": True,
        "content_policy": "derived_fact_prose_not_model_evidence",
    }
    assert MALICIOUS_PROSE not in raw
    assert "disease" not in raw
    assert ACTOR_ID not in raw
    assert not ({
        "subject", "verb", "object", "status", "fact_type", "what",
        "who", "when", "where", "why", "conversation_id",
    } & payload["facts"][0].keys())


@pytest.mark.parametrize(
    "speaker_context",
    [None, SpeakerRetrievalContext.ineligible()],
)
def test_query_facts_without_proved_audience_fails_closed(
    speaker_context: SpeakerRetrievalContext | None,
) -> None:
    engine = _engine_with_fact()

    raw = execute_vc_tool(
        engine,
        "vc_query_facts",
        {},
        speaker_context=speaker_context,
    )

    assert json.loads(raw) == {
        "count": 0,
        "facts": [],
        "fact_content_withheld": True,
        "content_policy": "derived_fact_prose_not_model_evidence",
        "request_scope": "ineligible",
    }
    assert MALICIOUS_PROSE not in raw
    assert ACTOR_ID not in raw
    engine.query_facts.assert_not_called()
