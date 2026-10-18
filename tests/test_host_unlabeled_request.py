"""A host prompt whose request follows the selected room history without a label (BUG-135).

A reused host thread appends the turn as metadata blocks, the selected
room-history section, a blank line and the request, with no "Current user
request:" label. The whole prompt was stored as the user's message.
"""
from __future__ import annotations

import json

import pytest

from virtual_context.core.raw_bodies import HOST_CONTEXT_OMITTED_LABEL, compact_raw_body
from virtual_context.proxy._envelope import _extract_envelope_metadata

HISTORY = "".join(
    f'#session:s{n} 2026-10-18 19:0{n}:00 UTC {{"id":"8049932331","name":"Y","channel":"telegram"}}: line {n} '
    + "x" * 600 + "\n"
    for n in range(9)
)
PROMPT = (
    "Conversation info: ⟦openclaw:ctx⟧\n```json\n"
    '{"chat_id":"telegram:-1","message_id":"24056","sender":{"id":"8049932331","name":"Y"}}\n```\n\n'
    "Conversation context (chronological, selected for current message): ⟦openclaw:ctx⟧\n"
    + HISTORY
    + "\nFinish the check"
)


@pytest.mark.regression("BUG-135")
def test_unlabeled_request_is_the_users_text():
    text, metadata = _extract_envelope_metadata(PROMPT)
    assert text == "Finish the check"
    assert metadata["conversation info"]["message_id"] == "24056"


@pytest.mark.regression("BUG-135")
def test_unlabeled_request_raw_body_omits_the_replayed_history():
    raw = json.dumps([{"type": "input_text", "text": PROMPT}])
    stored = json.loads(compact_raw_body(raw).text)[0]["text"]
    assert "#session:" not in stored
    assert HOST_CONTEXT_OMITTED_LABEL in stored
    assert stored.startswith("Conversation info: ⟦openclaw:ctx⟧")
    assert stored.endswith("Finish the check")


@pytest.mark.regression("BUG-135")
def test_selected_history_with_nothing_after_it_is_left_alone():
    prompt = PROMPT[: PROMPT.rindex("\nFinish the check")]
    text, _ = _extract_envelope_metadata(prompt)
    assert "line 8" in text
