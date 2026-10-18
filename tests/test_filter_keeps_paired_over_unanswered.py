"""The history filter keeps a protected question over an unanswered one.

When filtering left an unanswered user message (never paired with a reply)
directly before a kept question, role alternation dropped the later message,
so a protected turn lost its question and its reply followed the stale one.
"""

from __future__ import annotations

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import get_format
from virtual_context.proxy.message_filter import filter_body_messages
from virtual_context.types import TurnTagEntry


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _texts(body):
    return [(i["role"], "\n".join(part["text"] for part in i["content"]))
            for i in body["input"] if i.get("role") in ("user", "assistant")]


@pytest.mark.regression("BUG-105")
def test_an_unanswered_message_does_not_displace_a_protected_question():
    items = [_msg("user", "old q"), _msg("assistant", "old a"), _msg("user", "unanswered"),
             _msg("user", "q1"), _msg("assistant", "a1"), _msg("user", "q2"), _msg("assistant", "a2"),
             _msg("user", "current")]
    index = TurnTagIndex()
    for n in range(4):
        index.append(TurnTagEntry(turn_number=n, message_hash=f"h{n}", tags=["other"], primary_tag="other"))
    out, _dropped = filter_body_messages(
        {"model": "m", "input": items}, index, ["deadlift"], recent_turns=2,
        fmt=get_format("openai_responses"),
    )
    kept = _texts(out)
    asks = [n for n, (role, text) in enumerate(kept) if role == "user" and text.endswith("q1")]
    assert asks and kept[asks[0] + 1] == ("assistant", "a1")
    assert ("user", "q2") in kept
