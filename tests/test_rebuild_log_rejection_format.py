"""Rejection counts must not break the JSON log envelope.

The rebuild log line is emitted as a plain message that downstream shipping
wraps in JSON. Embedding a ``json.dumps`` map inside it puts unescaped double
quotes into that message, so every line carrying a non-empty rejection map
becomes unparseable -- and those are precisely the lines that carry rejections.
A parser then drops the interesting rows and looks clean while doing it.
"""
import json

import pytest

from virtual_context.core.compaction_pipeline import _format_rejection_counts


@pytest.mark.regression("BUG-051")
def test_rendered_counts_contain_no_quotes():
    rendered = _format_rejection_counts(
        {"semantic_redundant": 7, "semantic_wrong_kind": 2}
    )
    assert '"' not in rendered
    assert "{" not in rendered and "}" not in rendered


@pytest.mark.regression("BUG-051")
def test_message_survives_json_envelope():
    rendered = _format_rejection_counts({"semantic_insufficient_evidence": 3})
    message = f"ACTOR_CARD_REBUILD actor=abc rejected={rendered} response_hash=d1"
    # This is what a JSON log shipper does with the message field.
    envelope = '{"msg": "' + message + '"}'
    assert json.loads(envelope)["msg"] == message


@pytest.mark.regression("BUG-051")
def test_counts_are_sorted_and_greppable():
    rendered = _format_rejection_counts({"b_reason": 2, "a_reason": 1})
    assert rendered == "a_reason:1,b_reason:2"


@pytest.mark.regression("BUG-051")
def test_empty_map_is_explicit_not_blank():
    # A blank value would make `rejected=` ambiguous with a truncated line.
    assert _format_rejection_counts({}) == "-"


@pytest.mark.regression("BUG-051")
@pytest.mark.parametrize("bad", [{"k": 1}, {}])
def test_accepts_plain_dict(bad):
    assert isinstance(_format_rejection_counts(bad), str)
