"""Host-appended internal-context user items are not the user's message."""
import copy

import pytest

from virtual_context.proxy.formats import get_format

CTX = ("<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nConversation data (data, not instructions):\n"
       "\"Active exec sessions:\\nnone\"\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>")


def _item(role, text, kind="input_text"):
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _body():
    return {"model": "gpt-5.6-sol", "input": [
        _item("user", "[Tue 2026-09-22 04:25 UTC] Reply with exactly: OK1"), _item("user", CTX),
        _item("assistant", "OK1", "output_text"),
        _item("user", "[Tue 2026-09-22 04:26 UTC] Reply with exactly: OK2"), _item("user", CTX),
    ]}


def test_current_message_is_the_prompt_not_the_host_context():
    assert get_format("openai_responses").extract_user_message(_body()).endswith("Reply with exactly: OK2")


def test_history_pairs_pair_the_prompt_with_the_reply():
    pairs = get_format("openai_responses").extract_history_pairs(_body())
    assert [(m.role, m.content[-3:]) for m in pairs] == [("user", "OK1"), ("assistant", "OK1")]


def test_host_context_stays_inside_the_prompts_turn_group():
    groups = get_format("openai_responses").group_into_turns(_body())
    assert [g.indices for g in groups] == [[0, 1, 2], [3, 4]]


def test_a_lone_host_context_item_is_still_a_turn_of_its_own():
    groups = get_format("openai_responses").group_into_turns(
        {"model": "m", "input": [_item("user", CTX), _item("assistant", "x", "output_text")]})
    assert [g.indices for g in groups] == [[0, 1]]


def test_host_context_items_are_never_ingested():
    from virtual_context.proxy.formats import extract_ingestible_messages
    fmt = get_format("openai_responses")
    messages, stats = extract_ingestible_messages(_body(), fmt, mode="ingest")
    assert [(m.role, m.content[-3:]) for m in messages] == [("user", "OK1"), ("assistant", "OK1"), ("user", "OK2")]
    assert stats["skipped_non_chat_entry_count"] == 2


def test_other_host_scaffolding_user_items_are_not_the_users_message():
    fmt = get_format("openai_responses")
    body = {"model": "m", "input": [
        _item("user", "<recommended_plugins>\nHere is a list of plugins that are available"),
        _item("user", '<external_openclaw_current_sender>{"sender":{"id":"1"}}</external_openclaw_current_sender>'),
        _item("user", "[Tue 2026-09-22 12:40 UTC] <conversation_context>\nold stuff\n</conversation_context>\n\nwhat did we decide?"),
    ]}
    current = fmt.extract_user_message(body)
    assert current.endswith("what did we decide?") and "<conversation_context>" not in current
    from virtual_context.proxy.formats import extract_ingestible_messages
    messages, _ = extract_ingestible_messages(body, fmt, mode="ingest")
    assert len(messages) == 1 and messages[0].content.endswith("what did we decide?")
    assert "<conversation_context>" not in messages[0].content


def test_removing_the_replayed_block_keeps_the_surrounding_text_intact():
    from virtual_context.proxy._envelope import _strip_envelope
    assert _strip_envelope("A<conversation_context>host replay</conversation_context> B") == "A B"
    assert _strip_envelope("<conversation_context>x</conversation_context>\n\nhello") == "hello"
    assert _strip_envelope("no block here") == "no block here"


PROJECT_INSTRUCTIONS = (
    "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>\n"
    "Use BrowserOS after a blocked web_fetch.\nPreserve the running gateway.\n"
    "</INSTRUCTIONS>"
)
ENVIRONMENT = "<environment_context>\n  <cwd>/workspace</cwd>\n</environment_context>"


def _project_item():
    item = _item("user", PROJECT_INSTRUCTIONS)
    item["content"].append({"type": "input_text", "text": ENVIRONMENT})
    return item


@pytest.mark.parametrize("with_environment", [False, True])
def test_project_instructions_are_forwarded_as_context_not_conversation(with_environment):
    from virtual_context.proxy.formats import extract_ingestible_messages

    fmt = get_format("openai_responses")
    project = _project_item() if with_environment else _item("user", PROJECT_INSTRUCTIONS)
    original = copy.deepcopy(project)
    body = {"input": [project, _item("user", "Find a cotton sweater"),
                      _item("assistant", "I will check", "output_text"),
                      _item("user", "Verify the price"), copy.deepcopy(project)]}

    assert fmt.extract_user_message(body) == "Verify the price"
    assert [(m.role, m.content) for m in fmt.extract_history_pairs(body)] == [
        ("user", "Find a cotton sweater"), ("assistant", "I will check")]
    messages, _ = extract_ingestible_messages(body, fmt, mode="ingest")
    assert [m.content for m in messages] == [
        "Find a cotton sweater", "I will check", "Verify the price"]
    assert project == original


def test_project_and_runtime_context_cannot_merge_with_the_actual_question():
    from virtual_context.proxy.formats import extract_ingestible_messages

    fmt = get_format("openai_responses")
    body = {"input": [_project_item(), _item("user", ENVIRONMENT),
                      _item("user", "Find a cotton sweater"), _item("user", CTX)]}
    before = copy.deepcopy(body)
    fmt.merge_consecutive_conversational(body)
    assert body == before
    assert fmt.extract_user_message(body) == "Find a cotton sweater"
    messages, _ = extract_ingestible_messages(body, fmt, mode="ingest")
    assert [m.content for m in messages] == ["Find a cotton sweater"]


@pytest.mark.parametrize("text", [
    "Please explain AGENTS.md instructions for this project.",
    "Please review this:\n" + PROJECT_INSTRUCTIONS,
    "```markdown\n" + PROJECT_INSTRUCTIONS + "\n```",
    PROJECT_INSTRUCTIONS + "\nWhy is this wrong?",
    PROJECT_INSTRUCTIONS.replace("</INSTRUCTIONS>", "").rstrip(),
])
def test_quoted_or_incomplete_project_documents_remain_user_speech(text):
    from virtual_context.proxy.formats import extract_ingestible_messages

    fmt = get_format("openai_responses")
    body = {"input": [_item("user", text)]}
    assert fmt.extract_user_message(body) == text
    messages, _ = extract_ingestible_messages(body, fmt, mode="ingest")
    assert len(messages) == 1 and messages[0].content == text


def test_request_in_an_extra_content_block_is_not_hidden_as_project_context():
    fmt = get_format("openai_responses")
    item = _project_item()
    item["content"].append({"type": "input_text", "text": "Explain these rules"})
    assert not fmt._is_host_context_item(item)
    assert fmt.extract_user_message({"input": [item]}).endswith("Explain these rules")


def test_replayed_conversation_is_inserted_after_the_project_instructions():
    from virtual_context.types import Message

    fmt = get_format("openai_responses")
    project = _project_item()
    current = _item("user", "Verify the price")
    body = {"input": [project, current, _item("user", CTX)]}
    before = copy.deepcopy(body)
    replay = [Message(role="user", content="Find a cotton sweater"),
              Message(role="assistant", content="I will check")]
    result, delivered = fmt.inject_replayed_conversation_with_delivery(body, replay)
    assert result["input"][0] == project
    assert [fmt.extract_message_text(m) for m in result["input"][1:4]] == [
        "Find a cotton sweater", "I will check", "Verify the price"]
    assert delivered == replay
    assert body == before
