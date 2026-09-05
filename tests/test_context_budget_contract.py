"""Serialized context budgets, including paging and recovered native rows."""

from types import SimpleNamespace

import pytest

from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.retrieval_assembler import RetrievalAssembler
from virtual_context.token_counter import create_token_counter
from virtual_context.types import (
    AssemblerConfig, Message, RequestRoles, RetrievalResult, VirtualContextConfig,
)


@pytest.fixture
def counter():
    pytest.importorskip("tiktoken")
    return create_token_counter("tiktoken")


@pytest.mark.parametrize("text", ["你" * 400, "🧭" * 400, "x = {};\n" * 400])
def test_core_uses_actual_tokenizer_with_hard_window_limit(counter, text):
    assembler = ContextAssembler(
        AssemblerConfig(core_context_max_tokens=100), token_counter=counter,
    )
    result = assembler.assemble(
        text, RetrievalResult(), [], token_budget=100,
    )
    assert 0 < counter(result.prepend_text) <= 100
    assert result.total_tokens == counter(result.prepend_text)


@pytest.mark.parametrize("headroom", [0, 1, 50])
def test_headroom_caps_core_and_whole_hint(counter, headroom):
    hint = "<context-topics>" + "hint " * 200 + "</context-topics>"
    assembler = ContextAssembler(AssemblerConfig(), token_counter=counter)
    result = assembler.assemble(
        "core " * 20, RetrievalResult(), [], token_budget=1000,
        context_hint=hint, max_context_tokens=headroom,
    )
    assert counter(result.prepend_text) <= headroom
    assert result.context_hint in ("", hint)
    assert sum(result.budget_breakdown.values()) == result.total_tokens


def test_final_count_includes_escaping_and_separators(counter):
    assembler = ContextAssembler(AssemblerConfig(), token_counter=counter)
    result = assembler.assemble(
        "<host-attribution>" * 80,
        RetrievalResult(), [Message(role="user", content="latest instruction")],
        token_budget=150, max_context_tokens=100,
    )
    assert counter(result.prepend_text) <= 100
    assert result.total_tokens == counter(result.prepend_text) + sum(
        counter(message.content) for message in result.conversation_history
    )
    assert result.total_tokens <= 150


def test_recovered_native_pair_is_charged_against_injection_headroom(counter):
    roles = RequestRoles(
        requester_actor_id="actor:discord:42",
        owner_conversation_id="owner",
        audience_conversation_id="owner",
        origin_channel_id="channel",
        audience_channel_id="channel",
    )
    history = [
        Message(
            role=role, content=content,
            metadata={
                "source": "db_recent", "canonical_turn_id": f"ct-{index}",
                "turn_number": index, "turn_group_number": 1,
                "origin_channel_id": "channel", "audience_conversation_id": "owner",
                "sender_actor_id": "actor:discord:42" if role == "user" else "",
                "sender": {"name": "Alice"},
            },
        )
        for index, (role, content) in enumerate([
            ("user", "Keep this exact instruction. " * 20),
            ("assistant", "Understood. " * 20),
        ])
    ]
    assembler = ContextAssembler(AssemblerConfig(), token_counter=counter)
    roomy = assembler.assemble(
        "", RetrievalResult(), history, token_budget=1000, request_roles=roles,
    )
    assert len(roomy.recent_conversation_messages) == 2
    tight = assembler.assemble(
        "core", RetrievalResult(), history, token_budget=1000,
        request_roles=roles, max_context_tokens=20,
    )
    assert tight.recent_conversation_messages == []
    assert counter(tight.prepend_text) + tight.recent_conversation_message_tokens <= 20


def test_paging_reassembly_retains_original_request_headroom(counter, monkeypatch):
    config = VirtualContextConfig(context_window=1000)
    assembler = ContextAssembler(config.assembler, token_counter=counter)
    monkeypatch.setattr(assembler, "load_core_context", lambda: "core " * 20)
    bridge = RetrievalAssembler(
        retriever=SimpleNamespace(retrieve=lambda **kwargs: RetrievalResult(
            retrieval_metadata={"tags_from_message": ["topic"]},
        )),
        assembler=assembler,
        monitor=SimpleNamespace(build_snapshot=lambda history: SimpleNamespace(
            total_tokens=0, budget_tokens=1000,
        )),
        paging=SimpleNamespace(working_set={}), store=None,
        turn_tag_index=None,
        engine_state=SimpleNamespace(
            compacted_prefix_messages=0, flushed_prefix_messages=0,
            history_offset=lambda *args, **kwargs: 0,
        ),
        fact_curator=None, config=config, token_counter=counter,
    )
    monkeypatch.setattr(bridge, "_get_recent_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, "_build_context_hint", lambda **kwargs: "")
    first = bridge.on_message_inbound("question", [], max_context_tokens=50)
    assert counter(first.prepend_text) <= 50
    monkeypatch.setattr(assembler, "load_core_context", lambda: "你" * 200)
    reassembled = bridge.reassemble_context()
    assert 0 < counter(reassembled) <= 50
