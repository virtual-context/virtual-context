"""Tests for engine.sync_turns_from_payload."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from contextlib import contextmanager

import pytest
from virtual_context.proxy.formats import detect_format


def _make_anthropic_messages():
    """Anthropic conversation with a tool chain + 'consortium staffing'."""
    return {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "What are the three layers?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read",
                 "input": {"path": "rfp.pdf"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "RFP text..."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": (
                    "They are mixing up three different layers:\n"
                    "- RFP technical response writing\n"
                    "- bid qualification / documentary proof\n"
                    "- consortium staffing / resource roles"
                )},
            ]},
            {"role": "user", "content": "That makes sense"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Glad that clarifies it."},
            ]},
        ],
    }


def _make_gemini_messages():
    """Gemini conversation using role: 'model' and 'contents' key."""
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "What is VC?"}]},
            {"role": "model", "parts": [
                {"text": "Virtual context manages your LLM memory."},
            ]},
        ],
    }


@pytest.fixture
def engine(tmp_path):
    """Create engine with SQLite store for testing."""
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig

    from virtual_context.types import TagGeneratorConfig
    config = VirtualContextConfig(
        conversation_id="test-conv-sync",
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "test.db")),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    eng = VirtualContextEngine(config=config)
    return eng


def _search_via_store(engine, query):
    """Search canonical turn text through the CompositeStore delegate.

    Validates that the CompositeStore.search_canonical_turn_text delegation works,
    not just the raw concrete store method.
    """
    store = engine._store
    _search = getattr(store, "search_canonical_turn_text", None)
    assert _search is not None, (
        "search_canonical_turn_text not reachable through store chain — "
        "CompositeStore delegation missing")
    return _search(query, conversation_id="test-conv-sync")


def test_sync_persists_tool_chain_text(engine):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced >= 2

    # The key test: 'consortium staffing' must be searchable
    # through the full store chain (guarded → composite → concrete)
    results = _search_via_store(engine, "consortium staffing")
    assert len(results) >= 1
    assert "consortium staffing" in results[0].text.lower()


def test_sync_idempotent(engine):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    first = engine.sync_turns_from_payload(body, fmt)
    assert first >= 2
    second = engine.sync_turns_from_payload(body, fmt)
    assert second == 0  # already stored


def test_sync_incremental(engine):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    engine.sync_turns_from_payload(body, fmt)

    body["messages"].extend([
        {"role": "user", "content": "What about scoring?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "The scoring criteria are..."},
        ]},
    ])
    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 1


def test_sync_gemini_format(engine):
    body = _make_gemini_messages()
    fmt = detect_format(body)

    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 1

    results = _search_via_store(engine, "virtual context")
    assert len(results) >= 1


def test_sync_openai_chat_format(engine):
    """OpenAI Chat format: messages key, role=assistant."""
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Explain virtual memory"},
            {"role": "assistant", "content": "Virtual memory maps logical addresses to physical RAM."},
            {"role": "user", "content": "How does paging work?"},
            {"role": "assistant", "content": "Pages are fixed-size blocks swapped between RAM and disk."},
        ],
    }
    fmt = detect_format(body)
    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 2

    results = _search_via_store(engine, "paging")
    assert len(results) >= 1


def test_sync_openai_responses_format(engine):
    """OpenAI Responses format: input key, input_text/output_text blocks."""
    body = {
        "model": "gpt-4o",
        "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "What is context compression?"},
            ]},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "Context compression reduces token usage by summarizing old turns."},
            ]},
        ],
    }
    fmt = detect_format(body)
    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 1

    results = _search_via_store(engine, "context compression")
    assert len(results) >= 1


def test_sync_responses_string_input(engine):
    """OpenAI Responses string shorthand has no completed turn — returns 0."""
    body = {"model": "gpt-4o", "input": "What is virtual context?"}
    fmt = detect_format(body)
    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 0  # no assistant response yet, nothing to persist


def test_sync_single_turn(engine):
    """Even a single user+assistant pair must be persisted."""
    body = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hi there"},
            ]},
        ],
    }
    fmt = detect_format(body)
    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 1


def test_sync_prefix_widening_prepends_older_turns(engine):
    initial = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Turn 2 user"},
            {"role": "assistant", "content": "Turn 2 assistant"},
            {"role": "user", "content": "Turn 3 user"},
            {"role": "assistant", "content": "Turn 3 assistant"},
        ],
    }
    widened = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Turn 1 user"},
            {"role": "assistant", "content": "Turn 1 assistant"},
            {"role": "user", "content": "Turn 2 user"},
            {"role": "assistant", "content": "Turn 2 assistant"},
            {"role": "user", "content": "Turn 3 user"},
            {"role": "assistant", "content": "Turn 3 assistant"},
        ],
    }

    assert engine.sync_turns_from_payload(initial, detect_format(initial)) == 2
    assert engine.sync_turns_from_payload(widened, detect_format(widened)) == 1

    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert [row.user_content for row in rows] == [
        "Turn 1 user",
        "Turn 2 user",
        "Turn 3 user",
    ]


def test_sync_no_overlap_append_preserves_existing_rows(engine):
    first_window = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Alpha user"},
            {"role": "assistant", "content": "Alpha assistant"},
            {"role": "user", "content": "Beta user"},
            {"role": "assistant", "content": "Beta assistant"},
        ],
    }
    second_window = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Gamma user"},
            {"role": "assistant", "content": "Gamma assistant"},
            {"role": "user", "content": "Delta user"},
            {"role": "assistant", "content": "Delta assistant"},
        ],
    }

    assert engine.sync_turns_from_payload(first_window, detect_format(first_window)) == 2
    assert engine.sync_turns_from_payload(second_window, detect_format(second_window)) == 2

    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert [row.user_content for row in rows] == [
        "Alpha user",
        "Beta user",
        "Gamma user",
        "Delta user",
    ]


def test_sync_uses_conversation_reconcile_lock(engine, monkeypatch):
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Lock me"},
            {"role": "assistant", "content": "Locked"},
        ],
    }
    calls: list[tuple[str, str]] = []

    @contextmanager
    def _lock(conversation_id: str):
        calls.append(("enter", conversation_id))
        try:
            yield
        finally:
            calls.append(("exit", conversation_id))

    monkeypatch.setattr(engine._store, "conversation_reconcile", _lock)

    assert engine.sync_turns_from_payload(body, detect_format(body)) == 1
    assert calls == [
        ("enter", "test-conv-sync"),
        ("exit", "test-conv-sync"),
    ]
