"""Tests for engine.sync_turns_from_payload."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from concurrent.futures import ThreadPoolExecutor
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


def _inner_store(engine):
    store = engine._store
    return getattr(store, "_store", store)


def _entry_texts(rows):
    payload = []
    for row in rows:
        if row.user_content:
            payload.append(("user", row.user_content))
        elif row.assistant_content:
            payload.append(("assistant", row.assistant_content))
    return payload


def test_sync_persists_tool_chain_text(engine):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced >= 5

    # The key test: 'consortium staffing' must be searchable
    # through the full store chain (guarded → composite → concrete)
    results = _search_via_store(engine, "consortium staffing")
    assert len(results) >= 1
    assert "consortium staffing" in results[0].text.lower()


def test_sync_idempotent(engine):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    first = engine.sync_turns_from_payload(body, fmt)
    assert first >= 5
    first_ids = [row.canonical_turn_id for row in engine._store.get_all_canonical_turns("test-conv-sync")]
    second = engine.sync_turns_from_payload(body, fmt)
    assert second == 0  # already stored
    second_ids = [row.canonical_turn_id for row in engine._store.get_all_canonical_turns("test-conv-sync")]
    assert second_ids == first_ids


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
    assert synced == 2


def test_sync_warns_but_succeeds_when_group_recompute_fails(engine, monkeypatch, caplog):
    body = _make_anthropic_messages()
    fmt = detect_format(body)

    inner = _inner_store(engine)
    monkeypatch.setattr(
        inner,
        "recompute_canonical_turn_groups",
        lambda conversation_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with caplog.at_level("WARNING"):
        synced = engine.sync_turns_from_payload(body, fmt)

    assert synced >= 5
    assert "CANONICAL_TURN_GROUP_RECOMPUTE_FAILED" in caplog.text


def test_sync_gemini_format(engine):
    body = _make_gemini_messages()
    fmt = detect_format(body)

    synced = engine.sync_turns_from_payload(body, fmt)
    assert synced == 2

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
    assert synced == 4

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
    assert synced == 2

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
    assert synced == 2


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

    assert engine.sync_turns_from_payload(initial, detect_format(initial)) == 4
    assert engine.sync_turns_from_payload(widened, detect_format(widened)) == 2

    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert _entry_texts(rows) == [
        ("user", "Turn 1 user"),
        ("assistant", "Turn 1 assistant"),
        ("user", "Turn 2 user"),
        ("assistant", "Turn 2 assistant"),
        ("user", "Turn 3 user"),
        ("assistant", "Turn 3 assistant"),
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

    assert engine.sync_turns_from_payload(first_window, detect_format(first_window)) == 4
    assert engine.sync_turns_from_payload(second_window, detect_format(second_window)) == 4

    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert _entry_texts(rows) == [
        ("user", "Alpha user"),
        ("assistant", "Alpha assistant"),
        ("user", "Beta user"),
        ("assistant", "Beta assistant"),
        ("user", "Gamma user"),
        ("assistant", "Gamma assistant"),
        ("user", "Delta user"),
        ("assistant", "Delta assistant"),
    ]


def test_sync_no_overlap_append_logs_alignment_warning(engine, caplog):
    first_window = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Alpha user"},
            {"role": "assistant", "content": "Alpha assistant"},
        ],
    }
    second_window = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Gamma user"},
            {"role": "assistant", "content": "Gamma assistant"},
        ],
    }

    assert engine.sync_turns_from_payload(first_window, detect_format(first_window)) == 2
    with caplog.at_level("WARNING"):
        assert engine.sync_turns_from_payload(second_window, detect_format(second_window)) == 2

    assert "CANONICAL_TURN_NO_ALIGNMENT" in caplog.text


def test_sync_interior_overlap_reuses_existing_canonical_ids(engine):
    initial = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "A user"},
            {"role": "assistant", "content": "A assistant"},
            {"role": "user", "content": "B user"},
            {"role": "assistant", "content": "B assistant"},
            {"role": "user", "content": "C user"},
            {"role": "assistant", "content": "C assistant"},
            {"role": "user", "content": "D user"},
            {"role": "assistant", "content": "D assistant"},
            {"role": "user", "content": "E user"},
            {"role": "assistant", "content": "E assistant"},
            {"role": "user", "content": "F user"},
            {"role": "assistant", "content": "F assistant"},
        ],
    }
    overlap = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "X user"},
            {"role": "assistant", "content": "X assistant"},
            {"role": "user", "content": "C user"},
            {"role": "assistant", "content": "C assistant"},
            {"role": "user", "content": "D user"},
            {"role": "assistant", "content": "D assistant"},
            {"role": "user", "content": "E user"},
            {"role": "assistant", "content": "E assistant"},
            {"role": "user", "content": "Y user"},
            {"role": "assistant", "content": "Y assistant"},
        ],
    }

    assert engine.sync_turns_from_payload(initial, detect_format(initial)) == 12
    initial_rows = engine._store.get_all_canonical_turns("test-conv-sync")
    overlap_ids = {
        row.user_content or row.assistant_content: row.canonical_turn_id
        for row in initial_rows
        if (row.user_content or row.assistant_content) in {
            "C user", "C assistant",
            "D user", "D assistant",
            "E user", "E assistant",
        }
    }

    assert engine.sync_turns_from_payload(overlap, detect_format(overlap)) == 4
    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert _entry_texts(rows) == [
        ("user", "A user"),
        ("assistant", "A assistant"),
        ("user", "B user"),
        ("assistant", "B assistant"),
        ("user", "X user"),
        ("assistant", "X assistant"),
        ("user", "C user"),
        ("assistant", "C assistant"),
        ("user", "D user"),
        ("assistant", "D assistant"),
        ("user", "E user"),
        ("assistant", "E assistant"),
        ("user", "Y user"),
        ("assistant", "Y assistant"),
        ("user", "F user"),
        ("assistant", "F assistant"),
    ]
    assert {
        row.user_content or row.assistant_content: row.canonical_turn_id
        for row in rows
        if (row.user_content or row.assistant_content) in {
            "C user", "C assistant",
            "D user", "D assistant",
            "E user", "E assistant",
        }
    } == overlap_ids


def test_sync_repeated_common_turns_do_not_collapse(engine):
    repeated = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "ok"},
        ],
    }

    assert engine.sync_turns_from_payload(repeated, detect_format(repeated)) == 6
    first_rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert len(first_rows) == 6
    assert len({row.canonical_turn_id for row in first_rows}) == 6

    assert engine.sync_turns_from_payload(repeated, detect_format(repeated)) == 0
    second_rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert [row.canonical_turn_id for row in second_rows] == [
        row.canonical_turn_id for row in first_rows
    ]


def test_sync_empty_payload_logs(engine, caplog):
    body = {"model": "gpt-4o", "input": "What is virtual context?"}
    fmt = detect_format(body)

    with caplog.at_level("INFO"):
        assert engine.sync_turns_from_payload(body, fmt) == 0

    assert "INGEST_EMPTY_PAYLOAD" in caplog.text


def test_ingest_resync_preserves_stored_session_date_against_empty(engine):
    # Regression: a re-ingest batch that does not carry session_date must not
    # clobber the value the original ingestion extracted from payload headers.
    # Content-hash alignment means the same turn will re-enter the overlap or
    # exact_resend path on re-ingest; without the preserve policy, the naive
    # overwrite would wipe a populated session_date back to "".
    from virtual_context.core.ingest_reconciler import IngestReconciler

    reconciler = IngestReconciler(_inner_store(engine), engine._semantic)

    reconciler.ingest_single(
        "test-conv-preserve",
        user_content="Hello there",
        assistant_content="Hi, how can I help?",
        session_date="2026-03-20T15:49:00",
    )

    # Re-ingest same content without session_date (simulates a resync batch
    # built from stripped history that no longer includes the header line).
    reconciler.ingest_single(
        "test-conv-preserve",
        user_content="Hello there",
        assistant_content="Hi, how can I help?",
        session_date="",
    )

    rows = engine._store.get_all_canonical_turns("test-conv-preserve")
    assert len(rows) == 2
    assert all(r.session_date == "2026-03-20T15:49:00" for r in rows), (
        f"expected stored session_date to survive empty re-ingest, got {[r.session_date for r in rows]}"
    )


def test_ingest_resync_allows_upgrade_from_empty_to_populated(engine):
    # Symmetric to the preserve test: if the original ingestion had no
    # session_date (e.g. a live proxy write where the header wasn't present)
    # and a later resync DOES carry one, the populated value should win.
    from virtual_context.core.ingest_reconciler import IngestReconciler

    reconciler = IngestReconciler(_inner_store(engine), engine._semantic)

    reconciler.ingest_single(
        "test-conv-upgrade",
        user_content="Hello there",
        assistant_content="Hi, how can I help?",
        session_date="",
    )
    reconciler.ingest_single(
        "test-conv-upgrade",
        user_content="Hello there",
        assistant_content="Hi, how can I help?",
        session_date="2026-03-20T15:49:00",
    )

    rows = engine._store.get_all_canonical_turns("test-conv-upgrade")
    assert len(rows) == 2
    assert all(r.session_date == "2026-03-20T15:49:00" for r in rows)


def test_ingest_single_retry_dedups_recent_turn(engine):
    from virtual_context.core.ingest_reconciler import IngestReconciler

    reconciler = IngestReconciler(_inner_store(engine), engine._semantic)

    first = reconciler.ingest_single(
        "test-conv-sync",
        user_content="Retry user",
        assistant_content="Retry assistant",
    )
    second = reconciler.ingest_single(
        "test-conv-sync",
        user_content="Retry user",
        assistant_content="Retry assistant",
    )

    assert first.turns_written == 2
    assert second.merge_mode == "exact_resend"
    assert second.turns_written == 0
    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert len(rows) == 2


def test_ingest_single_invalid_timestamp_disables_dedup_with_warning(engine, caplog):
    from virtual_context.core.ingest_reconciler import IngestReconciler

    store = _inner_store(engine)
    store.save_canonical_turn("test-conv-sync", 0, "Retry user", "", last_seen_at="not-a-time")
    store.save_canonical_turn("test-conv-sync", 1, "", "Retry assistant", last_seen_at="not-a-time")
    reconciler = IngestReconciler(store, engine._semantic)

    with caplog.at_level("WARNING"):
        result = reconciler.ingest_single(
            "test-conv-sync",
            user_content="Retry user",
            assistant_content="Retry assistant",
        )

    assert result.turns_written == 0
    assert "CANONICAL_TURN_DEDUP_TIMESTAMP_INVALID" in caplog.text
    assert len(engine._store.get_all_canonical_turns("test-conv-sync")) == 2


def test_ingest_single_retry_outside_last_five_rows_is_not_deduped(engine):
    from virtual_context.core.ingest_reconciler import IngestReconciler

    reconciler = IngestReconciler(_inner_store(engine), engine._semantic)
    payloads = [
        ("Old user", "Old assistant"),
        ("Middle user 1", "Middle assistant 1"),
        ("Middle user 2", "Middle assistant 2"),
    ]
    for user_content, assistant_content in payloads:
        reconciler.ingest_single(
            "test-conv-sync",
            user_content=user_content,
            assistant_content=assistant_content,
        )

    result = reconciler.ingest_single(
        "test-conv-sync",
        user_content="Old user",
        assistant_content="Old assistant",
    )

    assert result.merge_mode == "no_overlap_append"
    assert result.turns_written == 2
    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert len(rows) == 8


def test_sync_concurrent_writers_do_not_duplicate_rows(engine, monkeypatch):
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Concurrent user 1"},
            {"role": "assistant", "content": "Concurrent assistant 1"},
            {"role": "user", "content": "Concurrent user 2"},
            {"role": "assistant", "content": "Concurrent assistant 2"},
        ],
    }
    fmt = detect_format(body)
    monkeypatch.setattr(engine._semantic, "embed_and_store_turn", lambda *args, **kwargs: None)

    def _run_once():
        return engine.sync_turns_from_payload(body, fmt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(lambda _: _run_once(), range(2)))

    assert results == [0, 4]
    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert len(rows) == 4
    assert len({row.canonical_turn_id for row in rows}) == 4


def test_proxy_single_ingest_vs_rest_batch_concurrent(engine, monkeypatch):
    # Two writers hit the same conversation at once: a proxy-style single
    # ingest and a REST-style batch sync covering overlapping turns.
    # conversation_reconcile serializes them per conversation, so the result
    # should be duplicate-free with unique canonical_turn_ids and distinct
    # turn_group_numbers per turn-group.
    from virtual_context.core.ingest_reconciler import IngestReconciler

    monkeypatch.setattr(engine._semantic, "embed_and_store_turn", lambda *args, **kwargs: None)

    rest_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Shared user 1"},
            {"role": "assistant", "content": "Shared assistant 1"},
            {"role": "user", "content": "Shared user 2"},
            {"role": "assistant", "content": "Shared assistant 2"},
        ],
    }
    fmt = detect_format(rest_body)

    def _proxy_writer():
        reconciler = IngestReconciler(_inner_store(engine), engine._semantic)
        return reconciler.ingest_single(
            "test-conv-sync",
            user_content="Shared user 1",
            assistant_content="Shared assistant 1",
        )

    def _rest_writer():
        return engine.sync_turns_from_payload(rest_body, fmt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_proxy = pool.submit(_proxy_writer)
        future_rest = pool.submit(_rest_writer)
        future_proxy.result()
        future_rest.result()

    rows = engine._store.get_all_canonical_turns("test-conv-sync")
    assert len(rows) == 4
    assert len({row.canonical_turn_id for row in rows}) == 4

    # turn_group_number should pair user+assistant into 2 groups. Each group
    # gets a unique non-negative number; proxy path writes a single row with
    # both user_content and assistant_content populated (its own group).
    group_numbers = {row.turn_group_number for row in rows}
    assert -1 not in group_numbers, "legacy placeholder should not leak from concurrent writes"
    assert len(group_numbers) >= 2, f"expected multiple distinct turn_group_numbers, got {group_numbers}"

    # Watermark invariant: flushed <= compacted at every hydration.
    assert engine._engine_state.flushed_prefix_messages <= engine._engine_state.compacted_prefix_messages


def test_sync_logs_embedding_failures_without_aborting_ingest(engine, monkeypatch, caplog):
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Embedding failure user"},
            {"role": "assistant", "content": "Embedding failure assistant"},
        ],
    }

    def _raise(*args, **kwargs):
        raise RuntimeError("embedding offline")

    monkeypatch.setattr(engine._semantic, "embed_and_store_turn", _raise)

    with caplog.at_level("WARNING"):
        assert engine.sync_turns_from_payload(body, detect_format(body)) == 2

    assert "CANONICAL_TURN_EMBED_FAILED" in caplog.text
    assert len(engine._store.get_all_canonical_turns("test-conv-sync")) == 2


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

    assert engine.sync_turns_from_payload(body, detect_format(body)) == 2
    assert calls == [
        ("enter", "test-conv-sync"),
        ("exit", "test-conv-sync"),
    ]


def test_restore_prefers_persisted_turn_groups(tmp_path):
    from virtual_context.config import VirtualContextConfig
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.types import EngineStateSnapshot, StorageConfig, TagGeneratorConfig

    db_path = tmp_path / "restore-groups.db"
    config = VirtualContextConfig(
        conversation_id="restore-groups",
        storage=StorageConfig(backend="sqlite", sqlite_path=str(db_path)),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )

    engine1 = VirtualContextEngine(config=config)
    store = _inner_store(engine1)
    tagged_at = "2026-04-16T00:00:00+00:00"
    store.save_canonical_turn(
        "restore-groups",
        0,
        "user-0",
        "",
        canonical_turn_id="00000000-0000-0000-0000-000000000001",
        sort_key=1000.0,
        tagged_at=tagged_at,
        turn_group_number=0,
    )
    store.save_canonical_turn(
        "restore-groups",
        1,
        "user-1",
        "",
        canonical_turn_id="00000000-0000-0000-0000-000000000002",
        sort_key=2000.0,
        tagged_at=tagged_at,
        turn_group_number=1,
    )
    store.save_canonical_turn(
        "restore-groups",
        2,
        "",
        "assistant-1",
        canonical_turn_id="00000000-0000-0000-0000-000000000003",
        sort_key=3000.0,
        tagged_at=tagged_at,
        turn_group_number=1,
    )
    store.save_canonical_turn(
        "restore-groups",
        3,
        "",
        "assistant-0",
        canonical_turn_id="00000000-0000-0000-0000-000000000004",
        sort_key=4000.0,
        tagged_at=tagged_at,
        turn_group_number=0,
    )
    store.save_engine_state(
        EngineStateSnapshot(
            conversation_id="restore-groups",
            compacted_prefix_messages=0,
            turn_count=2,
            turn_tag_entries=[],
            last_compacted_turn=-1,
            last_completed_turn=1,
            last_indexed_turn=1,
            checkpoint_version=3,
        )
    )
    engine1.close()

    engine2 = VirtualContextEngine(config=config)
    assert engine2._restored_conversation_history == [
        (0, "user-0", "assistant-0"),
        (1, "user-1", "assistant-1"),
    ]
    assert engine2._engine_state.checkpoint_version >= 4
    engine2.close()
