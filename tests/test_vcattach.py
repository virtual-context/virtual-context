# tests/test_vcattach.py
"""Tests for VCATTACH command."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sqlite_store(tmp_path):
    from virtual_context.storage.sqlite import SQLiteStore
    store = SQLiteStore(str(tmp_path / "test.db"))
    return store


def test_save_and_resolve_alias(sqlite_store):
    sqlite_store.save_conversation_alias("old-conv-123", "target-conv-456")
    resolved = sqlite_store.resolve_conversation_alias("old-conv-123")
    assert resolved == "target-conv-456"


def test_resolve_unknown_alias(sqlite_store):
    resolved = sqlite_store.resolve_conversation_alias("nonexistent")
    assert resolved is None


def test_alias_overwrite(sqlite_store):
    sqlite_store.save_conversation_alias("old-123", "target-a")
    sqlite_store.save_conversation_alias("old-123", "target-b")
    assert sqlite_store.resolve_conversation_alias("old-123") == "target-b"


def test_alias_chain_not_followed(sqlite_store):
    """Aliases are single-hop — no chain following."""
    sqlite_store.save_conversation_alias("a", "b")
    sqlite_store.save_conversation_alias("b", "c")
    assert sqlite_store.resolve_conversation_alias("a") == "b"


# --- VCATTACH regex tests ---

import re

_VCATTACH_RE = re.compile(r"^VCATTACH\s+(.+)$", re.IGNORECASE)


def test_vcattach_regex_label():
    m = _VCATTACH_RE.match("VCATTACH website")
    assert m and m.group(1) == "website"


def test_vcattach_regex_uuid():
    m = _VCATTACH_RE.match("VCATTACH d4f83259-4ffc-fa3f-5914-a266d0a4577c")
    assert m and m.group(1) == "d4f83259-4ffc-fa3f-5914-a266d0a4577c"


def test_vcattach_regex_prefix():
    m = _VCATTACH_RE.match("VCATTACH d4f83259")
    assert m and m.group(1) == "d4f83259"


def test_vcattach_regex_case_insensitive():
    m = _VCATTACH_RE.match("vcattach Website")
    assert m and m.group(1) == "Website"


def test_vcattach_regex_no_target():
    m = _VCATTACH_RE.match("VCATTACH")
    assert m is None


def test_vcattach_regex_not_triggered_by_history():
    """Only the last user message should trigger, not history."""
    m = _VCATTACH_RE.match("I said VCATTACH website earlier")
    assert m is None


def test_emit_fake_response_sse_anthropic():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    events = fmt.emit_fake_response_sse("Test response", "conv-123")
    assert isinstance(events, bytes)
    text = events.decode()
    assert "Test response" in text
    assert "vc:conversation=conv-123" in text
    assert "message_start" in text
    assert "message_stop" in text


def test_emit_fake_response_sse_openai():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    events = fmt.emit_fake_response_sse("Test response", "conv-123")
    text = events.decode()
    assert "Test response" in text
    assert "vc:conversation=conv-123" in text
    assert "[DONE]" in text


def test_build_fake_response_anthropic():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    resp = fmt.build_fake_response("Test response", "conv-123")
    assert isinstance(resp, dict)
    assert resp["role"] == "assistant"
    assert "Test response" in resp["content"][0]["text"]
    assert "vc:conversation=conv-123" in resp["content"][0]["text"]


def test_build_fake_response_openai():
    from virtual_context.proxy.formats import detect_format
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    fmt = detect_format(body)
    resp = fmt.build_fake_response("Test response", "conv-123")
    assert isinstance(resp, dict)
    assert "choices" in resp
    assert "Test response" in resp["choices"][0]["message"]["content"]
    assert "vc:conversation=conv-123" in resp["choices"][0]["message"]["content"]


def test_vcstatus_uses_effective_engine_conversation_id():
    from virtual_context.proxy.handlers import _handle_vcstatus

    class _TurnTagIndex:
        entries = [1, 2, 3]

        def get_active_tags(self, lookback=6):
            return {"alpha", "beta"}

    class _Store:
        def get_conversation_stats(self):
            return [
                SimpleNamespace(conversation_id="alias-conv", segment_count=1),
                SimpleNamespace(conversation_id="target-conv", segment_count=7),
            ]

    state = SimpleNamespace(
        engine=SimpleNamespace(
            config=SimpleNamespace(conversation_id="target-conv"),
            _engine_state=SimpleNamespace(compacted_prefix_messages=4, conversation_generation=9),
            _turn_tag_index=_TurnTagIndex(),
            _store=_Store(),
            _paging=SimpleNamespace(working_set={}),
        )
    )

    text = _handle_vcstatus("alias-conv", state, tenant_registry=None, tenant_id=None)

    assert "Conversation: target-conv" in text
    assert "Stored: 7 segments, 0 tag summaries" in text


def test_vcstatus_without_state_reports_shell_status():
    from virtual_context.proxy.handlers import _handle_vcstatus

    text = _handle_vcstatus(
        "56315149-9812-9cf5-21a7-ade5a2279ad8",
        None,
        tenant_registry=None,
        tenant_id=None,
    )

    assert "Conversation: 56315149-9812-9cf5-21a7-ade5a2279ad8" in text
    assert "Status: ready" in text
    assert "Ingestion: 0 / 0 (0.0%)" in text
    assert "Turn state: 0 turns, 0 live history messages" in text
    assert "Stored: 0 segments, 0 tag summaries" in text


def test_vcstatus_surfaces_ingestion_payload_and_cache_metrics():
    from virtual_context.proxy.handlers import _handle_vcstatus

    class _TurnTagIndex:
        entries = [1, 2, 3]

        def get_active_tags(self, lookback=6):
            return {"alpha", "beta"}

    class _Store:
        def get_conversation_stats(self):
            return [
                SimpleNamespace(conversation_id="target-conv", segment_count=7),
            ]

        def get_all_segments(self, *, conversation_id=None, limit=None):
            return []

        def get_all_tag_summaries(self, *, conversation_id=None):
            return [SimpleNamespace(created_at=datetime.now(timezone.utc), covers_through_turn=12)]

        def load_request_captures(self, limit=50, conversation_id=None):
            return [
                {
                    "turn": 12,
                    "turn_id": "t12",
                    "ts": "2026-04-14T21:15:16.642392+00:00",
                    "client_payload_message_count": 989,
                    "client_payload_user_prompt_count": 300,
                    "client_payload_timestamped_message_count": 290,
                    "client_payload_earliest_timestamp": "2026-03-15T16:10:00+00:00",
                    "client_payload_latest_timestamp": "2026-04-14T21:15:00+00:00",
                    "raw_payload_entry_count": 4632,
                    "ingestible_entry_count": 989,
                    "upstream_input_tokens": 100000,
                    "cache_read_input_tokens": 82000,
                }
            ]

        def read_progress_snapshot(self, conversation_id):
            return SimpleNamespace(
                conversation_id=conversation_id,
                phase="ingesting",
                done_ingestible=989,
                total_ingestible=989,
                active_episode=None,
                active_compaction=None,
            )

    class _Metrics:
        def get_captured_requests_summary(self, conversation_id=None):
            return [
                {
                    "raw_payload_entry_count": 4632,
                    "ingestible_entry_count": 989,
                    "upstream_input_tokens": 100000,
                    "cache_read_input_tokens": 82000,
                }
            ]

    state = SimpleNamespace(
        session_state=SimpleNamespace(value="ingesting"),
        _ingestion_progress=(99, 494),
        _ingestible_entry_count=989,
        _raw_payload_entry_count=4632,
        _last_payload_kb=22550.3,
        _last_payload_tokens=12331759,
        conversation_history=[SimpleNamespace(content="x")] * 989,
        metrics=_Metrics(),
        compaction_snapshot=lambda: {},
        engine=SimpleNamespace(
            config=SimpleNamespace(
                conversation_id="target-conv",
                monitor=SimpleNamespace(soft_threshold=80000, hard_threshold=120000),
            ),
            _engine_state=SimpleNamespace(
                compacted_prefix_messages=986,
                flushed_prefix_messages=983,
                conversation_generation=15,
            ),
            _turn_tag_index=_TurnTagIndex(),
            _store=_Store(),
            _paging=SimpleNamespace(working_set={}),
        ),
    )

    text = _handle_vcstatus("target-conv", state, tenant_registry=None, tenant_id=None)

    assert "Status: ingesting" in text
    assert "Ingestion: 989 / 989 (100.0%)" in text
    assert "Turn state: 989 canonical chat entries, 4632 raw payload entries" in text
    assert "Stored: 7 segments, 1 tag summaries" in text
    assert "Thresholds: soft 80,000 / hard 120,000" in text
    assert "Last payload: 22.022 MB, 989 ingestible entries, 12,331,759 tokens" in text
    assert "Last raw payload: 4,632 entries" in text
    assert "Cache hit (last 5): 82% (avg 82.0%)" in text


def test_vcattach_preserves_target_engine_state():
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    class _InnerStore:
        def __init__(self):
            self.aliases = []
            self.load_called = False
            self.save_called = False
            self.deleted = []

        def save_conversation_alias(self, alias_id, target_id):
            self.aliases.append((alias_id, target_id))

        def load_engine_state(self, conversation_id):
            self.load_called = True
            return {"conversation_id": conversation_id}

        def save_engine_state(self, snapshot):
            self.save_called = True

        def delete_conversation(self, conversation_id):
            self.deleted.append(conversation_id)

    inner = _InnerStore()
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    registry = SimpleNamespace(remove_conversation=lambda cid: None)
    result = SimpleNamespace(
        vcattach_label="Cloud Claude",
        conversation_id="old-shell-conv",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH Cloud Claude"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            labels={"target-conv": "Cloud Claude"},
            conv_ids=["old-shell-conv", "target-conv"],
        )
    )

    assert response.status_code == 200
    assert inner.aliases == [("old-shell-conv", "target-conv")]
    # load_engine_state IS now called for the Bug #3 existence check
    # (defense-in-depth against stale labels). That's a benign read; what
    # we still guard against is target state being SAVED (i.e. mutated /
    # restored / merged) by VCATTACH.
    assert inner.save_called is False


def test_vcattach_does_not_delete_old_conversation_in_cloud_mode():
    """VCATTACH is a durable redirect — the old conversation is preserved."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    inner = MagicMock()
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    registry = MagicMock()
    tenant_registry = MagicMock()
    result = SimpleNamespace(
        vcattach_label="d4f83259-4ffc-fa3f-5914-a266d0a4577c",
        conversation_id="56315149-9812-9cf5-21a7-ade5a2279ad8",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH d4f83259-4ffc-fa3f-5914-a266d0a4577c"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            conv_ids=["d4f83259-4ffc-fa3f-5914-a266d0a4577c"],
            tenant_registry=tenant_registry,
            tenant_id="tenant-123",
        )
    )

    assert response.status_code == 200
    # Old conversation must NOT be deleted via tenant registry
    tenant_registry.delete_conversation.assert_not_called()
    # Alias must be written old -> target
    inner.save_conversation_alias.assert_called_once_with(
        "56315149-9812-9cf5-21a7-ade5a2279ad8",
        "d4f83259-4ffc-fa3f-5914-a266d0a4577c",
    )
    # Any dangling alias FROM target must be cleared (self-attach unlock)
    inner.delete_conversation_alias.assert_called_once_with(
        "d4f83259-4ffc-fa3f-5914-a266d0a4577c",
    )


def test_execute_attach_clears_reverse_alias():
    """If alias A -> B exists and user VCATTACHes to A, the A -> B alias is cleared."""
    from virtual_context.proxy.vcattach import execute_attach

    store = MagicMock()
    execute_attach(
        old_id="current-shell-id",
        target_id="conversation-A",
        store=store,
    )

    store.delete_conversation_alias.assert_called_once_with("conversation-A")
    store.save_conversation_alias.assert_called_once_with(
        "current-shell-id",
        "conversation-A",
    )


def test_execute_attach_no_longer_accepts_delete_conversation():
    """Old delete_conversation parameter has been removed."""
    import inspect
    from virtual_context.proxy.vcattach import execute_attach

    sig = inspect.signature(execute_attach)
    assert "delete_conversation" not in sig.parameters


def test_delete_conversation_alias_sqlite_roundtrip(sqlite_store):
    """delete_conversation_alias removes the row so resolve returns None."""
    sqlite_store.save_conversation_alias("a", "b")
    assert sqlite_store.resolve_conversation_alias("a") == "b"
    sqlite_store.delete_conversation_alias("a")
    assert sqlite_store.resolve_conversation_alias("a") is None


def test_delete_conversation_alias_missing_is_noop(sqlite_store):
    """Deleting an alias that doesn't exist must not error."""
    sqlite_store.delete_conversation_alias("never-existed")
    assert sqlite_store.resolve_conversation_alias("never-existed") is None


# ---------------------------------------------------------------------
# Regression: cross-team incident 2026-04-26 — VCATTACH destroyed 358+
# canonical turns of a labeled live conversation. Three connected bugs:
# (1) REST handler called session_state_provider.delete() which calls
#     PostgresStore.delete_conversation() (purges 21 tables + media dir).
# (2) execute_attach only invalidated target_id, never the issuing chat's
#     stale ProxyState — so ingestion kept writing to old_id.
# (3) resolve_target accepted any label, even when target had no
#     persisted state.
# ---------------------------------------------------------------------


# --- Bug #2: execute_attach evicts BOTH old_id and target_id ---


def test_execute_attach_invalidates_both_old_and_target():
    """The registry_invalidate callback must be invoked for both old_id and
    target_id — old_id so the issuing chat's stale ProxyState is evicted and
    the next request falls through to alias resolution; target_id so any
    cached target state cannot shadow a fresh load."""
    from virtual_context.proxy.vcattach import execute_attach

    invalidated = []
    store = MagicMock()
    execute_attach(
        old_id="old-conv-aaaa",
        target_id="target-conv-bbbb",
        store=store,
        registry_invalidate=lambda cid: invalidated.append(cid),
    )

    assert "old-conv-aaaa" in invalidated, (
        "execute_attach must invalidate the issuing session (old_id) — without "
        "this, the issuing chat's ProxyState keeps engine.config.conversation_id "
        "== old_id and routing skips alias resolution."
    )
    assert "target-conv-bbbb" in invalidated


def test_execute_attach_invalidate_failure_is_isolated_per_id():
    """A failure invalidating one id must not block the other — both ids
    must always get a chance to be invalidated."""
    from virtual_context.proxy.vcattach import execute_attach

    seen = []

    def _invalidate(cid):
        seen.append(cid)
        if cid == "old-conv":
            raise RuntimeError("simulated transient error")

    store = MagicMock()
    execute_attach(
        old_id="old-conv",
        target_id="target-conv",
        store=store,
        registry_invalidate=_invalidate,
    )
    assert "old-conv" in seen
    assert "target-conv" in seen


# --- Bug #3: resolve_target rejects deleted/nonexistent targets ---


def test_resolve_target_rejects_nonexistent_target_via_label():
    """A label pointing to a deleted/tombstoned conversation must be rejected
    before any alias is written. Defends against stale labels.json entries."""
    from virtual_context.proxy.vcattach import resolve_target

    target_id, target_label, error = resolve_target(
        target_raw="Telegram DM",
        current_id="old-conv",
        conversation_ids=["old-conv", "deleted-conv"],
        labels={"deleted-conv": "Telegram DM"},
        target_exists=lambda tid: False,
    )

    assert target_id is None
    assert error  # non-empty
    # Label echoed back so user sees what failed
    assert target_label == "Telegram DM"


def test_resolve_target_accepts_existing_target_via_label():
    from virtual_context.proxy.vcattach import resolve_target

    target_id, target_label, error = resolve_target(
        target_raw="Telegram DM",
        current_id="old-conv",
        conversation_ids=["old-conv", "live-conv"],
        labels={"live-conv": "Telegram DM"},
        target_exists=lambda tid: True,
    )

    assert target_id == "live-conv"
    assert target_label == "Telegram DM"
    assert error == ""


def test_resolve_target_rejects_nonexistent_target_via_uuid():
    from virtual_context.proxy.vcattach import resolve_target

    target_id, _label, error = resolve_target(
        target_raw="deleted-conv",
        current_id="old-conv",
        conversation_ids=["old-conv", "deleted-conv"],
        labels={},
        target_exists=lambda tid: False,
    )
    assert target_id is None
    assert error


def test_resolve_target_rejects_nonexistent_target_via_prefix():
    from virtual_context.proxy.vcattach import resolve_target

    target_id, _label, error = resolve_target(
        target_raw="deleted",
        current_id="old-conv",
        conversation_ids=["old-conv", "deleted-conv-aaa"],
        labels={},
        target_exists=lambda tid: False,
    )
    assert target_id is None
    assert error


def test_resolve_target_no_existence_check_preserves_old_behavior():
    """When target_exists is None (default), no existence check happens —
    backwards compat for existing callers."""
    from virtual_context.proxy.vcattach import resolve_target

    target_id, target_label, error = resolve_target(
        target_raw="Telegram DM",
        current_id="old-conv",
        conversation_ids=["old-conv", "any-conv"],
        labels={"any-conv": "Telegram DM"},
    )
    assert target_id == "any-conv"
    assert error == ""


def test_resolve_target_existence_check_failsafe_open_on_exception():
    """If target_exists raises, fail open so transient DB errors don't
    block legitimate VCATTACHes."""
    from virtual_context.proxy.vcattach import resolve_target

    def _raises(tid):
        raise RuntimeError("transient")

    target_id, _label, error = resolve_target(
        target_raw="Telegram DM",
        current_id="old-conv",
        conversation_ids=["old-conv", "live-conv"],
        labels={"live-conv": "Telegram DM"},
        target_exists=_raises,
    )
    assert target_id == "live-conv"
    assert error == ""


# --- Bug #1: REST handler must NOT destroy target conversation ---


def _build_rest_registry(tenant_id="tenant-1", labels=None, conv_ids=None,
                         in_memory_states=None, tombstone=None):
    """Build a SimpleNamespace shaped like vc_cloud.tenant.TenantRegistry."""
    import threading
    provider = MagicMock()
    if tombstone is None:
        provider.load.return_value = None
    else:
        provider.load.return_value = SimpleNamespace(deleted=tombstone)
    return SimpleNamespace(
        _session_state_provider=provider,
        _states={tenant_id: dict(in_memory_states or {})},
        _lock=threading.Lock(),
        get_conversation_labels=lambda tid: dict(labels or {}),
        list_persisted_conversation_ids=lambda tid: list(conv_ids or []),
    ), provider


def test_rest_vcattach_does_not_call_session_state_provider_delete():
    """REST VCATTACH must NEVER invoke session_state_provider.delete().

    The provider's delete() calls store.delete_conversation() which purges
    21 Postgres tables (segments, canonical_turns, engine_state, …) plus
    the per-conversation media directory. VCATTACH is a durable redirect:
    target data must be preserved.
    """
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    inner = MagicMock()
    inner.load_engine_state.return_value = {"conversation_id": "target-conv"}
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))

    issuing_state = SimpleNamespace(shutdown=MagicMock())
    registry, provider = _build_rest_registry(
        labels={"target-conv": "Telegram DM"},
        conv_ids=["old-conv", "target-conv"],
        in_memory_states={"old-conv": issuing_state},
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Telegram DM",
        conversation_id="old-conv",
    )

    _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )

    # Bug #1 guard — destructive paths must NEVER fire from VCATTACH:
    provider.delete.assert_not_called()
    inner.delete_conversation.assert_not_called()
    # Alias must be persisted (the durable redirect).
    inner.save_conversation_alias.assert_called_once_with("old-conv", "target-conv")


def test_rest_vcattach_evicts_issuing_chat_state():
    """The issuing chat's in-memory ProxyState must be popped from
    registry._states[tenant_id] so the next request from that chat falls
    through to alias resolution instead of resolving back to the stale
    ProxyState whose engine.config.conversation_id == old_id."""
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    inner = MagicMock()
    inner.load_engine_state.return_value = {"conversation_id": "target-conv"}
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))

    issuing_state = SimpleNamespace(shutdown=MagicMock())
    registry, _provider = _build_rest_registry(
        labels={"target-conv": "Telegram DM"},
        conv_ids=["old-conv", "target-conv"],
        in_memory_states={"old-conv": issuing_state},
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Telegram DM",
        conversation_id="old-conv",
    )

    _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )

    # Bug #2 fix: issuing chat's state evicted from in-memory map.
    assert "old-conv" not in registry._states["tenant-1"], (
        "issuing chat's stale ProxyState was not evicted; the next request "
        "would route via chat_id/sys_hash back to the same state with the "
        "old conversation_id."
    )
    # Evicted state had its shutdown() called for clean teardown.
    issuing_state.shutdown.assert_called_once()


def test_rest_vcattach_clears_target_tombstone_via_undelete():
    """If the target had a prior Redis tombstone (e.g. from cloud's
    tenant.delete_conversation flow), the VCATTACH must clear it via
    undelete() so a fresh load can succeed against the alias resolver."""
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    inner = MagicMock()
    # Note: load_engine_state still returns a row — undelete is called on
    # the tombstone-clearing path regardless of whether one is present.
    inner.load_engine_state.return_value = {"conversation_id": "target-conv"}
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))

    registry, provider = _build_rest_registry(
        labels={"target-conv": "Telegram DM"},
        conv_ids=["old-conv", "target-conv"],
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Telegram DM",
        conversation_id="old-conv",
    )

    _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )

    # Both old_id and target_id receive an undelete (idempotent no-op when
    # no tombstone is present).
    undelete_args = [c.args for c in provider.undelete.call_args_list]
    assert ("target-conv",) in undelete_args
    assert ("old-conv",) in undelete_args


def test_rest_vcattach_rejects_label_pointing_to_deleted_target():
    """Bug #3 in REST path: a label pointing to a tombstoned/no-data
    conversation must be rejected before any alias is written."""
    import json as _json
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    inner = MagicMock()
    inner.load_engine_state.return_value = None  # target has no engine state
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))

    registry, provider = _build_rest_registry(
        labels={"deleted-conv": "Telegram DM"},
        conv_ids=["old-conv", "deleted-conv"],
        tombstone=True,
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Telegram DM",
        conversation_id="old-conv",
    )

    response = _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )

    # No alias was saved — the attach was rejected.
    inner.save_conversation_alias.assert_not_called()
    # Response carries an error string.
    body = _json.loads(response.body)
    assert body.get("error"), "expected error in JSONResponse for stale label"


# --- Bug #2 / Bug #3: proxy path must also evict and validate ---


def test_proxy_vcattach_rejects_label_pointing_to_deleted_target():
    """Proxy path must reject labels pointing to deleted targets via the
    same target_exists guard."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    inner = MagicMock()
    inner.load_engine_state.return_value = None
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    registry = SimpleNamespace(remove_conversation=lambda cid: None)
    result = SimpleNamespace(
        vcattach_label="Telegram DM",
        conversation_id="old-conv",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH Telegram DM"}],
    })

    asyncio.run(
        _handle_vcattach(
            result, fmt, state, registry,
            labels={"deleted-conv": "Telegram DM"},
            conv_ids=["old-conv", "deleted-conv"],
        )
    )

    # No alias saved when the target has no persisted state.
    inner.save_conversation_alias.assert_not_called()


def test_proxy_vcattach_evicts_issuing_chat_state():
    """The proxy path must also invalidate the issuing chat's state via
    registry.remove_conversation(old_id)."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    inner = MagicMock()
    inner.load_engine_state.return_value = {"conversation_id": "target-conv"}
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))
    invalidated = []
    registry = SimpleNamespace(remove_conversation=lambda cid: invalidated.append(cid))

    result = SimpleNamespace(
        vcattach_label="target-conv",
        conversation_id="old-conv",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH target-conv"}],
    })

    asyncio.run(
        _handle_vcattach(
            result, fmt, state, registry,
            labels={},
            conv_ids=["old-conv", "target-conv"],
        )
    )

    assert "old-conv" in invalidated, (
        "proxy path must evict old_id (issuing chat) — without this, "
        "the next request from this chat keeps routing to the stale state."
    )
    assert "target-conv" in invalidated
