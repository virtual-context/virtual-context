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


def test_execute_attach_no_longer_accepts_reset_engine_state():
    """Dormant reset_engine_state callback slot has been removed (F-2 audit).

    Background: the slot was a callable(target_id) callback documented as
    "reset target checkpoints", but its naming convention ("reset") is a
    historical synonym for "delete-and-rebuild" elsewhere in this codebase.
    All three call sites passed None; the slot was the same call shape as
    the VCATTACH seam (a non-destructively-named callback
    invoked on target_id), and a future PR could have wired it to a
    destructive primitive without explicit review.

    Removing it structurally prevents that class of regression. If a real
    use case for an engine-state reset hook appears later, it should be
    re-introduced with an explicit name and a docstring rule forbidding
    destructive primitives.
    """
    import inspect
    from virtual_context.proxy.vcattach import execute_attach

    sig = inspect.signature(execute_attach)
    assert "reset_engine_state" not in sig.parameters

    # Stronger guard: assert the entire parameter set is the expected
    # non-destructive shape. Any new kwarg added here triggers explicit
    # review. ``cross_worker_invalidate`` was added by engine commit-2
    # for cross-worker cache invalidation: it forwards an
    # observer-only ``AliasEvent`` callback to
    # ``save_conversation_alias`` / ``delete_conversation_alias`` and
    # cannot mutate or destroy persisted state. The guard is preserved
    # so any FUTURE kwarg still triggers review.
    # ``session_state_provider`` was added by the
    # vcattach-redis-marker-write spec for the post-alias-commit
    # SessionState derivation + write step (T2). It accepts the
    # cloud-side SessionStateProvider instance so target's Redis
    # markers reflect canonical_turns truth after VCATTACH. The
    # provider is observer-via-save-only at this entry point — the
    # alias write itself is unaffected by provider availability.
    assert set(sig.parameters.keys()) == {
        "old_id",
        "target_id",
        "store",
        "registry_invalidate",
        "cross_worker_invalidate",
        "session_state_provider",
    }


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
# Regression: cross-team incident — VCATTACH destroyed 358+
# canonical turns of a labeled live conversation. Three connected bugs:
# (1) REST handler called session_state_provider.delete() which calls
# PostgresStore.delete_conversation() (purges 21 tables + media dir).
# (2) execute_attach only invalidated target_id, never the issuing chat's
# stale ProxyState — so ingestion kept writing to old_id.
# (3) resolve_target accepted any label, even when target had no
# persisted state.
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


def test_rest_vcattach_error_response_populates_both_error_and_message():
    """Production-bug fix: REST VCATTACH error responses must populate BOTH
    `error` (programmatic) AND `message` (human-readable).

    Background: OpenClaw plugin clients render `prepareResult.message`, NOT
    `prepareResult.error`. A 200 response with only `error` populated is
    silently swallowed by the plugin. The user sees the LLM answer their
    raw "VCATTACH X" prompt as if no command was issued.

    Bug shipped via every CORE_CACHE_BUST since the prependContext
    switchover (`4c89fd9`). Fix: dual-populate so plugin's
    ``prepareResult.message`` rendering path lights up the error directly.

    This contract applies prospectively to all VC-command error responses
    (VCMERGE will follow the same shape per its ).
    """
    import json as _json
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    inner = MagicMock()
    inner.load_engine_state.return_value = None  # target has no engine state
    state = SimpleNamespace(engine=SimpleNamespace(_store=SimpleNamespace(_store=inner)))

    registry, _provider = _build_rest_registry(
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
    body = _json.loads(response.body)

    assert body.get("error"), "expected `error` field for plugin-side / programmatic consumers"
    assert body.get("message"), (
        "expected `message` field for plugin clients (OpenClaw and any future "
        "REST client) that render `prepareResult.message` via prependContext. "
        "Without `message` the plugin silently swallows the error and the "
        "user sees the LLM hallucinate as if no command was issued."
    )
    # Same content in both fields is acceptable for v1; future v2 may split
    # `error` into a short code and `message` into prose. The contract is
    # "both populated", not "byte-identical".
    assert body["error"] == body["message"], (
        "v1 contract: `error` and `message` carry the same human-readable "
        "string. Future versions may diverge but must preserve both fields."
    )


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
    inner.is_attachable_target.return_value = False
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


# ---------------------------------------------------------------------
# Store.is_attachable_target predicate — VCATTACH liveness gate
# ---------------------------------------------------------------------
# Replaces the post-cutover-empty engine_state row check. The
# predicate denies for missing / soft-deleted (deleted_at) /
# phase IN ('deleted','merged') / cross-tenant; allows for
# active/init/ingesting/compacting rows in the matching tenant.


def _seed_conversation(
    store, *, conversation_id: str, tenant_id: str,
    phase: str = "active", deleted_at: str | None = None,
) -> None:
    """Insert a row into ``conversations`` with the requested liveness fields.

    Uses the store's own pool/connection seam so the new row participates in
    the same transactional view as ``is_attachable_target`` reads.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = store._get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO conversations (
            conversation_id, tenant_id, lifecycle_epoch, phase,
            pending_raw_payload_entries, last_raw_payload_entries,
            last_ingestible_payload_entries,
            created_at, updated_at, deleted_at
        ) VALUES (?, ?, 1, ?, 0, 0, 0, ?, ?, ?)
        """,
        (conversation_id, tenant_id, phase, now, now, deleted_at),
    )
    conn.commit()


def test_is_attachable_target_active_no_tenant(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-active",
                       tenant_id="t1", phase="active")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-active") is True


def test_is_attachable_target_active_tenant_match(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-active",
                       tenant_id="t1", phase="active")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-active", tenant_id="t1") is True


def test_is_attachable_target_cross_tenant_denied(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-active",
                       tenant_id="t1", phase="active")
    # Same conv id, different tenant → must deny so a stale label can't
    # bridge across tenants.
    assert sqlite_store.is_attachable_target(
        conversation_id="c-active", tenant_id="t2") is False


def test_is_attachable_target_phase_deleted(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-del",
                       tenant_id="t1", phase="deleted")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-del", tenant_id="t1") is False


def test_is_attachable_target_phase_merged(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-merged",
                       tenant_id="t1", phase="merged")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-merged", tenant_id="t1") is False


def test_is_attachable_target_soft_deleted_via_deleted_at(sqlite_store):
    """phase='active' but deleted_at non-null = soft-deleted in transit
    (e.g. the deletion flow stamped deleted_at before flipping phase)."""
    now = datetime.now(timezone.utc).isoformat()
    _seed_conversation(sqlite_store, conversation_id="c-soft",
                       tenant_id="t1", phase="active", deleted_at=now)
    assert sqlite_store.is_attachable_target(
        conversation_id="c-soft", tenant_id="t1") is False


def test_is_attachable_target_missing_row(sqlite_store):
    assert sqlite_store.is_attachable_target(
        conversation_id="never-existed", tenant_id="t1") is False


def test_is_attachable_target_empty_conversation_id(sqlite_store):
    """Defensive: empty string must not match anything."""
    assert sqlite_store.is_attachable_target(
        conversation_id="", tenant_id="t1") is False


def test_is_attachable_target_phase_init_attachable(sqlite_store):
    """phase='init' is a freshly-created conversation; still a valid attach
    target (it owns its own identity even before first ingest)."""
    _seed_conversation(sqlite_store, conversation_id="c-init",
                       tenant_id="t1", phase="init")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-init", tenant_id="t1") is True


def test_is_attachable_target_phase_ingesting_attachable(sqlite_store):
    _seed_conversation(sqlite_store, conversation_id="c-ing",
                       tenant_id="t1", phase="ingesting")
    assert sqlite_store.is_attachable_target(
        conversation_id="c-ing", tenant_id="t1") is True


# ---------------------------------------------------------------------
# Regression: prod symptom 2026-05-09 — active conv with no engine_state
# ---------------------------------------------------------------------
# Pre-fix: _target_exists consulted load_engine_state(tid) which is empty
# across every REST-only tenant in the post-cutover schema, so VCATTACH
# rejected every legitimate target ("the conversation has no persisted
# state"). Post-fix: the conversations row is the liveness signal.


def test_rest_vcattach_attaches_when_engine_state_empty(sqlite_store):
    """REST VCATTACH against an active target must succeed even when no
    engine_state row exists — covers REST-only ingest tenants whose
    session-state save path was never invoked."""
    import json as _json
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    # Active target with NO engine_state row. The new predicate reads
    # ``conversations`` directly.
    _seed_conversation(sqlite_store, conversation_id="target-conv",
                       tenant_id="tenant-1", phase="active")
    _seed_conversation(sqlite_store, conversation_id="old-conv",
                       tenant_id="tenant-1", phase="active")

    state = SimpleNamespace(engine=SimpleNamespace(
        _store=SimpleNamespace(_store=sqlite_store)))

    registry, provider = _build_rest_registry(
        labels={"target-conv": "Telegram Group"},
        conv_ids=["old-conv", "target-conv"],
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Telegram Group",
        conversation_id="old-conv",
    )

    response = _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )
    body = _json.loads(response.body)

    # No error: the active target with no engine_state is attachable.
    assert not body.get("error"), (
        f"VCATTACH unexpectedly rejected active target with no engine_state row; "
        f"response was: {body}"
    )
    # Alias row written for the durable redirect.
    assert sqlite_store.resolve_conversation_alias("old-conv") == "target-conv"


def test_proxy_vcattach_attaches_when_engine_state_empty(sqlite_store):
    """Proxy path mirror of the REST regression — no engine_state row,
    active conversations row, attach must succeed."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    _seed_conversation(sqlite_store, conversation_id="target-conv",
                       tenant_id="tenant-1", phase="active")

    state = SimpleNamespace(engine=SimpleNamespace(
        _store=SimpleNamespace(_store=sqlite_store)))
    registry = SimpleNamespace(remove_conversation=lambda cid: None)

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
            tenant_id="tenant-1",
        )
    )

    assert sqlite_store.resolve_conversation_alias("old-conv") == "target-conv"


def test_rest_vcattach_rejects_merged_source(sqlite_store):
    """A merged source is a tombstone whose content is at the merge target;
    re-attaching to it would graft an alias onto a corpse. Must refuse."""
    import json as _json
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    _seed_conversation(sqlite_store, conversation_id="merged-src",
                       tenant_id="tenant-1", phase="merged")
    _seed_conversation(sqlite_store, conversation_id="old-conv",
                       tenant_id="tenant-1", phase="active")

    state = SimpleNamespace(engine=SimpleNamespace(
        _store=SimpleNamespace(_store=sqlite_store)))

    registry, _provider = _build_rest_registry(
        labels={"merged-src": "Old Topic"},
        conv_ids=["old-conv", "merged-src"],
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Old Topic",
        conversation_id="old-conv",
    )

    response = _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )
    body = _json.loads(response.body)

    assert body.get("error"), "expected error response for merged source"
    # No alias row written.
    assert sqlite_store.resolve_conversation_alias("old-conv") is None


def test_rest_vcattach_rejects_cross_tenant_target(sqlite_store):
    """A label resolved to a conversation owned by a different tenant must
    be denied so a stale labels.json entry cannot bridge tenants."""
    import json as _json
    from virtual_context.proxy.handlers import _handle_vc_command_rest

    # Target lives in tenant-OTHER, attacker is tenant-1.
    _seed_conversation(sqlite_store, conversation_id="other-tenant-conv",
                       tenant_id="tenant-OTHER", phase="active")
    _seed_conversation(sqlite_store, conversation_id="old-conv",
                       tenant_id="tenant-1", phase="active")

    state = SimpleNamespace(engine=SimpleNamespace(
        _store=SimpleNamespace(_store=sqlite_store)))

    # tenant-1's labels.json points (incorrectly / maliciously) to a conv
    # that belongs to tenant-OTHER. The list_persisted_conversation_ids
    # returns it because tenant-1's labels.json has it; the conversations
    # table predicate is the cross-tenant safety net.
    registry, _provider = _build_rest_registry(
        labels={"other-tenant-conv": "Some Label"},
        conv_ids=["old-conv", "other-tenant-conv"],
    )

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="Some Label",
        conversation_id="old-conv",
    )

    response = _handle_vc_command_rest(
        result, state, registry, tenant_id="tenant-1", vcconv="old-conv",
    )
    body = _json.loads(response.body)

    assert body.get("error"), "expected cross-tenant attach to be rejected"
    assert sqlite_store.resolve_conversation_alias("old-conv") is None


# ---------------------------------------------------------------------
# Multi-worker concurrency: two threads VCATTACH the same source
# ---------------------------------------------------------------------
# conversation_aliases.alias_id is PK; concurrent UPSERTs serialize cleanly
# with last-write-wins. No constraint violation; final state is one of the
# two target ids.


def test_concurrent_save_conversation_alias_no_constraint_violation(sqlite_store):
    """Last-write-wins under contention; no constraint violation. Models the
    multi-worker safety claim of the rigor pass."""
    import threading

    _seed_conversation(sqlite_store, conversation_id="t-a",
                       tenant_id="t1", phase="active")
    _seed_conversation(sqlite_store, conversation_id="t-b",
                       tenant_id="t1", phase="active")

    errors: list[BaseException] = []

    def _attach(target_id: str) -> None:
        try:
            sqlite_store.save_conversation_alias("src", target_id)
        except BaseException as exc:  # noqa: BLE001 — capture any race
            errors.append(exc)

    threads = [
        threading.Thread(target=_attach, args=("t-a",)),
        threading.Thread(target=_attach, args=("t-b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected errors under contention: {errors!r}"
    final = sqlite_store.resolve_conversation_alias("src")
    assert final in {"t-a", "t-b"}, (
        f"final alias target must be one of the contenders, got {final!r}"
    )


# ===========================================================================
# Structural guard: engine._store=None on VCATTACH handler entry (task #28).
#
# The unwrap site at handlers.py walks ``state.engine._store ->
# (._store)`` to extract the raw store passed to ``execute_attach``. In
# multi-worker prod a window exists where engine construction has
# completed enough for ``state.engine`` to be published into the cache
# but ``_store`` is still ``None`` (root cause tracked separately as a
# follow-up investigation; reproduction requires conditions a
# single-worker harness can't trigger).
#
# Without the guard, ``execute_attach`` reaches ``vcattach.py:187``'s
# ``store.save_conversation_alias(...)`` on ``None`` and raises
# ``AttributeError``, surfacing as an opaque HTTP 500 the client
# can't usefully retry on. The guard converts the condition into a
# retryable HTTP 503 so clients can retry past the transient window.
# ===========================================================================


@pytest.mark.parametrize(
    "state",
    [
        SimpleNamespace(engine=SimpleNamespace(_store=None)),
        SimpleNamespace(engine=None),
    ],
)
def test_vcattach_returns_503_when_engine_store_is_unavailable(state) -> None:
    """When ``state.engine._store`` is unavailable at handler entry, the
    VCATTACH path MUST return HTTP 503 with a retryable error payload,
    NOT raise an AttributeError out of ``execute_attach``."""
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    registry = SimpleNamespace(remove_conversation=lambda cid: None)
    result = SimpleNamespace(
        vcattach_label="some-target",
        conversation_id="src-conv-123",
        is_streaming=False,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH some-target"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            labels={"target-conv-id": "some-target"},
            conv_ids=["src-conv-123", "target-conv-id"],
        )
    )

    # 503 (retryable), not 500 (AttributeError surface).
    assert response.status_code == 503
    assert response.media_type == "application/json"
    assert response.body == (
        b'{"error":"engine state not ready; please retry","retryable":true}'
    )


def test_vcattach_returns_json_503_streaming_when_engine_store_is_none() -> None:
    """Streaming variant: when the request is_streaming and engine._store
    is None, response is a retryable JSON 503."""
    from starlette.responses import JSONResponse
    from virtual_context.proxy.handlers import _handle_vcattach
    from virtual_context.proxy.formats import detect_format

    state = SimpleNamespace(
        engine=SimpleNamespace(_store=None),
    )
    registry = SimpleNamespace(remove_conversation=lambda cid: None)
    result = SimpleNamespace(
        vcattach_label="some-target",
        conversation_id="src-conv-789",
        is_streaming=True,
    )
    fmt = detect_format({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "VCATTACH some-target"}],
    })

    response = asyncio.run(
        _handle_vcattach(
            result,
            fmt,
            state,
            registry,
            labels={"target-conv-id": "some-target"},
            conv_ids=["src-conv-789", "target-conv-id"],
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert response.body == (
        b'{"error":"engine state not ready; please retry","retryable":true}'
    )
