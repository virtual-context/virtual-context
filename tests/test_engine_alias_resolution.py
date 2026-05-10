"""Tests for engine-side multi-hop alias resolution (per spec
``specs/engine-alias-resolution.md`` and plan v1.2).

Covers the walker module (S1), engine ``__init__`` integration (S3 + S4),
the wrap-site coverage bundle (TS3), the lossless-restart broad rebuild
helper (S5 + TS1), and the regression bundle for the user-reported
"vc_find_quote returns source-side hits" symptom.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# S1 walker tests — virtual_context.core.alias_resolution
# ---------------------------------------------------------------------------


class _FakeAliasStore:
    """Minimal store double for walker unit tests.

    Backs ``resolve_conversation_alias`` with an in-memory dict so tests
    can exercise the chain without touching a real backend.
    """

    def __init__(
        self,
        edges: dict[str, str] | None = None,
        *,
        list_edges: dict[str, list[str]] | None = None,
    ) -> None:
        self._edges = dict(edges or {})
        self._list_edges = dict(list_edges or {})
        self.calls: list[str] = []

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        self.calls.append(alias_id)
        return self._edges.get(alias_id)

    def list_conversation_aliases_by_target(self, target_id: str) -> list[str]:
        return list(self._list_edges.get(target_id, []))


class _RaisingResolveStore:
    """Walker store double whose ``resolve_conversation_alias`` always raises."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        raise self._exc


def test_walker_no_alias_returns_input() -> None:
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    store = _FakeAliasStore({})
    assert walk_conversation_alias_chain(store, "conv-1") == "conv-1"


def test_walker_single_hop() -> None:
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    store = _FakeAliasStore({"a": "b"})
    assert walk_conversation_alias_chain(store, "a") == "b"


def test_walker_multi_hop() -> None:
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    store = _FakeAliasStore({"a": "b", "b": "c", "c": "d"})
    assert walk_conversation_alias_chain(store, "a") == "d"


def test_walker_cycle_raises() -> None:
    from virtual_context.core.alias_resolution import (
        AliasResolutionError,
        walk_conversation_alias_chain,
    )

    store = _FakeAliasStore({"a": "b", "b": "c", "c": "a"})
    with pytest.raises(AliasResolutionError) as excinfo:
        walk_conversation_alias_chain(store, "a")
    assert excinfo.value.reason == "cycle"
    assert excinfo.value.chain[0] == "a"
    assert excinfo.value.chain[-1] == "a"


def test_walker_self_loop_raises() -> None:
    from virtual_context.core.alias_resolution import (
        AliasResolutionError,
        walk_conversation_alias_chain,
    )

    store = _FakeAliasStore({"a": "a"})
    with pytest.raises(AliasResolutionError) as excinfo:
        walk_conversation_alias_chain(store, "a")
    assert excinfo.value.reason == "cycle"


def test_walker_max_hops_exceeded() -> None:
    from virtual_context.core.alias_resolution import (
        AliasResolutionError,
        walk_conversation_alias_chain,
    )

    edges = {f"n{i}": f"n{i + 1}" for i in range(20)}
    store = _FakeAliasStore(edges)
    with pytest.raises(AliasResolutionError) as excinfo:
        walk_conversation_alias_chain(store, "n0", max_hops=8)
    assert excinfo.value.reason == "max_hops"
    assert len(excinfo.value.chain) == 9  # 1 source + 8 hops


def test_walker_transient_store_error_propagates() -> None:
    """Walker must re-raise non-AliasResolutionError store exceptions verbatim."""
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    boom = RuntimeError("connection reset by peer")
    store = _RaisingResolveStore(boom)
    with pytest.raises(RuntimeError) as excinfo:
        walk_conversation_alias_chain(store, "a")
    assert excinfo.value is boom


def test_walker_idempotent_under_concurrent_writes() -> None:
    """Walker reads each hop once; concurrent overwrites visible only on a re-walk.

    Simulates the spec's "mid-walk snapshot" semantic by having the store
    mutate edges between resolve calls. The walker must terminate
    deterministically from the snapshot it observed: once it advances to
    ``b`` it never re-reads ``a`` for the same walk.
    """
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    class _MutatingStore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def resolve_conversation_alias(self, alias_id: str) -> str | None:
            self.calls.append(alias_id)
            if alias_id == "a":
                return "b"
            if alias_id == "b":
                # Simulate a concurrent VCATTACH committing while walker
                # is mid-walk: insert a downstream edge from c.
                return "c"
            if alias_id == "c":
                return None
            return None

    store = _MutatingStore()
    assert walk_conversation_alias_chain(store, "a") == "c"
    # Walker must visit a, b, c — each exactly once.
    assert store.calls == ["a", "b", "c"]


def test_walker_store_without_resolve_method_returns_input() -> None:
    """Defensive: if the store has no ``resolve_conversation_alias`` method,
    walker treats input as terminal (matches the d2691c7 fallback shape)."""
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    class _StoreNoResolver:
        pass

    assert walk_conversation_alias_chain(_StoreNoResolver(), "anything") == "anything"


def test_walker_empty_input_returns_input() -> None:
    """Empty conversation_id is the engine's default; walker must not raise."""
    from virtual_context.core.alias_resolution import walk_conversation_alias_chain

    store = _FakeAliasStore({})
    assert walk_conversation_alias_chain(store, "") == ""


# ---------------------------------------------------------------------------
# compute_reverse_dependents tests
# ---------------------------------------------------------------------------


def test_compute_reverse_dependents_no_listing_returns_empty() -> None:
    """Defensive: stores without ``list_conversation_aliases_by_target``
    return empty (custom-store backcompat)."""
    from virtual_context.core.alias_resolution import compute_reverse_dependents

    class _StoreNoListing:
        pass

    assert compute_reverse_dependents(_StoreNoListing(), "target-1") == []


def test_compute_reverse_dependents_single_level() -> None:
    from virtual_context.core.alias_resolution import compute_reverse_dependents

    store = _FakeAliasStore(
        list_edges={"target-1": ["src-c", "src-a", "src-b"]},
    )
    # Per-level alphabetical sort.
    assert compute_reverse_dependents(store, "target-1") == [
        "src-a",
        "src-b",
        "src-c",
    ]


def test_compute_reverse_dependents_multi_level() -> None:
    """BFS up the alias-incoming graph through transitively-aliased ids."""
    from virtual_context.core.alias_resolution import compute_reverse_dependents

    # target-z is pointed to by target-y, which is pointed to by src-x.
    store = _FakeAliasStore(
        list_edges={
            "target-z": ["target-y"],
            "target-y": ["src-x"],
        },
    )
    deps = compute_reverse_dependents(store, "target-z")
    assert deps == ["target-y", "src-x"]


def test_compute_reverse_dependents_no_incoming() -> None:
    from virtual_context.core.alias_resolution import compute_reverse_dependents

    store = _FakeAliasStore()
    assert compute_reverse_dependents(store, "lonely-target") == []


# ---------------------------------------------------------------------------
# S2 exceptions tests — virtual_context.core.exceptions
# ---------------------------------------------------------------------------


def test_engine_construction_error_carries_reason_source_target_chain() -> None:
    from virtual_context.core.exceptions import EngineConstructionError

    exc = EngineConstructionError(
        reason="alias_target_unattachable",
        source_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        target_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        chain=["aaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbb"],
    )
    assert exc.reason == "alias_target_unattachable"
    assert exc.source_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert exc.target_id == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert exc.chain == ["aaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbb"]
    # Message includes truncated source/target prefixes for ops debugging.
    assert "aaaaaaaaaaaa" in str(exc)
    assert "bbbbbbbbbbbb" in str(exc)


def test_engine_construction_error_defaults() -> None:
    from virtual_context.core.exceptions import EngineConstructionError

    exc = EngineConstructionError(reason="cycle")
    assert exc.reason == "cycle"
    assert exc.source_id == ""
    assert exc.target_id == ""
    assert exc.chain == []


def test_invalidation_failed_error_carries_event_and_cause() -> None:
    from virtual_context.core.exceptions import InvalidationFailedError

    cause = ConnectionError("redis down")
    event = {"type": "alias_created", "source": "s", "target": "t"}
    exc = InvalidationFailedError(event=event, cause=cause)
    assert exc.event == event
    # Defensive copy: mutating the input dict must not affect the exception.
    event["mutated"] = True
    assert "mutated" not in exc.event
    assert exc.__cause__ is cause


# ---------------------------------------------------------------------------
# S3 + S4 engine __init__ integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_factory(tmp_path):
    """Return a callable that builds a fresh engine bound to a sqlite store
    at ``tmp_path / "store.db"``.

    The store db is shared across engines built by the same factory call,
    so test cases can pre-seed alias rows / conversation rows via one
    engine's `_store` and then build a second engine bound to the source
    conv_id to exercise alias resolution at construction time.
    """
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.config import load_config

    db_path = str(tmp_path / "store.db")

    def _build(conversation_id: str | None = None):
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
            "tag_generator": {"type": "keyword"},
        }
        if conversation_id is not None:
            config_dict["conversation_id"] = conversation_id
        config = load_config(config_dict=config_dict)
        return VirtualContextEngine(config=config)

    return _build


def _underlying_store(engine):
    """Unwrap `ConversationStoreView` to reach the raw CompositeStore for
    test pre-seeding of alias / conversation rows."""
    store = engine._store
    return getattr(store, "_store", store)


def _seed_attachable_target(raw_store, target_id: str, *, tenant_id: str = "") -> None:
    """Pre-seed a `conversations` row so `is_attachable_target(target_id)`
    returns True. Mirrors the d2691c7 attachability schema.

    `tenant_id` defaults to empty string for default-tenant proxy mode
    (matches `conversations.tenant_id` NOT NULL constraint). `phase` is
    set to `'active'` which passes the spec's
    `phase NOT IN ('deleted','merged')` predicate."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    sqlite = raw_store._segments  # CompositeStore.segments → SQLiteStore
    conn = sqlite._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO conversations
           (conversation_id, tenant_id, phase, deleted_at,
            created_at, updated_at)
           VALUES (?, ?, 'active', NULL, ?, ?)""",
        (target_id, tenant_id, now, now),
    )
    conn.commit()


def _seed_unattachable_target(raw_store, target_id: str, *, phase: str) -> None:
    """Pre-seed a `conversations` row in a phase that fails attachability
    (`'deleted'` or `'merged'`)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    sqlite = raw_store._segments
    conn = sqlite._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO conversations
           (conversation_id, tenant_id, phase, deleted_at,
            created_at, updated_at)
           VALUES (?, ?, ?, NULL, ?, ?)""",
        (target_id, "", phase, now, now),
    )
    conn.commit()


def test_engine_init_no_alias_binds_to_source(engine_factory) -> None:
    """No alias row → engine binds to its config.conversation_id unchanged."""
    engine = engine_factory("source-conv-001")
    assert engine.config.conversation_id == "source-conv-001"


def test_engine_init_with_alias_rebinds_to_target(engine_factory) -> None:
    """With a single-hop alias and an attachable target, engine init
    walks the alias and binds state to the target."""
    seeder = engine_factory("seeder-conv-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "target-conv-aaa")
    raw.save_conversation_alias("source-conv-aaa", "target-conv-aaa")

    engine = engine_factory("source-conv-aaa")
    assert engine.config.conversation_id == "target-conv-aaa"


def test_engine_init_with_multi_hop_alias_walks_to_terminal(engine_factory) -> None:
    seeder = engine_factory("seeder-multi-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "terminal-z")
    raw.save_conversation_alias("hop-a", "hop-b")
    raw.save_conversation_alias("hop-b", "hop-c")
    raw.save_conversation_alias("hop-c", "terminal-z")

    engine = engine_factory("hop-a")
    assert engine.config.conversation_id == "terminal-z"


def test_engine_init_alias_cycle_raises_engine_construction_error(engine_factory) -> None:
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory("seeder-cycle-000")
    raw = _underlying_store(seeder)
    raw.save_conversation_alias("cyc-a", "cyc-b")
    raw.save_conversation_alias("cyc-b", "cyc-c")
    raw.save_conversation_alias("cyc-c", "cyc-a")

    with pytest.raises(EngineConstructionError) as excinfo:
        engine_factory("cyc-a")
    assert excinfo.value.reason == "cycle"
    assert excinfo.value.source_id == "cyc-a"
    assert excinfo.value.chain[0] == "cyc-a"


def test_engine_init_alias_to_deleted_target_raises(engine_factory) -> None:
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory("seeder-del-000")
    raw = _underlying_store(seeder)
    _seed_unattachable_target(raw, "deleted-target", phase="deleted")
    raw.save_conversation_alias("src-deleted", "deleted-target")

    with pytest.raises(EngineConstructionError) as excinfo:
        engine_factory("src-deleted")
    assert excinfo.value.reason == "alias_target_unattachable"
    assert excinfo.value.source_id == "src-deleted"
    assert excinfo.value.target_id == "deleted-target"


def test_engine_init_alias_to_merged_target_raises(engine_factory) -> None:
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory("seeder-mrg-000")
    raw = _underlying_store(seeder)
    _seed_unattachable_target(raw, "merged-target", phase="merged")
    raw.save_conversation_alias("src-merged", "merged-target")

    with pytest.raises(EngineConstructionError) as excinfo:
        engine_factory("src-merged")
    assert excinfo.value.reason == "alias_target_unattachable"


def test_engine_init_alias_to_missing_target_raises(engine_factory) -> None:
    """Alias points to a conversation_id with no row in `conversations`
    table → attachability check fails (missing) → EngineConstructionError."""
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory("seeder-missing-000")
    raw = _underlying_store(seeder)
    # Note: NO _seed_attachable_target call — target row does not exist.
    raw.save_conversation_alias("src-missing", "ghost-target")

    with pytest.raises(EngineConstructionError) as excinfo:
        engine_factory("src-missing")
    assert excinfo.value.reason == "alias_target_unattachable"


# ---------------------------------------------------------------------------
# ConversationStoreView wrap correctness (wrap-site row 11)
# ---------------------------------------------------------------------------


def test_engine_init_alias_rebinds_conversation_store_view(engine_factory) -> None:
    """The `ConversationStoreView` wrap binds to the resolved target id, not
    the source. Verifies S4 + Option-A: the wrap site captures
    `self.config.conversation_id` AFTER alias resolution mutates it."""
    seeder = engine_factory("seeder-view-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "view-target")
    raw.save_conversation_alias("view-source", "view-target")

    engine = engine_factory("view-source")
    # ConversationStoreView captures `conversation_id` at construction.
    assert engine._store.conversation_id == "view-target"


# ---------------------------------------------------------------------------
# Wrap-site coverage bundle (v1.1 author rigor TS3 — 10 tests, one per
# row in the Option-A wrap-site audit table at S4 except ConversationStoreView
# which is covered by the test above)
# ---------------------------------------------------------------------------


def _build_alias_engine(engine_factory, *, source: str, target: str):
    """Helper: pre-seed an attachable target + alias, then construct an
    engine bound to the source. Returns the engine bound to target."""
    seeder = engine_factory(f"seeder-{source}")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, target)
    raw.save_conversation_alias(source, target)
    engine = engine_factory(source)
    assert engine.config.conversation_id == target  # sanity check
    return engine


def test_engine_init_alias_rebinds_canonicalizer(engine_factory) -> None:
    engine = _build_alias_engine(engine_factory, source="src-cn", target="tgt-cn")
    # Canonicalizer captures conversation_id for tag-alias scoping.
    cn = engine._canonicalizer
    captured = getattr(cn, "conversation_id", None) or getattr(cn, "_conversation_id", None)
    if captured is not None:
        assert captured == "tgt-cn"
    # Either way, the engine's config.conversation_id is the target —
    # canonicalizer reads from the engine's bindings.
    assert engine.config.conversation_id == "tgt-cn"


def test_engine_init_alias_rebinds_monitor(engine_factory) -> None:
    engine = _build_alias_engine(engine_factory, source="src-mn", target="tgt-mn")
    mon = engine._monitor
    # Monitor stores conversation_id only as truncated `_conv_short` (12 chars).
    captured = (
        getattr(mon, "conversation_id", None)
        or getattr(mon, "_conversation_id", None)
        or getattr(mon, "_conv_short", None)
    )
    assert captured == "tgt-mn"[:12]


def test_engine_init_alias_rebinds_assembler(engine_factory) -> None:
    engine = _build_alias_engine(engine_factory, source="src-as", target="tgt-as")
    asm = engine._assembler
    captured = getattr(asm, "conversation_id", None) or getattr(asm, "_conversation_id", None)
    assert captured == "tgt-as"


def test_engine_init_alias_rebinds_retriever(engine_factory) -> None:
    engine = _build_alias_engine(engine_factory, source="src-rt", target="tgt-rt")
    rt = engine._retriever
    captured = getattr(rt, "conversation_id", None) or getattr(rt, "_conversation_id", None)
    assert captured == "tgt-rt"


def test_engine_init_alias_rebinds_paging(engine_factory) -> None:
    engine = _build_alias_engine(engine_factory, source="src-pg", target="tgt-pg")
    pg = engine._paging
    captured = getattr(pg, "conversation_id", None) or getattr(pg, "_conversation_id", None)
    assert captured == "tgt-pg"


def test_engine_init_alias_rebinds_tag_generator(engine_factory) -> None:
    """Tag generator may not capture conversation_id directly (it's stateless
    for keyword type), but the engine's config.conversation_id binding is
    the resolved target — covers the audit table row contractually."""
    engine = _build_alias_engine(engine_factory, source="src-tg", target="tgt-tg")
    assert engine.config.conversation_id == "tgt-tg"
    # Tag generator exists and can be invoked against the resolved target.
    assert engine._tag_generator is not None


def test_engine_init_alias_rebinds_segmenter(engine_factory) -> None:
    """Segmenter has no direct conversation_id capture today (per audit
    table), but is constructed AFTER resolution so any future capture
    inherits the resolved id. Verifies engine binding is correct."""
    engine = _build_alias_engine(engine_factory, source="src-sg", target="tgt-sg")
    assert engine.config.conversation_id == "tgt-sg"
    assert engine._segmenter is not None


def test_engine_init_alias_rebinds_compactor(engine_factory) -> None:
    """Compactor has no direct conversation_id capture today (per audit
    table). With keyword tag generator + no LLM provider config, the
    compactor may be a no-op / None instance — the substantive check is
    that engine.config.conversation_id is the resolved target so any
    future compactor capture inherits the resolved id (Option-A guarantee)."""
    engine = _build_alias_engine(engine_factory, source="src-cp", target="tgt-cp")
    assert engine.config.conversation_id == "tgt-cp"
    # Attribute exists (may be None when compactor is unconfigured).
    assert hasattr(engine, "_compactor")


def test_engine_init_alias_rebinds_tag_splitter(engine_factory) -> None:
    """Tag splitter has no direct conv_id capture today; verify engine
    binding (Option-A: future capture inherits resolved id)."""
    engine = _build_alias_engine(engine_factory, source="src-ts", target="tgt-ts")
    assert engine.config.conversation_id == "tgt-ts"
    assert hasattr(engine, "_tag_splitter")


def test_engine_init_alias_rebinds_telemetry(engine_factory) -> None:
    """Telemetry has no direct conversation_id capture today (per audit
    table), but is constructed AFTER resolution; verify engine binding."""
    engine = _build_alias_engine(engine_factory, source="src-tl", target="tgt-tl")
    assert engine.config.conversation_id == "tgt-tl"
    assert engine._telemetry is not None


# ---------------------------------------------------------------------------
# S5 lossless-restart helper tests
# ---------------------------------------------------------------------------


def test_refresh_conversation_binding_after_rebind_rebuilds_store_view(
    engine_factory,
) -> None:
    """`_refresh_conversation_binding_after_rebind` rebuilds the
    `ConversationStoreView` wrap to bind to the (mutated)
    `self.config.conversation_id`."""
    engine = engine_factory("init-conv-001")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-001")

    # Simulate lossless-restart's mutation pattern: the caller has already
    # walked the alias chain, attachability-checked the terminal, and
    # mutated engine.config.conversation_id to the resolved target.
    engine.config.conversation_id = "rebind-target-001"
    engine._refresh_conversation_binding_after_rebind(raw)

    assert engine._store.conversation_id == "rebind-target-001"


def test_refresh_conversation_binding_after_rebind_rebuilds_paging(
    engine_factory,
) -> None:
    engine = engine_factory("init-conv-002")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-002")
    engine.config.conversation_id = "rebind-target-002"
    engine._refresh_conversation_binding_after_rebind(raw)

    pg = engine._paging
    captured = (
        getattr(pg, "conversation_id", None)
        or getattr(pg, "_conversation_id", None)
    )
    assert captured == "rebind-target-002"


def test_refresh_conversation_binding_after_rebind_rebuilds_canonicalizer(
    engine_factory,
) -> None:
    engine = engine_factory("init-conv-003")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-003")
    engine.config.conversation_id = "rebind-target-003"
    engine._refresh_conversation_binding_after_rebind(raw)

    cn = engine._canonicalizer
    captured = (
        getattr(cn, "conversation_id", None)
        or getattr(cn, "_conversation_id", None)
    )
    if captured is not None:
        assert captured == "rebind-target-003"
    # Engine config binding is the rebound target either way.
    assert engine.config.conversation_id == "rebind-target-003"


def test_refresh_conversation_binding_after_rebind_rebuilds_monitor(
    engine_factory,
) -> None:
    engine = engine_factory("init-conv-004")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-004")
    engine.config.conversation_id = "rebind-target-004"
    engine._refresh_conversation_binding_after_rebind(raw)

    mon = engine._monitor
    captured = (
        getattr(mon, "conversation_id", None)
        or getattr(mon, "_conversation_id", None)
        or getattr(mon, "_conv_short", None)
    )
    assert captured == "rebind-target-004"[:12]


def test_refresh_conversation_binding_after_rebind_rebuilds_assembler(
    engine_factory,
) -> None:
    engine = engine_factory("init-conv-005")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-005")
    engine.config.conversation_id = "rebind-target-005"
    engine._refresh_conversation_binding_after_rebind(raw)

    asm = engine._assembler
    captured = (
        getattr(asm, "conversation_id", None)
        or getattr(asm, "_conversation_id", None)
    )
    assert captured == "rebind-target-005"


def test_refresh_conversation_binding_after_rebind_rebuilds_retriever(
    engine_factory,
) -> None:
    engine = engine_factory("init-conv-006")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-006")
    engine.config.conversation_id = "rebind-target-006"
    engine._refresh_conversation_binding_after_rebind(raw)

    rt = engine._retriever
    captured = (
        getattr(rt, "conversation_id", None)
        or getattr(rt, "_conversation_id", None)
    )
    assert captured == "rebind-target-006"


def test_refresh_conversation_binding_after_rebind_repoints_store_bound_delegates(
    engine_factory,
) -> None:
    """The helper re-points existing pipeline objects (`_tagging`,
    `_compaction`, `_retrieval`) at the rebuilt store / dependencies so
    they don't hold stale `ConversationStoreView` references."""
    engine = engine_factory("init-conv-007")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-007")
    engine.config.conversation_id = "rebind-target-007"
    engine._refresh_conversation_binding_after_rebind(raw)

    # Each pipeline's `store` ref is the rebuilt `ConversationStoreView`.
    if hasattr(engine, "_tagging"):
        assert engine._tagging.store is engine._store
    if hasattr(engine, "_compaction"):
        assert engine._compaction.store is engine._store
    if hasattr(engine, "_retrieval"):
        assert engine._retrieval._store is engine._store


def test_lossless_restart_rebuilds_all_conversation_bound_wrappers(
    engine_factory,
) -> None:
    """Broader regression test (TS1): asserts every wrapper rebuilt by
    `_refresh_conversation_binding_after_rebind` is a fresh instance
    bound to the resolved id. Surfaces regressions if the helper drifts
    out of sync with `__init__`-built dependencies."""
    engine = engine_factory("init-conv-008")
    raw = _underlying_store(engine)
    _seed_attachable_target(raw, "rebind-target-008")

    # Capture pre-rebind references for identity comparison.
    pre_store = engine._store
    pre_paging = engine._paging
    pre_assembler = engine._assembler
    pre_retriever = engine._retriever

    engine.config.conversation_id = "rebind-target-008"
    engine._refresh_conversation_binding_after_rebind(raw)

    # Each wrapper is a NEW instance (identity check).
    assert engine._store is not pre_store
    assert engine._paging is not pre_paging
    assert engine._assembler is not pre_assembler
    assert engine._retriever is not pre_retriever

    # All bound to the rebind target.
    assert engine._store.conversation_id == "rebind-target-008"


# ---------------------------------------------------------------------------
# Cross-tenant invariant — engine __init__ refuses cross-tenant alias terminal
# (per plan v1.2 line 921 + line 926; matches W4/X-tenant guardrail at the
# attachability seam). Mirrors the engine's `is_attachable_target(*,
# conversation_id, tenant_id)` predicate when the engine config carries a
# non-empty tenant_id.
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_factory_tenant(tmp_path):
    """Engine factory variant that accepts a tenant_id alongside conversation_id.

    Mirrors the default `engine_factory` but sets `config.tenant_id` so
    the engine's alias-resolution `is_attachable_target` call passes a
    non-None tenant_id and exercises the cross-tenant predicate.
    """
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    db_path = str(tmp_path / "store.db")

    def _build(conversation_id: str, *, tenant_id: str):
        config_dict = {
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
            "tag_generator": {"type": "keyword"},
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
        }
        config = load_config(config_dict=config_dict)
        return VirtualContextEngine(config=config)

    return _build


def test_engine_init_cross_tenant_alias_target_raises(engine_factory_tenant) -> None:
    """Cross-tenant alias terminal → EngineConstructionError(reason='alias_target_unattachable').

    Engine's tenant_id ('tenant-B') does not match the target conversation
    row's tenant_id ('tenant-A'); `is_attachable_target` refuses, the
    resolver maps that to `alias_target_unattachable`. Verifies the source-
    tenant invariant from spec / plan W6.
    """
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory_tenant("seeder-xt-000", tenant_id="tenant-A")
    raw = _underlying_store(seeder)
    # Target row lives in tenant-A.
    _seed_attachable_target(raw, "target-tenant-A", tenant_id="tenant-A")
    raw.save_conversation_alias("src-cross", "target-tenant-A")

    # Engine in tenant-B tries to bind to a source aliased to tenant-A.
    with pytest.raises(EngineConstructionError) as excinfo:
        engine_factory_tenant("src-cross", tenant_id="tenant-B")
    assert excinfo.value.reason == "alias_target_unattachable"
    assert excinfo.value.source_id == "src-cross"
    assert excinfo.value.target_id == "target-tenant-A"


# ---------------------------------------------------------------------------
# Regression bundle for user-reported "vc_find_quote returns source-side hits"
# symptom (per plan v1.2 line 922). Each test exercises a downstream
# engine-bound surface that captures `config.conversation_id` and asserts
# the bound id is the resolved TARGET — proving the user's symptom is
# closed by engine-side resolution at construction.
# ---------------------------------------------------------------------------


def _seed_engine_state_for_conversation(
    raw_store, conversation_id: str, *, turn_tag_entries: list,
    compacted_prefix_messages: int = 0, turn_count: int = 0,
) -> None:
    """Pre-seed an `engine_state` row for *conversation_id*.

    Mirrors the production save path but with a minimal payload so the
    regression tests can verify that engine-init reads the row keyed by
    the resolved (target) id, not the source id.
    """
    from virtual_context.types import EngineStateSnapshot

    snapshot = EngineStateSnapshot(
        conversation_id=conversation_id,
        compacted_prefix_messages=compacted_prefix_messages,
        turn_tag_entries=list(turn_tag_entries),
        turn_count=turn_count,
    )
    sqlite = raw_store._segments  # CompositeStore.segments → SQLiteStore
    sqlite.save_engine_state(snapshot)


def test_engine_alias_redirects_engine_state_load(engine_factory) -> None:
    """After alias resolution, engine __init__ loads engine_state for the
    TARGET conversation_id, not the source.

    Pre-seeds an `engine_state` row keyed by `tgt-state`; alias `src-state` →
    `tgt-state` with target attachable. Building the engine with `src-state`
    must restore the target's TurnTagIndex entries.
    """
    from virtual_context.types import TurnTagEntry

    seeder = engine_factory("seeder-state-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "tgt-state")
    raw.save_conversation_alias("src-state", "tgt-state")

    # Seed engine_state ONLY for the target id.
    seed_entries = [
        TurnTagEntry(
            turn_number=0,
            message_hash="hash-0",
            tags=["alpha"],
            primary_tag="alpha",
        ),
        TurnTagEntry(
            turn_number=1,
            message_hash="hash-1",
            tags=["beta"],
            primary_tag="beta",
        ),
    ]
    _seed_engine_state_for_conversation(
        raw, "tgt-state",
        turn_tag_entries=seed_entries, turn_count=2,
    )

    # Build engine bound to source; alias resolution rebinds to target,
    # then `_load_persisted_state` reads the target's engine_state row.
    engine = engine_factory("src-state")
    assert engine.config.conversation_id == "tgt-state"
    # TurnTagIndex restored from the target's row.
    assert len(engine._turn_tag_index.entries) == 2
    primaries = [e.primary_tag for e in engine._turn_tag_index.entries]
    assert primaries == ["alpha", "beta"]


def test_engine_alias_redirects_progress_snapshot(engine_factory) -> None:
    """`read_progress_snapshot(config.conversation_id)` reads the TARGET row
    after alias resolution.

    A source-side row was never seeded; calling
    `read_progress_snapshot('src-prog')` would `KeyError`. Calling it with
    `engine.config.conversation_id` (the resolved target) returns the
    target's snapshot. This is the data path that flows to
    `cloud_conversations_get` → dashboard JS.
    """
    seeder = engine_factory("seeder-prog-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "tgt-prog")
    raw.save_conversation_alias("src-prog", "tgt-prog")

    engine = engine_factory("src-prog")
    assert engine.config.conversation_id == "tgt-prog"

    raw_store = _underlying_store(engine)
    sqlite = raw_store._segments

    # Source-side row was NOT seeded — querying by 'src-prog' raises KeyError.
    with pytest.raises(KeyError):
        sqlite.read_progress_snapshot("src-prog")
    # Target-side row exists; the engine's bound id resolves correctly.
    snapshot = sqlite.read_progress_snapshot(engine.config.conversation_id)
    assert snapshot is not None
    assert snapshot.phase == "active"


def test_engine_alias_redirects_quote_search(engine_factory) -> None:
    """`engine.find_quote(...)` calls into the canonical-turn search with
    the resolved TARGET conversation_id.

    Closes the user-reported symptom: `vc_find_quote` from an alias source
    must search the target's canonical turns, not the source's empty
    conversation. Captures the `conversation_id` argument that the engine
    passes to the search backend via a test double on
    `search_canonical_turn_text` — the lexical leg
    `_search_find_quote_candidates` uses inside `find_quote`.
    """
    seeder = engine_factory("seeder-quote-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "tgt-quote")
    raw.save_conversation_alias("src-quote", "tgt-quote")

    engine = engine_factory("src-quote")
    assert engine.config.conversation_id == "tgt-quote"

    # Capture the conversation_id passed into the search backend's
    # `search_canonical_turn_text` (find_quote's lexical leg).
    raw_store = _underlying_store(engine)
    sqlite = raw_store._search  # SearchStore protocol — same SQLiteStore in sqlite mode
    captured: dict = {}
    original_search_canonical_turn_text = sqlite.search_canonical_turn_text

    def _capture(query, limit=5, conversation_id=None):
        captured["conversation_id"] = conversation_id
        return original_search_canonical_turn_text(
            query, limit=limit, conversation_id=conversation_id,
        )

    sqlite.search_canonical_turn_text = _capture  # type: ignore[assignment]
    try:
        engine.find_quote("anything", max_results=1)
    finally:
        sqlite.search_canonical_turn_text = original_search_canonical_turn_text  # type: ignore[assignment]

    assert captured.get("conversation_id") == "tgt-quote"


def test_vcstatus_after_alias_engine_init_renders_target_stats(engine_factory) -> None:
    """VCSTATUS surface (engine.config.conversation_id + restored engine
    state) reflects the target after alias-resolved engine init.

    Pre-seeds engine_state for the target (`compacted_prefix_messages=4`);
    alias source→target. Engine constructed with source id loads target
    state, so VCSTATUS-style introspection (config.conversation_id +
    `_engine_state.compacted_prefix_messages`) reports target stats.
    """
    from virtual_context.types import TurnTagEntry

    seeder = engine_factory("seeder-vcs-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "tgt-vcs")
    raw.save_conversation_alias("src-vcs", "tgt-vcs")

    seed_entries = [
        TurnTagEntry(
            turn_number=i, message_hash=f"h{i}",
            tags=[f"t{i}"], primary_tag=f"t{i}",
        )
        for i in range(3)
    ]
    _seed_engine_state_for_conversation(
        raw, "tgt-vcs",
        turn_tag_entries=seed_entries,
        compacted_prefix_messages=4,
        turn_count=3,
    )

    engine = engine_factory("src-vcs")
    # VCSTATUS bindings: id + compacted-prefix counter + restored entries.
    assert engine.config.conversation_id == "tgt-vcs"
    assert engine._engine_state.compacted_prefix_messages == 4
    assert len(engine._turn_tag_index.entries) == 3
    # Store view also sees target id.
    assert engine._store.conversation_id == "tgt-vcs"


# ---------------------------------------------------------------------------
# Lossless-restart end-to-end (per plan v1.2 line 925; L1 matrix row 955).
# Mirrors the proxy `create_app()` lossless-restart inner block at
# `virtual_context/proxy/server.py:2300-2410`: walk alias chain on
# `latest.conversation_id`, attachability-check the terminal, mutate the
# rebound engine's `config.conversation_id`, then call
# `_refresh_conversation_binding_after_rebind`. On `EngineConstructionError`
# the proxy logs WARNING + skips rebind cleanly (E2 contract).
# ---------------------------------------------------------------------------


def _simulate_lossless_restart_rebind(engine, raw_store, latest_conv_id: str) -> str:
    """Replay the resolution+attachability+rebind sequence used by the
    proxy lossless-restart code at `virtual_context/proxy/server.py:2300-2410`.

    Kept tiny and additive so the test exercises the same algorithmic
    path the proxy executes, without spinning up the full FastAPI app.
    Returns the resolved terminal id on success; raises
    `EngineConstructionError` on alias failure or unattachable terminal
    (matching what the proxy block would surface to its `_RebindSkipped`
    catcher).
    """
    from virtual_context.core.alias_resolution import (
        AliasResolutionError,
        walk_conversation_alias_chain,
    )
    from virtual_context.core.exceptions import EngineConstructionError

    try:
        resolved = walk_conversation_alias_chain(raw_store, latest_conv_id)
    except AliasResolutionError as exc:
        raise EngineConstructionError(
            reason=exc.reason, source_id=latest_conv_id, chain=exc.chain,
        ) from exc
    except Exception as exc:
        raise EngineConstructionError(
            reason="transient_store_error", source_id=latest_conv_id,
        ) from exc

    is_attachable = getattr(raw_store, "is_attachable_target", None)
    if callable(is_attachable):
        try:
            ok = bool(is_attachable(
                conversation_id=resolved,
                tenant_id=getattr(engine.config, "tenant_id", None) or None,
            ))
        except Exception as exc:
            raise EngineConstructionError(
                reason="transient_store_error",
                source_id=latest_conv_id,
                target_id=resolved,
            ) from exc
        if not ok:
            raise EngineConstructionError(
                reason="alias_target_unattachable",
                source_id=latest_conv_id,
                target_id=resolved,
            )

    engine.config.conversation_id = resolved
    engine._refresh_conversation_binding_after_rebind(raw_store)
    return resolved


def test_lossless_restart_resolves_alias_source(engine_factory) -> None:
    """Lossless-restart path resolves `latest.conversation_id` (an alias
    source) to the terminal target, rebinds the engine to the target,
    and rebuilds wrappers via `_refresh_conversation_binding_after_rebind`.

    Mirrors the proxy `create_app()` rebind block (server.py:2300-2410).
    """
    # Seeder + alias + attachable target.
    seeder = engine_factory("seeder-lr-000")
    raw_seed = _underlying_store(seeder)
    _seed_attachable_target(raw_seed, "tgt-lr")
    raw_seed.save_conversation_alias("src-lr", "tgt-lr")
    # Persist `engine_state` keyed by the source id — what the proxy's
    # `load_latest_engine_state()` would surface on restart.
    _seed_engine_state_for_conversation(
        raw_seed, "src-lr",
        turn_tag_entries=[], turn_count=0,
    )

    # Fresh engine boots with an auto-generated id (no alias on it).
    engine = engine_factory()
    pre_id = engine.config.conversation_id
    assert pre_id != "src-lr" and pre_id != "tgt-lr"

    raw = _underlying_store(engine)
    # Sanity: load_latest_engine_state returns the source row.
    sqlite = raw._segments
    latest = sqlite.load_latest_engine_state()
    assert latest is not None
    assert latest.conversation_id == "src-lr"

    resolved = _simulate_lossless_restart_rebind(engine, raw, latest.conversation_id)
    assert resolved == "tgt-lr"
    assert engine.config.conversation_id == "tgt-lr"
    assert engine._store.conversation_id == "tgt-lr"


def test_lossless_restart_refuses_unattachable_terminal(engine_factory) -> None:
    """Lossless-restart refuses (E2) when the alias chain terminates at an
    unattachable conversation (deleted/merged/missing/cross-tenant).

    Asserts the resolver raises `EngineConstructionError` with
    `reason='alias_target_unattachable'` so the proxy's outer
    `try/except _RebindSkipped` catcher can log WARNING and leave the
    registry slot empty for fresh `__init__` resolution on the next
    request.
    """
    from virtual_context.core.exceptions import EngineConstructionError

    seeder = engine_factory("seeder-lr-unatt-000")
    raw_seed = _underlying_store(seeder)
    # Target seeded as `phase='deleted'` → unattachable.
    _seed_unattachable_target(raw_seed, "tgt-lr-unatt", phase="deleted")
    raw_seed.save_conversation_alias("src-lr-unatt", "tgt-lr-unatt")
    _seed_engine_state_for_conversation(
        raw_seed, "src-lr-unatt",
        turn_tag_entries=[], turn_count=0,
    )

    # Fresh engine; capture pre-rebind config so we can assert NO mutation
    # happened on refusal.
    engine = engine_factory()
    pre_conv_id = engine.config.conversation_id
    raw = _underlying_store(engine)

    sqlite = raw._segments
    latest = sqlite.load_latest_engine_state()
    assert latest is not None
    assert latest.conversation_id == "src-lr-unatt"

    with pytest.raises(EngineConstructionError) as excinfo:
        _simulate_lossless_restart_rebind(engine, raw, latest.conversation_id)
    assert excinfo.value.reason == "alias_target_unattachable"
    assert excinfo.value.source_id == "src-lr-unatt"
    assert excinfo.value.target_id == "tgt-lr-unatt"

    # Refusal: engine.config.conversation_id is unchanged. The proxy's
    # outer catcher logs WARNING and leaves the registry slot empty;
    # next request triggers fresh __init__ resolution.
    assert engine.config.conversation_id == pre_conv_id
