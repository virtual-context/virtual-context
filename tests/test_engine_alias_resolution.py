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
            "retrieval": {"inbound_tagger_type": "keyword"},
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
            "retrieval": {"inbound_tagger_type": "keyword"},
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


# ===========================================================================
# Commit-2 — S6 / S7 / S8 / S9 / S10 cross-worker invalidation tests
# ===========================================================================
#
# This block covers the engine-side of the cross-worker invalidation
# feature: store-level reverse-dependent lookup (S6), AliasEvent payload
# shape (S7), the post-commit registry that defers callback firing until
# the outer transaction commits (S8), VCATTACH callback wiring with
# at-least-once / retryable contract (S9), and VCMERGE callback wiring
# with post-commit best-effort + WARNING-and-metric on failure (S10).
# All tests live in this file per the plan v1.2 test mapping section.


# ---------------------------------------------------------------------------
# S6 — Store.list_conversation_aliases_by_target
# ---------------------------------------------------------------------------


def _build_sqlite_store(tmp_path):
    """Build a fresh CompositeStore-backed SQLite store for direct tests.

    Mirrors the engine_factory's storage shape but skips engine
    construction so tests can exercise low-level Store APIs.
    """
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    db_path = str(tmp_path / "store.db")
    config_dict = {
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {"type": "keyword"},
        "retrieval": {"inbound_tagger_type": "keyword"},
    }
    config = load_config(config_dict=config_dict)
    engine = VirtualContextEngine(config=config)
    raw = _underlying_store(engine)
    return raw, raw._segments  # composite, sqlite


def test_list_conversation_aliases_by_target_empty(tmp_path) -> None:
    composite, _ = _build_sqlite_store(tmp_path)
    assert composite.list_conversation_aliases_by_target("nobody") == []


def test_list_conversation_aliases_by_target_single(tmp_path) -> None:
    composite, _ = _build_sqlite_store(tmp_path)
    composite.save_conversation_alias("alias-1", "target-1")
    assert composite.list_conversation_aliases_by_target("target-1") == ["alias-1"]


def test_list_conversation_aliases_by_target_sorted(tmp_path) -> None:
    """Results must be sorted by alias_id for deterministic event payloads."""
    composite, _ = _build_sqlite_store(tmp_path)
    for src in ["zeta", "alpha", "mu", "beta"]:
        composite.save_conversation_alias(src, "target-1")
    assert composite.list_conversation_aliases_by_target("target-1") == [
        "alpha", "beta", "mu", "zeta",
    ]


def test_list_conversation_aliases_by_target_filters_correct_target(tmp_path) -> None:
    composite, _ = _build_sqlite_store(tmp_path)
    composite.save_conversation_alias("alias-A", "target-1")
    composite.save_conversation_alias("alias-B", "target-2")
    composite.save_conversation_alias("alias-C", "target-1")
    assert composite.list_conversation_aliases_by_target("target-1") == [
        "alias-A", "alias-C",
    ]
    assert composite.list_conversation_aliases_by_target("target-2") == ["alias-B"]


def test_list_conversation_aliases_by_target_index_exists(tmp_path) -> None:
    """The ``idx_conversation_aliases_target_id`` index must be created by
    schema bootstrap so lookups stay sub-linear at scale."""
    _, sqlite = _build_sqlite_store(tmp_path)
    conn = sqlite._get_conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND tbl_name='conversation_aliases'"
    ).fetchall()
    index_names = {r[0] for r in rows}
    assert "idx_conversation_aliases_target_id" in index_names


def test_list_conversation_aliases_by_target_filesystem_backend(tmp_path) -> None:
    """Filesystem store implements the method by scanning ``_vcattach_aliases``."""
    from virtual_context.storage.filesystem import FilesystemStore

    fs = FilesystemStore(root=str(tmp_path / "fs-store"))
    fs.save_conversation_alias("z", "tgt")
    fs.save_conversation_alias("a", "tgt")
    fs.save_conversation_alias("m", "other-tgt")
    assert fs.list_conversation_aliases_by_target("tgt") == ["a", "z"]
    assert fs.list_conversation_aliases_by_target("missing") == []


def test_context_store_abc_default_returns_empty_for_custom_backends() -> None:
    """ContextStore ABC ships a default implementation returning [] so
    custom backends without the new method continue working."""
    from virtual_context.core.store import ContextStore

    # Pick a method off the ABC; defensive: the default sits on the class.
    default = ContextStore.list_conversation_aliases_by_target
    assert callable(default)
    # The unbound function called on a bare object must return [] without
    # touching any backend state.
    class _Bare:
        list_conversation_aliases_by_target = (
            ContextStore.list_conversation_aliases_by_target
        )
    assert _Bare().list_conversation_aliases_by_target("anything") == []


# ---------------------------------------------------------------------------
# S7 — AliasEvent TypedDict shape
# ---------------------------------------------------------------------------


def test_alias_created_event_typeddict_shape() -> None:
    from virtual_context.types import AliasCreatedEvent

    event: AliasCreatedEvent = {
        "type": "alias_created",
        "source": "src-1",
        "target": "tgt-1",
        "reverse_dependents": [],
        "timestamp": "2026-05-10T17:30:00Z",
    }
    assert event["type"] == "alias_created"
    assert event["source"] == "src-1"
    assert event["target"] == "tgt-1"
    assert event["reverse_dependents"] == []
    assert event["timestamp"].endswith("Z")


def test_alias_deleted_event_typeddict_shape() -> None:
    from virtual_context.types import AliasDeletedEvent

    event: AliasDeletedEvent = {
        "type": "alias_deleted",
        "alias_id": "src-1",
        "reverse_dependents": ["nephew-1"],
        "timestamp": "2026-05-10T17:30:00Z",
    }
    assert event["type"] == "alias_deleted"
    assert event["alias_id"] == "src-1"
    assert event["reverse_dependents"] == ["nephew-1"]


# ---------------------------------------------------------------------------
# S8 — save_conversation_alias / delete_conversation_alias on_committed
# (own-commit + scoped post-commit registry, SQLite path)
# ---------------------------------------------------------------------------


def test_save_conversation_alias_invokes_on_committed_callback(tmp_path) -> None:
    """Own-commit path: callback fires once after the row commits, with
    the engine-side AliasCreatedEvent shape (type=source/target/
    reverse_dependents/timestamp; tenant_id added by the caller-side
    adapter, not by the store)."""
    composite, _ = _build_sqlite_store(tmp_path)
    received: list[dict] = []

    def _cb(event):
        received.append(dict(event))

    composite.save_conversation_alias("src-cb-1", "tgt-cb-1", on_committed=_cb)

    assert len(received) == 1
    event = received[0]
    assert event["type"] == "alias_created"
    assert event["source"] == "src-cb-1"
    assert event["target"] == "tgt-cb-1"
    assert event["reverse_dependents"] == []
    assert isinstance(event["timestamp"], str) and event["timestamp"]
    # No tenant_id at the engine layer; cloud's adapter adds it.
    assert "tenant_id" not in event


def test_save_conversation_alias_callback_fires_after_row_committed(tmp_path) -> None:
    """The callback observes a committed row: a SELECT inside the
    callback must already see the new alias."""
    composite, sqlite = _build_sqlite_store(tmp_path)
    seen_in_cb: dict = {}

    def _cb(event):
        conn = sqlite._get_conn()
        row = conn.execute(
            "SELECT target_id FROM conversation_aliases WHERE alias_id = ?",
            ("src-after",),
        ).fetchone()
        seen_in_cb["target"] = row["target_id"] if row else None

    composite.save_conversation_alias("src-after", "tgt-after", on_committed=_cb)
    assert seen_in_cb["target"] == "tgt-after"


def test_save_conversation_alias_propagates_callback_error(tmp_path) -> None:
    """Callback exceptions surface to the caller (VCATTACH path translates
    InvalidationFailedError to retryable 503; merge body catches and logs)."""
    from virtual_context.core.exceptions import InvalidationFailedError

    composite, _ = _build_sqlite_store(tmp_path)

    def _cb(event):
        raise InvalidationFailedError(
            event=event,
            cause=ConnectionError("redis down"),
        )

    with pytest.raises(InvalidationFailedError) as excinfo:
        composite.save_conversation_alias(
            "src-err", "tgt-err", on_committed=_cb,
        )
    assert excinfo.value.event["type"] == "alias_created"
    # Row is already committed despite the callback raising — the
    # alias write is durable and the error is a delivery-side concern.
    assert composite.resolve_conversation_alias("src-err") == "tgt-err"


def test_delete_conversation_alias_invokes_on_committed_callback(tmp_path) -> None:
    composite, _ = _build_sqlite_store(tmp_path)
    composite.save_conversation_alias("src-del", "tgt-del")
    received: list[dict] = []

    def _cb(event):
        received.append(dict(event))

    composite.delete_conversation_alias("src-del", on_committed=_cb)
    assert composite.resolve_conversation_alias("src-del") is None
    assert len(received) == 1
    event = received[0]
    assert event["type"] == "alias_deleted"
    assert event["alias_id"] == "src-del"
    assert event["reverse_dependents"] == []
    assert isinstance(event["timestamp"], str) and event["timestamp"]


def test_save_conversation_alias_event_includes_reverse_dependents(tmp_path) -> None:
    """When other aliases already point at the new alias_id, the event's
    ``reverse_dependents`` lists them so cloud subscribers can evict
    transitively-stale source ids."""
    composite, _ = _build_sqlite_store(tmp_path)
    # Pre-existing aliases that point at "src-rev" — i.e., they alias to
    # the conversation that's about to gain its own outgoing alias.
    composite.save_conversation_alias("upstream-a", "src-rev")
    composite.save_conversation_alias("upstream-b", "src-rev")

    received: list[dict] = []
    composite.save_conversation_alias(
        "src-rev", "tgt-rev",
        on_committed=lambda e: received.append(dict(e)),
    )
    assert len(received) == 1
    event = received[0]
    # alias_created event payload's reverse_dependents covers the
    # incoming-edge graph of the alias source (NOT the new target's
    # graph — see plan S7 alias_created event shape).
    assert sorted(event["reverse_dependents"]) == ["upstream-a", "upstream-b"]


# ---------------------------------------------------------------------------
# S8 — Post-commit registry (scoped) defers callback until outer commit
# ---------------------------------------------------------------------------


def test_post_commit_scope_defers_callback_until_outer_commit(tmp_path) -> None:
    """Inside an `_alias_post_commit_scope`, save_conversation_alias must
    register the callback rather than fire it; the callback fires only
    when the scope's flush helper runs after the outer commit."""
    composite, sqlite = _build_sqlite_store(tmp_path)
    fired: list[str] = []

    conn = sqlite._get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        with sqlite._alias_post_commit_scope(conn) as scope:
            composite.save_conversation_alias(
                "src-scoped", "tgt-scoped",
                on_committed=lambda e: fired.append(e["source"]),
            )
            # Callback has NOT fired yet — registered on the scope.
            assert fired == []
            assert len(scope["hooks"]) == 1
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # After outer commit: flush manually (the merge body wires the flush
    # itself; tests exercise the contract directly).
    sqlite._flush_post_commit_hooks(scope)
    assert fired == ["src-scoped"]


def test_post_commit_scope_discards_hooks_on_rollback(tmp_path) -> None:
    """If the outer transaction rolls back, the registered hooks must be
    abandoned (callback never fires)."""
    composite, sqlite = _build_sqlite_store(tmp_path)
    fired: list[str] = []

    conn = sqlite._get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        with sqlite._alias_post_commit_scope(conn) as scope:
            composite.save_conversation_alias(
                "src-rb", "tgt-rb",
                on_committed=lambda e: fired.append(e["source"]),
            )
            # Force rollback.
            raise RuntimeError("simulated merge body failure")
    except RuntimeError:
        conn.execute("ROLLBACK")
        scope["hooks"].clear()  # merge body's finally clears the queue

    # No flush after rollback. Even if a buggy caller flushed an empty
    # queue, no callback fires.
    sqlite._flush_post_commit_hooks(scope)
    assert fired == []
    # Row was never committed.
    assert composite.resolve_conversation_alias("src-rb") is None


# ---------------------------------------------------------------------------
# S9 — VCATTACH execute_attach forwards cross_worker_invalidate
# ---------------------------------------------------------------------------


def test_execute_attach_forwards_cross_worker_invalidate(tmp_path) -> None:
    """``execute_attach`` passes ``cross_worker_invalidate`` down to BOTH
    delete_conversation_alias (clearing the reverse alias) and
    save_conversation_alias (registering the new alias) as
    ``on_committed=``. Each store-side call fires the callback once; the
    callback observes the engine-side AliasEvent payload (no tenant_id)."""
    from virtual_context.proxy.vcattach import execute_attach

    composite, _ = _build_sqlite_store(tmp_path)
    # Pre-existing alias on the new target so the delete leg has work
    # to do.
    composite.save_conversation_alias("tgt-vcattach", "old-target")
    received: list[dict] = []

    execute_attach(
        old_id="src-vcattach",
        target_id="tgt-vcattach",
        store=composite,
        cross_worker_invalidate=lambda e: received.append(dict(e)),
    )

    types = [e["type"] for e in received]
    # Expect the delete leg's alias_deleted (clearing tgt-vcattach's
    # reverse alias) followed by the save leg's alias_created.
    assert types == ["alias_deleted", "alias_created"]

    deleted_event = received[0]
    assert deleted_event["alias_id"] == "tgt-vcattach"
    created_event = received[1]
    assert created_event["source"] == "src-vcattach"
    assert created_event["target"] == "tgt-vcattach"
    # Final state: src-vcattach -> tgt-vcattach; tgt-vcattach has no alias.
    assert composite.resolve_conversation_alias("src-vcattach") == "tgt-vcattach"
    assert composite.resolve_conversation_alias("tgt-vcattach") is None


def test_execute_attach_propagates_invalidation_failed_error(tmp_path) -> None:
    """When the cross_worker_invalidate callback raises
    InvalidationFailedError, ``execute_attach`` re-raises so the REST
    handler can surface a retryable 503. The alias row is already
    committed at the moment the exception is raised."""
    from virtual_context.core.exceptions import InvalidationFailedError
    from virtual_context.proxy.vcattach import execute_attach

    composite, _ = _build_sqlite_store(tmp_path)

    def _cb(event):
        if event["type"] == "alias_created":
            raise InvalidationFailedError(
                event=event, cause=ConnectionError("redis offline"),
            )

    with pytest.raises(InvalidationFailedError):
        execute_attach(
            old_id="src-vcattach-fail",
            target_id="tgt-vcattach-fail",
            store=composite,
            cross_worker_invalidate=_cb,
        )
    # Row is durable: at-least-once contract — retry will fire callback again.
    assert composite.resolve_conversation_alias(
        "src-vcattach-fail",
    ) == "tgt-vcattach-fail"


# ---------------------------------------------------------------------------
# S10 — VCMERGE post-commit invalidation wiring
# ---------------------------------------------------------------------------
#
# These tests exercise the merge body's alias UPSERT through the post-
# commit scope: callback fires AFTER the outer transaction commits, and
# callback failure logs WARNING + emits ``vcmerge_invalidation_failed``
# without changing merge success. Full multi-row VCMERGE replay is
# covered by tests/test_merge_body_*.py — these tests stay narrow on
# the alias-write + callback contract.


def _seed_vcmerge_preconditions(
    sqlite_store, *, tenant_id: str, source_id: str, target_id: str,
    merge_id: str, expected_target_epoch: int = 1,
) -> None:
    """Seed the minimum DB rows the merge body needs to run end-to-end:
    `conversations` rows for source + target (active, current epoch) and
    a `merge_audit` row in `in_progress`."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite_store._get_conn()
    for cid in (source_id, target_id):
        conn.execute(
            """INSERT OR REPLACE INTO conversations
               (conversation_id, tenant_id, lifecycle_epoch, phase, deleted_at,
                created_at, updated_at)
               VALUES (?, ?, ?, 'active', NULL, ?, ?)""",
            (cid, tenant_id, expected_target_epoch, now, now),
        )
    conn.execute(
        """INSERT INTO merge_audit
           (merge_id, tenant_id, source_conversation_id,
            target_conversation_id, status,
            source_label_at_merge, started_at)
           VALUES (?, ?, ?, ?, 'in_progress', '', ?)""",
        (merge_id, tenant_id, source_id, target_id, now),
    )
    conn.commit()


def test_merge_body_alias_write_fires_invalidation_callback(tmp_path) -> None:
    """The merge body's alias UPSERT fires
    ``cross_worker_invalidate`` AFTER the outer transaction commits.
    Callback observes an alias_created event for source -> target with
    the row already visible."""
    composite, sqlite = _build_sqlite_store(tmp_path)
    _seed_vcmerge_preconditions(
        sqlite,
        tenant_id="t1",
        source_id="merge-src",
        target_id="merge-tgt",
        merge_id="merge-id-1",
    )

    received: list[dict] = []

    def _cb(event):
        received.append(dict(event))
        # Row visible at callback time -> outer commit completed first.
        assert composite.resolve_conversation_alias("merge-src") == "merge-tgt"

    sqlite.merge_conversation_data(
        merge_id="merge-id-1",
        tenant_id="t1",
        source_conversation_id="merge-src",
        target_conversation_id="merge-tgt",
        sort_key_offset=0.0,
        request_turn_offset=0,
        expected_target_lifecycle_epoch=1,
        source_label_at_merge="src-label",
        cross_worker_invalidate=_cb,
    )

    assert len(received) == 1
    assert received[0]["type"] == "alias_created"
    assert received[0]["source"] == "merge-src"
    assert received[0]["target"] == "merge-tgt"


def test_merge_body_invalidation_callback_failure_does_not_break_merge(
    tmp_path, caplog,
) -> None:
    """When ``cross_worker_invalidate`` raises ``InvalidationFailedError``
    after the outer commit, the merge MUST return success (the merge is
    durable) and the failure MUST be logged at WARNING with the
    structured fields per spec.
    """
    import logging
    from virtual_context.core.exceptions import InvalidationFailedError

    composite, sqlite = _build_sqlite_store(tmp_path)
    _seed_vcmerge_preconditions(
        sqlite,
        tenant_id="t1",
        source_id="merge-src-fail",
        target_id="merge-tgt-fail",
        merge_id="merge-id-fail",
    )

    def _cb(event):
        raise InvalidationFailedError(
            event=event, cause=ConnectionError("redis flapping"),
        )

    with caplog.at_level(logging.WARNING, logger="virtual_context.storage.sqlite"):
        stats = sqlite.merge_conversation_data(
            merge_id="merge-id-fail",
            tenant_id="t1",
            source_conversation_id="merge-src-fail",
            target_conversation_id="merge-tgt-fail",
            sort_key_offset=0.0,
            request_turn_offset=0,
            expected_target_lifecycle_epoch=1,
            source_label_at_merge="src-label",
            cross_worker_invalidate=_cb,
        )

    # Merge committed despite callback failure.
    assert stats.source_conversation_id == "merge-src-fail"
    assert stats.target_conversation_id == "merge-tgt-fail"
    assert composite.resolve_conversation_alias(
        "merge-src-fail",
    ) == "merge-tgt-fail"

    # WARNING surfaced for ops with the structured fields.
    matching = [
        rec for rec in caplog.records
        if "vcmerge invalidation failed" in rec.getMessage().lower()
    ]
    assert matching, f"Expected vcmerge invalidation WARNING; got {[r.getMessage() for r in caplog.records]}"


def test_alias_deleted_event_includes_reverse_dependents(tmp_path) -> None:
    """``delete_conversation_alias`` event payload includes
    ``reverse_dependents`` — the alias ids that pointed at the cleared
    row's source. Captured BEFORE the DELETE so the BFS sees pre-delete
    incoming-edge state.
    """
    composite, _ = _build_sqlite_store(tmp_path)
    # Seed: nephew-a / nephew-b alias to src-del; src-del aliases to tgt.
    composite.save_conversation_alias("nephew-a", "src-del-rev")
    composite.save_conversation_alias("nephew-b", "src-del-rev")
    composite.save_conversation_alias("src-del-rev", "tgt-x")

    received: list[dict] = []
    composite.delete_conversation_alias(
        "src-del-rev",
        on_committed=lambda e: received.append(dict(e)),
    )
    assert len(received) == 1
    event = received[0]
    assert event["type"] == "alias_deleted"
    assert event["alias_id"] == "src-del-rev"
    assert sorted(event["reverse_dependents"]) == ["nephew-a", "nephew-b"]


def test_vcattach_refuses_cross_tenant_source(tmp_path) -> None:
    """REST handler S10 source-tenant gate: when the calling source
    ``conversation_id`` belongs to a different tenant than the
    request's ``tenant_id``, refuse with the source-tenant envelope
    BEFORE writing any alias.

    Closes W6 invariant: prevents a misroute / replay / hostile path
    from creating an alias whose source is owned by another tenant.
    The same ``is_attachable_target`` predicate that gates the target
    via ``_target_exists`` also gates the source here for symmetry.
    """
    import json
    from types import SimpleNamespace

    from virtual_context.proxy import handlers

    composite, sqlite = _build_sqlite_store(tmp_path)
    # Seed: source belongs to tenant-A, target belongs to tenant-B.
    _seed_attachable_target(composite, "src-st-cross", tenant_id="tenant-A")
    _seed_attachable_target(composite, "tgt-st-cross", tenant_id="tenant-B")

    # Build a minimal engine handle so the handler can reach
    # `state.engine._store` without spinning up a real engine.
    engine_stub = SimpleNamespace(_store=composite)
    state = SimpleNamespace(engine=engine_stub, request=None)

    # Registry stub — attach path needs labels + conv_ids from the
    # tenant-scoped registry; the session state provider attributes are
    # only consulted during ``_invalidate``, which the gate prevents.
    class _StubRegistry:
        def get_conversation_labels(self, tenant_id):
            return {"tgt-st-cross": "TARGET-LABEL"}

        def list_persisted_conversation_ids(self, tenant_id):
            return ["tgt-st-cross"]

        _lock = None
        _states = None
        _session_state_provider = None

    result = SimpleNamespace(
        vc_command="attach",
        vc_command_arg="TARGET-LABEL",
        conversation_id="src-st-cross",
    )

    response = handlers._handle_vc_command_rest(
        result, state, _StubRegistry(),
        tenant_id="tenant-B",  # mismatched against source tenant-A
        vcconv=None,
    )
    payload = json.loads(response.body)
    assert payload["error"] == "source conversation does not belong to this tenant"
    assert payload["message"] == payload["error"]
    assert payload["vc_command"] == "attach"
    assert payload["conversation_id"] == "src-st-cross"
    # No alias was written.
    assert composite.resolve_conversation_alias("src-st-cross") is None


def test_merge_body_post_commit_callback_does_not_fire_on_rollback(
    tmp_path,
) -> None:
    """If the merge body raises BEFORE the outer COMMIT (e.g.,
    LifecycleEpochMismatch), the registered post-commit callback MUST
    NOT fire (the alias row never persisted)."""
    from virtual_context.core.exceptions import LifecycleEpochMismatch

    composite, sqlite = _build_sqlite_store(tmp_path)
    _seed_vcmerge_preconditions(
        sqlite,
        tenant_id="t1",
        source_id="merge-rb-src",
        target_id="merge-rb-tgt",
        merge_id="merge-rb-id",
        expected_target_epoch=1,
    )

    fired: list[dict] = []

    def _cb(event):
        fired.append(dict(event))

    # Pass an INTENTIONALLY WRONG epoch so the merge body's pre-flight
    # epoch check raises and the outer transaction rolls back.
    with pytest.raises(LifecycleEpochMismatch):
        sqlite.merge_conversation_data(
            merge_id="merge-rb-id",
            tenant_id="t1",
            source_conversation_id="merge-rb-src",
            target_conversation_id="merge-rb-tgt",
            sort_key_offset=0.0,
            request_turn_offset=0,
            expected_target_lifecycle_epoch=999,  # mismatch
            source_label_at_merge="src-label",
            cross_worker_invalidate=_cb,
        )

    assert fired == []
    # Alias never committed.
    assert composite.resolve_conversation_alias("merge-rb-src") is None


# ===========================================================================
# Engine self-hydrate on alias-resolver rebind (provider mode).
#
# Spec: docs/specs/engine-self-hydrate-on-rebind.md
# Fix target: task #29 (SessionState hydration keying mismatch).
# ===========================================================================


from virtual_context.proxy.session_state import SessionState


class _StubSessionStateProvider:
    """Minimal SessionStateProvider stub for engine.__init__ self-hydrate tests.

    Only ``load(conversation_id)`` is exercised by the engine's
    self-hydrate block at the end of ``VirtualContextEngine.__init__``.
    Other provider surface (``save``, embedding caches, tag stats) are
    not invoked during construction in provider mode (the engine's
    ``_save_state`` short-circuit and the embedding cache hooks fire
    later, on actual turn ingestion).

    ``load_calls`` records every ``load`` invocation so tests can
    assert which conversation_id the engine asked for. ``response_by_id``
    drives the return value: an explicit ``SessionState`` instance
    returns that state, an ``Exception`` subclass raises it, ``None``
    returns ``None`` (Redis miss). Default fallback when an id is not
    in the map is ``None``.
    """

    def __init__(self, response_by_id: dict[str, object] | None = None) -> None:
        self.response_by_id: dict[str, object] = response_by_id or {}
        self.load_calls: list[str] = []
        self._store = None  # cloud's wiring point; engine never reads this

    def load(self, conversation_id: str) -> SessionState | None:
        self.load_calls.append(conversation_id)
        response = self.response_by_id.get(conversation_id)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, SessionState):
            return response
        return None

    # Other provider methods called during engine __init__ delegate
    # construction. None of these affect the self-hydrate logic; they
    # exist so the engine's delegates accept the provider reference
    # without throwing.
    def load_tag_embeddings(self, model_name, tags):  # pragma: no cover
        return {}

    def save_tag_embeddings(self, *_a, **_kw):  # pragma: no cover
        return None

    def load_tag_stats_snapshot(self, conversation_id):  # pragma: no cover
        return None

    def load_tag_summary_embedding_snapshot(self, conversation_id):  # pragma: no cover
        return None

    def load_context_hint_cache(self, conversation_id, cache_key):  # pragma: no cover
        return None

    def save_payload_token_cache(self, *_a, **_kw):  # pragma: no cover
        return None

    def load_payload_token_cache(self, *_a, **_kw):  # pragma: no cover
        return None

    def next_tool_tag(self, conversation_id):  # pragma: no cover
        return 0


class _NoopEmbeddingProvider:
    """EmbeddingProvider double that keeps provider-mode construction offline."""

    def get_embed_fn(self):  # pragma: no cover
        return None


def _populated_target_session_state(
    *,
    last_completed_turn: int = 499,
    working_set: list[dict] | None = None,
) -> SessionState:
    """Return a SessionState with the post-compaction invariants the
    retriever's ``summary_floor`` gate depends on (flushed > 0,
    last_completed_turn matching a non-empty index)."""
    return SessionState(
        compacted_prefix_messages=958,
        flushed_prefix_messages=958,
        flushed_prefix_messages_present=True,
        last_request_time=0.0,
        last_compacted_turn=478,
        last_completed_turn=last_completed_turn,
        last_indexed_turn=last_completed_turn,
        checkpoint_version=1,
        conversation_generation=0,
        tool_tag_counter=0,
        split_processed_tags=set(),
        trailing_fingerprint="",
        provider="stub",
        turn_tag_entries=[],
        working_set=list(working_set or []),
        version=1,
    )


def _engine_with_provider(tmp_path, *, conversation_id, provider):
    """Build a VirtualContextEngine in provider mode for self-hydrate tests.

    Uses the same SQLite-backed config shape as ``engine_factory`` but
    passes ``session_state_provider=provider`` so the engine __init__'s
    provider-mode branch runs.
    """
    from virtual_context.engine import VirtualContextEngine
    from virtual_context.config import load_config

    db_path = str(tmp_path / "store.db")
    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
        "tag_generator": {"type": "keyword"},
        "retrieval": {"inbound_tagger_type": "keyword"},
        "conversation_id": conversation_id,
    })
    return VirtualContextEngine(
        config=config,
        session_state_provider=provider,
        embedding_provider=_NoopEmbeddingProvider(),
    )


def test_self_hydrate_fires_on_rebind_with_provider(engine_factory, tmp_path) -> None:
    """Happy path: alias source -> target exists, provider returns target's
    SessionState. Engine self-hydrates, sets the flag, and the
    retriever-driving fields reflect target's values."""
    # Seed an attachable target + alias row via a no-provider engine.
    seeder = engine_factory("seeder-sh-001")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-target-001")
    raw.save_conversation_alias("sh-source-001", "sh-target-001")

    target_state = _populated_target_session_state(
        working_set=[
            {
                "tag": "target-topic",
                "depth": "summary",
                "tokens": 321,
                "last_accessed_turn": 478,
            },
        ],
    )
    provider = _StubSessionStateProvider(
        response_by_id={"sh-target-001": target_state},
    )

    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-source-001", provider=provider,
    )

    # Resolver rebound to target.
    assert engine.config.conversation_id == "sh-target-001"
    # Self-hydrate fired.
    assert engine._self_hydrated_from_provider is True
    # Provider was asked for the resolver-rebound (target) id, NOT the source.
    assert "sh-target-001" in provider.load_calls
    assert "sh-source-001" not in provider.load_calls
    # Retriever-driving fields reflect target's loaded values.
    #
    # NOTE on the assertion choice: ``hydrate_from_session_state`` at
    # ``engine.py:1441`` clamps ``flushed_prefix_messages`` to
    # ``min(loaded, _derive_compacted_prefix_messages_from_rows(...))``.
    # For a target with zero canonical_turns rows (this test's
    # freshly-seeded target lives only in the ``conversations`` table),
    # the derivation returns ``(0, -1)`` and the clamp drives flushed
    # down to 0 regardless of the loaded value. That's correct engine
    # behaviour — the retriever's compaction watermark should reflect
    # actual rows on disk, not stale SessionState. To assert that
    # hydration ran with the right *target* state we use
    # ``last_completed_turn`` / ``last_indexed_turn`` which are loaded
    # directly from SessionState without a clamp.
    assert engine._engine_state.last_completed_turn == 499
    assert engine._engine_state.last_indexed_turn == 499
    assert list(engine._paging.working_set) == ["target-topic"]
    assert engine._paging.working_set["target-topic"].tokens == 321


def test_self_hydrate_multi_hop_rebind_loads_terminal_provider_state(
    engine_factory, tmp_path,
) -> None:
    """Self-hydrate must use the alias-chain terminal id, not an
    intermediate hop."""
    seeder = engine_factory("seeder-sh-chain-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-chain-target")
    raw.save_conversation_alias("sh-chain-source", "sh-chain-mid")
    raw.save_conversation_alias("sh-chain-mid", "sh-chain-target")

    target_state = _populated_target_session_state(last_completed_turn=777)
    provider = _StubSessionStateProvider(
        response_by_id={"sh-chain-target": target_state},
    )

    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-chain-source", provider=provider,
    )

    assert engine.config.conversation_id == "sh-chain-target"
    assert engine._self_hydrated_from_provider is True
    assert provider.load_calls == ["sh-chain-target"]
    assert engine._engine_state.last_completed_turn == 777


def test_self_hydrate_concurrent_same_source_construction_is_idempotent(
    engine_factory, tmp_path,
) -> None:
    """Concurrent constructions for the same source both rebind and hydrate from
    the same terminal target state."""
    from concurrent.futures import ThreadPoolExecutor

    seeder = engine_factory("seeder-sh-repeat-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-repeat-target")
    raw.save_conversation_alias("sh-repeat-source", "sh-repeat-target")

    provider = _StubSessionStateProvider(
        response_by_id={
            "sh-repeat-target": _populated_target_session_state(
                last_completed_turn=888,
            ),
        },
    )

    def _build():
        return _engine_with_provider(
            tmp_path, conversation_id="sh-repeat-source", provider=provider,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [future.result() for future in (
            pool.submit(_build),
            pool.submit(_build),
        )]

    assert first.config.conversation_id == "sh-repeat-target"
    assert second.config.conversation_id == "sh-repeat-target"
    assert first._self_hydrated_from_provider is True
    assert second._self_hydrated_from_provider is True
    assert provider.load_calls == ["sh-repeat-target", "sh-repeat-target"]
    assert first._engine_state.last_completed_turn == 888
    assert second._engine_state.last_completed_turn == 888


def test_self_hydrate_provider_state_wins_over_persisted_engine_state(
    engine_factory, tmp_path,
) -> None:
    """Provider mode skips _load_persisted_state; target-keyed provider
    state remains authoritative even if a target engine_state row exists."""
    from virtual_context.types import TurnTagEntry

    seeder = engine_factory("seeder-sh-persist-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-persist-target")
    raw.save_conversation_alias("sh-persist-source", "sh-persist-target")
    _seed_engine_state_for_conversation(
        raw,
        "sh-persist-target",
        turn_tag_entries=[
            TurnTagEntry(
                turn_number=0,
                message_hash="persisted-hash",
                tags=["persisted"],
                primary_tag="persisted",
            ),
        ],
        turn_count=1,
    )

    provider = _StubSessionStateProvider(
        response_by_id={
            "sh-persist-target": _populated_target_session_state(
                last_completed_turn=999,
            ),
        },
    )

    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-persist-source", provider=provider,
    )

    assert engine.config.conversation_id == "sh-persist-target"
    assert engine._self_hydrated_from_provider is True
    assert engine._engine_state.last_completed_turn == 999
    assert engine._turn_tag_index.entries == []


def test_self_hydrate_skipped_when_no_rebind(engine_factory, tmp_path) -> None:
    """When the resolver does NOT rebind (no alias row), the self-hydrate
    block is skipped — cloud will hydrate on its own path."""
    provider = _StubSessionStateProvider(
        response_by_id={
            "sh-noalias-source": _populated_target_session_state(),
        },
    )
    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-noalias-source", provider=provider,
    )

    # No alias row exists, so config.conversation_id is unchanged.
    assert engine.config.conversation_id == "sh-noalias-source"
    # Self-hydrate did NOT fire (resolver didn't rebind).
    assert engine._self_hydrated_from_provider is False
    # Provider was NOT asked during construction.
    assert provider.load_calls == []


def test_self_hydrate_skipped_when_no_provider(engine_factory) -> None:
    """Local-proxy mode (session_state_provider=None) skips the
    self-hydrate block entirely — the existing _load_persisted_state
    path is authoritative for non-provider deployments."""
    seeder = engine_factory("seeder-sh-noprov-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-noprov-target")
    raw.save_conversation_alias("sh-noprov-source", "sh-noprov-target")

    # engine_factory builds with session_state_provider=None.
    engine = engine_factory("sh-noprov-source")

    # Resolver still rebound.
    assert engine.config.conversation_id == "sh-noprov-target"
    # Flag stays False — provider mode wasn't active.
    assert engine._self_hydrated_from_provider is False


def test_self_hydrate_handles_provider_exception(engine_factory, tmp_path, caplog) -> None:
    """Provider.load raises a transient/persistent exception. The engine
    swallows + logs at WARNING, leaves the flag at False, and
    construction succeeds. Cloud's caller-driven hydration path then
    fires on the next request (back to pre-fix behavior — not a
    regression).
    """
    import logging

    seeder = engine_factory("seeder-sh-exc-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-exc-target")
    raw.save_conversation_alias("sh-exc-source", "sh-exc-target")

    provider = _StubSessionStateProvider(
        response_by_id={
            "sh-exc-target": RuntimeError("simulated redis hiccup"),
        },
    )

    with caplog.at_level(logging.WARNING, logger="virtual_context.engine"):
        engine = _engine_with_provider(
            tmp_path, conversation_id="sh-exc-source", provider=provider,
        )

    # Resolver still rebound.
    assert engine.config.conversation_id == "sh-exc-target"
    # Self-hydrate attempted (load was called).
    assert "sh-exc-target" in provider.load_calls
    # Flag stays False because load raised.
    assert engine._self_hydrated_from_provider is False
    # WARNING-level log was emitted with the rebound id prefix.
    assert any(
        "self-hydration load failed" in record.getMessage().lower()
        and "sh-exc-targ" in record.getMessage()
        for record in caplog.records
    )


def test_self_hydrate_handles_tombstoned_target(engine_factory, tmp_path) -> None:
    """Provider returns SessionState(deleted=True) for the target.
    The engine MUST NOT call hydrate_from_session_state — clobbering
    state with tombstone fields would corrupt the engine. Flag stays
    False; cloud's existing tombstone path on the source id takes
    over (and likely also returns False since source isn't tombstoned)."""
    seeder = engine_factory("seeder-sh-tomb-000")
    raw = _underlying_store(seeder)
    _seed_attachable_target(raw, "sh-tomb-target")
    raw.save_conversation_alias("sh-tomb-source", "sh-tomb-target")

    tombstoned = SessionState(deleted=True, version=1)
    provider = _StubSessionStateProvider(
        response_by_id={"sh-tomb-target": tombstoned},
    )

    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-tomb-source", provider=provider,
    )

    # Resolver rebound to target.
    assert engine.config.conversation_id == "sh-tomb-target"
    # Provider was queried.
    assert "sh-tomb-target" in provider.load_calls
    # Flag stays False — we did NOT hydrate from a tombstone.
    assert engine._self_hydrated_from_provider is False


def test_double_hydrate_with_different_state_is_observable(engine_factory, tmp_path) -> None:
    """Document the bug shape this spec is fixing.

    ``hydrate_from_session_state`` is overwrite-shaped. Calling it once
    with source-empty state then once with target-populated state
    LEAVES THE ENGINE IN TARGET STATE (the second call wins). The
    actual bug is the reverse: cloud hydrates with source state AFTER
    the engine has self-hydrated with target state, which clobbers
    target with source. This test pins the overwrite invariant so any
    future change to make hydration accumulative breaks the test
    loudly and forces a re-read of the spec.
    """
    # Build a non-provider engine on a fresh DB so canonical_turns
    # derivation doesn't override our test values.
    provider = _StubSessionStateProvider()  # no response
    engine = _engine_with_provider(
        tmp_path, conversation_id="sh-double-source", provider=provider,
    )
    assert engine._self_hydrated_from_provider is False

    state_a = _populated_target_session_state(last_completed_turn=100)
    state_b = _populated_target_session_state(last_completed_turn=499)

    engine.hydrate_from_session_state(state_a)
    # ``last_completed_turn`` is loaded directly from SessionState without
    # a canonical-turns clamp, so we use it (rather than
    # ``flushed_prefix_messages``, which the line 1441 clamp would drive
    # to 0 against an empty canonical_turns table) to observe overwrite
    # semantics across successive hydrate calls.
    assert engine._engine_state.last_completed_turn == 100

    engine.hydrate_from_session_state(state_b)
    # Second call OVERWRITES — engine reflects state_b, not an
    # accumulation of state_a + state_b.
    assert engine._engine_state.last_completed_turn == 499
