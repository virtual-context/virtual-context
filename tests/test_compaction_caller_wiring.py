"""Compaction-fence Phase 3 caller-wiring tests.

Covers fencing plan §5.6 caller-side propagation and fail-closed
exception handling:

* T3.17a: ``SemanticSearchManager.embed_and_store_chunks`` forwards
  every guard kwarg (operation_id, owner_worker_id, lifecycle_epoch,
  conversation_id) to ``store_chunk_embeddings``.
* T3.17b: ``CompactionPipeline._propagate_tool_output_links`` forwards
  the guard triple to ``link_segment_tool_output`` for each turn.
* T3.17c: ``FactSupersessionChecker.check_and_supersede`` forwards the
  guard triple to ``set_fact_superseded``.
* T3.17d: ``FactLinkChecker.check_and_link`` (graph mode) forwards
  guard triple + conversation_id to ``store_fact_links`` and
  ``set_fact_superseded`` and downstream ``promote_planned_facts`` ->
  ``update_fact_fields``.
* T3.17e: ``promote_planned_facts`` forwards the guard triple to
  ``update_fact_fields``.

* T3.18a: ``CompactionLeaseLost`` raised inside
  ``_propagate_tool_output_links`` propagates past its broad ``except
  Exception`` handler.
* T3.18b: ``CompactionLeaseLost`` raised by ``store_fact_links`` /
  ``set_fact_superseded`` inside ``FactLinkChecker.check_and_link``
  propagates past the supersession/linking ``except Exception``
  handler.
* T3.18c: ``CompactionLeaseLost`` raised during tag-summary embedding
  propagates past that block's broad ``except Exception`` handler.

The tests use stub ``store`` objects so the test does not depend on
SQLite schema seeding. The point is to assert kwarg propagation and
exception passthrough, not the storage SQL fences themselves (those
are covered by tests/test_compaction_per_write_fence.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.ingest.supersession import (
    FactLinkChecker,
    FactSupersessionChecker,
    promote_planned_facts,
)
from virtual_context.types import (
    ChunkEmbedding,
    CompactionResult,
    CompactionLeaseLost,
    Fact,
    FactLink,
    StoredSegment,
    SupersessionConfig,
    TagSummary,
)


# ---------------------------------------------------------------------------
# Stub store + helpers
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    method: str
    args: tuple
    kwargs: dict


class _SpyStore:
    """Captures every method call's args and kwargs.

    Optionally raises ``CompactionLeaseLost`` from a specified method
    so callers can verify fail-closed propagation.
    """

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self._raise_on: set[str] = set()
        self._planned_facts: list[Fact] = []
        self._candidate_facts: list[Fact] = []

    def raise_lease_lost_on(self, method: str) -> None:
        self._raise_on.add(method)

    def seed_planned_fact(self, fact: Fact) -> None:
        self._planned_facts.append(fact)

    def seed_candidate(self, fact: Fact) -> None:
        self._candidate_facts.append(fact)

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        self.calls.append(_Call(method=method, args=args, kwargs=kwargs))
        if method in self._raise_on:
            raise CompactionLeaseLost(
                operation_id=kwargs.get("operation_id") or "spy",
                write_site=method,
            )

    # -- Methods called by the wired helpers --

    def store_chunk_embeddings(self, *args: Any, **kwargs: Any) -> None:
        self._record("store_chunk_embeddings", args, kwargs)

    def link_segment_tool_output(self, *args: Any, **kwargs: Any) -> None:
        self._record("link_segment_tool_output", args, kwargs)

    def store_fact_links(self, *args: Any, **kwargs: Any) -> int:
        self._record("store_fact_links", args, kwargs)
        return 0

    def set_fact_superseded(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_fact_superseded", args, kwargs)

    def update_fact_fields(self, *args: Any, **kwargs: Any) -> None:
        self._record("update_fact_fields", args, kwargs)

    def get_tool_outputs_for_turn(self, conv: str, turn: int) -> list[str]:
        # Single tool ref per turn for simplicity.
        self.calls.append(_Call("get_tool_outputs_for_turn", (conv, turn), {}))
        return [f"tool-ref-{turn}"]

    def query_facts(self, **kwargs: Any) -> list[Fact]:
        self.calls.append(_Call("query_facts", (), kwargs))
        status = kwargs.get("status")
        if status == "planned":
            return list(self._planned_facts)
        return list(self._candidate_facts)

    def call_names(self) -> list[str]:
        return [c.method for c in self.calls]

    def calls_of(self, method: str) -> list[_Call]:
        return [c for c in self.calls if c.method == method]


# ---------------------------------------------------------------------------
# T3.17a: embed_and_store_chunks forwards guard kwargs
# ---------------------------------------------------------------------------


class TestT317a_EmbedAndStoreChunks:
    def test_forwards_all_guard_kwargs_to_store(self):
        store = _SpyStore()
        # Construct a SemanticSearchManager with a stub embed_fn so we
        # exercise the kwarg-forwarding path without loading a real
        # sentence-transformers model.
        from virtual_context.types import VirtualContextConfig
        sm = SemanticSearchManager(
            store=store, config=VirtualContextConfig(),
        )
        sm._embed_fn = lambda texts: [[0.1, 0.2] for _ in texts]

        stored = StoredSegment(
            ref="seg-1",
            conversation_id="conv-A",
            primary_tag="_general",
            tags=["_general"],
            summary="hello world",
            summary_tokens=2,
            full_text=("hello world test phrase " * 20),
            full_tokens=80,
            messages=[],
            metadata=None,
            compaction_model="m",
            compression_ratio=1.0,
            start_timestamp="2026-05-29T00:00:00+00:00",
            end_timestamp="2026-05-29T00:00:00+00:00",
        )
        sm.embed_and_store_chunks(
            stored,
            operation_id="op-1", owner_worker_id="w-1",
            lifecycle_epoch=2, conversation_id="conv-A",
        )
        sce = store.calls_of("store_chunk_embeddings")
        assert sce, "store_chunk_embeddings was not called"
        kw = sce[0].kwargs
        assert kw["operation_id"] == "op-1"
        assert kw["owner_worker_id"] == "w-1"
        assert kw["lifecycle_epoch"] == 2
        assert kw["conversation_id"] == "conv-A"

    def test_legacy_call_passes_none_kwargs(self):
        store = _SpyStore()
        from virtual_context.types import VirtualContextConfig
        sm = SemanticSearchManager(
            store=store, config=VirtualContextConfig(),
        )
        sm._embed_fn = lambda texts: [[0.1, 0.2] for _ in texts]
        stored = StoredSegment(
            ref="seg-leg", conversation_id="conv-X",
            primary_tag="_general", tags=["_general"],
            summary="x", summary_tokens=1,
            full_text=("legacy chunk content here " * 20),
            full_tokens=6, messages=[], metadata=None,
            compaction_model="m", compression_ratio=1.0,
            start_timestamp="2026-05-29T00:00:00+00:00",
            end_timestamp="2026-05-29T00:00:00+00:00",
        )
        sm.embed_and_store_chunks(stored)
        sce = store.calls_of("store_chunk_embeddings")
        assert sce
        kw = sce[0].kwargs
        assert kw["operation_id"] is None
        assert kw["owner_worker_id"] is None
        assert kw["lifecycle_epoch"] is None
        assert kw["conversation_id"] is None


# ---------------------------------------------------------------------------
# T3.17b + T3.18a: _propagate_tool_output_links forwards guard kwargs
# and propagates CompactionLeaseLost past its broad except handler.
# ---------------------------------------------------------------------------


class _StubConfig:
    def __init__(self, conv_id: str) -> None:
        self.conversation_id = conv_id


class _MinimalPipeline:
    """The smallest object that exposes the methods
    ``_propagate_tool_output_links`` reads from ``self``.

    We bypass the real ``CompactionPipeline.__init__`` (which expects
    a full engine state, compactor, segmenter, etc.) by binding the
    method as a free function in tests below.
    """

    def __init__(self, store, conv_id: str) -> None:
        self._store = store
        self._config = _StubConfig(conv_id)


def _bound_propagate(store, conv_id: str):
    from virtual_context.core.compaction_pipeline import CompactionPipeline
    pipe = _MinimalPipeline(store, conv_id)
    return CompactionPipeline._propagate_tool_output_links.__get__(pipe)


class TestT317b_PropagateToolOutputLinks:
    def test_forwards_guard_kwargs_for_each_turn(self):
        store = _SpyStore()
        propagate = _bound_propagate(store, "conv-A")
        propagate(
            "seg-1", 0, 3,
            operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=2,
        )
        lsto = store.calls_of("link_segment_tool_output")
        # 3 turns -> 3 link calls.
        assert len(lsto) == 3
        for call in lsto:
            assert call.args == ("conv-A", "seg-1", call.args[2])
            assert call.kwargs["operation_id"] == "op-1"
            assert call.kwargs["owner_worker_id"] == "w-1"
            assert call.kwargs["lifecycle_epoch"] == 2

    def test_legacy_call_passes_none_kwargs(self):
        store = _SpyStore()
        propagate = _bound_propagate(store, "conv-X")
        propagate("seg-leg", 5, 6)
        lsto = store.calls_of("link_segment_tool_output")
        assert len(lsto) == 1
        kw = lsto[0].kwargs
        assert kw["operation_id"] is None
        assert kw["owner_worker_id"] is None
        assert kw["lifecycle_epoch"] is None


class TestT318a_PropagateToolOutputLeaseLostPropagates:
    def test_lease_lost_breaks_through_broad_except(self):
        store = _SpyStore()
        store.raise_lease_lost_on("link_segment_tool_output")
        propagate = _bound_propagate(store, "conv-A")
        with pytest.raises(CompactionLeaseLost) as exc:
            propagate(
                "seg-1", 0, 3,
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=2,
            )
        assert exc.value.write_site == "link_segment_tool_output"


# ---------------------------------------------------------------------------
# T3.17c: FactSupersessionChecker.check_and_supersede forwards guard
# triple to set_fact_superseded.
# ---------------------------------------------------------------------------


class _LLMSayingSupersedeAll:
    """Stub LLMProvider whose response supersedes every candidate."""

    def complete(self, *, system: str, user: str, max_tokens: int = 0):
        # Return JSON array of all candidate indices. The supersession
        # checker scans up to batch_size; for tests we always have 1.
        return "[0]", 0


def _make_supersession_checker(store):
    cfg = SupersessionConfig(enabled=True, batch_size=20)
    return FactSupersessionChecker(
        llm_provider=_LLMSayingSupersedeAll(),
        model="stub",
        store=store,
        config=cfg,
    )


class TestT317c_CheckAndSupersede:
    def test_forwards_guard_triple_to_set_fact_superseded(self):
        store = _SpyStore()
        store.seed_candidate(Fact(id="old-1", subject="alice",
                                  verb="likes", object="tea",
                                  what="tea preference",
                                  conversation_id="conv-A"))
        checker = _make_supersession_checker(store)
        new_fact = Fact(id="new-1", subject="alice", verb="likes",
                        object="coffee", what="coffee preference",
                        conversation_id="conv-A")
        checker.check_and_supersede(
            [new_fact],
            operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=2,
        )
        sfs = store.calls_of("set_fact_superseded")
        assert sfs, "set_fact_superseded was not called"
        kw = sfs[0].kwargs
        assert kw["operation_id"] == "op-1"
        assert kw["owner_worker_id"] == "w-1"
        assert kw["lifecycle_epoch"] == 2

    def test_legacy_call_passes_none_kwargs(self):
        store = _SpyStore()
        store.seed_candidate(Fact(id="old-leg", subject="bob",
                                  verb="visits", object="paris",
                                  what="paris visit",
                                  conversation_id="conv-Y"))
        checker = _make_supersession_checker(store)
        checker.check_and_supersede([Fact(
            id="new-leg", subject="bob", verb="visits", object="london",
            what="london visit", conversation_id="conv-Y",
        )])
        sfs = store.calls_of("set_fact_superseded")
        assert sfs
        kw = sfs[0].kwargs
        assert kw["operation_id"] is None
        assert kw["owner_worker_id"] is None
        assert kw["lifecycle_epoch"] is None


# ---------------------------------------------------------------------------
# T3.17d: FactLinkChecker.check_and_link forwards guard kwargs to all
# downstream writes (set_fact_superseded, store_fact_links,
# promote_planned_facts -> update_fact_fields).
# ---------------------------------------------------------------------------


class _LLMForGraphLinks:
    """Stub LLMProvider that produces a single link suggesting
    'supersedes' on the only candidate."""

    def complete(self, *, system: str, user: str, max_tokens: int = 0):
        # Used both by FactLinkChecker._check_links and by
        # promote_planned_facts. Default to an empty array for any
        # call so promote_planned_facts falls back to status flip.
        return "[]", 0


class _LLMNeverPromotes:
    """LLMProvider that never returns anything; ``promote_planned_facts``
    falls back to status flip which still hits ``update_fact_fields``.
    """

    def complete(self, *, system: str, user: str, max_tokens: int = 0):
        return "", 0


def _make_link_checker(store, graph: bool = True):
    cfg = SupersessionConfig(enabled=True, batch_size=20)
    return FactLinkChecker(
        llm_provider=_LLMNeverPromotes(),
        model="stub",
        store=store,
        config=cfg,
        graph_links=graph,
    )


class TestT317d_CheckAndLink:
    def test_promote_planned_pass_forwards_guard_kwargs(self):
        """check_and_link runs promote_planned_facts as a pre-pass; the
        guard kwargs must reach update_fact_fields when promote rewrites
        a planned fact."""
        store = _SpyStore()
        # Seed a planned fact whose when_date has already passed.
        past_fact = Fact(
            id="planned-1", subject="carol", verb="will visit",
            object="rome", what="rome trip", status="planned",
            when_date="2020-01-01", conversation_id="conv-A",
        )
        store.seed_planned_fact(past_fact)
        checker = _make_link_checker(store, graph=False)
        # Graph mode False: promote runs, then check_and_supersede; the
        # candidate list is empty so no set_fact_superseded fires.
        checker.check_and_link(
            [Fact(id="trigger", subject="z", verb="v", object="o",
                  what="w", conversation_id="conv-A")],
            operation_id="op-2", owner_worker_id="w-2", lifecycle_epoch=3,
            conversation_id="conv-A",
        )
        uff = store.calls_of("update_fact_fields")
        assert uff, "update_fact_fields was not called from promote_planned_facts"
        kw = uff[0].kwargs
        assert kw["operation_id"] == "op-2"
        assert kw["owner_worker_id"] == "w-2"
        assert kw["lifecycle_epoch"] == 3

    def test_graph_mode_store_fact_links_carries_conversation_id(self):
        """In graph mode, the FactLinkChecker writes a FactLink and that
        store_fact_links call must carry conversation_id alongside the
        triple per fencing plan §5.3."""

        class _LLMOneLink:
            def complete(self, *, system: str, user: str, max_tokens: int = 0):
                # Return a JSON object that the _check_links parser will
                # turn into a single link with relation_type 'related_to'.
                return (
                    '{"links": [{"source": "N0", "target": "E0", '
                    '"relation": "related_to", "confidence": 0.9, '
                    '"context": "demo"}], "superseded": []}'
                ), 0

        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-1", subject="dave", verb="works_at", object="acme",
            what="employment", conversation_id="conv-A",
        ))
        cfg = SupersessionConfig(enabled=True, batch_size=20)
        checker = FactLinkChecker(
            llm_provider=_LLMOneLink(),
            model="stub", store=store, config=cfg, graph_links=True,
        )
        new_fact = Fact(
            id="new-link", subject="dave", verb="visits", object="paris",
            what="trip", conversation_id="conv-A",
        )
        checker.check_and_link(
            [new_fact],
            operation_id="op-3", owner_worker_id="w-3", lifecycle_epoch=4,
            conversation_id="conv-A",
        )
        sfl = store.calls_of("store_fact_links")
        assert sfl, "store_fact_links was not called"
        for call in sfl:
            assert call.kwargs["operation_id"] == "op-3"
            assert call.kwargs["owner_worker_id"] == "w-3"
            assert call.kwargs["lifecycle_epoch"] == 4
            assert call.kwargs["conversation_id"] == "conv-A"


# ---------------------------------------------------------------------------
# T3.17e: promote_planned_facts forwards guard triple
# ---------------------------------------------------------------------------


class TestT317e_PromotePlannedFacts:
    def test_forwards_guard_triple_to_update_fact_fields(self):
        store = _SpyStore()
        store.seed_planned_fact(Fact(
            id="planned-2", subject="erin", verb="will run", object="5k",
            what="5k plan", status="planned",
            when_date="2020-01-01", conversation_id="conv-Z",
        ))
        promoted = promote_planned_facts(
            store,
            operation_id="op-9", owner_worker_id="w-9", lifecycle_epoch=7,
        )
        assert promoted == 1
        uff = store.calls_of("update_fact_fields")
        assert uff
        kw = uff[0].kwargs
        assert kw["operation_id"] == "op-9"
        assert kw["owner_worker_id"] == "w-9"
        assert kw["lifecycle_epoch"] == 7

    def test_legacy_call_passes_none_kwargs(self):
        store = _SpyStore()
        store.seed_planned_fact(Fact(
            id="planned-leg", subject="x", verb="will", object="y",
            what="z", status="planned",
            when_date="2020-01-01", conversation_id="conv-Q",
        ))
        promote_planned_facts(store)
        uff = store.calls_of("update_fact_fields")
        assert uff
        kw = uff[0].kwargs
        assert kw["operation_id"] is None
        assert kw["owner_worker_id"] is None
        assert kw["lifecycle_epoch"] is None


# ---------------------------------------------------------------------------
# T3.18b: CompactionLeaseLost raised by set_fact_superseded /
# store_fact_links inside FactLinkChecker.check_and_link propagates
# past the per-fact broad except handler.
# ---------------------------------------------------------------------------


class TestT318b_FactLinkCheckerLeaseLostPropagates:
    """T3.18b: CompactionLeaseLost must propagate past every broad
    ``except Exception`` handler that wraps a fenced write path.

    Covers three handlers:

    * ``FactSupersessionChecker.check_and_supersede`` per-fact body
      (lease loss from the direct ``set_fact_superseded`` call).
    * ``FactLinkChecker.check_and_link`` per-fact body which wraps
      ``_check_links`` in a broad except (lease loss from
      ``store_fact_links`` / ``set_fact_superseded`` AFTER the inner
      try). The inner ``_check_links`` is LLM-only so its broad except
      cannot itself emit ``CompactionLeaseLost``; the relevant
      regression is that the SUBSEQUENT fenced writes inside the
      per-fact body propagate past the inner handler.
    * Indirect coverage: the compaction-pipeline supersession/linking
      block re-raises ``CompactionLeaseLost`` before its generic
      handler. That block is exercised end-to-end by the per-write
      fence tests in tests/test_compaction_per_write_fence.py; the
      re-raise edge is asserted by code inspection (an explicit
      ``except CompactionLeaseLost: raise`` clause precedes the
      generic handler).
    """

    def test_lease_lost_on_set_fact_superseded_propagates(self):
        """Lease loss from the direct ``set_fact_superseded`` call in
        ``check_and_supersede`` propagates (no broad except wraps it).
        """
        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-x", subject="frank", verb="owns", object="car",
            what="car", conversation_id="conv-A",
        ))
        store.raise_lease_lost_on("set_fact_superseded")
        checker = _make_supersession_checker(store)
        with pytest.raises(CompactionLeaseLost):
            checker.check_and_supersede(
                [Fact(id="new-x", subject="frank", verb="owns",
                      object="bike", what="bike",
                      conversation_id="conv-A")],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=1,
            )

    def test_lease_lost_on_store_fact_links_propagates(self):
        """In graph mode, the per-fact body wraps ``_check_links`` in a
        broad except. After that try block returns, the body issues
        ``store_fact_links``. A lease loss from that fenced write must
        propagate out of ``check_and_link`` rather than be silently
        swallowed by the per-fact loop.
        """

        class _LLMOneLink:
            def complete(self, *, system: str, user: str, max_tokens: int = 0):
                return (
                    '{"superseded": [], "links": [{"source": "N0", '
                    '"target": "E0", "relation": "related_to", '
                    '"confidence": 0.9, "context": "demo"}]}'
                ), 0

        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-y", subject="grace", verb="works_at", object="acme",
            what="employment", conversation_id="conv-A",
        ))
        store.raise_lease_lost_on("store_fact_links")
        cfg = SupersessionConfig(enabled=True, batch_size=20)
        checker = FactLinkChecker(
            llm_provider=_LLMOneLink(),
            model="stub", store=store, config=cfg, graph_links=True,
        )
        with pytest.raises(CompactionLeaseLost):
            checker.check_and_link(
                [Fact(id="new-y", subject="grace", verb="visits",
                      object="paris", what="trip",
                      conversation_id="conv-A")],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=1, conversation_id="conv-A",
            )

    def test_lease_lost_from_check_links_inner_try_propagates(self):
        """Lease loss raised inside the ``_check_links`` try block must
        pass through the local broad exception handler.
        """
        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-inner", subject="helen", verb="uses", object="python",
            what="tooling", conversation_id="conv-A",
        ))
        checker = _make_link_checker(store, graph=True)

        def _raise_lease_lost(*_args, **_kwargs):
            raise CompactionLeaseLost(
                operation_id="op-1", write_site="store_fact_links",
            )

        checker._check_links = _raise_lease_lost
        with pytest.raises(CompactionLeaseLost):
            checker.check_and_link(
                [Fact(id="new-inner", subject="helen", verb="uses",
                      object="go", what="tooling",
                      conversation_id="conv-A")],
                operation_id="op-1", owner_worker_id="w-1",
                lifecycle_epoch=1, conversation_id="conv-A",
            )


class _EmptyTurnTagIndex:
    entries: list = []

    def compute_cover_set(self) -> list[str]:
        return []


class _TagSummaryCompactor:
    def compact_tag_summaries(self, **_kwargs):
        return [TagSummary(tag="_general", summary="summary text")]


class _SemanticEmbeddingStub:
    def get_embed_fn(self):
        return lambda texts: [[0.1, 0.2] for _ in texts]


class _TagSummaryStore(_SpyStore):
    def get_summaries_by_tags(self, **kwargs: Any) -> list:
        self.calls.append(_Call("get_summaries_by_tags", (), kwargs))
        return []

    def get_tag_summary(self, *args: Any, **kwargs: Any):
        self.calls.append(_Call("get_tag_summary", args, kwargs))
        return None

    def save_tag_summary(self, *args: Any, **kwargs: Any) -> None:
        self._record("save_tag_summary", args, kwargs)

    def store_tag_summary_embedding(self, *args: Any, **kwargs: Any) -> None:
        self._record("store_tag_summary_embedding", args, kwargs)


class _TagSummaryPipelineStub:
    def __init__(self, store) -> None:
        self._store = store
        self._config = _StubConfig("conv-A")
        self._engine_state = _StubEngineState(epoch=1)
        self._worker_id = "w-1"
        self._turn_tag_index = _EmptyTurnTagIndex()
        self._compactor = _TagSummaryCompactor()
        self._semantic = _SemanticEmbeddingStub()


def _bound_build_tag_summaries(store):
    from virtual_context.core.compaction_pipeline import CompactionPipeline
    pipe = _TagSummaryPipelineStub(store)
    pipe._compaction_guard_kwargs = (
        CompactionPipeline._compaction_guard_kwargs.__get__(pipe)
    )
    return CompactionPipeline._build_tag_summaries.__get__(pipe)


class TestT318c_TagSummaryEmbeddingLeaseLostPropagates:
    def test_lease_lost_breaks_through_tag_embedding_broad_except(self):
        store = _TagSummaryStore()
        store.raise_lease_lost_on("store_tag_summary_embedding")
        build_tag_summaries = _bound_build_tag_summaries(store)

        with pytest.raises(CompactionLeaseLost):
            build_tag_summaries(
                results=[CompactionResult(
                    segment_id="seg-1",
                    primary_tag="_general",
                    tags=["_general"],
                )],
                compact_rows=None,
                operation_id="op-1",
            )


# ---------------------------------------------------------------------------
# T3.17f: cross-conversation candidate scoping. When the caller is
# fenced (operation_id supplied), supersession + planned-fact queries
# must be scoped to the active op's conversation so a candidate from
# another conversation does not trigger a guarded write rejection.
# ---------------------------------------------------------------------------


class TestT317f_FencedQueriesScopedToConversation:
    def test_check_and_supersede_passes_conversation_id_when_fenced(self):
        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-z", subject="hank", verb="lives-in", object="NYC",
            what="address", conversation_id="conv-A",
        ))
        checker = _make_supersession_checker(store)
        new = Fact(id="new-z", subject="hank", verb="lives-in",
                   object="Chicago", what="address",
                   conversation_id="conv-A")
        checker.check_and_supersede(
            [new],
            operation_id="op-1", owner_worker_id="w-1", lifecycle_epoch=1,
        )
        # Every query_facts call must carry conversation_id=conv-A.
        qf = store.calls_of("query_facts")
        assert qf, "query_facts was not called"
        for call in qf:
            assert call.kwargs.get("conversation_id") == "conv-A", (
                "fenced caller must scope candidate queries to the active "
                "op's conversation"
            )

    def test_check_and_supersede_legacy_does_not_pass_conversation_id(self):
        store = _SpyStore()
        store.seed_candidate(Fact(
            id="cand-leg", subject="iris", verb="likes", object="tea",
            what="pref", conversation_id="conv-X",
        ))
        checker = _make_supersession_checker(store)
        checker.check_and_supersede([Fact(
            id="new-leg", subject="iris", verb="likes", object="coffee",
            what="pref", conversation_id="conv-X",
        )])
        qf = store.calls_of("query_facts")
        assert qf
        for call in qf:
            # Legacy path: conversation_id must NOT be passed so behavior
            # matches pre-P3 global scan.
            assert "conversation_id" not in call.kwargs

    def test_promote_planned_facts_passes_conversation_id_when_fenced(self):
        store = _SpyStore()
        store.seed_planned_fact(Fact(
            id="planned-scoped", subject="jay", verb="will",
            object="walk", what="task", status="planned",
            when_date="2020-01-01", conversation_id="conv-A",
        ))
        promote_planned_facts(
            store,
            operation_id="op-1", owner_worker_id="w-1",
            lifecycle_epoch=1, conversation_id="conv-A",
        )
        qf = store.calls_of("query_facts")
        assert qf
        assert qf[0].kwargs.get("conversation_id") == "conv-A"

    def test_promote_planned_facts_legacy_does_not_pass_conversation_id(self):
        store = _SpyStore()
        store.seed_planned_fact(Fact(
            id="planned-leg", subject="k", verb="will", object="x",
            what="y", status="planned",
            when_date="2020-01-01", conversation_id="conv-Q",
        ))
        promote_planned_facts(store)
        qf = store.calls_of("query_facts")
        assert qf
        assert "conversation_id" not in qf[0].kwargs


# ---------------------------------------------------------------------------
# T3.17g: compaction-pipeline guard kwargs are all-or-nothing. When the
# pipeline has no active operation_id but DOES have a worker_id, the
# helper must emit (None, None, None) -- NOT (None, worker, None).
# ---------------------------------------------------------------------------


class _StubEngineState:
    def __init__(self, epoch: int = 1) -> None:
        self.lifecycle_epoch = epoch


class _GuardPipelineStub:
    """The smallest object that ``_compaction_guard_kwargs`` reads from
    ``self``. We bind the bound method to this instance to exercise the
    contract without spinning up the full CompactionPipeline.
    """

    def __init__(self, worker_id: str | None, conv_id: str = "conv-A",
                 epoch: int = 1) -> None:
        self._worker_id = worker_id
        self._engine_state = _StubEngineState(epoch=epoch)

        class _Cfg:
            conversation_id = conv_id
        self._config = _Cfg()


def _bound_guard(worker_id: str | None):
    from virtual_context.core.compaction_pipeline import CompactionPipeline
    stub = _GuardPipelineStub(worker_id=worker_id)
    return CompactionPipeline._compaction_guard_kwargs.__get__(stub)


class TestT317g_GuardKwargsAllOrNothing:
    def test_no_operation_id_returns_all_none_even_with_worker(self):
        guard = _bound_guard(worker_id="w-active")
        kw = guard(None)
        assert kw == {
            "operation_id": None,
            "owner_worker_id": None,
            "lifecycle_epoch": None,
        }

    def test_no_worker_returns_all_none_even_with_operation_id(self):
        guard = _bound_guard(worker_id=None)
        kw = guard("op-1")
        assert kw == {
            "operation_id": None,
            "owner_worker_id": None,
            "lifecycle_epoch": None,
        }

    def test_both_set_returns_full_guard_triple(self):
        guard = _bound_guard(worker_id="w-1")
        kw = guard("op-1")
        assert kw == {
            "operation_id": "op-1",
            "owner_worker_id": "w-1",
            "lifecycle_epoch": 1,
        }

    def test_include_conversation_id_adds_conv_kwarg(self):
        guard = _bound_guard(worker_id="w-1")
        kw = guard("op-1", include_conversation_id=True)
        assert kw == {
            "operation_id": "op-1",
            "owner_worker_id": "w-1",
            "lifecycle_epoch": 1,
            "conversation_id": "conv-A",
        }

    def test_include_conversation_id_legacy_all_none(self):
        guard = _bound_guard(worker_id="w-1")
        kw = guard(None, include_conversation_id=True)
        assert kw == {
            "operation_id": None,
            "owner_worker_id": None,
            "lifecycle_epoch": None,
            "conversation_id": None,
        }
