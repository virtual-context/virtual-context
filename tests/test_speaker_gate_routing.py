"""Gate routing and context forwarding for speaker-aware retrieval.

``SearchEngine.find_quote`` is the gate router: a supplied
``SpeakerRetrievalContext`` reaches candidate generation only when
``speaker_annotations_enabled`` is on AND the context proved its audience.
Otherwise the context is normalized to ``None`` before candidate
generation, which selects the complete legacy retrieval branch — an
ineligible context is never repaired to the resolved owner.

Callers thread the context, they never gate: the VC tool executor, the
engine's synchronous tool-loop wrapper, and the VCRECALL command handler
all forward the request-derived context into the gated entrypoint.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from virtual_context.config import VirtualContextConfig
from virtual_context.core.search_engine import SearchEngine
from virtual_context.types import (
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
)


def _ctx(**kw) -> SpeakerRetrievalContext:
    base = dict(
        tenant_id="t",
        owner_conversation_id="c",
        audience_conversation_id="c",
    )
    base.update(kw)
    return SpeakerRetrievalContext(**base)


class TestGateRouter:
    def _engine(self, *, enabled: bool) -> SearchEngine:
        config = VirtualContextConfig(
            conversation_id="c",
            storage=StorageConfig(backend="sqlite"),
            tag_generator=TagGeneratorConfig(type="keyword"),
        )
        config.search.speaker_annotations_enabled = enabled
        return SearchEngine(
            store=MagicMock(), semantic=MagicMock(),
            turn_tag_index=MagicMock(), config=config,
        )

    def test_gate_off_normalizes_an_eligible_context_to_none(self):
        engine = self._engine(enabled=False)
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q", speaker_context=_ctx())
        assert spy.call_args.kwargs["speaker_context"] is None

    def test_gate_on_forwards_the_exact_eligible_context(self):
        engine = self._engine(enabled=True)
        context = _ctx()
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context

    def test_gate_on_still_drops_an_ineligible_context(self):
        engine = self._engine(enabled=True)
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote(
                "q", speaker_context=_ctx(audience_conversation_id=""),
            )
        assert spy.call_args.kwargs["speaker_context"] is None

    def test_no_context_stays_none(self):
        engine = self._engine(enabled=True)
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q")
        assert spy.call_args.kwargs["speaker_context"] is None


class TestContextForwarding:
    def test_execute_vc_tool_forwards_a_derived_context(self):
        from virtual_context.core.tool_loop import execute_vc_tool

        engine = MagicMock()
        engine.config.conversation_id = "c"
        engine.config.search.find_quote_max_results = 20
        engine.config.search.tool_guard_enabled = False
        engine.find_quote.return_value = {"found": False, "results": []}
        context = _ctx()
        execute_vc_tool(
            engine, "vc_find_quote", {"query": "x"}, speaker_context=context,
        )
        assert engine.find_quote.call_args.kwargs["speaker_context"] is context

    def test_execute_vc_tool_keeps_the_legacy_call_shape_without_one(self):
        from virtual_context.core.tool_loop import execute_vc_tool

        engine = MagicMock()
        engine.config.conversation_id = "c"
        engine.config.search.find_quote_max_results = 20
        engine.config.search.tool_guard_enabled = False
        engine.find_quote.return_value = {"found": False, "results": []}
        execute_vc_tool(engine, "vc_find_quote", {"query": "x"})
        assert "speaker_context" not in engine.find_quote.call_args.kwargs

    def test_engine_query_with_tools_forwards_the_context(self):
        from virtual_context.engine import VirtualContextEngine

        recorder = MagicMock()
        context = _ctx()
        VirtualContextEngine.query_with_tools(
            SimpleNamespace(_tool_query=recorder),
            [{"role": "user", "content": "hi"}],
            speaker_context=context,
        )
        forwarded = recorder.query_with_tools.call_args.kwargs
        assert forwarded["speaker_context"] is context

    def test_vcrecall_routes_through_the_engine_entrypoint(self):
        from virtual_context.proxy.handlers import _handle_vcrecall

        engine = MagicMock()
        engine.find_quote.return_value = {"found": False, "results": []}
        state = SimpleNamespace(engine=engine)
        context = _ctx()
        text = _handle_vcrecall("boston", state, speaker_context=context)
        assert "No matches" in text
        assert engine.find_quote.call_args.kwargs["speaker_context"] is context
        assert engine.find_quote.call_args.kwargs["max_results"] == 10
