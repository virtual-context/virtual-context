"""Gate routing and context forwarding for speaker-aware retrieval.

``SearchEngine.find_quote`` always sends explicit request authority to
candidate generation.  ``speaker_annotations_enabled`` gates presentation
only; absent or ineligible authority reaches quote search as an ineligible
sentinel and fails closed instead of selecting the legacy branch.

Callers thread the context, they never gate: the VC tool executor, the
engine's synchronous tool-loop wrapper, and the VCRECALL command handler
all forward the request-derived context into the gated entrypoint.
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from virtual_context.config import VirtualContextConfig
from virtual_context.core.search_engine import SearchEngine
from virtual_context.core.summary_identity import SUMMARY_ATTRIBUTION_QUARANTINE
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


def _syntax_only_canonical_envelope() -> str:
    payload = {
        "source": "canonical_turns",
        "generated_summary_prose_used": False,
        "lanes": [{
            "source_speaker_ref": "historical_0123456789abcdef",
            "display_name": "BigTex",
            "role": "historical_human",
            "content": "I currently take tesamorelin.",
            "session_date": "2026-08-18",
            "current_requester_match": "unproved",
        }],
    }
    return (
        "<historical-source-transcript>\n"
        f"{json.dumps(payload, separators=(',', ':'))}\n"
        "</historical-source-transcript>"
    )


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

    def test_gate_off_keeps_authority_but_disables_annotations(self):
        engine = self._engine(enabled=False)
        context = _ctx()
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context
        assert spy.call_args.kwargs["speaker_annotations"] is False

    def test_gate_on_forwards_the_exact_eligible_context(self):
        engine = self._engine(enabled=True)
        context = _ctx()
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context
        assert spy.call_args.kwargs["speaker_annotations"] is True

    def test_gate_on_preserves_ineligible_context_for_refusal(self):
        engine = self._engine(enabled=True)
        context = _ctx(audience_conversation_id="")
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote(
                "q", speaker_context=context,
            )
        assert spy.call_args.kwargs["speaker_context"] is context
        assert spy.call_args.kwargs["speaker_annotations"] is False

    def test_direct_no_context_keeps_isolated_legacy_path(self):
        engine = self._engine(enabled=True)
        with patch(
            "virtual_context.core.search_engine._find_quote",
            return_value={"found": False},
        ) as spy:
            engine.find_quote("q")
        assert spy.call_args.kwargs["speaker_context"] is None
        assert spy.call_args.kwargs["speaker_annotations"] is False

    def test_summary_search_routes_the_same_gate_to_core_search(self):
        engine = self._engine(enabled=True)
        context = _ctx()
        with patch(
            "virtual_context.core.search_engine._search_summaries",
            return_value={"found": False, "results": []},
        ) as spy:
            engine.search_summaries("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context

    def test_summary_search_gate_off_still_forwards_incident_containment_context(self):
        engine = self._engine(enabled=False)
        context = _ctx()
        with patch(
            "virtual_context.core.search_engine._search_summaries",
            return_value={"found": False, "results": []},
        ) as spy:
            engine.search_summaries("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context

    def test_summary_search_preserves_explicit_ineligible_context_for_refusal(self):
        engine = self._engine(enabled=False)
        context = _ctx(audience_conversation_id="")
        with patch(
            "virtual_context.core.search_engine._search_summaries",
            return_value={"found": False, "results": []},
        ) as spy:
            engine.search_summaries("q", speaker_context=context)
        assert spy.call_args.kwargs["speaker_context"] is context

    def test_summary_search_does_not_launder_syntax_only_envelope(self):
        engine = self._engine(enabled=True)
        with patch(
            "virtual_context.core.search_engine._search_summaries",
            return_value={
                "found": True,
                "results": [{"excerpt": _syntax_only_canonical_envelope()}],
            },
        ):
            result = engine.search_summaries("q", speaker_context=_ctx())

        assert result["results"][0]["excerpt"] == SUMMARY_ATTRIBUTION_QUARANTINE


class TestContextForwarding:
    def test_engine_search_summaries_forwards_the_context(self):
        from virtual_context.engine import VirtualContextEngine

        recorder = MagicMock()
        context = _ctx()
        VirtualContextEngine.search_summaries(
            SimpleNamespace(_search=recorder),
            "q",
            speaker_context=context,
        )
        assert recorder.search_summaries.call_args.kwargs["speaker_context"] is context

    def test_engine_remember_when_forwards_the_context(self):
        from virtual_context.engine import VirtualContextEngine

        recorder = MagicMock()
        recorder.remember_when.return_value = {"found": False, "results": []}
        context = _ctx()
        VirtualContextEngine.remember_when(
            SimpleNamespace(_temporal=recorder),
            "q",
            {"last_n_days": 30},
            speaker_context=context,
        )
        assert recorder.remember_when.call_args.kwargs["speaker_context"] is context

    def test_engine_remember_when_does_not_launder_syntax_only_envelope(self):
        from virtual_context.engine import VirtualContextEngine

        recorder = MagicMock()
        recorder.remember_when.return_value = {
            "found": True,
            "results": [{"excerpt": _syntax_only_canonical_envelope()}],
        }

        result = VirtualContextEngine.remember_when(
            SimpleNamespace(_temporal=recorder),
            "tesamorelin",
            {"last_n_days": 30},
            speaker_context=_ctx(),
        )

        assert result["results"][0]["excerpt"] == SUMMARY_ATTRIBUTION_QUARANTINE

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

    def test_execute_vc_tool_supplies_ineligible_context_without_one(self):
        from virtual_context.core.tool_loop import execute_vc_tool

        engine = MagicMock()
        engine.config.conversation_id = "c"
        engine.config.search.find_quote_max_results = 20
        engine.config.search.tool_guard_enabled = False
        engine.find_quote.return_value = {"found": False, "results": []}
        execute_vc_tool(engine, "vc_find_quote", {"query": "x"})
        context = engine.find_quote.call_args.kwargs["speaker_context"]
        assert isinstance(context, SpeakerRetrievalContext)
        assert context.eligible is False

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
