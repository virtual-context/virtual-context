"""Offline memory contracts exercise real SQLite and model-facing proof paths."""

from dataclasses import replace

import pytest

from benchmarks.context_contracts.runner import ABLATIONS, evaluate, load_corpus
from virtual_context.core.assembler import ContextAssembler


@pytest.fixture(scope="module")
def report():
    return evaluate()


def _case(report, scenario, ablation="full_layers"):
    return next(
        row
        for row in report["results"]
        if row["scenario"] == scenario and row["ablation"] == ablation
    )


def test_real_sqlite_corpus_passes_all_safety_contracts_and_positive_baseline(report):
    assert len(report["results"]) == 11 * len(ABLATIONS)
    assert all(row["contract_passed"] for row in report["results"])
    assert all(
        row["recall"] in (1.0, None)
        for row in report["results"]
        if row["ablation"] == "full_layers"
    )
    assert all(
        value
        for probe in report["retrieval_probes"]
        for key, value in probe.items()
        if key.endswith("_passed")
    )


def test_corrections_require_current_versions_and_real_text_not_just_matching_ids(report):
    current = _case(report, "corrected_old_source")
    stale_claims = _case(report, "corrected_old_source", "segment_claims")
    assert current["presented_source_ids"] == ["old"]
    assert current["source_versions"]["old"] == current["expected_source_versions"]["old"]
    assert current["expanded_from_summary"] and current["correction_accuracy"] == 1
    assert stale_claims["presented_source_ids"] == []  # stale v1 may not launder a correction
    assert stale_claims["recall"] == 0 and stale_claims["correction_accuracy"] == 1
    cancelled = _case(report, "cancelled_plan")
    assert cancelled["presented_source_ids"] == ["cancel", "plan"]
    assert cancelled["correction_accuracy"] == 1
    probe = next(p for p in report["retrieval_probes"] if p["scenario"] == "cancelled_plan")
    assert probe["fact_temporal_query_passed"] and probe["fact_query_ids"] == []


def test_channel_tool_and_colliding_unicode_names_abstain_for_every_layer(report):
    for name in ("hidden_channel", "tool_artifact", "unicode_label_collision"):
        for ablation in ABLATIONS:
            result = _case(report, name, ablation)
            assert result["abstained"] and result["abstention_accuracy"] == 1
            assert result["paging_success"]


def test_assistant_code_stays_role_local_and_unicode_uses_measured_tokens(report):
    code = _case(report, "assistant_coding")
    assert code["expanded_from_summary"] and code["attribution_precision"] == 1
    assert code["evidence_kinds"] == ["canonical_transcript"]
    assert _case(report, "unicode_evidence")["delivered_tokens"] > 0
    assert all(row["paging_success"] for row in report["results"])
    probe = next(p for p in report["retrieval_probes"] if p["scenario"] == "assistant_coding")
    assistants = [hit for hit in probe["canonical_retrieval"] if hit["source_role"] == "assistant"]
    assert assistants and all(hit["actor_id"] == "" for hit in assistants)


def test_duplicate_chunk_control_requires_continuation_to_later_physical_source(report):
    probe = next(
        p for p in report["retrieval_probes"] if p["scenario"] == "duplicate_chunk_continuation"
    )
    assert probe["embedding_page_sizes"] == [200, 6, 0]
    assert {hit["canonical_turn_id"] for hit in probe["canonical_retrieval"]} == {
        "repeated",
        "wanted",
    }
    assert probe["pagination_passed"] and probe["page_bound_passed"]


def test_ablations_disclose_evidence_loss_and_zero_external_model_cost(report):
    assert report["ablations"]["tags_only"]["recall"] == 0
    assert report["ablations"]["facts_only"]["recall"] == 0
    assert 0 < report["ablations"]["segment_claims"]["recall"] < 1
    assert report["cost"]["external_llm_calls"] == report["cost"]["external_embedding_calls"] == 0
    assert report["cost"]["api_cost_usd"] == 0
    assert "not LLM answer quality" in report["limitations"]
    assert report["tokenizer"] == "cl100k_base"


def test_negative_control_stale_version_fails_even_when_text_and_source_ids_match(monkeypatch):
    original = ContextAssembler.assemble

    def stale(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.rendered_memories = tuple(
            replace(
                memory,
                sources=tuple(replace(source, version="0" * 64) for source in memory.sources),
            )
            for memory in result.rendered_memories
        )
        return result

    monkeypatch.setattr(ContextAssembler, "assemble", stale)
    corpus = load_corpus()
    corpus["scenarios"] = [corpus["scenarios"][0]]
    result = evaluate(corpus=corpus, ablations=("canonical_only",))["results"][0]
    assert result["presented_source_ids"] == ["old"] and result["recall"] == 1
    assert not result["contract_passed"] and "stale_source_version" in result["failures"]


def test_negative_control_wrong_human_label_fails_even_with_valid_source_metadata(monkeypatch):
    original = ContextAssembler.assemble

    def wrong_label(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.prepend_text = result.prepend_text.replace("Alice", "Betty")
        return result

    monkeypatch.setattr(ContextAssembler, "assemble", wrong_label)
    corpus = load_corpus()
    corpus["scenarios"] = [
        case for case in corpus["scenarios"] if case["id"] == "same_speaker_history"
    ]
    result = evaluate(corpus=corpus, ablations=("canonical_only",))["results"][0]
    assert result["recall"] == 1 and not result["contract_passed"]
    assert "lane_attribution" in result["failures"]
