"""Run offline source contracts and presentation ablations against real SQLite.

Usage: .venv/bin/python -m benchmarks.context_contracts.runner --output /tmp/context-contracts.json

This corpus tests deterministic storage, admission, assembly and paging contracts.
It does not measure LLM answer accuracy, learned embedding recall, production
latency, PostgreSQL plans, or real API cost. Synthetic embeddings deliberately
score every source equally, exercising stable pagination and duplicate handling.
Token counts use the local cl100k_base tokenizer, including delivered wrappers.
Each layer receives the same exact persisted corpus and request-owned scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from virtual_context.core.assembler import ContextAssembler
from virtual_context.core.fact_query import FactQueryEngine
from virtual_context.core.paging_manager import PagingManager
from virtual_context.core.rendered_memory import MemorySourceVersion
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.structured_summary import (
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.token_counter import create_token_counter
from virtual_context.types import (
    AssemblerConfig,
    CanonicalTurnChunkEmbedding,
    DepthLevel,
    Fact,
    RetrievalResult,
    SegmentMetadata,
    SpeakerRetrievalContext,
    StoredSegment,
    StructuredSummary,
    SummaryClaim,
    SummarySource,
    TagSummary,
    VirtualContextConfig,
    WorkingSetEntry,
)

ABLATIONS = (
    "full_layers",
    "canonical_only",
    "tags_only",
    "facts_only",
    "segment_claims",
    "tag_claims",
)
OWNER, AUDIENCE, CHANNEL = "contract-owner", "contract-guild", "public"
ACTOR, LABEL = "actor:discord:alice", "Alice"
STAMP = "2026-09-05T12:00:00+00:00"


class ControlledEmbeddings:
    """Ranking control only; never a learned model or a relevance judge."""

    def __init__(self):
        self.calls = 0

    def get_embed_fn(self):
        def embed(texts):
            self.calls += 1
            return [[1.0, 0.0] for _text in texts]

        return embed


class ObservedSQLite(SQLiteStore):
    def __init__(self, path):
        super().__init__(path)
        self.embedding_pages = 0
        self.embedding_rows = 0
        self.embedding_page_sizes = []

    def get_canonical_turn_chunk_embedding_page(self, **kwargs):
        page = super().get_canonical_turn_chunk_embedding_page(**kwargs)
        self.embedding_pages += 1
        self.embedding_rows += len(page)
        self.embedding_page_sizes.append(len(page))
        return page


def load_corpus(path: Path | None = None) -> dict:
    path = path or Path(__file__).with_name("corpus.yaml")
    corpus = yaml.safe_load(path.read_text())
    if corpus.get("version") != 1 or not isinstance(corpus.get("scenarios"), list):
        raise ValueError("Unsupported context contract corpus")
    names = [case["id"] for case in corpus["scenarios"]]
    if len(names) != len(set(names)):
        raise ValueError("Scenario ids must be unique")
    return corpus


def _record(row) -> dict:
    return {
        "canonical_turn_id": row.canonical_turn_id,
        "source_role": "requester",
        "actor_id": row.sender_actor_id,
        "speaker_label": row.sender,
        "content": row.user_content,
        "session_date": row.session_date,
        "audience_conversation_id": row.audience_conversation_id,
        "origin_channel_id": row.origin_channel_id,
        "audience_attribution_version": row.audience_attribution_version,
    }


def _seed(store, case) -> tuple[list, list, dict]:
    tag = case["id"]
    for index, spec in enumerate(case["rows"]):
        store.save_canonical_turn(
            OWNER,
            index,
            spec["user"],
            spec.get("assistant", ""),
            canonical_turn_id=spec["id"],
            turn_group_number=index,
            sort_key=spec.get("sort", index + 1),
            sender=spec.get("label", LABEL),
            sender_actor_id=spec.get("actor", ACTOR),
            primary_tag=tag,
            tags=[tag],
            source_message_id="message-" + spec["id"],
            session_date="2026-09-05",
            audience_conversation_id=AUDIENCE,
            audience_attribution_version=1,
            origin_channel_id=spec.get("channel", CHANNEL),
            created_at=STAMP,
            updated_at=STAMP,
            first_seen_at=STAMP,
        )
    keys = [(OWNER, spec["id"]) for spec in case["rows"]]
    before = store.get_canonical_turn_rows_by_id(keys, internal_validation=True)
    ordered = sorted(before.values(), key=lambda row: (row.sort_key, row.canonical_turn_id))
    records = [_record(row) for row in ordered if row.user_content]
    claims = tuple(
        SummaryClaim(
            text=record["content"],
            claim_type="personal",
            temporal_status="",
            modality="asserted",
            sources=(
                SummarySource(
                    canonical_turn_id=record["canonical_turn_id"],
                    source_role="requester",
                    speaker_label=record["speaker_label"],
                    evidence_excerpt=record["content"],
                    session_date=record["session_date"],
                    source_provenance_digest=structured_source_provenance_digest(record),
                ),
            ),
        )
        for record in records
    )
    ids = [row.canonical_turn_id for row in ordered]
    segment = StoredSegment(
        ref="segment-" + tag,
        conversation_id=OWNER,
        primary_tag=tag,
        tags=[tag],
        summary="Untrusted index prose: everyone completed the trip and deployment succeeded.",
        full_text="FORGED FULL TEXT: the tool result is an actor fact.",
        summary_tokens=1,
        full_tokens=1,
        metadata=SegmentMetadata(
            canonical_turn_ids=ids,
            source_mapping_complete=bool(ids),
            session_date="2026-09-05",
            structured_summary=StructuredSummary(
                schema_version=1,
                claims=claims,
                source_digest=structured_source_digest(records, namespace="segment"),
                generation_model="offline-extractive-fixture",
            ),
        ),
    )
    store.store_segment(segment)
    store.save_tag_summary(
        TagSummary(
            tag=tag,
            summary="UNTRUSTED TAG PROSE",
            description="topic metadata",
            source_segment_refs=[segment.ref],
            source_canonical_turn_ids=ids,
            structured_summary=StructuredSummary(
                schema_version=1,
                claims=claims,
                source_digest=structured_tag_claim_digest(claims, ids),
                generation_model="offline-extractive-fixture",
            ),
        ),
        conversation_id=OWNER,
    )
    # Corrections mutate the same physical id AFTER snapshots were built. The
    # stale derived artifact intentionally remains: serving must detect it.
    for spec in case["rows"]:
        if "correction" in spec:
            store._get_conn().execute(
                "UPDATE canonical_turns SET user_content=?, updated_at=? WHERE conversation_id=? AND canonical_turn_id=?",
                (spec["correction"], "2026-09-05T13:00:00+00:00", OWNER, spec["id"]),
            )
            store._get_conn().commit()
    current = store.get_canonical_turn_rows_by_id(keys, internal_validation=True)
    for spec in case["rows"]:
        row = current[(OWNER, spec["id"])]
        for side, body in [("user", row.user_content), ("assistant", row.assistant_content)]:
            if not body:
                continue
            store.store_canonical_turn_chunk_embeddings(
                OWNER,
                row.turn_number,
                side,
                [
                    CanonicalTurnChunkEmbedding(
                        conversation_id=OWNER,
                        canonical_turn_id=row.canonical_turn_id,
                        side=side,
                        chunk_index=i,
                        text=body,
                        embedding=[1.0, 0.0],
                    )
                    for i in range(spec.get("copies", 1))
                ],
                canonical_turn_id=row.canonical_turn_id,
                embedding_model="offline-controlled-2d",
            )
    facts = [
        Fact(
            id=spec["id"],
            conversation_id=OWNER,
            subject=LABEL,
            verb=spec["verb"],
            object=spec["object"],
            status=spec.get("status", ""),
            tags=[tag],
            what="UNTRUSTED FACT PROSE",
            segment_ref=segment.ref,
            author_actor_id=ACTOR,
            author_attribution_version=2,
            author_source_role="requester",
            mentioned_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        )
        for spec in case.get("facts", [])
    ]
    if facts:
        store.store_facts(facts)
    if case.get("tool_output"):
        store.store_tool_output(
            "tool-artifact",
            OWNER,
            "shell",
            "deploy",
            0,
            case["tool_output"],
            len(case["tool_output"]),
        )
    return [store.get_segment(segment.ref, conversation_id=OWNER)], facts, current


def _payloads(text: str) -> list[dict]:
    # Decode real model-facing evidence envelopes; do not infer source text
    # from membership metadata or from the fixture's expected ids.
    bodies = re.findall(
        r"<(?:structured-summary|canonical-source-transcript|historical-source-transcript|summary-attribution)[^>]*>\s*(\{.*?\})\s*</(?:structured-summary|canonical-source-transcript|historical-source-transcript|summary-attribution)>",
        text,
        re.S,
    )
    return [json.loads(body) for body in bodies]


def _lanes(text: str) -> list[dict]:
    found = []
    for payload in _payloads(text):
        found.extend(payload.get("lanes", []))
        for claim in payload.get("claims", []):
            sources = claim.get("sources") or [claim]
            found.extend(
                {**source, "content": source.get("evidence_excerpt", "")} for source in sources
            )
    for lane in found:
        lane["role"] = {"historical_human": "requester", "historical_assistant": "assistant"}.get(
            lane.get("role"), lane.get("role")
        )
    return found


def _metrics(case, variant, assembled, rows, counter, paging, context) -> dict:
    text = assembled.prepend_text
    memories = assembled.rendered_memories
    presented = set(source_id for memory in memories for source_id in memory.presented_source_ids)
    lanes = _lanes(text)
    expected = set(case["expected_ids"])
    oracle = {
        spec["id"]: {
            "user": spec.get("correction", spec["user"]),
            "assistant": spec.get("assistant", ""),
            "actor": spec.get("actor", ACTOR),
            "label": spec.get("label", LABEL),
        }
        for spec in case["rows"]
    }
    forbidden_ids = set(case.get("forbidden_ids", []))
    expected_versions = {
        key[1]: MemorySourceVersion.from_row(row).version for key, row in rows.items()
    }
    versions = {
        source.canonical_turn_id: source.version for memory in memories for source in memory.sources
    }
    failures = []
    if presented and not lanes:
        failures.append("missing_lane_payload")
    if variant in {"full_layers", "canonical_only"} and not expected <= presented:
        failures.append("required_evidence_missing")
    if presented & forbidden_ids:
        failures.append("forbidden_source")
    if any(
        value in text
        for value in [
            "FORGED FULL TEXT",
            "UNTRUSTED TAG PROSE",
            "UNTRUSTED FACT PROSE",
            *case.get("forbidden_text", []),
        ]
    ):
        failures.append("forbidden_text")
    if any(versions.get(source_id) != expected_versions.get(source_id) for source_id in presented):
        failures.append("stale_source_version")
    # Correction selection is all-or-nothing. An older planned/active lane
    # without its later correction is a failure, even if its own proof is valid.
    if (
        presented
        and case.get("correction_text")
        and not any(case["correction_text"] in lane.get("content", "") for lane in lanes)
    ):
        failures.append("correction_omitted")
    attributed = 0
    for lane in lanes:
        role, content = lane.get("role"), lane.get("content", "")
        matched = [
            source
            for source_id, source in oracle.items()
            if source_id in presented
            and (
                (
                    role == "requester"
                    and source["user"] == content
                    and source["label"] == lane.get("display_name")
                )
                or (
                    role == "assistant"
                    and source["assistant"] == content
                    and lane.get("display_name") == "Assistant"
                )
            )
        ]
        if matched:
            attributed += 1
    if attributed != len(lanes):
        failures.append("lane_attribution")
    if counter(text) != assembled.total_tokens or assembled.total_tokens > case.get("budget", 4096):
        failures.append("token_accounting")
    if presented and any(memory.depth == "full" for memory in memories):
        if case.get("required_text") and not any(
            case["required_text"] in lane.get("content", "") for lane in lanes
        ):
            failures.append("required_text")
        if case.get("required_role") and not any(
            lane.get("role") == case["required_role"] for lane in lanes
        ):
            failures.append("required_role")
        order = case.get("ordered_text", [])
        contents = [lane.get("content") for lane in lanes]
        positions = [contents.index(body) if body in contents else -1 for body in order]
        if positions and (min(positions) < 0 or positions != sorted(positions)):
            failures.append("source_order")
    page_result = paging.expand_topic(case["id"], "full", speaker_context=context)
    page = paging.rendered_memories.get(case["id"])
    page_ok = (
        page is not None
        and "error" not in page_result
        and paging.working_set[case["id"]].tokens == counter(page.text)
        and set(page.presented_source_ids) == expected
    )
    if not expected:
        page_ok = "error" in page_result and page is None
    if not page_ok:
        failures.append("paging_recovery")
    return {
        "scenario": case["id"],
        "ablation": variant,
        "expected_source_ids": sorted(expected),
        "presented_source_ids": sorted(presented),
        "source_versions": versions,
        "expected_source_versions": expected_versions,
        "recall": len(presented & expected) / len(expected) if expected else None,
        "attribution_precision": attributed / len(lanes) if lanes else None,
        "correction_accuracy": int(
            "correction_omitted" not in failures and "forbidden_text" not in failures
        )
        if case.get("correction_text")
        else None,
        "abstained": not bool(presented),
        "abstention_accuracy": int(not presented) if not expected else None,
        "delivered_tokens": counter(text),
        "paging_success": page_ok,
        "contract_passed": not failures,
        "failures": failures,
        "evidence_kinds": sorted({memory.evidence_kind for memory in memories}),
    }


def evaluate(*, corpus: dict | None = None, ablations=ABLATIONS) -> dict:
    corpus = corpus or load_corpus()
    if set(ablations) - set(ABLATIONS):
        raise ValueError("Unknown ablation")
    counter = create_token_counter("tiktoken")
    results, probes = [], []
    with tempfile.TemporaryDirectory(prefix="vc-context-contracts-") as directory:
        for index, case in enumerate(corpus["scenarios"]):
            store = ObservedSQLite(str(Path(directory) / f"{index}.db"))
            try:
                segments, facts, rows = _seed(store, case)
                context = SpeakerRetrievalContext(
                    tenant_id="contract-tenant",
                    owner_conversation_id=OWNER,
                    audience_conversation_id=AUDIENCE,
                    audience_channel_id=CHANNEL,
                    requester_actor_id=ACTOR,
                )
                config = VirtualContextConfig(conversation_id=OWNER)
                provider = ControlledEmbeddings()
                semantic = SemanticSearchManager(store, config, embedding_provider=provider)
                matches = semantic.semantic_canonical_turn_search(
                    case["query"],
                    max_results=20,
                    conversation_id=OWNER,
                    speaker_context=context,
                )
                source_hits = [asdict(match.provenance) for match in matches if match.provenance]
                probe = {
                    "scenario": case["id"],
                    "canonical_retrieval": source_hits,
                    "embedding_pages": store.embedding_pages,
                    "embedding_rows": store.embedding_rows,
                    "embedding_page_sizes": store.embedding_page_sizes,
                    "page_bound_passed": all(size <= 200 for size in store.embedding_page_sizes),
                    "embedding_calls": provider.calls,
                    "external_llm_calls": 0,
                    "pagination_passed": store.embedding_pages
                    >= case.get("minimum_embedding_pages", 1),
                    "assistant_attribution_passed": all(
                        hit["actor_id"] == ""
                        for hit in source_hits
                        if hit["source_role"] == "assistant"
                    ),
                    "channel_scope_passed": all(
                        hit["origin_channel_id"] == CHANNEL for hit in source_hits
                    ),
                    "actor_provenance_passed": all(
                        hit["canonical_turn_id"] in {key[1] for key in rows}
                        and hit["actor_id"]
                        == (
                            rows[(OWNER, hit["canonical_turn_id"])].sender_actor_id
                            if hit["source_role"] == "requester"
                            else ""
                        )
                        for hit in source_hits
                    ),
                }
                if "fact_probe" in case:
                    query = dict(case["fact_probe"])
                    expected_count = query.pop("expected_count")
                    found = FactQueryEngine(store=store, semantic=semantic, config=config).query(
                        **query
                    )
                    probe["fact_temporal_query_passed"] = len(found) == expected_count
                    probe["fact_query_ids"] = [fact.id for fact in found]
                probes.append(probe)
                for variant in ablations:
                    assembler = ContextAssembler(
                        AssemblerConfig(), counter, store=store, conversation_id=OWNER
                    )
                    paging = PagingManager(
                        store,
                        counter,
                        tag_context_max_tokens=case.get("budget", 4096),
                        conversation_id=OWNER,
                    )
                    paging.set_memory_renderer(assembler.render_topic_memory)
                    depth = {
                        "segment_claims": DepthLevel.SEGMENTS,
                        "tag_claims": DepthLevel.SUMMARY,
                        "full_layers": DepthLevel.SUMMARY,
                    }.get(variant, DepthLevel.FULL)
                    with_sources = variant not in {"tags_only", "facts_only"}
                    retrieval = RetrievalResult(
                        tags_matched=[case["id"]],
                        summaries=[
                            store.get_summary(segment.ref, conversation_id=OWNER)
                            for segment in segments
                        ]
                        if with_sources
                        else [],
                        facts=facts if variant in {"facts_only", "full_layers"} else [],
                    )
                    assembled = assembler.assemble(
                        "",
                        retrieval,
                        [],
                        case.get("budget", 4096),
                        context_hint=f"Available topic: {case['id']}"
                        if variant == "tags_only"
                        else "",
                        working_set={case["id"]: WorkingSetEntry(tag=case["id"], depth=depth)}
                        if with_sources
                        else {},
                        full_segments={case["id"]: segments} if with_sources else {},
                        speaker_context=context,
                    )
                    initial_tokens = counter(assembled.prepend_text)
                    expanded = False
                    if variant == "full_layers" and (
                        not any(
                            memory.presented_source_ids for memory in assembled.rendered_memories
                        )
                        or case.get("request_depth") == "full"
                    ):
                        page = paging.expand_topic(case["id"], "full", speaker_context=context)
                        if "error" not in page:
                            expanded = True
                            assembled = assembler.assemble(
                                "",
                                retrieval,
                                [],
                                case.get("budget", 4096),
                                working_set=paging.working_set,
                                full_segments={case["id"]: segments},
                                speaker_context=context,
                            )
                    measured = _metrics(case, variant, assembled, rows, counter, paging, context)
                    measured["expanded_from_summary"] = expanded
                    measured["initial_delivery_tokens"] = initial_tokens
                    measured["staged_delivery_tokens"] = (
                        initial_tokens + counter(assembled.prepend_text)
                        if expanded
                        else initial_tokens
                    )
                    results.append(measured)
            finally:
                store.close()
    aggregates = {}
    for variant in ablations:
        selected = [row for row in results if row["ablation"] == variant]
        averages = {}
        for name in [
            "recall",
            "attribution_precision",
            "correction_accuracy",
            "abstention_accuracy",
            "delivered_tokens",
            "paging_success",
        ]:
            values = [row[name] for row in selected if row[name] is not None]
            averages[name] = sum(values) / len(values) if values else None
        aggregates[variant] = {
            **averages,
            "cases": len(selected),
            "contract_failures": sum(not row["contract_passed"] for row in selected),
        }
    return {
        "evaluation": "offline-context-contracts",
        "version": 1,
        "corpus_version": corpus["version"],
        "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
        "limitations": "Deterministic real-SQLite contract controls; not LLM answer quality, learned embedding recall, live PostgreSQL validation or production latency.",
        "tokenizer": "cl100k_base",
        "recall_unit": "Expected physical canonical source id; full-depth role/text requirements and exact fixture actor/label are checked independently.",
        "candidate_policy": "Each scenario supplies one topic; presentation receives its exhaustive stored candidates. Recall measures evidence delivery, not semantic relevance ranking.",
        "paging_metric": "Independent explicit FULL recovery from each initial presentation; canonical sources remain in storage for every ablation.",
        "ablation_policy": {
            "full_layers": "SUMMARY first, FULL on an empty proved summary or explicit full-depth request",
            "canonical_only": "FULL canonical projection",
            "tags_only": "Topic catalog with no evidence sources",
            "facts_only": "Derived facts presented to the canonical-proof boundary",
            "segment_claims": "SEGMENTS structured claims with ordinary source-proof fallback",
            "tag_claims": "SUMMARY tag claims with ordinary source-proof fallback",
        },
        "cost": {
            "external_llm_calls": 0,
            "external_embedding_calls": 0,
            "api_cost_usd": 0.0,
            "kind": "offline controls, no model-cost comparison",
        },
        "ablations": aggregates,
        "results": results,
        "retrieval_probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ablation", choices=ABLATIONS, action="append")
    args = parser.parse_args()
    report = evaluate(corpus=load_corpus(args.corpus), ablations=args.ablation or ABLATIONS)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    return int(
        any(not row["contract_passed"] for row in report["results"])
        or any(
            not value
            for probe in report["retrieval_probes"]
            for key, value in probe.items()
            if key.endswith("_passed")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
