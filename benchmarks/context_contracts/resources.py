"""Controlled archive resource/parity experiment, without LLM/model downloads.

Run each mode in a fresh process so fixture creation and previous requests do
not contaminate peak RSS. PostgreSQL is opt-in via a DSN environment variable;
its vector schema must already be migrated. Only generated fixture owners are
written/deleted. No database, extension, or shared migration is created here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

from virtual_context.core.math_utils import cosine_similarity
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore, _merge_canonical_turn_rows
from virtual_context.types import (
    CanonicalTurnChunkEmbedding, ChunkEmbedding, SpeakerRetrievalContext,
    StoredSegment, VirtualContextConfig,
)

MODEL = "all-MiniLM-L6-v2"


def vector(score=1.0):
    return [score, math.sqrt(1 - score * score)] + [0.0] * 382


def _open(spec):
    if spec["backend"] == "sqlite":
        return SQLiteStore(spec["path"])
    from virtual_context.storage.postgres import PostgresStore
    dsn = os.environ.get(spec["dsn_env"], "")
    if not dsn:
        raise RuntimeError("PostgreSQL resource run requires the configured DSN environment variable")
    return PostgresStore(dsn)


def _owner(spec, foreign=False):
    return "vc-resource-" + spec["fixture_id"] + ("-foreign" if foreign else "")


def _key(spec, index, foreign=False):
    return str(uuid.uuid5(uuid.UUID(spec["fixture_id"]), f"{foreign}:{index}"))


def _context(spec):
    return SpeakerRetrievalContext(
        tenant_id="resource-fixture", owner_conversation_id=_owner(spec),
        audience_conversation_id=_owner(spec), audience_channel_id="resource-public",
        request_origin_channel_id="resource-public", requester_actor_id="actor:discord:fixture",
    )


def seed(spec):
    store = _open(spec)
    try:
        if spec["backend"] == "postgres" and not store.vector_search_ready(MODEL):
            raise RuntimeError("Migrate the isolated PostgreSQL benchmark database before this run")
        for foreign in (False, True):
            owner = _owner(spec, foreign)
            store.upsert_conversation(tenant_id="resource-fixture", conversation_id=owner)
            count = max(1, spec["rows"] // 10) if foreign else spec["rows"]
            for index in range(count):
                key = _key(spec, index, foreign)
                # Unique ranks; foreign and wrong-audience records include
                # strong hits and must be excluded before result limits.
                values = vector(.999 if foreign else .51 + .35 * (index + 1) / (count + 1))
                text = f"fixture row {index} " + "x" * max(0, spec["body_bytes"] - 40)
                public = index % 7 != 0
                store.save_canonical_turn(
                    owner, index, text, "paired assistant source", canonical_turn_id=key,
                    sort_key=float(index), turn_group_number=index,
                    primary_tag="resource-topic", tags=["resource-topic"], sender="FixtureMember",
                    sender_actor_id="actor:discord:fixture", origin_channel_id="resource-public",
                    audience_conversation_id=owner if public else owner + "-private",
                    audience_attribution_version=1, compacted_at="2026-09-05" if index < count - 12 else None,
                )
                store.store_segment(StoredSegment(ref=key, conversation_id=owner, primary_tag="resource-topic", tags=["resource-topic"]))
                store.store_chunk_embeddings(key, [ChunkEmbedding(
                    segment_ref=key, chunk_index=0, text=f"fixture chunk {index}", embedding=values,
                )], embedding_model=MODEL)
                store.store_canonical_turn_chunk_embeddings(owner, index, "user", [CanonicalTurnChunkEmbedding(
                    conversation_id=owner, canonical_turn_id=key, turn_number=index,
                    side="user", chunk_index=0, text=f"fixture chunk {index}", embedding=values,
                )], canonical_turn_id=key, embedding_model=MODEL)
        return {"seeded_rows": spec["rows"], "fixture_id": spec["fixture_id"]}
    finally:
        store.close()


def cleanup(spec):
    store = _open(spec)
    try:
        # Owners are derived solely from a validated generated UUID. Never
        # accept a caller-provided existing conversation id for cleanup.
        uuid.UUID(spec["fixture_id"])
        for foreign in (False, True):
            store.delete_conversation(_owner(spec, foreign))
        return {"cleaned": True}
    finally:
        store.close()


def _body_bytes(row):
    return sum(len((getattr(row, name, "") or "").encode())
               for name in ("user_content", "assistant_content", "reply_target_body"))


class ReadCounts:
    def __init__(self, store):
        self.stage = "setup"
        self.values = {}
        self.native_queries = []
        decoder = store._canonical_decoder()
        def observed(row):
            parsed = decoder(row)
            self.bodies([parsed])
            return parsed
        store._canonical_decoder = lambda: observed
        physical = store.get_canonical_turn_rows_by_id
        def physical_rows(*args, **kwargs):
            rows = physical(*args, **kwargs)
            self.bodies(rows.values())
            return rows
        store.get_canonical_turn_rows_by_id = physical_rows
        get_segment = store.get_segment
        def segment(*args, **kwargs):
            self.record()["segment_source_reads"] += 1
            return get_segment(*args, **kwargs)
        store.get_segment = segment
        for method in ("get_segment_chunk_embedding_page", "get_canonical_turn_chunk_embedding_page",
                       "search_segment_chunks_by_embedding", "search_speaker_turn_chunks_by_embedding"):
            original = getattr(store, method, None)
            if not callable(original):
                continue
            def page(*args, _fn=original, **kwargs):
                rows = _fn(*args, **kwargs)
                self.page(rows)
                return rows
            setattr(store, method, page)
        if hasattr(store, "pool"):
            store.pool = _ObservedPool(store.pool, self.native_queries)

    def record(self):
        return self.values.setdefault(self.stage, dict(
            canonical_rows=0, canonical_body_bytes=0, embedding_rows=0,
            embedding_scalars=0, maximum_embedding_page=0, segment_source_reads=0,
        ))

    def bodies(self, rows):
        record = self.record()
        for row in rows:
            record["canonical_rows"] += 1
            record["canonical_body_bytes"] += _body_bytes(row)

    def page(self, rows):
        record = self.record()
        record["embedding_rows"] += len(rows)
        record["maximum_embedding_page"] = max(record["maximum_embedding_page"], len(rows))
        record["embedding_scalars"] += sum(len(row.get("embedding", ())) for row in rows)


class _ObservedPool:
    def __init__(self, pool, queries):
        self.pool, self.queries = pool, queries

    @contextmanager
    def connection(self):
        with self.pool.connection() as conn:
            queries = self.queries
            class Connection:
                def execute(self, query, params=None, **kwargs):
                    if isinstance(query, str) and query.startswith("WITH scored AS MATERIALIZED") and len(queries) < 2 and not any(query == text for text, _params in queries):
                        queries.append((query, params))
                    return conn.execute(query, params, **kwargs)
                def __getattr__(self, name):
                    return getattr(conn, name)
            yield Connection()

    def __getattr__(self, name):
        return getattr(self.pool, name)


def _peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _signature(result):
    return [(hit.segment_ref, hashlib.sha256(hit.text.encode()).hexdigest()) for hit in result]


def _baseline_search(store, manager, counts, spec, canonical=False):
    owner, query = _owner(spec), vector()
    chunks = (store.get_all_canonical_turn_chunk_embeddings(conversation_id=owner, speaker_context=_context(spec))
              if canonical else store.get_all_chunk_embeddings(conversation_id=owner))
    record = counts.record()
    record["embedding_rows"] += len(chunks)
    record["embedding_scalars"] += sum(len(chunk.embedding) for chunk in chunks)
    record["maximum_embedding_page"] = max(record["maximum_embedding_page"], len(chunks))
    ranked = sorted(((cosine_similarity(query, chunk.embedding), chunk) for chunk in chunks),
                    key=lambda item: item[0], reverse=True)
    if canonical:
        physical = store.get_canonical_turn_rows_by_id(
            [(owner, chunk.canonical_turn_id) for _score, chunk in ranked], speaker_context=_context(spec),
        )
        return [manager._format_physical_semantic_result(score, chunk, physical[(owner, chunk.canonical_turn_id)], channel="")
                for score, chunk in ranked[:spec["top_k"]]]
    from virtual_context.types import QuoteResult
    return [QuoteResult(text=chunk.text, tag="resource-topic", segment_ref=chunk.segment_ref, similarity=round(score, 3))
            for score, chunk in ranked[:spec["top_k"]]]


def measure(spec):
    store = _open(spec)
    counts = ReadCounts(store)
    mode = spec["mode"]
    config = VirtualContextConfig(conversation_id=_owner(spec))
    config.retriever.vector_search_enabled = mode == "native"
    manager = SemanticSearchManager(store, config)
    manager._embed_fn = lambda texts: [vector() for _ in texts]
    before = _peak_rss_bytes()
    latencies, signatures, rss_samples = {}, {}, {}
    try:
        for stage in ("pending_compaction", "logical_read", "segment_search", "speaker_search"):
            counts.stage = stage
            timings = []
            rss_samples[stage] = []
            for _ in range(spec["repeats"]):
                start = time.perf_counter()
                if stage in ("pending_compaction", "logical_read"):
                    if mode == "legacy":
                        archive = store._load_canonical_turn_rows(_owner(spec))
                        counts.bodies(archive)
                        merged = _merge_canonical_turn_rows(archive)
                        if stage == "pending_compaction":
                            result = [row for row in merged.values() if not row.compacted_at
                                      and row.user_content.strip() and row.assistant_content.strip()][:-2]
                        else:
                            result = [merged[spec["rows"] - 1]]
                    elif stage == "pending_compaction":
                        result = store.get_uncompacted_canonical_turns(_owner(spec), protected_recent_turns=2)
                    else:
                        result = list(store.get_canonical_turn_rows(_owner(spec), [spec["rows"] - 1]).values())
                    signatures[stage] = [(row.turn_group_number, hashlib.sha256(row.user_content.encode()).hexdigest()) for row in result]
                    if mode == "legacy":
                        del archive, merged
                elif mode == "legacy":
                    result = _baseline_search(store, manager, counts, spec, canonical=stage == "speaker_search")
                    signatures[stage] = _signature(result)
                elif stage == "segment_search":
                    result = manager.semantic_search("fixture query", max_results=spec["top_k"], conversation_id=_owner(spec))
                    signatures[stage] = _signature(result)
                else:
                    result = manager.semantic_canonical_turn_search("fixture query", max_results=spec["top_k"],
                                                                   conversation_id=_owner(spec), speaker_context=_context(spec))
                    signatures[stage] = _signature(result)
                timings.append((time.perf_counter() - start) * 1000)
                rss_samples[stage].append(_peak_rss_bytes())
            # Empirical nearest-rank percentile. Ten samples put p95 at the
            # maximum; this is an observation, not a population-tail estimate.
            latencies[stage] = {"median_ms": statistics.median(timings),
                                "p95_ms": sorted(timings)[math.ceil(.95 * len(timings)) - 1],
                                "maximum_ms": max(timings), "sample_count": len(timings)}
        plans = []
        for query, params in counts.native_queries:
            with store.pool.connection() as conn:
                row = conn.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query, params).fetchone()
                plans.append(row["QUERY PLAN"])
        return {"mode": mode, "peak_rss_bytes": _peak_rss_bytes(), "baseline_rss_bytes": before,
                "peak_growth_bytes": max(0, _peak_rss_bytes() - before), "latency": latencies,
                "reads": counts.values, "signatures": signatures, "native_explain": plans,
                "peak_rss_after_calls_bytes": rss_samples}
    finally:
        store.close()


def _child(spec, timeout=180):
    completed = subprocess.run([sys.executable, "-m", "benchmarks.context_contracts.resources", "--worker"],
                               input=json.dumps(spec), text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"Resource worker {spec['phase']} failed: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def run_experiment(*, backend="sqlite", sizes=(500, 5000), body_bytes=8192,
                   repeats=3, top_k=5, dsn_env="VC_RESOURCE_POSTGRES_DSN", timeout=180):
    if backend not in {"sqlite", "postgres"} or any(size <= 2 for size in sizes):
        raise ValueError("Choose sqlite/postgres and archive sizes greater than two")
    if body_bytes < 0 or repeats <= 0 or top_k <= 0:
        raise ValueError("Invalid resource experiment dimensions")
    runs = []
    with tempfile.TemporaryDirectory(prefix="vc-resource-") as directory:
        for size in sizes:
            spec = dict(backend=backend, rows=size, body_bytes=body_bytes, repeats=repeats, top_k=top_k,
                        fixture_id=uuid.uuid4().hex, path=str(Path(directory) / f"archive-{size}.db"), dsn_env=dsn_env)
            try:
                _child(dict(spec, phase="seed"), timeout)
                modes = ("legacy", "streaming", "native") if backend == "postgres" else ("legacy", "streaming")
                measurements = [_child(dict(spec, phase="measure", mode=mode), timeout) for mode in modes]
                expected = measurements[0]["signatures"]
                parity = all(result["signatures"] == expected for result in measurements[1:])
                allowed = {_key(spec, index) for index in range(size)}
                speaker_allowed = {_key(spec, index) for index in range(size) if index % 7 != 0}
                # Quote identities carry turn:<UUID>:user on the speaker path.
                scoped = all(
                    len(result["signatures"]["segment_search"]) == min(top_k, len(allowed))
                    and len(result["signatures"]["speaker_search"]) == min(top_k, len(speaker_allowed))
                    and all(identifier in allowed for identifier, _digest in result["signatures"]["segment_search"])
                    and all(any(key in identifier for key in speaker_allowed)
                            for identifier, _digest in result["signatures"]["speaker_search"])
                    for result in measurements
                )
                bounded = all(
                    result["reads"]["logical_read"]["canonical_rows"] == repeats
                    and result["reads"]["pending_compaction"]["canonical_rows"] <= 12 * repeats
                    and all(stage["maximum_embedding_page"] <= 200 for stage in result["reads"].values())
                    and (result["mode"] != "streaming" or result["reads"]["segment_search"]["segment_source_reads"] == 0)
                    and (result["mode"] != "native" or all(stage["embedding_scalars"] == 0 for stage in result["reads"].values()))
                    for result in measurements[1:]
                )
                runs.append(dict(rows=size, body_bytes=body_bytes, parity=parity, strict_scope=scoped,
                                 bounded_reads=bounded, measurements=measurements))
            finally:
                _child(dict(spec, phase="cleanup"), timeout)
    return dict(schema_version=2, backend=backend, fixture="generated unique scoped owners",
                repeats=repeats, top_k=top_k, measurements_are="fresh-process ru_maxrss and exact hydrated counts",
                percentile_method="empirical nearest-rank ceil(0.95 * sample_count)",
                limitations=["Controlled vectors test ranking parity, not learned relevance.",
                             "RSS is process high-water memory; latency depends on host and cache state.",
                             "Legacy mode deliberately reproduces former archive materialization."],
                passed=all(run["parity"] and run["strict_scope"] and run["bounded_reads"] for run in runs), runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 5000])
    parser.add_argument("--body-bytes", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dsn-env", default="VC_RESOURCE_POSTGRES_DSN")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        spec = json.load(sys.stdin)
        result = {"seed": seed, "measure": measure, "cleanup": cleanup}[spec["phase"]](spec)
    else:
        result = run_experiment(backend=args.backend, sizes=args.sizes, body_bytes=args.body_bytes,
                                repeats=args.repeats, dsn_env=args.dsn_env, timeout=args.timeout)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
        print(json.dumps({"output": str(args.output), "passed": result["passed"]}))
    else:
        print(encoded)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
