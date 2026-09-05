"""Small subprocess validation of the resource experiment itself."""

from benchmarks.context_contracts.resources import run_experiment


def test_resource_harness_proves_parity_scope_and_lower_hydration():
    report = run_experiment(sizes=(18,), body_bytes=1024, repeats=1, top_k=3, timeout=30)
    assert report["passed"]
    run, = report["runs"]
    legacy, streamed = run["measurements"]
    assert legacy["peak_rss_bytes"] > 0 and streamed["peak_rss_bytes"] > 0
    assert legacy["signatures"] == streamed["signatures"]
    assert legacy["reads"]["pending_compaction"]["canonical_rows"] == 18
    assert streamed["reads"]["pending_compaction"]["canonical_rows"] == 12
    assert legacy["reads"]["logical_read"]["canonical_rows"] == 18
    assert streamed["reads"]["logical_read"]["canonical_rows"] == 1
    assert streamed["reads"]["speaker_search"]["maximum_embedding_page"] <= 200
    assert streamed["reads"]["segment_search"]["segment_source_reads"] == 0
    assert all(stage["median_ms"] >= 0 for stage in streamed["latency"].values())
    assert all(stage["sample_count"] == 1 and stage["p95_ms"] == stage["maximum_ms"]
               for stage in streamed["latency"].values())
    assert all(len(samples) == 1 for samples in streamed["peak_rss_after_calls_bytes"].values())
