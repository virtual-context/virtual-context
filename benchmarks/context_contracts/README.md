# Offline context contracts

This corpus runs eleven adversarial scenarios through real SQLite source lookup,
semantic pagination, structured rendering, `ContextAssembler`, and `PagingManager`.
The embedding provider is an intentional equal-score control. No external model,
API, PostgreSQL service, or Docker container is used.

The scenarios cover corrections to old source IDs, cancelled plans, repeated and
different speakers, hidden channels, assistant-authored code, untrusted tool
artifacts, Unicode text and name collisions, out-of-order ingestion, and duplicate
chunks that require another retrieval page. Structured summaries are extractive
fixtures with valid provenance digests; the correction scenario deliberately
leaves those artifacts stale after changing the physical source.

Run from the repository root using the shared test slot:

```bash
set -e
mkdir /tmp/vc-suite-slot.lock
trap 'rmdir /tmp/vc-suite-slot.lock' EXIT
nice -n 19 .venv/bin/python -m benchmarks.context_contracts.runner \
  --output /tmp/context-contracts.json
```

For a single presentation policy, add `--ablation canonical_only`. The available
policies are `full_layers`, `canonical_only`, `tags_only`, `facts_only`,
`segment_claims`, and `tag_claims`. `full_layers` starts at SUMMARY and requests FULL
when a proved summary is unavailable or the fixture explicitly requests full
detail. Each other policy isolates its named input or depth. The same underlying
canonical evidence remains available for the separate explicit FULL paging probe.

The JSON report contains scenario results, aggregate metrics, source IDs and
current version fingerprints, real page sizes, and the corpus hash. Recall means
expected physical-source delivery, not learned semantic relevance. Attribution
also checks the actual rendered lane against the independent fixture's role,
label, and exact text. Correction checks fail when any admitted evidence omits a
required later correction; they allow an explicit abstention at compressed depths.
FULL and layered recovery must return all expected sources. Token counts use the
local `cl100k_base` tokenizer and include wrappers; API calls and API cost are zero.
The runner exits nonzero for a contract or retrieval-probe failure.

This is a deterministic contract evaluation, not an answer-quality score or a
production performance comparison. Tag metadata and generated facts intentionally
carry no canonical evidence into the model-facing prompt, so their isolated recall
can be zero while their safety checks pass. Negative controls in
`tests/test_context_evaluation.py` prove that matching IDs alone cannot conceal a
stale source version or a wrong human label.
