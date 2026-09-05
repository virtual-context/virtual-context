# PostgreSQL semantic ranking

Status (2026-09-05): implemented in the working tree and disabled by default.
Generated parity and resource checks ran against an isolated remote PostgreSQL
database. These checks do not describe production package/schema state or
authorize enabling the flag on production data. The rollout checks below still
apply to each deployment target.

## Purpose

The default semantic path transfers scoped JSON embeddings to Python in keyset
pages and calculates cosine similarity for every candidate. It retains one page
and a bounded set of best admitted results, so application memory no longer grows
with an archive-sized embedding list. Total vector transfer and comparison work
still grow with the scoped archive. Native ranking keeps vector comparison in
PostgreSQL and transfers bounded pages of scored text and metadata.

This implementation performs exact database scans. It does not use approximate
nearest-neighbor indexes. PostgreSQL still does work proportional to the candidate
archive on each page; the improvement to verify is bounded application-side
vector allocation and transfer, followed by measured latency and RSS behavior.

## Storage and activation contract

`embedding_json` remains the source for a rebuildable cache on `segment_chunks`
and `canonical_turn_chunks`. The additive columns are `embedding vector(384)`,
`embedding_zero`, `embedding_source_hash`, and `embedding_model`.

Only `all-MiniLM-L6-v2` is supported. Chunk writers pass the model explicitly.
A transaction-local setting conveys that attestation to a database trigger,
which validates and derives the cache atomically from the JSON. A legacy writer
without an attestation invalidates readiness. Invalid dimensions, nonnumeric or
nonfinite values, missing model information, and stale source hashes prevent
activation. Valid zero vectors have a NULL vector with `embedding_zero=true`;
they cannot pass the existing positive similarity threshold.

The runtime probe checks catalog capability and absence of invalid residue.
Partial indexes make the healthy residue check independent of archive size.
Every ranked page checks readiness and selects candidates within the same
repeatable-read snapshot. Enabled reads fail visibly on incomplete migration or
query errors; they do not silently return to Python vector enumeration.

DDL is excluded from ordinary vector-search bootstrap. The explicit operation is:

```bash
virtual-context --config virtual-context.yaml admin migrate-semantic-vectors
virtual-context --config virtual-context.yaml admin migrate-semantic-vectors --apply
```

The default reports readiness without applying vector DDL. Apply requires the
server's pgvector extension package to be available and operator confirmation
that legacy JSON embeddings came from the supported model. Existing rows have
no historical model attestation; dimensionality alone cannot prove that claim.
Apply uses an advisory lock, short DDL lock timeout, bounded transactional
backfill, concurrent residue indexes, and a final readiness report. Invalid rows
remain reported rather than being invented or discarded.

## Deployment order and readiness scope

Readiness covers both chunk tables across the entire database. A single row
written without model attestation, even for another conversation, prevents every
enabled native semantic read from proceeding. This is deliberate: the reader
must not rank vectors whose model or cache correctness is unknown.

Upgrade every chunk writer while `retrieval.vector_search_enabled` is false,
including background workers, importers, administrative rebuilds, and old
processes that may still hold database connections. Drain the old writers before
the final migration pass. Then run `migrate-semantic-vectors --apply`, repeat the
dry run, and require `ready=true` and zero residue in both tables before enabling
the flag. A mixed-version rollout with the flag already enabled can cause
semantic searches on upgraded workers to fail until the remaining legacy writes
are repaired. Disabling the flag remains the immediate rollback.

Each table's report contains `residue_by_model`, a list of `{model, rows}` counts
for rows that prevent activation. An empty model identifies an unattested writer;
a different model requires re-embedding with the supported model, not relabeling
existing JSON. Invalid supported-model rows require source/cache correction.
Before the cache schema exists, `residue_by_model` is `null`: the database has no
model metadata from which to infer that distribution. Apply can attest unknown
legacy rows only when the operator has independently established their model.

The canonical group-read index is also an explicit PostgreSQL migration. Worker
startup no longer performs this potentially blocking build:

```bash
virtual-context --config virtual-context.yaml admin migrate-read-indexes
virtual-context --config virtual-context.yaml admin migrate-read-indexes --apply
```

Apply requires an idle autocommit connection, builds with `CREATE INDEX
CONCURRENTLY`, and restores its temporary lock timeout afterward. Reads remain
correct before the migration, but large group lookups may be slower. An interrupted
concurrent build may leave an invalid index; re-running the operation detects and
rebuilds that specific index. SQLite continues to create its local index during
normal schema setup.

## Retrieval contract

Three explicit store methods preserve their separate source identities:

| Method | Source identity and ordering after distance |
|---|---|
| `search_segment_chunks_by_embedding` | segment reference, chunk index |
| `search_canonical_turn_chunks_by_embedding` | conversation, physical sort key, side, chunk index, physical turn ID |
| `search_speaker_turn_chunks_by_embedding` | conversation, physical sort key, side, chunk index, physical turn ID |

Queries use `<=>` inside a materialized scored CTE and a keyset cursor, returning
up to 200 rows per core request. Neither embeddings nor their JSON representation
are included. Canonical legacy rows include physical source metadata required by
the existing rendering/admission path. Speaker rows hydrate bounded batches with
the original request-owned authority.

The caller continues through duplicates, unavailable sources, and rejected
channels until it has enough admissible results or exhausts candidates. A fixed
candidate cutoff cannot silently reduce source recall. Existing quote renderers,
side provenance, physical-row requirements, and speaker/channel admission still
apply after ranking. Tenant wrappers require an owned conversation before the
backend reads candidates and validate speaker authority against that scope.

Set `retrieval.vector_search_enabled: true` only after validation. Configuration
requires PostgreSQL and the supported model. With the flag false, SQLite and
PostgreSQL use the bounded Python path. Filesystem storage incrementally decodes
legacy embedding arrays and explicitly rejects an individual array item larger
than 4 MiB. Native capability is false for stores that do not implement it.

Segment pages obtain tag/date metadata from the live scoped source with bounded
tag batches, without loading a full segment for every candidate. Filesystem
metadata is invalidated when its source file changes and orphan files are
excluded. The relational cursor lower bound is applied to both sides of the
segment/chunk join. This prevents a retained PostgreSQL plan from rescanning
earlier sources on each continuation. It does not change session planner policy.

## Required rollout evidence

1. Run the DSN-gated PostgreSQL transaction and native-vector contract tests on
   the remote test fleet with pgvector available. Include rollback/locking,
   migration reruns, legacy writes, invalid/zero vectors, and cursor continuation.
   Skipped database tests do not pass this rollout gate; provision the required
   test role and extension before accepting it.
   Set `VC_REQUIRE_PGVECTOR_TESTS=1` when running
   `tests/test_pgvector_storage_postgres.py` with the remote test DSN so missing
   prerequisites fail visibly. Include
   `tests/test_postgres_fact_mutation_atomicity.py` in that remote check. Also run
   `tests/test_storage_domain_contracts.py`, `tests/test_storage_bounded_contracts.py`,
   and `tests/test_postgres_read_index_migration.py` with
   `VC_REQUIRE_STORAGE_DOMAIN_TESTS=1` to exercise the shared PostgreSQL contracts,
   including named-cursor legacy backfills, watermarks, and proposal round-trips.
2. Compare legacy and native results on representative canonical, speaker, and
   segment queries. Include hidden channels, missing rows, duplicate chunks,
   exact-value queries, ties, and similarity-threshold boundaries. pgvector uses
   float32: assert tolerance-based similarity parity and inspect ordering near
   ties rather than claiming bit-identical arithmetic.
3. Measure rows/bytes returned, peak allocation, RSS across repeated calls, and
   p50/p95 latency as archive size grows. Offline doubles prove routing and
   admission behavior, not SQL correctness or production resource improvement.
4. Deploy compatible readers and explicit-model writers with the flag off;
   inspect target capability, migrate/backfill, verify residue and real-data
   parity, then enable deliberately. Rollback disables the flag while retaining
   canonical JSON and additive cache columns.

Fact/tag-summary ranking is separate from these chunk-search methods. Optional
approximate indexing requires its own recall evaluation and is not part of this
change.

## Reproducing the generated resource check

Use an isolated database with the semantic vector migration already applied.
The harness creates unique conversation owners and deletes only those owners in
its cleanup. It does not create databases or migrate a supplied database.

```bash
# Set VC_RESOURCE_POSTGRES_DSN to the isolated test database through your
# normal environment configuration; do not use a production database.
python -m benchmarks.context_contracts.resources \
  --backend postgres --sizes 500 5000 --body-bytes 8192 \
  --repeats 10 --timeout 180 --output /tmp/vc-resource-postgres.json
```

Each mode starts in a fresh subprocess after a separate fixture-seeding process.
The archive baseline deliberately reproduces the former embedding-list and
canonical-body materialization. The default streamed mode and the opt-in native
mode run the current public retrieval methods. Each size has that many owned
canonical pairs and segments, approximately 8 KiB of canonical source text per
pair, one 384-value segment chunk and one canonical chunk per pair, plus 10%
foreign-owner records. Every seventh owned canonical source has a private
audience; foreign and private sources include strong matches. Ranking vectors
are deterministic generated values with distinct scores, without model downloads
or LLM calls. This tests ranking, source identity and scope parity, not learned
retrieval quality.

The JSON artifact records source/text top-five parity, exact source-body and
embedding hydration counts, per-operation median/p95/max latency and sample
counts, process high-water RSS after every call, and actual native
`EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` plans. The p95 is the empirical
nearest-rank sample at `ceil(0.95 * sample_count)`; with ten calls it is the
largest observed sample. It is not a stable estimate of production tail latency.

The 2026-09-05 remote host had 8 logical CPUs reporting AMD EPYC 7713, 15.6 GiB
RAM, Linux 6.8.0-136, PostgreSQL 16.15 and pgvector 0.6.0. OS and database caches
were not flushed or controlled. Modes run in archive/streamed/native order;
shared cache state and concurrent host activity can affect their timings.
Process RSS spans all four measured operations, including pending compaction
and logical source reads; PostgreSQL backend memory is not included.

The final ten-call run passed top-five source/text parity, strict conversation
and audience scope, and the hydration bounds at both sizes:

| Owned pairs | Archive baseline peak RSS | Streamed peak RSS | Native peak RSS |
|---|---:|---:|---:|
| 500 | 86.66 MiB | 80.60 MiB | 67.80 MiB |
| 5,000 | 352.48 MiB | 80.81 MiB | 67.80 MiB |

Observed latency below is **median / empirical p95**, in milliseconds, with ten
samples in every cell:

| Pairs | Operation | Archive baseline | Streamed | Native |
|---|---|---:|---:|---:|
| 500 | Segment search | 43.50 / 50.95 | 66.25 / 92.23 | 16.13 / 27.05 |
| 500 | Speaker search | 128.61 / 137.57 | 303.47 / 366.64 | 78.45 / 97.30 |
| 5,000 | Pending compaction | 295.73 / 389.65 | 8.16 / 19.92 | 7.48 / 17.45 |
| 5,000 | One logical source | 281.35 / 306.72 | 2.95 / 3.74 | 2.89 / 3.93 |
| 5,000 | Segment search | 482.94 / 520.94 | 567.40 / 651.50 | 33.10 / 52.69 |
| 5,000 | Speaker search | 746.65 / 10077.19 | 1352.58 / 1425.61 | 49.16 / 57.15 |

At 5,000 pairs, pending compaction hydrated 12 source bodies per call instead of
5,000, while a single logical read hydrated one. Streamed segment search made
zero full-segment lookups and returned embedding pages of at most 200 rows.
Native segment and speaker search each returned 200 scored rows per call and
zero embedding scalars. Speaker source hydration was 200 bodies per native call,
versus 4,285 across bounded streamed pages. The retained top results remained
identical after scope admission.

The streamed path trades additional page/proof overhead for bounded memory; its
semantic latency was still higher than the archive baseline in this fixture.
Native ranking gave the larger improvement. Across the ten 5,000-pair speaker
calls, process high-water RSS went from 80.66 to 80.81 MiB for streamed reads and
64.55 to 67.80 MiB for native reads. These short sequences do not establish a
long-running memory plateau or prove absence of a leak. The archive speaker
baseline also had a 10.08-second outlier, illustrating why these controlled
observations should not be presented as production latency guarantees.
