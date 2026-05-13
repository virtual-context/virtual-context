# VCATTACH harness pre-seed fixtures

Artifacts produced by `tests/fixtures/seed_realistic_target_conv.py` in
the **virtual-context** repo, copied here for consumption by the
virtual-context-cloud VCATTACH end-to-end harness. Each test run
restores these into the harness's local Postgres + Redis to give POST 1
(VCATTACH) a realistic target to attach to and POST 2 ("what topics?") a
populated state to retrieve from.

## Files

| File | Size | Purpose |
| --- | --- | --- |
| `target_baseline.sql` | 1.86 MB | `pg_dump --data-only --inserts` of the target conv across `canonical_turns` / `segments` / `tag_summaries` / `conversations` / `canonical_turn_anchors` / `segment_tags` / `ingest_batches`. **`engine_state` is intentionally excluded** (matches prod's empty state). |
| `target_baseline_redis.json` | 172 KB | `{redis_key: utf-8 json string}` mapping for Redis SessionState. Restore each entry with `redis.set(key, value)`. |
| `README.md` | — | This file. |

The generator lives at `tests/fixtures/seed_realistic_target_conv.py` in
the **virtual-context** repo (engine side). Regen instructions at the
bottom.

## Target conversation parameters

| Field | Value |
| --- | --- |
| `conversation_id` | `harness-target-77f110` |
| `tenant_id` | `""` (single-tenant seed; harness conftest can stamp tenant via its own loader) |
| Turn pairs ingested | 500 |
| Distinct primary tags | 40 (subset of `topic-001`..`topic-050` — some merged during compaction) |
| Compactions fired | 17 |
| Tag content marker | `BASELINE-MARKER-NNN` (string embedded in every segment summary + tag_summary) |

## Expected row counts post-restore

| Table | Count |
| --- | --- |
| `conversations` | 1 |
| `conversation_lifecycle` | 1 |
| `canonical_turns` | 966 |
| `canonical_turn_anchors` | 2889 |
| `segments` | 40 |
| `segment_tags` | 84 |
| `tag_summaries` | 48 |
| `ingest_batches` | 483 |
| `tag_aliases` | 0 |
| `conversation_aliases` | 0 |
| `engine_state` | **0** (intentional — see below) |

### Why `engine_state` is empty

**Dominant cause**: in cloud's production deployment the engine runs in
provider mode (`session_state_provider` non-None). The engine's
`_save_state` at `virtual_context/engine.py:1654-1663` short-circuits
before reaching `_store.save_engine_state` in provider mode, so the
table is never written during normal operation. SessionState (Redis)
is the authoritative store.

**Secondary contributing factor** (parallel issue, tracked as task #32):
`postgres.py:5028` INSERTs the post-`e677bba` column names
(`compacted_prefix_messages` / `flushed_prefix_messages`) but the prod
schema still has the pre-rename names (`compacted_through` /
`flushed_through`). On the rare path where cloud's
`SessionStateProvider._save_to_store` exercises Postgres backup save,
the INSERT raises `UndefinedColumn` and gets swallowed by the bare
`except Exception` at `proxy/session_state.py:786`. This independently
keeps the table empty in those paths.

For seeding: the generator runs the engine in **non-provider mode**,
which would normally trigger `_save_state` to actually write. To match
prod's empty-state invariant the generator passes
`--suppress-engine-state-write` (no-ops `save_engine_state` on the
store). The Postgres dump excludes `engine_state` from `--table` flags
regardless.

## Expected SessionState after restore

The Redis blob lives at:

```
Key:   vc:session:harness-target-77f110
Value: SessionState.to_json() bytes (UTF-8 JSON)
TTL:   none (persistent — set without an expiry)
```

The Redis key format comes from
`virtual_context/proxy/session_state.py:152`:
```python
return f"vc:session:{conversation_id}"
```

`target_baseline_redis.json` contains a single entry today:
```json
{
  "vc:session:harness-target-77f110": "<utf-8 json string>"
}
```

The harness restores it as raw bytes:
```python
import json, redis
client = redis.Redis.from_url(test_redis_url)
payload = json.load(open("tests/harness/fixtures/target_baseline_redis.json"))
for key, value in payload.items():
    client.set(key, value)
```

Decoded SessionState fields:

| Field | Value |
| --- | --- |
| `compacted_prefix_messages` | 958 |
| `flushed_prefix_messages` | 958 |
| `last_completed_turn` | 499 |
| `last_indexed_turn` | 499 |
| `turn_tag_entries` | 500 entries |

The `flushed_prefix_messages > 0` invariant matters: it's what the
retriever's `post_compaction` gate at
`virtual_context/core/retrieval_assembler.py:148-185` reads to decide
whether to fire the `summary_floor` prefetch. After the bug-#29 fix
lands and the harness loads target's SessionState (not source's), this
is what makes POST 2 surface the BASELINE-MARKER strings in the
upstream LLM payload.

## Harness restore recipe

```python
import json
import subprocess
import redis

# 1. Schema is bootstrapped by the harness conftest (PostgresStore(dsn))
#    BEFORE this restore runs. Conftest may downgrade engine_state column
#    names to pre-`e677bba`; that's fine -- this dump doesn't touch
#    engine_state.

# 2. Restore conversation rows from the SQL dump
subprocess.check_call(
    ["psql", test_pg_url, "-f", "tests/harness/fixtures/target_baseline.sql"],
)

# 3. Restore Redis SessionState
client = redis.Redis.from_url(test_redis_url)
payload = json.load(open("tests/harness/fixtures/target_baseline_redis.json"))
for key, value in payload.items():
    client.set(key, value)
```

## Regeneration

To regenerate both artifacts from the engine-side generator:

```bash
# In virtual-context repo:
cd ~/projects/virtual-context

# Ensure harness Postgres is up (cloud's docker-compose.test.yml provides
# harness-test-postgres-1 on 127.0.0.1:5433):
docker ps | grep harness-test-postgres-1   # should be Up

# Run the generator (uses cloud's harness Postgres + docker exec for pg_dump):
.venv/bin/python tests/fixtures/seed_realistic_target_conv.py \
    --pg-url "postgresql://vc_test:vc_test@127.0.0.1:5433/vc_harness" \
    --docker-container harness-test-postgres-1 \
    --conversation-id harness-target-77f110 \
    --turns 500 --segments 50 --summaries 50 \
    --out-pg ./tests/fixtures/target_baseline.sql \
    --out-redis ./tests/fixtures/target_baseline_redis.json \
    --suppress-engine-state-write

# Copy the produced artifacts into the cloud repo's fixtures dir:
cp tests/fixtures/target_baseline.sql \
   ~/projects/virtual-context-cloud/tests/harness/fixtures/
cp tests/fixtures/target_baseline_redis.json \
   ~/projects/virtual-context-cloud/tests/harness/fixtures/
```

Generator flags:

- `--docker-container` — invoke `pg_dump`/`psql` inside the named Docker
  container. Omit when `libpq` is on the host PATH.
- `--suppress-engine-state-write` — no-op `save_engine_state` to match
  prod's empty-state invariant. Required when the generator runs in
  non-provider mode against the harness Postgres.

## Phase-1 / Phase-2 assertion mechanics

The harness verifies the VCATTACH flow by grepping the upstream LLM
payload captured during POST 2. The dump populates 48 `tag_summary`
rows and 40 `segment` rows whose `summary` text embeds
`BASELINE-MARKER-001` .. `BASELINE-MARKER-050`. The SQL dump contains
**1050 hits** of `BASELINE-MARKER-` substring total.

**Phase 1 (failing harness, no engine fixes yet)**:

- POST 1 (VCATTACH): may fail with `state.engine._store is None`
  AttributeError at `handlers.py:1865` (bug #28). If bug #28 doesn't
  reproduce, POST 1 succeeds.
- POST 2 ("what topics?"): cloud's `SessionStateProvider.load(conv_id)`
  uses the request's conv_id (source). Fresh source's state is empty.
  Engine hydrates with empty state, `flushed_prefix_messages = 0`.
  Retriever's `post_compaction = (flushed_prefix_messages > 0) = False`
  gate at `retrieval_assembler.py:185` is False →
  `summary_floor` doesn't fire → no tag_summaries injected → upstream
  payload contains no `BASELINE-MARKER-`.
- Phase 1 assertion: `assert b"BASELINE-MARKER-" not in upstream_payload`.

**Phase 2 (after bug-#28 + bug-#29 fixes)**:

- POST 1 succeeds; alias `source → target` written.
- POST 2: the bug-#29 fix routes hydration through the resolver-rebound
  target id (or the engine self-hydrates internally, depending on the
  chosen fix surface). Engine has `flushed_prefix_messages = 958`.
  `post_compaction = True` → `summary_floor` injects all 48
  `tag_summary` rows → upstream payload contains many
  `BASELINE-MARKER-NNN` strings.
- Phase 2 assertion: `assert b"BASELINE-MARKER-" in upstream_payload`.

## Notes / quirks

- **40 segments / 48 tag_summaries vs target 50/50**: engine's natural
  segmenter merges adjacent topics when tag sets overlap, and tag
  summaries can accumulate multiple `covers_through_turn` rows per tag
  across successive compaction passes. Both are correct engine
  behaviors; the harness assertions are robust to both since they grep
  for `BASELINE-MARKER-` substring presence rather than exact counts.
- **`tag_aliases` empty**: no manual aliasing during the seed. If the
  harness needs a pre-loaded `source → target` row in
  `conversation_aliases`, that's a separate INSERT the conftest can
  emit, or the seed script can be parameterized.
- **`pg_dump` output uses `\restrict` markers** (Postgres 17+ pg_dump
  format). `psql -f` on PG16+ handles these natively.
