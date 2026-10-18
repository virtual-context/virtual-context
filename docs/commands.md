# Commands

Virtual-context provides user-facing commands that work across both the proxy (Anthropic streaming and non-streaming) and REST paths. Commands are detected case-insensitively in user messages.

## Command Reference

| Command | Purpose |
|---------|---------|
| `VCATTACH <label\|id>` | Reattach the current session to another conversation by label or ID |
| `VCLABEL <name>` | Label the current conversation (no argument shows the current label) |
| `VCSTATUS` | Show conversation ID, label, turns, segments, working set, active tags |
| `VCRECALL <query>` | Search stored context and promote matching topics into the working set |
| `VCCOMPACT` | Force immediate compaction |
| `VCLIST` | List conversations with labels and turn counts |
| `VCFORGET <topic>` | Forget a topic and rebuild what was derived from it |
| `VCMERGE INTO <target>` | Merge this conversation into another (reserved; see below) |
| `VCMERGESTATUS` | Report merge progress (reserved; see below) |

## VCATTACH

Reattach the current session to another stored conversation, identified by its label or conversation ID. This is the mechanism for cross-platform memory: a conversation on Claude Code can pick up the memory built in a Telegram bot session, and vice versa.

### Usage

```
VCATTACH <label-or-id>
```

### Use Cases

- **Cross-platform continuity**: Build context on Telegram via OpenClaw, then `VCATTACH` from Claude Code. Both clients now enrich the same conversation.
- **Multi-agent memory**: Multiple agents attach to the same conversation and share its fact base and topic index.
- **Session recovery**: Reconnect to a previous conversation's context after identity detaches (system prompt change, client truncation, redeploy).

### How It Works

`VCATTACH` writes a durable alias that redirects the current conversation identity to the target conversation. Subsequent requests that arrive under the old identity resolve through the alias to the target, so retrieval, compaction, and new turns all operate on the target conversation. The old conversation is **not** deleted; its identity simply routes to the target from then on. Stale markers embedded in older assistant responses keep working because they follow the alias.

Attaching is a redirect, not a merge: it does not copy segments or facts between conversations. To combine two conversations' stored data into one, see `VCMERGE`.

## VCLABEL

Label the current conversation with a human-readable name. Labels are used for identification in `VCLIST` output and the dashboard sessions panel.

### Usage

```
VCLABEL MyProjectName
```

The label is stored as session metadata and persists across restarts (when using durable storage).

## VCSTATUS

Display the current state of the context window, including:

- Conversation ID and session label
- Turn count and compaction watermark
- Context window fill level (tokens used / total)
- Number of stored segments and their total token count
- Active tags (the working set from recent turns)
- All known tags in the store

### Usage

```
VCSTATUS
```

The output is returned as structured text that the model can read and relay to the user.

## VCRECALL

Search stored context for a query and promote the matching topics into the working set, so the next turn's assembly includes them. Useful when the user wants to revisit a specific earlier discussion.

### Usage

```
VCRECALL the database migration discussion
```

The engine runs lexical and semantic search over stored conversation text, reports the matches, and promotes up to five matched tags into the working set for the next request.

## VCCOMPACT

Force immediate compaction regardless of threshold levels. Useful when you know you're about to enter a long conversation and want to free up budget proactively.

### Usage

```
VCCOMPACT
```

This triggers the same compaction pipeline that runs automatically at thresholds, but executes immediately. Protected recent turns are still preserved.

## VCLIST

List stored conversations with their labels and turn counts.

### Usage

```
VCLIST
```

## VCFORGET

Forget a topic. The topic is removed from every turn that carries it, and a turn left with no topic at all is removed. The topic's summary is deleted, and the segments, facts and summaries derived from the changed turns are rebuilt in the background. `VCSTATUS` shows the rebuild while it runs.

### Usage

```
VCFORGET <topic>
```

The topic is matched case-insensitively, and an alias resolves to its topic. If the topic is not found, the response lists the available topics.

Every change is kept in an audit trail: each untagged or removed turn records its text and tags before the change, who made it and when. A host replaying old history does not bring a removed turn back. To erase content entirely, delete the conversation.

## Reviewing and editing the record

Memory is reviewed where it comes from. A virtual-context admin inspects the facts
held for a conversation, each beside the turns it was extracted from, and corrects
the record when something is wrong: edit a turn, remove one, or forget a topic. The
facts and summaries derived from those turns are rebuilt to match. Review is
available, not required: nothing waits in a pending queue, because a fact is a cached
copy of the conversation and its trust state already says whether the conversation
still supports it (see [fact lifecycle](fact-lifecycle.md#trust-state)).

The dashboard's **Memory Review** panel does all of this in the browser: it lists a
conversation's facts with their trust state beside the turns they came from, with
buttons to reject or restore a fact, edit or remove a turn, forget a topic, and read
the edit history, verdicts and rebuilds. The same operations are on the command line:

```
virtual-context admin record facts   <conversation_id> [<topic>]
virtual-context admin record verify  <conversation_id>
virtual-context admin record reject  <conversation_id> <fact_id> --reason "<why>"
virtual-context admin record restore <conversation_id> <fact_id>
virtual-context admin record verdicts <conversation_id>
virtual-context admin record forget  <conversation_id> <topic>
virtual-context admin record edit    <conversation_id> <turn_id> --user "<text>" [--assistant "<text>"]
virtual-context admin record remove  <conversation_id> <turn_id>
virtual-context admin record history <conversation_id> [<turn_id>]
virtual-context admin record status  <conversation_id> [<operation_id>]
virtual-context admin record process <conversation_id> <operation_id>
```

`facts` lists current facts with their trust state and source turns; `verify`
re-derives every fact's trust state from its source turns. `reject` withholds a fact
from every read, and the same statement extracted again from the same turns is not
stored; `restore` lifts the rejection and brings the fact back; `verdicts` lists them with who decided,
when and why. `forget`, `edit` and
`remove` change the turns, write the audit trail, retract the facts of the affected
segments, then rebuild the segments, facts and topic summaries derived from them.
`history` is the audit trail: for each edited, untagged or removed turn, its text
and tags before the change, the text after it, who made it, when and why. `--actor`
and `--reason` are stored with each change. `edit` drops any source attestation the
turn carried, since the turn no longer matches its source. `remove` removes both
halves of the exchange. `process` finishes a rebuild that was interrupted. Deciding
who may run these is up to the deployment.

## VCMERGE and VCMERGESTATUS

`VCMERGE INTO <target-label-or-id>` combines two conversations' stored data into one; `VCMERGE PREVIEW <target>` previews the operation; `VCMERGESTATUS` reports progress. Unlike `VCATTACH` (a redirect), a merge moves the source conversation's stored data into the target.

In the standalone proxy these commands are recognized but fail closed with an explanatory message instead of executing: merge execution requires a deployment that intercepts the command ahead of the proxy, such as the hosted service. This fail-closed behavior is deliberate; a merge must never be silently dropped or forwarded to the upstream model as ordinary text.

## Detection

Commands are detected by pattern matching in the user message text. Detection is:

- **Case-insensitive**: `vcattach mylabel`, `VCATTACH MyLabel`, and `VcAttach mylabel` all work
- **Whole-message**: the command (plus its argument) must be the entire message, aside from transport metadata that the proxy strips automatically
- **Works on both paths**: Proxy (streaming and non-streaming) and direct REST API calls

When a command is detected, the engine intercepts the request before forwarding to the upstream provider and handles it internally.

## CLI Reference

The `virtual-context` command line is a separate surface from the in-conversation commands above.

| Subcommand | Purpose |
|------------|---------|
| `proxy` | Start the HTTP proxy for LLM enrichment |
| `daemon install\|status\|start\|stop\|restart\|uninstall` | Manage the proxy as a background service |
| `onboard` | Guided setup: create/validate config, optionally install the daemon |
| `init <preset>` | Generate a config from a preset |
| `presets list\|show <name>` | List or inspect config presets |
| `config validate` | Validate the config file |
| `status` | Show tag stats and token usage |
| `tags` | List all tags in the store |
| `recall <tag>` | Recall stored context by tag |
| `retrieve -m <msg>` | Retrieve context for a message (JSON output) |
| `transform -m <msg>` | Retrieve and assemble a context block |
| `compact` | Trigger manual compaction |
| `aliases list\|suggest\|add` | Manage tag aliases |
| `chat` | Interactive TUI chat (`--headless --replay` for scripted runs) |
| `import` | Import conversation history from exports (see below) |
| `telemetry` / `cost-report` | Show conversation telemetry and cost |
| `admin <subcommand>` | Operational primitives (backfills, repairs) |

### Importing Conversation History

`virtual-context import` ingests conversation exports from other assistants into the store, so a user migrating to virtual-context arrives with their history already indexed, tagged, and retrievable:

```bash
virtual-context import --provider chatgpt --input conversation.json
virtual-context import --provider claude  --input ~/exports/        # directory: every *.json inside
virtual-context import --provider grok    --input export.json --compact
```

- `--provider` / `-p` (required): one of `chatgpt`, `claude`, `grok`. Each adapter parses that service's native conversation-export JSON.
- `--input` / `-i` (required): a single export file, or a directory whose `*.json` files are imported in sorted order. Files with no messages are skipped and counted.
- `--compact`: run compaction after the import (off by default). Without it, imported turns are ingested and tagged but summarization happens later through the normal thresholds.

Each imported conversation keeps its own conversation ID from the export, so separate source conversations stay separate in the store, and progress is printed per conversation as turns are ingested.

### Admin Subcommands

`virtual-context admin` hosts guarded operational commands for backfilling and repairing stored data. They operate directly on the store, support explicit database targeting, and are designed to be idempotent:

| Subcommand | Purpose |
|------------|---------|
| `record` | Review facts beside their source turns; forget a topic, edit or remove a turn, with an audit trail (see [Reviewing and editing the record](#reviewing-and-editing-the-record)) |
| `backfill-tag-summaries` | Materialize missing tag summaries |
| `backfill-fact-embeddings` | Write embeddings for facts that predate embedding storage |
| `backfill-senders` | Recover sender labels on canonical turns |
| `backfill-channels` | Recover channel provenance on canonical turns |
| `compact-raw-bodies` | Store each inline image once and drop replayed host context from stored raw bodies (`--batch-size`, default 5 rows per transaction) |
| `backfill-actors` | Recover durable actor IDs from retained raw user text |
| `backfill-reply-roles` | Recover reply roles and audience provenance |
| `backfill-fact-authors` | Re-distill facts with canonical actor provenance |
| `backfill-session-state-markers` | Rebuild session-restore markers from canonical rows |
| `rebuild-actor-cards` | Rebuild per-actor card caches |
| `rebuild-derived-data` | Rebuild derived data for a conversation |
| `reattribute-audience` | Correct audience attribution on stored rows |
| `resummarize-segments` | Re-run summarization for stored segments |
| `migrate-structured-summaries` | Rebuild source-bound segment claims and deterministic tag rollups |
| `resequence-canonical-turns` | Repair canonical turn ordering |
| `normalize-canonical-actor-ids` | Normalize actor ID formats on canonical rows |
| `reindex-canonical-turn-embeddings` | Rebuild canonical turn embedding indexes |
| `retag-canonical-turns` | Re-tag stored turns with pair context via the configured tagger |

Run `virtual-context admin <subcommand> --help` for the flags each command takes.

#### Common admin flags

**Two safety models exist; know which one your command uses.**

- Commands that **write by default** and take `--dry-run` to report instead: `backfill-senders`, `backfill-channels`, `compact-raw-bodies`, `backfill-actors`, `backfill-reply-roles`, `backfill-fact-authors`, `rebuild-actor-cards`, `retag-canonical-turns`, `backfill-session-state-markers`.
- Commands that **dry-run by default** and take `--apply` to write: `reattribute-audience`, `rebuild-derived-data`, `resummarize-segments`, `migrate-structured-summaries`, `resequence-canonical-turns`, `normalize-canonical-actor-ids`, `reindex-canonical-turn-embeddings`.
- `backfill-tag-summaries` and `backfill-fact-embeddings` take `--force-rebuild` to regenerate rows that already exist.

Storage and scope targeting, shared across the admin surface:

| Flag | Meaning |
|------|---------|
| `--postgres-dsn <dsn>` | Target Postgres; overrides the config's `storage.postgres.dsn` |
| `--sqlite-path <path>` | Target SQLite; overrides the config's `storage.sqlite.path` |
| `--tenant-id <id>` | Tenant to operate on (default: empty, single-tenant) |
| `--all-convs-for-tenant` | Enumerate every conversation owning canonical rows for the tenant instead of naming one conversation ID |
| `--limit N` | Cap rows upgraded per conversation, or conversations enumerated (default: no cap) |
| `--platform <name>` | Actor commands only: operator-asserted platform (e.g. `telegram`) for conversations whose caller keys never named one; never inferred, applied only to identity blocks carrying a sender ID without platform proof |

Storage resolution follows the precedence chain: explicit flag > `-c` config > the `DATABASE_URL` environment variable (see [configuration](configuration.md#environment-variables)); the env fallback is consulted only when neither a storage flag nor `-c` was given, so a bare invocation works inside a container that has `DATABASE_URL` set and no config file mounted.

#### Structured summary migration

`migrate-structured-summaries` upgrades both summary layers to the current
source-bound structured-claim schema. It is intentionally Postgres-only,
defaults to `--phase all`, and dry-runs by default:

```bash
virtual-context -c /app/default-tenant-config.yaml admin \
  migrate-structured-summaries "$CONVERSATION_ID" \
  --tenant-id "$TENANT_ID" --postgres-dsn "$DATABASE_URL"

virtual-context -c /app/default-tenant-config.yaml admin \
  migrate-structured-summaries "$CONVERSATION_ID" \
  --tenant-id "$TENANT_ID" --postgres-dsn "$DATABASE_URL" --apply \
  --phase segments --limit 25 \
  --journal /data/tenants/diagnostics/structured-summary-migration.jsonl

# Finish the segment phase first. Then migrate tags in bounded resumable runs;
# a bounded --phase all run intentionally refuses to start tags while segment
# candidates remain.
virtual-context -c /app/default-tenant-config.yaml admin \
  migrate-structured-summaries "$CONVERSATION_ID" \
  --tenant-id "$TENANT_ID" --postgres-dsn "$DATABASE_URL" --apply \
  --phase tags --after-tag "$LAST_COMPLETED_TAG"
```

The dry run uses a server-enforced read-only connection and constructs neither
an engine nor a store. Apply mode reconstructs each segment only from the exact
canonical turn IDs in a proved-complete source mapping; it never reads the
stored `summary`, `full_text`, or `messages_json` as model input (only an
in-database checksum of the old synopsis is selected for the journal). It then
builds the structured envelope deterministically from exact requester lanes;
no summarization provider is called. The compare-and-set changes only
`metadata_json.structured_summary`. Existing retrieval synopsis bytes,
`summary_tokens`, compression/model metadata, `summary_tsv`, and full-text
chunk embeddings remain unchanged. Each write is preceded by an `fsync`'d
JSONL journal entry containing the preserved synopsis checksum and the new
structured-envelope checksum, and is guarded by segment `xmin`, tenant,
lifecycle epoch, lifecycle generation, active-operation, canonical-source
digest, and retained source-alias checks.

Schema v1 admits requester evidence only. Every persisted excerpt must equal
the complete trimmed canonical `user_content` lane and carry its exact actor,
speaker, audience, channel, date, and canonical-turn provenance. Assistant
output, reply-target copies, partial substrings, and legacy summary prose are
never admitted as evidence. Every admissible requester lane within the bounded
excerpt size becomes a claim. Safety-critical corrections are ordered newest
first, followed by the remaining lanes in deterministic physical order. An
overlong safety-critical lane uses the exact canonical fallback when it fits
the bounded projection; claim-cap overflow quarantines the compressed artifact
rather than truncating or presenting a partial source set.

The tag phase runs only after a full, unbounded segment inventory proves that
every eligible source segment has a current non-empty claim envelope. It copies
and deduplicates those exact claims, applies the deterministic safety floor,
and stores a bounded ordered selection. No tag summarizer or embedding provider
is called. For an existing tag row, the transaction changes only its source
coordinates and `structured_summary_json`; its synopsis, description, token
metadata, code references, generation metadata, timestamps, and embedding bytes
are preserved exactly. Lifecycle, source set, segment `xmin`, canonical source
digests, and existing tag/embedding row versions are revalidated under locks.
A missing tag row is reported as blocked and uses the exact SEGMENTS fallback;
the emergency migration never invents a retrieval synopsis or embedding. The
tag `source_digest` authenticates the ordered selected claims and complete
source coordinates; serving independently rehydrates the canonical rows that
authorize every claim.

`--limit` caps attempted candidates, not successful writes. A concurrent
lifecycle/source/row-version change freezes that phase's `resume_after_ref` or
`resume_after_tag` so the undecided item is retried. Resume with the reported
value, then finish each phase with one run without its cursor; already-current
rows are skipped without any provider call. The JSONL journal records
`migration_mode=deterministic_metadata_only_v1` so it cannot be confused with
an older provider-backed run.

After any accepted tag write, the JSON result reports a required serving-cache
action. Delete `vc:tag_summary_embeddings:<conversation>`,
`vc:tag_stats:<conversation>`, and matching
`vc:context_hint:<conversation>:*` Redis entries, then recycle every serving
worker to clear process-local snapshots. The historical upgrade is not
serving-complete until those actions and the final verification run succeed.

#### Other notable flags

- `retrieve` / `transform`: `-m` / `--message` is the inbound message to retrieve for; `--active-tags` supplies a comma-separated working set to simulate; `transform` also takes `--budget` to override the token budget.
- `recall`: `--limit` caps returned segments (default 5).
- `init`: `--force` overwrites an existing config file.

### Check and repair SQLite search indexes

`virtual-context -c config.yaml admin repair-search-indexes` compares FTS indexes
against their stored content without rebuilding them. Add `--apply` to rebuild
only corrupt indexes. Use `--sqlite-path PATH` to select an existing database
and repeat `--index NAME` to restrict the check. Output reports `ok`,
`needs_rebuild`, or `rebuilt` for each checked index. This repairs search indexes;
it cannot recover fact relationships removed by older replacement writes.

### Prepare PostgreSQL canonical group reads

`virtual-context -c config.yaml admin migrate-read-indexes` reports whether the
canonical group index is ready. Add `--apply` to build it concurrently through
an idle connection. PostgreSQL worker startup does not build this large index;
schedule the command as an explicit upgrade step. A conflicting migration or
lock wait fails visibly and can be retried. SQLite creates its local index at
startup.

### Prepare native PostgreSQL semantic ranking

`virtual-context -c config.yaml admin migrate-semantic-vectors` checks the
optional pgvector cache schema and reports readiness. Add `--apply` to install
and backfill it, with `--batch-size N` controlling each batch. PostgreSQL must
have the pgvector extension package available. The configured embedding model
must be `all-MiniLM-L6-v2`; verify that existing chunk embeddings were generated
with that model before backfilling. Raw embedding JSON remains authoritative.

Upgrade all writers, then rerun the backfill and check the per-model residue
report before enabling native reads. A mixed-version writer can make the cache
incomplete for every conversation. After real-database parity and resource checks, set
`retrieval.vector_search_enabled: true` to use exact DB-side ranking. The flag
is off by default; migration does not activate it. Enabled reads require a ready
cache and raise if it becomes incomplete rather than falling back to an
archive-sized Python scan. Set the flag to false to restore the existing path.

### Reassign a historical memory audience

The client chooses memory audiences. These administrative commands apply an
explicit historical correction after source and target audiences have been
registered as direct aliases of the same active storage owner. Identifiers
are opaque; the commands do not infer groups from channels or platform names.
Finish any storage merges and drain ingestion and compaction before planning.

```bash
virtual-context admin plan-audience-reassignment \
  "$OWNER_ID" "$SOURCE_AUDIENCE" "$TARGET_AUDIENCE" \
  --tenant-id "$TENANT_ID" --expected-lifecycle-epoch "$OWNER_EPOCH" \
  --operation-id "$OPERATION_ID" --manifest /private/operations/audience.json \
  --postgres-dsn-env DATABASE_URL

virtual-context admin reassign-audience \
  --manifest /private/operations/audience.json --postgres-dsn-env DATABASE_URL

virtual-context admin reassign-audience \
  --manifest /private/operations/audience.json --postgres-dsn-env DATABASE_URL --apply
```

Use `--sqlite-db PATH` instead to select an existing SQLite database. Database
selection is required and does not fall back to application configuration.
The schema must already be upgraded; these commands do not run startup data
migrations, construct an engine, or call a model provider.

Planning creates a new mode-0600 JSON manifest and refuses to overwrite an
existing path. Store manifests outside source control. Verification is the
default; only `reassign-audience --apply` writes. The manifest pins every selected
canonical row and each attested user/assistant pair. A selected row without
exact source membership refuses the entire plan; complete its ordinary source
admission before planning. Apply rechecks tenant,
lifecycle, owner routing, active work, source hashes and complete membership
before recording immutable authorizations and changing effective audience.
Original source-ledger evidence, message bodies and physical channel provenance
are preserved. A repeated identical apply reports zero updates; conflicting
manifests or a second reassignment of the same canonical row are refused.

Output reports selected and updated rows, invalidated cards and any remaining
rows in the source audience. Apply hides affected actor cards while retaining
their immutable entries and citations for re-admission. It does not rebuild
derived data. Reproject exact card-source pointers from current canonical/fact
proof before normal card rebuilding; stale pointers must never be marked clean.
Reproject affected structured summaries, allocate any missing speaker handles,
and clear serving caches as separate cutover steps. Preserve a private card
checkpoint before any ordinary storage merge, which can remove cached entries.
