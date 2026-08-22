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
| `VCFORGET <tag>` | Delete the segments and summaries stored for a tag |
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

Delete the stored segments and summaries for a specific tag. Useful for removing a topic that is stale or sensitive.

### Usage

```
VCFORGET <tag>
```

The argument must be an existing tag name (case-insensitive). If the tag is not found, the response lists the available tags. This is a permanent deletion of that tag's segments and summaries.

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
| `backfill-tag-summaries` | Materialize missing tag summaries |
| `backfill-fact-embeddings` | Write embeddings for facts that predate embedding storage |
| `backfill-senders` | Recover sender labels on canonical turns |
| `backfill-channels` | Recover channel provenance on canonical turns |
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

- Commands that **write by default** and take `--dry-run` to report instead: `backfill-senders`, `backfill-channels`, `backfill-actors`, `backfill-reply-roles`, `backfill-fact-authors`, `rebuild-actor-cards`, `retag-canonical-turns`, `backfill-session-state-markers`.
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
runs the normal strict segment summarizer and atomically writes the newly
generated retrieval synopsis, its token/model metadata, and
`metadata_json.structured_summary`. The Postgres summary-FTS trigger refreshes
`summary_tsv` in that same write; full-text chunk embeddings are unchanged
because their canonical source text is unchanged. Each write is preceded by an
`fsync`'d JSONL journal entry containing old/new synopsis checksums and the
structured-envelope checksum, and is guarded by segment `xmin`, tenant,
lifecycle epoch, lifecycle generation, active-operation, canonical-source
digest, and retained source-alias checks.

Schema v1 admits requester evidence only. Every persisted excerpt must equal
the complete trimmed canonical `user_content` lane and carry its exact actor,
speaker, audience, channel, date, and canonical-turn provenance. Assistant
output, reply-target copies, partial substrings, and legacy summary prose are
never admitted as evidence.

The tag phase runs only after a full, unbounded segment inventory proves that
every eligible source segment has a current non-empty claim envelope. It copies
and deduplicates those exact claims newest-first (up to 256), then calls the
strict normal tag rollup only to regenerate the free retrieval synopsis. It
never supplies the old tag synopsis to the model. The tag row and the embedding
of its new synopsis are written in one transaction after the lifecycle, source
set, segment `xmin` values, canonical source digests, and existing tag/embedding
row versions are revalidated under locks. If provider generation, embedding,
or any compare-and-set fails, neither row is changed. The tag `source_digest`
is a deterministic claim-set integrity/idempotency checksum; serving performs
the independent canonical-row rehydration that authorizes each claim.

`--limit` caps attempted candidates (and therefore model cost), not successful
writes in each selected phase. A provider failure or concurrent change freezes
that phase's `resume_after_ref` or `resume_after_tag` so the undecided item is
retried. Resume with the reported value, then finish each phase with one run
without its cursor; already-current rows are skipped without a model call.

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
