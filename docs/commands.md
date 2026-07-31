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
| `import` | Import conversation history from exports |
| `telemetry` / `cost-report` | Show conversation telemetry and cost |
| `admin <subcommand>` | Operational primitives (backfills, repairs) |

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
| `resequence-canonical-turns` | Repair canonical turn ordering |
| `normalize-canonical-actor-ids` | Normalize actor ID formats on canonical rows |
| `reindex-canonical-turn-embeddings` | Rebuild canonical turn embedding indexes |
| `retag-canonical-turns` | Re-tag stored turns with pair context via the configured tagger |

Run `virtual-context admin <subcommand> --help` for the flags each command takes.
