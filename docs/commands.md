# Commands

Virtual-context provides user-facing commands that work across both the proxy (Anthropic streaming and non-streaming) and REST paths. Commands are detected case-insensitively in user messages.

## Command Reference

| Command | Purpose |
|---------|---------|
| `VCATTACH` | Attach cross-platform memory to the current session |
| `VCLABEL` | Label the current conversation for future retrieval |
| `VCSTATUS` | Show context window state, active tags, and compression stats |
| `VCRECALL` | Recall a specific topic or time range |
| `VCCOMPACT` | Force immediate compaction |
| `VCLIST` | List available stored contexts and sessions |
| `VCFORGET` | Remove specific topics or facts from memory |

## VCATTACH

Attach stored memory from another session or platform to the current conversation. This is the mechanism for cross-platform memory: a conversation on Claude Code can access memories from a Telegram bot session, and vice versa.

### Usage

```
VCATTACH
```

When the model encounters this command, it loads all available stored contexts (segments, facts, tag summaries) from the shared store and makes them available for retrieval in the current session. Previously separate conversations become part of the same memory pool.

### Use Cases

- **Cross-platform continuity**: Start a conversation on Telegram via OpenClaw, continue it on Claude Code. Both sessions share the same underlying memory.
- **Multi-agent memory**: Multiple agents using the same virtual-context store share a unified fact base and topic index.
- **Session recovery**: Reconnect to a previous session's context after a fresh start.
- **Context bootstrapping**: Load a corpus of pre-indexed knowledge into a new conversation.

### How It Works

1. The command triggers `engine.attach_contexts()`
2. The engine queries the store for all sessions with stored segments
3. Segments are loaded into the retrieval index, facts into the fact store
4. The TurnTagIndex is updated to include tags from attached sessions
5. Subsequent retrieval queries can now surface content from any attached session

The attachment is additive. Attaching does not remove or modify existing session content.

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

Recall specific content by topic or time range. This triggers retrieval outside the normal request flow, useful when the user wants to revisit a specific earlier discussion.

### Usage

```
VCRECALL the database migration discussion
VCRECALL what we talked about last Tuesday
```

The engine parses the recall target, runs retrieval (or temporal resolution for time-based queries), and surfaces matching segments.

## VCCOMPACT

Force immediate compaction regardless of threshold levels. Useful when you know you're about to enter a long conversation and want to free up budget proactively.

### Usage

```
VCCOMPACT
```

This triggers the same compaction pipeline that runs automatically at thresholds, but executes immediately. Protected recent turns are still preserved.

## VCLIST

List all available stored contexts, sessions, and their metadata. Shows:

- Session IDs with labels
- Segment counts per session
- Tag clouds per session
- Total stored tokens
- Compression ratios

### Usage

```
VCLIST
```

## VCFORGET

Remove specific topics or facts from memory. Useful for correcting stale information or removing sensitive content.

### Usage

```
VCFORGET the old API key discussion
VCFORGET fact: user lives in NYC
```

The engine identifies matching segments or facts and removes them from the store. This is a permanent deletion.

## Detection

Commands are detected by pattern matching in the user message text. Detection is:

- **Case-insensitive**: `vcattach`, `VCATTACH`, and `VcAttach` all work
- **Position-independent**: The command can appear anywhere in the message
- **Works on both paths**: Proxy (streaming and non-streaming) and direct REST API calls

When a command is detected, the engine intercepts the request before forwarding to the upstream provider and handles it internally.
