# Speaker Attribution and Actor Cards

In a group conversation (a Telegram group, a Discord server), many humans talk to one assistant. Raw chat history flattens them into an undifferentiated stream of "user" messages, so the model cannot answer "what has *Alice* said about this?", cannot tell whose preference a stored fact records, and will happily attribute one person's statement to another. Virtual-context solves this with a layered attribution subsystem: durable per-message provenance, stable actor identity, authorship-safe fact extraction, per-person memory cards, and speaker-aware search.

Everything in this subsystem is gated and **ships dark by default**: with the gates off, output and tool schemas are byte-identical to a build without the feature.

## Provenance on Canonical Turns

Attribution starts at ingestion. The envelope parser claims identity metadata from the transport wrapper *before* the wrapper is stripped from the model-visible text, and the claims land as columns on the canonical turn row:

| Column | Meaning |
|--------|---------|
| `sender` | Display label of the human who sent the message |
| `sender_actor_id` | Stable actor ID resolved for the sender |
| `origin_channel_id`, `origin_channel_label` | Which channel the message arrived on |
| `reply_target_message_id` | The message this one replies to, when the transport carries it |
| `reply_subject_actor_id` | The actor whose message is being replied to |
| `audience_conversation_id` | The audience (conversation) this row is proved to belong to |

Because provenance is stored on the durable row rather than inferred later from text, every downstream layer (facts, cards, search) can attribute content without trusting model output.

## Actor Identity

An **actor** is a durable identity for one human, stored in `actor_profiles` and keyed by a stable key derived from the transport identity (nested sender objects and multiple stable-key kinds are accepted). Display labels can change; the actor ID does not. Sender labels can also be recovered from actor profiles when a transport stops sending them.

## Reply Lanes and Fact Authorship

Fact extraction is authorship-safe by construction:

- **Facts are attributed from canonical rows, never from model text.** The author of a fact is the actor recorded on the row the fact came from, not whatever name the summarization model happens to emit.
- **Reply lanes** separate the two people involved in a reply. When Alice replies to Bob, the *requester lane* (Alice's statement) is attributed to Alice's row actor, and the *subject lane* (the quoted material being replied to) is attributed only to the resolved reply subject (Bob). A requester lane never contains its own quote block, and an unresolved actor stays empty rather than being guessed.

This is what makes "who said X" answerable from storage instead of from the model's recollection.

## Person Cards (Actor Cards)

A **person card** is a small, curated, per-actor digest of durable identity facts (preferences, biography, standing context), rebuilt from that actor's canonical speech and injected into assembly for the requester. When Alice sends a message, the model receives Alice's card; a new member with no history gets nothing, and starts clean.

### Lifecycle

1. **Candidate extraction**: the fact pipeline proposes card-worthy entries from the actor's own recorded speech.
2. **Semantic admission**: a dedicated admission model decides whether a candidate is a *durable identity fact* rather than a passing remark or a temporary instruction. Admission requires citations to the segments the evidence came from, rejects ambiguous entries, and preserves qualifiers rather than flattening them. **This is deliberately fail-closed at the configuration level**: enabling the card gate without configuring `actor_card_admission_model` is a configuration error, because the cheap fact extractor must not be the judge of what becomes durable identity memory.
3. **Curation**: a curation pass reads the actor's admitted evidence (bounded by `actor_card_fact_limit` and, per policy audience, `actor_card_turn_limit`, so a busy guild cannot crowd out an actor's DM evidence) and emits the card, capped at `actor_card_entries_per_kind` entries per kind.
4. **Consolidation at the compaction boundary**: cards marked dirty by new evidence are consolidated after compaction, batched with a per-run limit; a rebuild-status table tracks due cards.
5. **Refresh safety**: during a live refresh the last good card keeps serving; malformed model responses are retried; a no-op evidence update does not dirty the card; provider-level refusals can fall back to `actor_card_admission_fallback_model`, but a valid semantic rejection is final.

### Injection Rules

Card injection fails closed. Nothing is injected when: the gate is off, the requester is unknown, the audience is unproved, the card is invalid, or no store is available. Cards are also fenced by audience, tenancy, and lifecycle, so a card can never leak across tenants, across unproved audiences, or across a conversation reset. DM evidence and public-channel evidence sit behind a privacy boundary that fails closed.

### Configuration

```yaml
assembly:
  actor_card_enabled: false          # master gate; ships dark
  actor_card_max_tokens: 400         # rendered card budget
  actor_card_fact_limit: 60          # facts the curation pass may read
  actor_card_turn_limit: 500         # exact canonical messages per policy audience
  actor_card_entries_per_kind: 3
  actor_card_curation_model: ""      # unset = general compaction model (configure a stronger one in production)
  actor_card_curation_fallback_model: ""
  actor_card_admission_model: ""     # REQUIRED when the gate is on; fail-closed otherwise
  actor_card_admission_fallback_model: ""  # provider-level failures only, never valid rejections
```

## Speaker Roster

The **speaker roster** is an audience-scoped list of the participants in the current conversation, with durable handles (stored in `speaker_handles`), rendered into assembly so the model knows who is present and how to refer to them. It is independent of the actor card and ships dark: with the gate off, no roster read happens and rendered output and tool schemas are byte-identical.

```yaml
assembly:
  speaker_roster_enabled: false
  speaker_roster_max_tokens: 300     # wrapper-inclusive cap
```

## Speaker-Conditioned Search

Two independent gates under `search:` make the search tools speaker-aware:

- **`speaker_annotations_enabled`** (default `false`): search results carry speaker labels, so the model sees *who said* each returned excerpt.
- **`speaker_selection_enabled`** (default `false`): activates the speaker-input unit as one atomic gate: a request-local `speaker` hint and a strict `speaker_only` filter on the search tools, execution-time validation of the selection against the roster snapshot, affinity ordering, and requester-intent conditioning (a first-person question like "what did *I* say?" resolves to the requester). While off, an arriving `speaker` or `speaker_only` argument is never consumed and results carry no conditioning metadata.

With both on, "what has this person said?" resolves against rows attributed to that actor rather than against the whole conversation, and `speaker_only: true` guarantees no other speaker's content appears in the result.

**Audience scope**: `search.speaker_audience_scope` controls the read boundary. `"channel"` (default) preserves a strict per-channel boundary; `"conversation"` keeps the proved owner/audience boundary but treats the origin channel as provenance rather than a filter, which fits servers where every public channel deliberately shares one conversation.

```yaml
search:
  speaker_annotations_enabled: false
  speaker_selection_enabled: false
  speaker_audience_scope: "channel"   # or "conversation"
```

## Operator Surface

Existing conversations predate these columns, so the admin CLI ships guarded, idempotent backfills (see [commands](commands.md) for the full table):

| Command | Restores |
|---------|----------|
| `admin backfill-senders` | Sender labels on canonical turns |
| `admin backfill-channels` | Channel provenance |
| `admin backfill-actors` | Durable actor IDs from retained raw user text |
| `admin backfill-reply-roles` | Reply roles and audience provenance |
| `admin backfill-fact-authors` | Re-distills facts with canonical actor provenance |
| `admin reattribute-audience` | Corrects audience attribution |
| `admin normalize-canonical-actor-ids` | Normalizes actor ID formats |
| `admin rebuild-actor-cards` | Rebuilds card caches |

### How to Tell It's Working

- With annotations on, `vc_find_quote` results carry speaker labels.
- With the card gate on and a known requester, the assembled context contains the requester's rendered card (within its token budget); an unknown requester gets none.
- With the roster on, the assembled context contains the roster block for the current audience.
- `admin rebuild-actor-cards` reports which cards it rebuilt; the rebuild-status table records due and completed cards.
- New canonical turns in group conversations carry non-empty `sender` and `sender_actor_id` columns; rows written before the backfills will show empty ones until backfilled.
