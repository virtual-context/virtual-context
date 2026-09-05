# Proxy request and tool continuations

Every proxy request carries an immutable `RequestContext`: authenticated tenant,
resolved conversation and audience route, provider/model identity, request and
operation IDs, speaker authority, roster snapshot, lifecycle epoch, metrics
collector, provider limits, output reservation and admitted source history.
Response persistence uses that history snapshot. A later concurrent request
cannot substitute its active user or metrics destination.

`ContinuationSession` owns memory-tool execution, deduplication, telemetry,
provider continuation admission, durable mixed-tool handoffs and completion.
JSON and intercepted SSE responses use the same service and provider adapters.
Public text produced before internal tool calls is retained in the final native
response. A mixed handoff displays that text once and checkpoints it for eventual
completion persistence; resuming does not display it again. Internal tool calls
and memory results are excluded from this public-text history.
Inserted text follows leading signed thinking or reasoning blocks, which retain
their original order and bytes. Responses carries the earlier provider-issued
public message items and IDs. Completion extraction includes every public text
block and excludes Gemini parts marked as thoughts.
Public fragments from separate provider rounds are separated by a blank line;
parts and deltas within one provider round retain their original concatenation.
Only successfully completed responses enter the completion pipeline. Cancelled
provider calls release their lease, close their connection, and do not persist
an incomplete answer. A synchronous tool already running in a worker thread is
allowed to finish before its lease is released.

## Mixed client and memory tools

The proxy executes only memory tools. For a mixed batch it atomically stores the
full original provider exchange and memory results, and returns only the client's
own tool calls. Their names and arguments remain intact; opaque `vcx_` call IDs
correlate the successor request. No client result is invented.

A successor may arrive at another worker. The store atomically claims the
checkpoint. The proxy checks tenant, conversation, audience, provider, model,
lifecycle epoch, exact client-visible transcript and complete result batch before
restoring the original IDs and hidden memory results. Hidden exchanges bypass
canonical source ingestion. Responses API `previous_response_id` continuations
are supported; Gemini clients that omit function-response IDs can match the
retained function-call IDs or the bounded exact transcript. A client must preserve
its assistant call history when using that older Gemini shape. Transcript discovery
uses read-only snapshots before claiming a lease. An unrelated Gemini tool response
passes through even when another exchange is pending or the store has no durable
continuation capability. If multiple checkpoints match an ID-less transcript, the
client must retain the opaque call ID to disambiguate them.

Checkpoints expire after ten minutes. Each conversation may retain at most four,
sharing a two MiB payload budget. Claims last two minutes and renew every thirty
seconds. Missing, expired, mismatched or concurrently claimed opaque IDs return an
explicit 409. Capacity failures return 503. SQLite and PostgreSQL share the durable
contract; unsupported backends fail explicitly. Conversation deletion removes
pending exchanges. API query-string credentials are excluded from checkpoints.
If a conversation is deleted or changes epoch during raw SSE forwarding, the
completed bytes remain intact and the stale completion is not ingested.

## Streaming tradeoff and validation

Requests without memory-tool interception retain raw byte forwarding. Intercepted
requests collect each provider turn before exposing its tool calls; collection is
capped at eight MiB, including an unterminated SSE event. This prevents a late
mixed batch from leaking an internal call. It also delays the first model-content
event until the provider turn and internal continuations finish. No claim of
unchanged first-token latency is made. A future incremental implementation must
retain this admission boundary for tool blocks and full response envelopes.

Final SSE includes native text deltas, tool arguments, signed thinking and opaque
provider items. Focused tests use real SQLite across separate store instances,
fake provider transports, and the installed Anthropic and OpenAI Responses SDKs.
They cover authority/transcript tampering, result matching, cancellation,
continuation budgeting, multibyte text, source-history isolation and byte caps.
No live model or PostgreSQL performance measurements are implied by these tests.
