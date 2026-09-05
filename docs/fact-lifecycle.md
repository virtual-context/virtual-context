# Fact lifecycle admission

Fact comparison proposes relationships; it does not establish ownership,
authorship, chronology or completion. `core.fact_lifecycle` defines immutable
`FactProposal` and `AdmissionDecision` values and the shared deterministic policy.
The checker applies the policy before a candidate reaches an embedding model or
comparison model, and SQL storage applies it again while locking the affected
facts and physical source rows.

Supersession requires distinct active facts, one nonempty conversation owner,
the same subject, compatible author attribution and source role, current matching
audience proof, and non-reversing chronology. A plan cannot replace an observed
state merely because its date is later. Unknown dates remain explicitly labelled
in decisions. Legacy facts with no author attribution retain a labelled local
compatibility policy; any canonical audience already known still constrains them.

Current author proof is physical and role-specific. Version-one attribution
requires a complete one-human roster without reply ambiguity. Version-two
requester attribution must resolve one source message to the recorded sender.
Subject attribution resolves the quoted target through a current reply edge and
nonempty quote, never through the requester's identity. Duplicate external IDs,
missing source rows, changed authors and assistant text cannot prove a human
claim. Old version-two records without message identity require re-derivation
from source before they can participate in supersession.

Stamped attribution with an unresolved actor is intentionally ineligible for
automatic supersession and exact deduplication. Assistant-lane facts are also
ineligible: assistant prose is not evidence of a human author's claim, and this
policy has no separate agent-authorship ledger. Those records retain their
history, including possible duplicates or contradictions, until explicit
source-backed re-derivation establishes eligible authorship. They do not fall
back to the legacy unattributed policy. Broadening either lane requires an
explicit authority model and corresponding provenance tests.

Exact deduplication groups conversation, author, role, audience and statement.
It retains the latest dated duplicate only when the same admission policy allows
it. Rejected database writes do not increment counts or publish supersession
edges. The winning fact remains source-derived: model-written consolidation prose
is no longer allowed to rewrite its statement or temporal status.

SQL decisions record the proposal, acceptance reason and policy version, observed
and event dates, source versions and before/after values.
Model inputs are bound to fact fingerprints and canonical/segment versions before
comparison. Each incoming fact's candidate selection shares one snapshot per
candidate across embedding and comparison; this cache never crosses requests.
The transaction compares those expected versions under its locks;
changes during the model call are rejected and audited as `stale_proposal`.
Canonical fingerprints cover provenance-bearing fields and exclude maintenance
timestamps, so tagging, compaction and last-seen updates do not invalidate a
proposal without a source change. `proposal_json` retains expected fact and
source versions; `source_versions_json` and `observed_fact_versions_json` record
the versions actually locked for the decision. Database triggers protect audit
content and original ownership from updates. A merge may relocate its current
`conversation_id`, and conversation deletion removes its retained audit; this is
not an undeletable ledger.
SQLite tests cover real
canonical sources, cross-owner/author/audience/channel rejection before model
calls, chronological rejection, accepted evidence versions, duplicate isolation,
rejected writes and immutable source statements. PostgreSQL shares the domain
operation; database-specific locking and rollback validation runs on the remote
fleet rather than starting a local database.
