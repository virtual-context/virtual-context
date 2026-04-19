"""ContextStore abstract base class — tag-based storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime, timedelta

from ..types import ChunkEmbedding, CompactionLeaseClaim, ConversationStats, DepthLevel, EngineStateSnapshot, Fact, FactSignal, CanonicalTurnChunkEmbedding, CanonicalTurnRow, QuoteResult, StoredSegment, StoredSummary, TagStats, TagSummary, WorkingSetEntry
from .progress_snapshot import ProgressSnapshot


class ContextStore(ABC):
    """Pluggable storage backend for compacted conversation segments."""

    @abstractmethod
    def store_segment(self, segment: StoredSegment) -> str:
        """Upsert by ref. Returns ref."""

    def update_segment(self, segment: StoredSegment) -> None:
        """Update an existing segment in-place (same ref). Falls back to store_segment (upsert)."""
        self.store_segment(segment)

    @abstractmethod
    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None: ...

    @abstractmethod
    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None: ...

    @abstractmethod
    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        """Retrieve summaries matching tags by overlap count, newest first."""

    @abstractmethod
    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]: ...

    @abstractmethod
    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        """List all tags with statistics.

        If *conversation_id* is given, only return tags from segments
        belonging to that conversation.
        """

    @abstractmethod
    def get_conversation_stats(self) -> list[ConversationStats]:
        """Return aggregate statistics grouped by conversation_id, newest first."""

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        """Return full stored segments, newest first when supported."""
        return []

    @abstractmethod
    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]: ...

    @abstractmethod
    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None: ...

    @abstractmethod
    def delete_segment(self, ref: str) -> bool: ...

    @abstractmethod
    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int: ...

    @abstractmethod
    def save_tag_summary(
        self,
        tag_summary: TagSummary,
        conversation_id: str = "",
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        """Upsert on (tag, conversation_id)."""

    @abstractmethod
    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None: ...

    @abstractmethod
    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        """Retrieve all tag summaries, ordered by tag name.

        If *conversation_id* is given, only return tag summaries whose
        source segments belong to that conversation.
        """

    @abstractmethod
    def search_full_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        """Search full_text across all segments.

        Returns QuoteResult objects with excerpts (~200 chars context around match).
        """

    def search_canonical_turn_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        """Search canonical turn text across stored conversation turns."""
        return []

    def conversation_reconcile(self, conversation_id: str):
        """Optional per-conversation write lock for merge-style ingest paths."""
        return nullcontext()

    @abstractmethod
    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]: ...

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        """Idempotent: replaces any existing chunks for this segment."""

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        return []

    def store_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int,
        side: str,
        chunks: list[CanonicalTurnChunkEmbedding],
        canonical_turn_id: str | None = None,
    ) -> None:
        """Idempotent: replaces any existing embedded chunks for this turn side."""

    def get_all_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str | None = None,
    ) -> list[CanonicalTurnChunkEmbedding]:
        return []

    def delete_canonical_turn_chunk_embeddings(
        self,
        conversation_id: str,
        turn_number: int | None = None,
        canonical_turn_id: str | None = None,
    ) -> int:
        return 0

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        """Upsert by conversation_id."""

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        return None

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        return None

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        """Return {trailing_fingerprint: conversation_id} for all persisted conversations.

        Used by SessionRegistry on restart to match inbound requests to
        existing conversations when conversation markers are unavailable.
        """

    # ------------------------------------------------------------------
    # Canonical turn ledger
    # ------------------------------------------------------------------

    def save_canonical_turn(
        self,
        conversation_id: str,
        turn_number: int,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
        primary_tag: str = "_general",
        tags: list[str] | None = None,
        session_date: str = "",
        sender: str = "",
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        canonical_turn_id: str | None = None,
        sort_key: float | None = None,
        turn_hash: str = "",
        hash_version: int = 0,
        normalized_user_text: str = "",
        normalized_assistant_text: str = "",
        tagged_at: str | None = None,
        compacted_at: str | None = None,
        first_seen_at: str | None = None,
        last_seen_at: str | None = None,
        source_batch_id: str | None = None,
        turn_group_number: int = -1,
    ) -> None:
        """Upsert a canonical turn, using ``turn_number`` only as an ordinal hint."""

    def recompute_canonical_turn_groups(
        self,
        conversation_id: str,
    ) -> int:
        return 0

    def get_canonical_turn_rows(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, CanonicalTurnRow]:
        return {}

    def get_all_canonical_turns(
        self,
        conversation_id: str,
    ) -> list[CanonicalTurnRow]:
        return []

    def get_uncompacted_canonical_turns(
        self,
        conversation_id: str,
        *,
        protected_recent_turns: int = 0,
    ) -> list[CanonicalTurnRow]:
        rows = [row for row in self.get_all_canonical_turns(conversation_id) if not row.compacted_at]
        if protected_recent_turns > 0 and len(rows) > protected_recent_turns:
            return rows[:-protected_recent_turns]
        if protected_recent_turns > 0:
            return []
        return rows

    def mark_canonical_turns_tagged(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        tagged_at: str | None = None,
    ) -> int:
        return 0

    def iter_untagged_canonical_rows(
        self,
        *,
        conversation_id: str,
        expected_lifecycle_epoch: int,
        batch_size: int = 32,
    ) -> list[CanonicalTurnRow]:
        """Return up to ``batch_size`` untagged canonical rows for a
        conversation, scoped to the given ``expected_lifecycle_epoch``.

        Implementations MUST JOIN against ``conversations.lifecycle_epoch``
        so a stale caller whose in-memory epoch no longer matches the
        authoritative row simply sees an empty list (no exception) — that
        keeps a background tagger loop from mutating rows that belong to a
        new lifecycle after a delete/resurrect. Rows MUST be ordered by
        ``sort_key ASC`` (never ``turn_number``, which is a VIEW column
        and not stable under concurrent writes). Concrete backends MUST
        implement.
        """
        raise NotImplementedError

    def mark_canonical_row_tagged(
        self,
        *,
        canonical_turn_id: str,
        conversation_id: str,
        expected_lifecycle_epoch: int,
    ) -> bool:
        """Flip a single canonical row's ``tagged_at`` to ``utcnow`` iff it is
        currently untagged AND the conversation still sits at
        ``expected_lifecycle_epoch``.

        Returns ``True`` when exactly one row was updated, else ``False``.
        The EXISTS subclause binds the write to the authoritative lifecycle
        epoch so a stale tagger thread cannot accidentally touch a row that
        belongs to a newer lifecycle after a delete/resurrect. Already-tagged
        rows also return ``False`` (the ``tagged_at IS NULL`` predicate
        filters them out), which makes the call safely idempotent on retry.
        Concrete backends MUST implement.
        """
        raise NotImplementedError

    def mark_canonical_turns_compacted(
        self,
        conversation_id: str,
        canonical_turn_ids: list[str],
        *,
        compacted_at: str | None = None,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        return 0

    def delete_canonical_turns(
        self,
        conversation_id: str,
        turn_number: int | None = None,
    ) -> int:
        return 0

    def delete_canonical_turns_by_batch_id(
        self,
        *,
        conversation_id: str,
        batch_id: str,
    ) -> int:
        """Delete rows from ``canonical_turns`` matching both ``conversation_id``
        AND ``source_batch_id``. Used by ``IngestReconciler`` for commit-time
        rollback on epoch race — surgically removes only the rows a single
        ingest call just wrote without touching concurrent new-lifecycle rows.

        Returns the number of rows deleted. Concrete backends MUST implement.
        """
        raise NotImplementedError

    def replace_canonical_turn_anchors(
        self,
        conversation_id: str,
        anchors: list[tuple[int, str, str]],
    ) -> int:
        return 0

    def get_canonical_turn_anchor_positions(
        self,
        conversation_id: str,
        window_size: int,
    ) -> dict[str, list[int]]:
        return {}

    def save_ingest_batch(self, batch: dict) -> str:
        return str(batch.get("batch_id", "") or "")

    # ------------------------------------------------------------------
    # Conversation lifecycle fencing
    # ------------------------------------------------------------------

    def activate_conversation(self, conversation_id: str) -> int:
        """Mark a conversation as live and return its current generation."""
        return 0

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        """Fence future writes for a conversation by advancing its generation."""
        return self.activate_conversation(conversation_id)

    def get_conversation_generation(self, conversation_id: str) -> int:
        """Return the current durable generation for a conversation."""
        return 0

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        """Whether ``generation`` is the live write generation for ``conversation_id``."""
        return generation == self.get_conversation_generation(conversation_id)

    # ------------------------------------------------------------------
    # Conversation row lifecycle (progress-bar redesign `conversations` table)
    # ------------------------------------------------------------------
    # These methods operate on the per-tenant `conversations` table that
    # carries `lifecycle_epoch` + `phase` for the progress tracker and the
    # delete+resurrect invariants. Distinct from the legacy lifecycle
    # fencing above (`activate_conversation` et al.) which uses the older
    # `conversation_lifecycle` table.

    # Lifecycle-epoch methods diverge from this base class's no-op default
    # convention: silent defaults are unsafe here — a store that returns 0
    # for get_lifecycle_epoch or no-ops mark_conversation_deleted would
    # corrupt epoch-based stale-write protection. Forcing callers to get
    # a clear NotImplementedError ensures any new store backend implements
    # these before it can be used in a progress-bar context.
    def upsert_conversation(self, *, tenant_id: str, conversation_id: str) -> None:
        """Create the conversations row if missing; otherwise refresh updated_at.

        Epoch starts at 1 on new rows; never bumped by this method.
        """
        raise NotImplementedError

    def get_lifecycle_epoch(self, conversation_id: str) -> int:
        """Return the current lifecycle_epoch. Raises KeyError if no row exists."""
        raise NotImplementedError

    def get_conversation_phase(self, conversation_id: str) -> str:
        """Return the current phase for the conversation.

        Returns one of ``"init" | "ingesting" | "compacting" | "active" |
        "deleted"``. Raises ``KeyError`` if no row exists.
        """
        raise NotImplementedError

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        """Admin-flow delete: sets phase='deleted' and stamps deleted_at."""
        raise NotImplementedError

    def increment_lifecycle_epoch_on_resurrect(self, conversation_id: str) -> int:
        """Bump lifecycle_epoch ONLY when phase == 'deleted'.

        TOCTOU-safe: concurrent resurrect calls cannot double-bump.
        Raises KeyError if no row exists.
        """
        raise NotImplementedError

    def read_progress_snapshot(self, conversation_id: str) -> ProgressSnapshot:
        """Return a point-in-time ProgressSnapshot for ``conversation_id``.

        ``total_ingestible`` / ``done_ingestible`` are derived at read
        time from SUM(covered_ingestible_entries) over ``canonical_turns``
        (filtered by ``tagged_at IS NOT NULL`` for the numerator) so they
        can never drift from canonical truth.  Raises ``KeyError`` if the
        conversation row doesn't exist.
        """
        raise NotImplementedError

    def upsert_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        raw_payload_entries: int,
    ) -> None:
        """Ownership-free upsert of the running ingestion_episode row.

        On INSERT, creates a running episode with the given worker as
        initial owner. On CONFLICT (another running row exists at the
        same (conversation, lifecycle_epoch)), ONLY widens
        ``raw_payload_entries`` via MAX — does NOT change ownership or
        other fields. Idempotent.
        """
        raise NotImplementedError

    def claim_ingestion_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> bool:
        """Claim the ingestion lease for this (conversation, lifecycle_epoch).

        Returns True iff the caller now owns the lease. Rules:
          - If caller already owns it: refresh heartbeat, return True.
          - If current heartbeat is stale (older than ``lease_ttl_s``):
            take over, return True.
          - Otherwise: another worker owns a fresh lease; return False.
        Epoch-scoped: filters on ``lifecycle_epoch``.
        """
        raise NotImplementedError

    def refresh_ingestion_heartbeat(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Epoch-scoped heartbeat refresh.

        A stale thread carrying an old epoch cannot refresh a new
        lifecycle's heartbeat. Returns True iff a matching row was updated.
        """
        raise NotImplementedError

    def complete_ingestion_episode(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Race-guarded completion of the running ingestion_episode row.

        Epoch-scoped. Returns True iff:
          - A running episode exists at the caller's ``lifecycle_epoch``.
          - The caller is the current owner.
          - No untagged canonical rows remain (NOT EXISTS guard).
        Returns False if any condition fails.
        """
        raise NotImplementedError

    def start_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_count: int,
        phase_name: str,
    ) -> str:
        """Insert a fresh ``compaction_operation`` row in ``'queued'``
        status. Returns the new ``operation_id`` (UUID string).

        Raises the backend-specific unique-violation error (enforced by
        the partial unique index on ``status IN ('queued','running')``)
        if another active operation already exists for this
        (conversation, lifecycle_epoch). Callers should retry or wait.
        """
        raise NotImplementedError

    def claim_compaction_lease(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        lease_ttl_s: float,
    ) -> "CompactionLeaseClaim":
        """Claim the compaction lease for this (conversation, epoch).

        Works on queued OR running operations. Returns a CompactionLeaseClaim
        with claimed=True iff:
          - Caller already owns the row, OR
          - Current heartbeat is stale (older than ``lease_ttl_s``).
        Returns claimed=False if another worker holds a fresh lease or no
        active row exists at the given ``lifecycle_epoch``.
        The prev_operation_id and prev_owner_worker_id fields reflect the row
        observed before the UPDATE (None when no active row existed).
        """
        raise NotImplementedError

    def advance_compaction_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_index: int,
        phase_name: str,
    ) -> bool:
        """Epoch-scoped phase advance. Transitions status from
        ``'queued'`` to ``'running'`` and records the new
        phase_index/phase_name/heartbeat_ts. The epoch filter is
        double-scoped with a correlated subquery against
        ``conversations.lifecycle_epoch`` so a stale thread is
        rejected at SQL level. Returns True iff a matching row was
        updated.
        """
        raise NotImplementedError

    def complete_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> bool:
        """Epoch-scoped compaction completion. Transitions status to
        ``'completed'`` and stamps ``completed_at``. Returns True iff
        an active (queued or running) row exists, the caller owns it,
        and the caller's epoch matches the authoritative
        ``conversations.lifecycle_epoch``.
        """
        raise NotImplementedError

    def fail_compaction_operation(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        error_message: str,
    ) -> bool:
        """Epoch-scoped compaction failure. Records ``error_message``,
        stamps ``completed_at``, and transitions status to
        ``'failed'``. Same ownership + epoch guards as
        ``complete_compaction_operation``.
        """
        raise NotImplementedError

    def cleanup_abandoned_compaction(
        self,
        *,
        conversation_id: str,
        dead_operation_id: str,
        new_operation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
        phase_count: int,
    ) -> bool:
        """Atomic takeover-cleanup transaction.

        Returns True iff this call performed the transition (dead_op was
        'running' and we abandoned it + inserted new_op). Returns False
        when the dead_op was already abandoned/completed — idempotent
        re-run; the new-row INSERT is skipped to preserve the
        one-active invariant enforced by ``idx_compaction_operation_active``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Request metadata + phase helpers (epoch-guarded)
    # ------------------------------------------------------------------
    # These four methods mutate the per-conversation progress metadata on
    # the ``conversations`` table and are ALL epoch-guarded at the SQL
    # layer. A stale caller whose in-memory ``lifecycle_epoch`` no longer
    # matches the authoritative row sees a ``False``/``None`` return and
    # never stomps a new lifecycle's counters/phase.

    def update_request_metadata(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        last_raw_payload_entries: int,
        last_ingestible_payload_entries: int,
    ) -> bool:
        """Overwrite the per-request snapshot counters
        (``last_raw_payload_entries`` + ``last_ingestible_payload_entries``)
        on the conversations row. Epoch-guarded. Returns ``True`` iff the
        UPDATE matched a row at the caller's ``lifecycle_epoch``. A stale
        caller gets ``False``.
        """
        raise NotImplementedError

    def widen_pending_raw_payload_entries(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        value: int,
    ) -> bool:
        """Monotonic widener for ``pending_raw_payload_entries``. SQLite
        uses scalar ``MAX()``; Postgres uses ``GREATEST()``. Never walks
        the counter backwards — concurrent writers coalesce to the
        largest seen value. Epoch-guarded. Returns ``True`` iff the
        UPDATE matched a row at the caller's ``lifecycle_epoch``.
        """
        raise NotImplementedError

    def set_phase(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        phase: str,
    ) -> bool:
        """Epoch-guarded phase write. Returns ``True`` iff the UPDATE
        matched a row at the caller's epoch. A stale thread cannot stomp
        a new lifecycle's phase.
        """
        raise NotImplementedError

    def set_phase_and_drain_pending_raw(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        new_phase: str,
    ) -> int | None:
        """Atomically transition phase AND return the drained
        ``pending_raw_payload_entries``. Transactional (SQLite uses
        ``BEGIN IMMEDIATE``; Postgres uses ``conn.transaction()``).
        Returns the drained integer on success, or ``None`` when the
        caller's ``lifecycle_epoch`` doesn't match the authoritative
        conversations row.
        """
        raise NotImplementedError

    def drain_compaction_exit(
        self,
        *,
        conversation_id: str,
        lifecycle_epoch: int,
        worker_id: str,
    ) -> str | None:
        """Atomic compaction-exit decision + pending drain.

        Inside a single transaction:

        1. Verify ``conversations.lifecycle_epoch`` matches the caller's —
           return ``None`` on mismatch (no writes).
        2. ``EXISTS (SELECT 1 FROM canonical_turns WHERE conversation_id = ?
           AND tagged_at IS NULL)`` inside the same transaction.
        3. If any untagged canonical rows remain → transition phase to
           ``'ingesting'``, zero ``pending_raw_payload_entries``, and INSERT
           a fresh ``ingestion_episode`` row in ``'running'`` status whose
           ``raw_payload_entries`` equals the drained ``pending_raw``.
           Else → transition phase to ``'active'`` and zero
           ``pending_raw_payload_entries``.

        Returns the new phase (``'ingesting'`` or ``'active'``) on success,
        or ``None`` when the caller's ``lifecycle_epoch`` does not match the
        authoritative conversations row. Epoch-guarded.

        Callers (e.g. ``ProxyState.exit_compaction``) must NOT rely on
        ``read_progress_snapshot`` for this decision — the EXISTS check
        runs in the same transaction as the phase UPDATE so a concurrent
        tagger cannot flip the answer between read and write.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tag summary search (used by RRF retrieval scoring)
    # ------------------------------------------------------------------

    def search_tag_summaries_fts(
        self, query: str, limit: int = 20, conversation_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """FTS search on tag summary text. Returns [(tag, bm25_score)]."""
        return []

    def store_tag_summary_embedding(
        self,
        tag: str,
        conversation_id: str,
        embedding: list[float],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> None:
        """Store embedding vector for a tag summary."""
        pass

    def load_tag_summary_embeddings(
        self, conversation_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Load all tag summary embeddings. Returns {tag: embedding_vector}."""
        return {}

    # ------------------------------------------------------------------
    # Cross-cutting queries (used by consolidator, tool loop, etc.)
    # ------------------------------------------------------------------

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        """Return snippet descriptions for tags that have no tag_summary entry.

        Each dict has keys: ``tag`` (str), ``snippet`` (str -- first ~100
        chars of the segment summary for one segment carrying that tag).

        Used by the tag consolidator to provide descriptions for orphan
        tags so the LLM can make informed consolidation decisions.

        Backends that do not support this may return an empty list.
        """
        return []

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        """Return facts that were superseded *by* the given fact IDs.

        Each dict has keys: ``superseded_by`` (str — the ID of the newer
        fact), ``subject``, ``verb``, ``object`` (all str).

        This is the reverse lookup: given a set of current (non-superseded)
        fact IDs, find the older facts they replaced.

        Backends that do not support facts may return an empty list.
        """
        return []

    # ------------------------------------------------------------------
    # D1: Fact Extraction
    # ------------------------------------------------------------------

    def store_facts(
        self,
        facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        return 0

    def query_facts(
        self,
        *,
        subject: str | None = None,
        verb: str | None = None,
        verbs: list[str] | None = None,
        object_contains: str | None = None,
        status: str | None = None,
        fact_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        return []

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        return []

    def get_facts_by_segment(self, segment_ref: str) -> list[Fact]:
        return []

    def replace_facts_for_segment(
        self,
        conversation_id: str,
        segment_ref: str,
        facts: list,
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> tuple[int, int]:
        """Atomically replace all facts for a segment. Returns (deleted, inserted)."""
        return 0, self.store_facts(
            facts,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list[Fact]:
        """FTS search across fact fields. Returns non-superseded facts."""
        return []

    def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        pass

    def update_fact_fields(
        self, fact_id: str, verb: str, object: str, status: str, what: str
    ) -> None:
        pass

    def get_actionable_fact_tags(
        self, tags: list[str], conversation_id: str | None = None,
    ) -> set[str]:
        """Return subset of tags that have non-superseded active/completed/personal facts."""
        return set()

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        return {}

    def query_experience_facts_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[Fact]:
        """Return facts within a when_date range.

        *start_date* and *end_date* are ``YYYY-MM-DD`` ISO strings compared
        lexicographically against the ``when_date`` column.  Returns
        facts ordered by when_date ASC.
        """
        return []

    # ------------------------------------------------------------------
    # Turn / Segment ↔ Tool Output linkage (join tables)
    # ------------------------------------------------------------------

    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None:
        """Link a tool output ref to a specific turn."""
        pass  # default no-op for backwards compatibility

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        """Return tool_output refs linked to a turn."""
        return []

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        """Link a tool output ref to a segment."""
        pass

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        """Return tool_output refs linked to a segment."""
        return []

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        """Return refs from the tool_outputs table for a given conversation + turn.

        Unlike ``get_tool_outputs_for_turn`` (which reads the join table),
        this queries the ``tool_outputs`` table directly by its ``turn`` column.
        Used during ingestion to discover intercepted tool outputs that should
        be linked to a canonical turn.
        """
        return []

    # ------------------------------------------------------------------
    # Media Output Storage
    # ------------------------------------------------------------------

    def store_media_output(
        self,
        ref: str,
        conversation_id: str,
        media_type: str,
        width: int,
        height: int,
        original_bytes: int,
        compressed_bytes: int,
        file_path: str,
    ) -> None:
        """Store metadata for a compressed media output. Default no-op."""
        pass

    def get_media_output(self, conversation_id: str, ref: str) -> dict | None:
        """Look up media output metadata by conversation_id and ref.

        Returns dict with keys {ref, conversation_id, media_type, width, height,
        original_bytes, compressed_bytes, file_path} or None if not found.
        Default returns None.
        """
        return None

    # ------------------------------------------------------------------
    # Tool Output Storage
    # ------------------------------------------------------------------

    def delete_conversation(self, conversation_id: str) -> int:
        return 0

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        """Delete aliases owned by ``conversation_id``.

        Backends may keep legacy/global aliases under the empty conversation id.
        """
        return 0

    def store_tool_output(
        self,
        ref: str,
        conversation_id: str,
        tool_name: str,
        command: str,
        turn: int,
        content: str,
        original_bytes: int,
    ) -> None:
        pass

    def search_tool_outputs(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list:
        return []

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        """Look up a stored tool output by conversation_id and ref.

        Returns the full stored content string, or None if not found.
        """
        return None

    # ------------------------------------------------------------------
    # Chain Snapshots (turn chain collapse)
    # ------------------------------------------------------------------

    def store_chain_snapshot(
        self,
        ref: str,
        conversation_id: str,
        turn_number: int,
        chain_json: str,
        message_count: int,
        tool_output_refs: str = "",
    ) -> None:
        """Upsert a chain snapshot by ref."""
        pass

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        """Retrieve a chain snapshot by conversation_id and ref.

        Returns {ref, conversation_id, turn_number, chain_json,
        message_count, tool_output_refs} or None.
        """
        return None

    def get_chain_snapshots_for_conversation(
        self,
        conversation_id: str,
        min_turn: int = 0,
    ) -> list[dict]:
        """Return metadata for chain snapshots where turn_number >= min_turn.
        Returns list of {ref, turn_number, tool_output_refs, message_count}. No chain_json."""
        return []

    def get_chain_recovery_manifest(
        self,
        conversation_id: str,
        min_turn: int = 0,
    ) -> list[dict]:
        """Return recovery metadata for collapsed chain stubs.

        Default implementation falls back to snapshot metadata plus one tool-name
        lookup per snapshot. Storage backends can override this with a single
        optimized query.
        """
        manifest: list[dict] = []
        for snap in self.get_chain_snapshots_for_conversation(
            conversation_id,
            min_turn=min_turn,
        ):
            raw_refs = [
                ref.strip()
                for ref in str(snap.get("tool_output_refs", "")).split(",")
                if ref.strip()
            ]
            tool_names = self.get_tool_names_for_refs(raw_refs) if raw_refs else []
            manifest.append({
                "ref": snap.get("ref", ""),
                "turn_number": snap.get("turn_number", -1),
                "tool_output_refs": snap.get("tool_output_refs", ""),
                "message_count": snap.get("message_count", 0),
                "tool_names": ", ".join(tool_names) if tool_names else "",
            })
        return manifest

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        return []

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        """Return distinct tool names linked to a segment via segment_tool_outputs.

        Performs a JOIN between segment_tool_outputs and tool_outputs to get
        distinct tool_name values.
        """
        return []

    def save_tool_call(self, call: dict) -> None:
        """Persist a tool call record."""
        pass

    def load_tool_calls(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Load recent tool call records for a conversation."""
        return []

    def load_tool_call(self, call_id: int) -> dict | None:
        """Load a single tool call by ID."""
        return None

    def save_request_context(self, context: dict) -> int:
        """Persist retrieval/assembly context for a request.

        Returns the durable per-conversation request sequence assigned to this
        context. Implementations may honor an explicit ``request_turn`` value
        when provided for migrations/backfills, but live request handling
        should rely on the returned value rather than supplying its own.
        """
        return int(context.get("request_turn", 0) or 0)

    def load_request_contexts(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Load recent request contexts for a conversation."""
        return []

    def save_request_capture(self, capture: dict) -> None:
        pass

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        return []
