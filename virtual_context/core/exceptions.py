"""Exception types for the VCMERGE surface.

These exceptions are raised by storage methods and the engine entry point
in `virtual_context/engine.py::Engine.merge_conversation`. The cloud-side
REST handler (`vc_cloud/rest_api.py`'s `handle_vc_merge_cloud`) catches
them and translates each into a dual-populated error envelope per spec
section 12.9.

Sub-codes for `MergeBusy` follow spec section 6.2's classification:
`merge_busy_compact`, `merge_busy_ingest`, `merge_busy_lock`, etc. The
sub-code lets cloud render an actionable message rather than a generic
"merge busy" string.
"""

from __future__ import annotations


class MergeAuditMissing(Exception):
    """Raised by the body method (`Store.merge_conversation_data`) when the
    pre-flight `SELECT 1 FROM merge_audit ... FOR UPDATE` (D1) finds no
    matching in_progress reservation row.

    Most common cause: the stale-reservation sweeper rolled back the
    reservation row between cloud's reservation step and the body call.
    Cloud's REST handler catches and returns the
    `merge_audit_missing` error envelope (with idempotency-retry guidance
    if the call should be retried).
    """


class MergeBusy(Exception):
    """Raised when a concurrent operation blocks the merge: an active
    compaction, a running ingestion episode, or an unreleasable lock on
    the source / target conversation.

    `code` discriminates the cause for cloud's error envelope. Values:
        - "merge_busy_compact": active compaction_operation row
          (status in queued/running) for source or target.
        - "merge_busy_ingest": active ingestion_episode row
          (status = running) for source or target.
        - "merge_busy_lock": failed to acquire conversation_reconcile
          lock within the body transaction's lock-timeout window.
    """

    def __init__(self, message: str, code: str = "merge_busy"):
        super().__init__(message)
        self.code = code


class LifecycleEpochMismatch(Exception):
    """Raised when the source's or target's `lifecycle_epoch` advanced
    between cloud's reservation step (which captured the epochs) and the
    body method's pre-condition check.

    The merge body refuses rather than committing against a stale view
    of lifecycle. Cloud renders the `lifecycle_epoch_mismatch` error
    envelope; client should re-fetch the conversation state and retry.
    """


class CrossTenantMergeError(Exception):
    """Raised when source's `tenant_id` does not match target's `tenant_id`
    at the engine entry point.

    Cloud's REST handler also pre-checks this at C2.4, but the engine-side
    refuse is defense-in-depth: it catches a misroute that bypasses
    cloud's pre-check (see plan section 13 anti-subversion summary).
    """


class MergeTooLarge(Exception):
    """Raised when source's `canonical_turns_count` exceeds
    `config.merge.max_sync_source_turns` (default 10000 per spec).

    Cloud's REST handler pre-checks this at C2.3 and returns the
    `merge_too_large` envelope without invoking the engine. Engine
    raise is defense-in-depth.
    """


class MergedSourceWithoutAliasError(Exception):
    """Raised by `SessionRegistry.get_or_create` (E1.3) when a row has
    `phase = 'merged'` but no corresponding `conversation_aliases` row
    exists to redirect to the target.

    This indicates corrupt state: the merge body committed the source's
    phase flip to 'merged' but failed to UPSERT the alias. Manual
    intervention required; the engine refuses to materialize a fresh
    engine on the source rather than silently creating a divergent state.
    """


class SchemaMismatchError(Exception):
    """Raised when an existing schema artifact (table, column, constraint,
    index) does not match the version the merge surface expects.

    Today the engine's `_ensure_schema()` is forward-only and idempotent;
    this exception is reserved for future use if a downgrade-then-upgrade
    cycle ever leaves the schema in an inconsistent state. Plan section
    9.6 calls out forward-fix as the preferred recovery.
    """
