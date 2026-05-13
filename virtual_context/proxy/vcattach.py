"""VCATTACH command execution — shared by proxy and REST paths."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_target(
    target_raw: str,
    current_id: str,
    conversation_ids: list[str],
    labels: dict[str, str],
    target_exists=None,
) -> tuple[str | None, str, str]:
    """Resolve VCATTACH target to a conversation ID.

    Args:
        target_raw: what the user typed after VCATTACH
        current_id: the current conversation ID
        conversation_ids: all known conversation IDs
        labels: {conversation_id: label} dict
        target_exists: optional callable(conversation_id) -> bool. When
            provided, a resolved target_id is rejected with an error if
            the callable returns False. Defends against labels.json
            entries pointing to deleted/tombstoned conversations
            (defense-in-depth — a stale label must not let VCATTACH
            silently graft onto a corpse).
            Exceptions inside the callable fail open: a transient DB
            blip must not block legitimate VCATTACHes.

    Returns (target_id, target_label, error_message).
    target_id is None on error; error_message is empty on success.
    """
    # Resolve to (tid, tlabel) via one of three paths, then run common
    # validations (same-conv, existence). Single return point keeps the
    # validation chain DRY.
    tid: str | None = None
    tlabel: str = target_raw

    # 1. Label match (case-insensitive)
    label_to_id = {v.lower(): (k, v) for k, v in labels.items()}
    if target_raw.lower() in label_to_id:
        tid, tlabel = label_to_id[target_raw.lower()]
    # 2. Exact UUID match
    elif target_raw in conversation_ids:
        tid = target_raw
        tlabel = labels.get(target_raw, target_raw)
    else:
        # 3. UUID prefix match
        matches = [c for c in conversation_ids if c.startswith(target_raw)]
        if len(matches) == 1:
            tid = matches[0]
            tlabel = labels.get(tid, target_raw)
        elif len(matches) > 1:
            return (
                None,
                target_raw,
                f"Ambiguous prefix '{target_raw}' matches {len(matches)} conversations.",
            )
        else:
            return (
                None,
                target_raw,
                f"No conversation found matching '{target_raw}'. "
                "Use a conversation label or ID from the dashboard.",
            )

    # Same-conv check (caller is already on the target).
    if tid == current_id:
        return None, tlabel, f"Already on conversation {tlabel}."

    # Existence check — defense-in-depth against stale labels pointing at
    # deleted/tombstoned conversations.
    if callable(target_exists):
        try:
            ok = bool(target_exists(tid))
        except Exception:
            # Fail open on transient errors so legitimate attaches don't break.
            ok = True
        if not ok:
            return (
                None,
                tlabel,
                f"Cannot attach to '{tlabel}' ({tid[:12]}) — "
                "the conversation has no persisted state (deleted or never "
                "existed). The label may be stale; remove it from the "
                "dashboard or pick another target.",
            )

    return tid, tlabel, ""


def _alias_deleted_event_for(store, target_id: str) -> dict:
    """Construct the AliasDeletedEvent dict for the explicit-invoke shape.

    Delegates to the store's ``_build_alias_deleted_event`` when
    available so the event shape matches what the legacy
    ``on_committed`` plumbing produced (including
    ``reverse_dependents`` BFS over incoming edges). Falls back to a
    minimal dict when the store doesn't expose the helper — preserves
    forward-compat against older or non-canonical store implementations.
    """
    builder = _resolve_store_attr(store, "_build_alias_deleted_event")
    if callable(builder):
        try:
            return builder(target_id)
        except Exception:
            logger.warning(
                "VCATTACH: _build_alias_deleted_event raised for %s; "
                "falling back to minimal event",
                target_id[:12], exc_info=True,
            )
    # Fallback shape matches the canonical store helpers' output keys
    # (``alias_id`` for the deleted-side, see
    # ``storage/sqlite.py:_build_alias_deleted_event``). Keeps
    # downstream subscribers' key access stable when the store layer
    # doesn't expose the helper directly.
    return {"type": "alias_deleted", "alias_id": target_id}


def _alias_created_event_for(store, old_id: str, target_id: str) -> dict:
    """Construct the AliasCreatedEvent dict.

    Same store-helper-with-fallback shape as
    ``_alias_deleted_event_for``. The store's helper resolves
    ``target_id`` to its terminal (walks the alias chain) and computes
    ``reverse_dependents`` for the source side.
    """
    builder = _resolve_store_attr(store, "_build_alias_created_event")
    if callable(builder):
        try:
            return builder(old_id, target_id)
        except Exception:
            logger.warning(
                "VCATTACH: _build_alias_created_event raised for %s->%s; "
                "falling back to minimal event",
                old_id[:12], target_id[:12], exc_info=True,
            )
    return {"type": "alias_created", "source": old_id, "target": target_id}


def _resolve_store_attr(store, attr_name: str):
    """Return ``attr_name`` from ``store`` or an inner concrete store.

    ``CompositeStore`` delegates alias APIs explicitly but does NOT
    forward private helpers like ``_build_alias_*_event``. Reach
    through to the inner concrete stores (which are usually
    ``SQLiteStore`` or ``PostgresStore``) when the composite is the
    visible interface.
    """
    direct = getattr(store, attr_name, None)
    if direct is not None:
        return direct
    for inner_name in ("_segments", "segments", "_state", "state"):
        inner = getattr(store, inner_name, None)
        if inner is None:
            continue
        resolved = getattr(inner, attr_name, None)
        if resolved is not None:
            return resolved
    return None


def execute_attach(
    old_id: str,
    target_id: str,
    store,
    registry_invalidate=None,
    cross_worker_invalidate=None,
    session_state_provider=None,
) -> None:
    """Execute the attach: clear any reverse alias, register new alias, invalidate.

    Old conversations are **preserved** — VCATTACH is a durable redirect, not a
    merge. The alias row in storage is the single source of truth: future
    requests with *old_id* route to *target_id* via the resolver, regardless of
    whether *old_id* still has its own stored conversation row. This means a
    user can VCATTACH away from a labeled conversation and later return to it
    via VCATTACH <that label> without losing history.

    To support "return to" flows, any existing alias with alias_id == target_id
    is cleared before saving the new alias. Example: if alias A -> B exists and
    the user types VCATTACH A, we remove A -> B so A can resolve to itself.

    Args:
        old_id: conversation being abandoned (its alias row is updated)
        target_id: conversation being attached to
        store: unwrapped store (composite or concrete) for alias persistence
        registry_invalidate: callable(conversation_id) — invoked twice (once
            for old_id, once for target_id) to evict any cached runtime
            state. The two invocations cover (1) the issuing chat's
            in-memory ProxyState whose engine.config.conversation_id still
            equals old_id (without this eviction the next request from
            the issuing chat keeps routing to the stale state and bypasses
            alias resolution) and (2) any cached state for target_id so
            it can be re-loaded fresh. The callback MUST NOT delete
            persisted conversation data — VCATTACH preserves both rows.
            A failure on one id does not prevent invocation for the other.
        cross_worker_invalidate: callable(``AliasEvent``) — engine-side
            cross-worker invalidation callback. Invoked explicitly after
            the alias write commits and the best-effort SessionState
            marker write runs. Cloud's adapter wraps this to capture
            ``tenant_id`` and publish a Redis event so cached engine
            instances on sibling workers evict their stale source-bound
            state on every alias write. Per spec S9
            at-least-once contract: callback failures raise
            ``InvalidationFailedError`` which propagates back through
            this function so the REST handler can return a retryable
            503 to the user. The alias row is already committed when
            the callback fires; retrying the request fires the
            callback again.

    Removed (F-2 audit):
        ``reset_engine_state`` callable(target_id) was a dormant callback
        slot. All call sites passed None and the slot was shape-identical
        to the VCATTACH seam (a non-destructively-named
        callback invoked on target_id). Removed structurally to prevent
        a future PR from wiring a destructive primitive into it without
        explicit review. If a real use case for an engine-state reset
        hook appears later, re-introduce with an explicit name and a
        docstring rule forbidding destructive primitives.
    """
    # T0a. Clear any alias FROM target_id so target_id resolves to itself.
    # This unlocks "return to A" flows where A was previously aliased to B.
    # ``on_committed`` is INTENTIONALLY NOT passed — cross-worker
    # invalidation moves to an explicit post-marker-write step (T1) so
    # the strict T2-before-T1 ordering invariant holds for every
    # observer. See ``docs/specs/vcattach-redis-marker-write-and-cross-worker-invalidation.md``
    # CRITICAL invariant (3).
    delete_alias = getattr(store, "delete_conversation_alias", None)
    if callable(delete_alias):
        try:
            delete_alias(target_id)
        except Exception:
            logger.warning(
                "VCATTACH: failed to clear existing alias for %s",
                target_id[:12],
            )

    # T0b. Register new alias old_id -> target_id. Same no-on_committed
    # rule as T0a — explicit cross-worker invocation at T1 below.
    store.save_conversation_alias(old_id, target_id)
    logger.info("VCATTACH: alias %s -> %s", old_id[:12], target_id[:12])

    # T2. Derive correct SessionState markers from canonical_turns and
    # write to Redis BEFORE publishing cross-worker invalidation. This
    # corrects target's SessionState in one shot, replacing any
    # corrupted state where compaction-time fields silently failed to
    # persist (the system-wide class fixed in v0.4.5's clamp inversion
    # at hydrate-time, here fixed at the write-time entry point so
    # future hydrations have correct data on disk too).
    #
    # Failures here are best-effort — alias row is already committed
    # at T0b, and T1 below still fires to invalidate sibling caches.
    # The hydrate-time defensive recovery (Step C in spec) catches any
    # workers that read stale Redis between T0b and a successful T2.
    if session_state_provider is not None:
        try:
            from ..core.state_recovery import derive_session_state_markers
            existing = None
            try:
                existing = session_state_provider.load(target_id)
            except Exception:
                logger.warning(
                    "VCATTACH: failed to load existing SessionState for "
                    "%s; deriving from canonical_turns without "
                    "carrying forward non-derivable fields",
                    target_id[:12], exc_info=True,
                )
            derived = derive_session_state_markers(
                store, target_id, existing_state=existing,
            )
            if derived is not None:
                saved_version = session_state_provider.save(target_id, derived)
                if saved_version is None:
                    logger.warning(
                        "VCATTACH: SessionStateProvider.save did not persist "
                        "derived markers for target %s; alias is committed and "
                        "cross-worker invalidation will still fire",
                        target_id[:12],
                    )
                else:
                    logger.info(
                        "VCATTACH: persisted derived SessionState markers for "
                        "target %s (compacted=%d, flushed=%d, "
                        "last_completed_turn=%d, last_indexed_turn=%d, "
                        "turn_tag_entries=%d)",
                        target_id[:12],
                        derived.compacted_prefix_messages,
                        derived.flushed_prefix_messages,
                        derived.last_completed_turn,
                        derived.last_indexed_turn,
                        len(derived.turn_tag_entries),
                    )
        except Exception:
            logger.warning(
                "VCATTACH: SessionState marker write failed for target %s; "
                "alias is committed and cross-worker invalidation will "
                "still fire — sibling workers will apply hydrate-time "
                "defensive recovery from canonical_turns",
                target_id[:12], exc_info=True,
            )

    # T1. Cross-worker invalidation — fires whether T2 succeeded or
    # not. The alias row IS committed at this point and sibling
    # workers MUST learn about it regardless of marker-write outcome.
    # When T2 succeeded: siblings re-hydrate from corrected Redis on
    # next request. When T2 failed: siblings re-hydrate from stale
    # Redis and apply hydrate-time defensive recovery.
    if callable(cross_worker_invalidate):
        from ..core.exceptions import InvalidationFailedError
        for event in (
            _alias_deleted_event_for(store, target_id),
            _alias_created_event_for(store, old_id, target_id),
        ):
            try:
                cross_worker_invalidate(event)
            except InvalidationFailedError:
                # Preserve at-least-once contract when the callback
                # itself signals retryable failure. Caller (REST
                # handler) translates to 503 so the client retries.
                # Alias row is already committed; marker write at T2
                # already ran. A retry of the VCATTACH refires this
                # publish step. If the retried marker write derives
                # from stale state, provider.save's Redis version check
                # rejects it; if it derives from the current state, the
                # write is a harmless checkpoint bump.
                raise
            except Exception:
                logger.warning(
                    "VCATTACH: cross-worker invalidation failed for "
                    "event %s",
                    event.get("type", "?"),
                    exc_info=True,
                )

    # T1.5. Local-worker registry eviction. old_id eviction is
    # the critical fix for the routing bug: without it, the issuing
    # chat's ProxyState (whose engine.config.conversation_id == old_id)
    # keeps matching chat_id/sys_hash routing on subsequent requests,
    # so ingestion continues writing to old_id even though the alias
    # row redirects old_id -> target_id. target_id eviction is a
    # cache-coherence guard so a stale cached target state cannot
    # shadow a fresh load.
    if callable(registry_invalidate):
        for cid in (old_id, target_id):
            try:
                registry_invalidate(cid)
            except Exception:
                logger.warning("VCATTACH: failed to invalidate session %s", cid[:12])
