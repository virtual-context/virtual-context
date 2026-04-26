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


def execute_attach(
    old_id: str,
    target_id: str,
    store,
    registry_invalidate=None,
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

    Removed (F-2 audit, 2026-04-26):
        ``reset_engine_state`` callable(target_id) was a dormant callback
        slot. All call sites passed None and the slot was shape-identical
        to the 2026-04-26 VCATTACH seam (a non-destructively-named
        callback invoked on target_id). Removed structurally to prevent
        a future PR from wiring a destructive primitive into it without
        explicit review. If a real use case for an engine-state reset
        hook appears later, re-introduce with an explicit name and a
        docstring rule forbidding destructive primitives.
    """
    # 1. Clear any alias FROM target_id so target_id resolves to itself again.
    #    This unlocks "return to A" flows where A was previously aliased to B.
    delete_alias = getattr(store, "delete_conversation_alias", None)
    if callable(delete_alias):
        try:
            delete_alias(target_id)
        except Exception:
            logger.warning("VCATTACH: failed to clear existing alias for %s", target_id[:12])

    # 2. Register new alias old_id -> target_id
    store.save_conversation_alias(old_id, target_id)
    logger.info("VCATTACH: alias %s -> %s", old_id[:12], target_id[:12])

    # 3. Invalidate cached runtime state for BOTH ids. old_id eviction is
    #    the critical fix for the routing bug: without it, the issuing
    #    chat's ProxyState (whose engine.config.conversation_id == old_id)
    #    keeps matching chat_id/sys_hash routing on subsequent requests,
    #    so ingestion continues writing to old_id even though the alias
    #    row redirects old_id -> target_id. target_id eviction is a
    #    cache-coherence guard so a stale cached target state cannot
    #    shadow a fresh load.
    if callable(registry_invalidate):
        for cid in (old_id, target_id):
            try:
                registry_invalidate(cid)
            except Exception:
                logger.warning("VCATTACH: failed to invalidate session %s", cid[:12])
