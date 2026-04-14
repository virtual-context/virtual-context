"""VCATTACH command execution — shared by proxy and REST paths."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_target(
    target_raw: str,
    current_id: str,
    conversation_ids: list[str],
    labels: dict[str, str],
) -> tuple[str | None, str, str]:
    """Resolve VCATTACH target to a conversation ID.

    Args:
        target_raw: what the user typed after VCATTACH
        current_id: the current conversation ID
        conversation_ids: all known conversation IDs
        labels: {conversation_id: label} dict

    Returns (target_id, target_label, error_message).
    target_id is None on error; error_message is empty on success.
    """
    # 1. Label match (case-insensitive)
    label_to_id = {v.lower(): (k, v) for k, v in labels.items()}
    if target_raw.lower() in label_to_id:
        tid, tlabel = label_to_id[target_raw.lower()]
        if tid == current_id:
            return None, tlabel, f"Already on conversation {tlabel}."
        return tid, tlabel, ""

    # 2. Exact UUID match
    if target_raw in conversation_ids:
        tlabel = labels.get(target_raw, target_raw)
        if target_raw == current_id:
            return None, tlabel, f"Already on conversation {tlabel}."
        return target_raw, tlabel, ""

    # 3. UUID prefix match
    matches = [c for c in conversation_ids if c.startswith(target_raw)]
    if len(matches) == 1:
        tid = matches[0]
        tlabel = labels.get(tid, target_raw)
        if tid == current_id:
            return None, tlabel, f"Already on conversation {tlabel}."
        return tid, tlabel, ""
    if len(matches) > 1:
        return None, target_raw, f"Ambiguous prefix '{target_raw}' matches {len(matches)} conversations."

    return None, target_raw, f"No conversation found matching '{target_raw}'. Use a conversation label or ID from the dashboard."


def execute_attach(
    old_id: str,
    target_id: str,
    store,
    registry_invalidate=None,
    reset_engine_state=None,
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
        registry_invalidate: callable(conversation_id) to evict + Redis invalidate
        reset_engine_state: callable(target_id) to reset target checkpoints
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

    # 3. Reset target checkpoint state (opt-in via caller)
    if callable(reset_engine_state):
        try:
            reset_engine_state(target_id)
        except Exception:
            logger.warning("VCATTACH: failed to reset target state %s", target_id[:12])

    # 4. Invalidate target session (local eviction + Redis key)
    if callable(registry_invalidate):
        try:
            registry_invalidate(target_id)
        except Exception:
            logger.warning("VCATTACH: failed to invalidate target session %s", target_id[:12])
