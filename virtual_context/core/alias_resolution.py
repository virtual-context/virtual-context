"""Multi-hop conversation_aliases resolver.

Distinct from ``Store.resolve_conversation_alias`` (single-hop). Used by
``VirtualContextEngine.__init__`` and lossless-restart rebind sites to
walk the alias chain to its terminal target before binding state.

Per the engine-alias-resolution spec, alias resolution lives in engine
content semantics; transport layers (cloud REST, MCP, library embedders,
proxy registries) stay pass-through. This module is the single
authoritative resolution layer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Cap per-hop fan-out for ``compute_reverse_dependents`` BFS.
#:
#: Bounds the worst-case event payload size when a hot target accumulates
#: thousands of incoming aliases. Truncation is logged at WARNING so ops
#: can surface the discovered-vs-kept gap; truncated dependents heal on
#: next state-construct via the engine ``__init__`` resolver.
MAX_REVERSE_DEPENDENT_FAN_OUT = 256


class AliasResolutionError(RuntimeError):
    """Raised by ``walk_conversation_alias_chain`` on cycle / max_hops.

    Carries the chain of conversation_ids visited so far (12-char-truncated
    when surfaced via ``EngineConstructionError``) for ops debugging.
    """

    def __init__(self, *, reason: str, chain: list[str]) -> None:
        super().__init__(
            f"alias resolution failed: {reason} "
            f"(chain {' → '.join(c[:12] for c in chain)})"
        )
        self.reason = reason
        self.chain = list(chain)


def walk_conversation_alias_chain(
    store: Any,
    conversation_id: str,
    *,
    max_hops: int = 8,
) -> str:
    """Walk single-hop ``conversation_aliases`` edges to the terminal target.

    Returns the terminal id. Returns input unchanged when the input has
    no outgoing alias OR when the store does not implement
    ``resolve_conversation_alias`` (defensive backwards compat for
    minimal Store implementations such as filesystem-only test fixtures).

    Args:
        store: Any object exposing ``resolve_conversation_alias(alias_id) ->
            str | None``. Stores without the method are treated as
            no-alias (return input).
        conversation_id: The id to resolve. Empty string returns empty.
        max_hops: Maximum chain depth before raising ``max_hops``. Default 8.

    Raises:
        AliasResolutionError(reason='cycle', chain=...): when the walk
            revisits a node (including self-loop on the input).
        AliasResolutionError(reason='max_hops', chain=...): when the chain
            exceeds ``max_hops``.
        Any exception raised by ``store.resolve_conversation_alias`` is
            re-raised verbatim — caller decides whether to map to
            ``EngineConstructionError(reason='transient_store_error')``
            or to retry.
    """
    if not conversation_id:
        return conversation_id
    resolve = getattr(store, "resolve_conversation_alias", None)
    if not callable(resolve):
        return conversation_id

    seen: set[str] = {conversation_id}
    chain: list[str] = [conversation_id]
    current = conversation_id
    for _ in range(max_hops):
        nxt = resolve(current)
        if not nxt:
            return current
        if nxt in seen:
            chain.append(nxt)
            raise AliasResolutionError(reason="cycle", chain=chain)
        seen.add(nxt)
        chain.append(nxt)
        current = nxt
    raise AliasResolutionError(reason="max_hops", chain=chain)


def compute_reverse_dependents(
    store: Any,
    target_id: str,
    *,
    max_hops: int = 8,
) -> list[str]:
    """Return every conv_id whose terminal walk would reach ``target_id`` today.

    Used to populate the ``alias_created`` / ``alias_deleted`` invalidation
    event payload's ``reverse_dependents`` field. Walks the incoming-edge
    graph BFS from ``target_id``, capped at ``max_hops`` levels of depth
    and ``MAX_REVERSE_DEPENDENT_FAN_OUT`` per-level fan-out (truncation
    logged at WARNING).

    Returns BFS-order with per-level alphabetical sort. Determinism is
    per-level, not globally sorted; consumers needing lexicographic order
    sort at receive time. Mid-walk-snapshot semantic per spec class-2:
    concurrent VCATTACH commits visible only on the next walk; missed
    aliases are at-least-once safe (heal via the resolver on next request).

    Stores without ``list_conversation_aliases_by_target`` return an empty
    list (defensive backcompat for custom Store backends).

    Args:
        store: Any object exposing
            ``list_conversation_aliases_by_target(target_id) -> list[str]``.
        target_id: The terminal id to walk incoming aliases from.
        max_hops: Maximum BFS depth. Default 8 (matches walker bound).

    Returns:
        BFS-order list of source ids. Empty if no incoming aliases or the
        store lacks the lookup method.
    """
    list_by_target = getattr(store, "list_conversation_aliases_by_target", None)
    if not callable(list_by_target):
        return []
    seen: set[str] = {target_id}
    out: list[str] = []
    frontier: list[str] = [target_id]
    for hop_idx in range(max_hops):
        if not frontier:
            break
        next_frontier: list[str] = []
        for node in frontier:
            incoming = sorted(list_by_target(node) or [])
            kept = 0
            for src in incoming:
                if src in seen:
                    continue
                if kept >= MAX_REVERSE_DEPENDENT_FAN_OUT:
                    logger.warning(
                        "alias reverse-dependent fan-out truncated",
                        extra={
                            "target_id": node[:12],
                            "hop": hop_idx,
                            "discovered": len(incoming),
                            "kept": MAX_REVERSE_DEPENDENT_FAN_OUT,
                        },
                    )
                    break
                seen.add(src)
                out.append(src)
                next_frontier.append(src)
                kept += 1
        frontier = next_frontier
    return out
