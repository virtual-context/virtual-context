"""Every store method the protocol declares must survive the composite.

``CompositeStore`` forwards by hand, one explicit method at a time, and has no
``__getattr__``. So a method added to a concrete backend and to the protocol,
but not here, does not fail at import or at construction. It fails at the call
site, in production, as an ``AttributeError`` on a store that looks complete.

That is how three methods of a shipped feature came to be unreachable in the
deployment while every test passed: the feature was exercised against a
concrete store, and the composite is only assembled in the deployed chain.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.store import ContextStore


def _public_methods(cls) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_the_composite_forwards_every_protocol_method():
    """The guard against the whole class, not against the three that broke.

    A hand-maintained forwarder drifts silently by construction. This is the
    only thing that makes the drift loud, and it must be an exhaustive
    comparison rather than a list someone remembers to extend.
    """
    missing = sorted(_public_methods(ContextStore) - _public_methods(CompositeStore))

    assert not missing, (
        "CompositeStore does not forward these protocol methods, so every call "
        "through the deployed store chain raises AttributeError: "
        + ", ".join(missing)
    )


def test_the_agent_quote_methods_are_reachable_through_the_composite():
    """Named explicitly as well as covered above, because these three are the
    ones whose absence made a deployed feature inert with no failing test."""
    for name in (
        "record_bot_outbound_messages",
        "is_bot_authored_message",
        "resolve_channel_namespace",
        "get_lifecycle_epoch_started_at",
    ):
        assert hasattr(CompositeStore, name), f"{name} is not forwarded"


def test_a_backend_without_the_ledger_yields_unknown_rather_than_success():
    """The fallbacks must not let a missing backend read as a working one."""
    composite = object.__new__(CompositeStore)
    composite._segments = object()  # a backend with none of these methods

    assert composite.is_bot_authored_message(
        tenant_id="t", agent_scope_id="s", conversation_id="c",
        platform="discord", account_id="a", channel_id="ch", message_id="m",
    ) is False
    assert composite.resolve_channel_namespace(
        conversation_id="c", channel_id="ch",
    ) is None
    assert composite.get_lifecycle_epoch_started_at("c") is None

    outcome = composite.record_bot_outbound_messages(
        tenant_id="t", agent_scope_id="s", conversation_id="c",
        observed=[{"platform": "discord"}],
    )
    assert outcome["accepted"] == 0, "a missing ledger reported an acceptance"
    assert outcome["unsupported"] == 1
