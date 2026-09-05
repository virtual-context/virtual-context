"""Explicit storage guarantees required by the context engine.

Method presence alone does not establish a guarantee: legacy adapters inherit
no-op methods. Backends advertise guarantees and consumers require the subset
they actually need.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class StoreCapabilities:
    conversation_scope: bool = False
    canonical_sources: bool = False
    lifecycle_fencing: bool = False
    atomic_fact_mutation: bool = False
    audience_proofs: bool = False
    actor_cards: bool = False
    fact_links: bool = False
    streaming_embeddings: bool = False
    durable_exchanges: bool = False
    native_vectors: bool = False

    def require(self, *names: str) -> None:
        supported = {field.name for field in fields(self)}
        unknown = set(names) - supported
        if unknown:
            raise ValueError(f"Unknown storage capabilities: {', '.join(sorted(unknown))}")
        missing = [name for name in names if not getattr(self, name)]
        if missing:
            raise ValueError(f"Storage does not provide: {', '.join(missing)}")


RELATIONAL_CAPABILITIES = StoreCapabilities(
    conversation_scope=True,
    canonical_sources=True,
    lifecycle_fencing=True,
    atomic_fact_mutation=True,
    audience_proofs=True,
    actor_cards=True,
    fact_links=True,
    streaming_embeddings=True,
    durable_exchanges=True,
)


def capabilities_of(store) -> StoreCapabilities:
    value = getattr(store, "capabilities", None)
    return value if isinstance(value, StoreCapabilities) else StoreCapabilities()
