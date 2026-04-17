class LifecycleEpochMismatch(Exception):
    """Raised when an in-memory lifecycle_epoch does not match the DB's
    current epoch. Indicates the caller's handle is stale — the conversation
    was deleted and resurrected (or the epoch was externally bumped) since
    the caller loaded its view. Callers should rehydrate before retrying."""

    def __init__(self, *, conversation_id: str, expected: int, observed: int) -> None:
        self.conversation_id = conversation_id
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"Lifecycle epoch mismatch for {conversation_id[:12]}: "
            f"expected={expected}, observed={observed}. "
            "In-memory handle is stale; rehydrate before retrying."
        )


def verify_epoch(*, conversation_id: str, expected: int, observed: int) -> None:
    """Raise LifecycleEpochMismatch iff expected != observed."""
    if expected != observed:
        raise LifecycleEpochMismatch(
            conversation_id=conversation_id,
            expected=expected,
            observed=observed,
        )
