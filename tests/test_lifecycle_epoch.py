import pytest
from virtual_context.core.lifecycle_epoch import (
    LifecycleEpochMismatch, verify_epoch,
)


def test_mismatch_exception_carries_details():
    err = LifecycleEpochMismatch(conversation_id="c", expected=1, observed=2)
    assert err.conversation_id == "c"
    assert err.expected == 1
    assert err.observed == 2
    assert "1" in str(err)
    assert "2" in str(err)


def test_mismatch_message_truncates_conversation_id():
    err = LifecycleEpochMismatch(
        conversation_id="aaaaaaaaaaaabbbbbbbbbbbb",
        expected=1, observed=2,
    )
    assert "aaaaaaaaaaaa" in str(err)
    # Full 24-char id should NOT appear — only first 12.
    assert "aaaaaaaaaaaabbbbbbbbbbbb" not in str(err)


def test_verify_epoch_passes_when_match():
    # No exception.
    verify_epoch(conversation_id="c", expected=1, observed=1)


def test_verify_epoch_raises_on_mismatch():
    with pytest.raises(LifecycleEpochMismatch) as exc_info:
        verify_epoch(conversation_id="c", expected=1, observed=2)
    assert exc_info.value.expected == 1
    assert exc_info.value.observed == 2


def test_verify_epoch_raises_when_observed_is_lower():
    """Rare but possible if DB was rolled back or has a stale snapshot."""
    with pytest.raises(LifecycleEpochMismatch):
        verify_epoch(conversation_id="c", expected=2, observed=1)


def test_exception_is_subclass_of_base_exception():
    # Callers may broadly catch Exception; ensure our exception is normal.
    err = LifecycleEpochMismatch(conversation_id="c", expected=1, observed=2)
    assert isinstance(err, Exception)
