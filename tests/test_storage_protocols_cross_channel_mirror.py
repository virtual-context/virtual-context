"""Storage-protocol parity tests for the cross-channel-mirror surface.

Asserts ``CompositeStore`` forwards ``has_any_alias`` and
``get_recent_canonical_turns`` to its segment-store delegate and
preserves the first-argument ``conversation_id`` shape (required by
cloud's tenant wrapper allowlist).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.store import ContextStore


def _composite_with_segments(segments) -> CompositeStore:
    # Minimal construct: only the segment-store delegate is exercised
    # for these two methods.
    composite = CompositeStore.__new__(CompositeStore)
    composite._segments = segments
    composite._search = MagicMock()
    composite._facts = MagicMock()
    return composite


def test_composite_forwards_has_any_alias_with_positional_conv_id() -> None:
    segments = MagicMock()
    segments.has_any_alias.return_value = True
    composite = _composite_with_segments(segments)
    assert composite.has_any_alias("conv-1") is True
    segments.has_any_alias.assert_called_once_with("conv-1")


def test_composite_forwards_get_recent_canonical_turns() -> None:
    segments = MagicMock()
    segments.get_recent_canonical_turns.return_value = ["row1", "row2"]
    composite = _composite_with_segments(segments)
    rows = composite.get_recent_canonical_turns("conv-1", limit=5)
    assert rows == ["row1", "row2"]
    segments.get_recent_canonical_turns.assert_called_once_with("conv-1", limit=5)


def test_composite_has_any_alias_returns_false_when_segment_lacks_method() -> None:
    """Defensive fallback for segment stores predating the protocol
    additions: the composite returns ``False`` rather than raising
    ``AttributeError``."""
    segments = SimpleNamespace()  # no has_any_alias attr
    composite = _composite_with_segments(segments)
    assert composite.has_any_alias("conv-1") is False


def test_composite_get_recent_returns_empty_when_segment_lacks_method() -> None:
    segments = SimpleNamespace()  # no get_recent_canonical_turns attr
    composite = _composite_with_segments(segments)
    assert composite.get_recent_canonical_turns("conv-1", limit=5) == []


def test_context_store_defensive_defaults() -> None:
    """Abstract ``ContextStore`` defines defensive defaults so backends
    that don't implement the methods (filesystem-only fixtures, graph
    secondary stores) still satisfy the protocol."""
    # Cannot instantiate ABC directly; verify via subclass that doesn't
    # override.

    class MinimalStore(ContextStore):
        # Implement only the abstractmethods needed to instantiate.
        def store_segment(self, *_a, **_kw): pass
        def get_segment(self, *_a, **_kw): return None
        def get_summary(self, *_a, **_kw): return None
        def update_segment(self, *_a, **_kw): pass
        def delete_segment(self, *_a, **_kw): pass
        def get_all_segments(self, *_a, **_kw): return []
        def get_all_tags(self, *_a, **_kw): return []
        def get_summaries_by_tags(self, *_a, **_kw): return []
        def search(self, *_a, **_kw): return []
        def search_full_text(self, *_a, **_kw): return []
        def get_segments_by_tags(self, *_a, **_kw): return []
        def get_conversation_stats(self, *_a, **_kw): return []
        def get_tag_aliases(self, *_a, **_kw): return []
        def set_tag_alias(self, *_a, **_kw): pass
        def cleanup(self, *_a, **_kw): pass
        def save_tag_summary(self, *_a, **_kw): pass
        def get_tag_summary(self, *_a, **_kw): return None
        def get_all_tag_summaries(self, *_a, **_kw): return []

    s = MinimalStore()
    assert s.has_any_alias("conv-x") is False
    assert s.get_recent_canonical_turns("conv-x", limit=5) == []
