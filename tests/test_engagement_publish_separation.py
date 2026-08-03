"""The compose path must have no route to a community channel.

This is the safety property of the staging design. With one widened list, a
bug anywhere in compose could put a question into a community channel with no
approval — the owner's approval would be a convention rather than a route.
Two lists make his approval the only way in by construction.

The tests that matter here are the ones that fail when someone merges the
lists or imports the wrong one, not the ones that pass today.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from virtual_context.core.engagement import PostRefused
from virtual_context.core.engagement.allowlist import (
    PUBLISH_CHANNEL_IDS, SOURCE_CHANNEL_IDS, STAGING_CHANNEL_IDS,
)

POSTER = pathlib.Path("virtual_context/core/engagement/poster.py")


def _imported_names(path: pathlib.Path) -> set[str]:
    """Names this module actually imports. Prose in a docstring is not one."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
    return names


class TestTheListsAreDisjoint:
    def test_no_channel_is_both(self):
        assert not set(STAGING_CHANNEL_IDS) & set(PUBLISH_CHANNEL_IDS), (
            "staging is reachable without approval; publishing must not be"
        )

    def test_publishing_targets_the_community_channels(self):
        assert set(PUBLISH_CHANNEL_IDS) == set(SOURCE_CHANNEL_IDS)

    def test_staging_is_a_single_private_channel(self):
        assert len(STAGING_CHANNEL_IDS) == 1


class TestTheComposePathCannotReachAPublishChannel:
    """Structural, not conventional."""

    def test_the_poster_does_not_import_the_publish_list(self):
        """The check that fails if someone merges or swaps the lists.

        Asserted on the import graph rather than by searching the text,
        because the module names PUBLISH_CHANNEL_IDS in an error message and
        a substring search would call that a violation — the same
        source-grep-for-a-behaviour mistake this suite keeps finding.
        """
        assert "PUBLISH_CHANNEL_IDS" not in _imported_names(POSTER), (
            "the compose path imported the publish list; it can now reach a "
            "community channel without approval"
        )

    def test_the_poster_imports_the_staging_list(self):
        assert "STAGING_CHANNEL_IDS" in _imported_names(POSTER)

    @pytest.mark.parametrize("channel", PUBLISH_CHANNEL_IDS)
    def test_every_publish_channel_is_refused_by_the_compose_path(
        self, channel, monkeypatch,
    ):
        """Behaviour, not just structure. Each community channel, by id."""
        import virtual_context.core.engagement.poster as poster_module

        monkeypatch.setattr(poster_module, "POSTING_ENABLED", True)
        with pytest.raises(PostRefused, match="staging destination"):
            poster_module.post_question(
                candidate=object(), question="anything?",
                channel_id=channel, verification=None,
                history=None, sender=lambda **kw: "1",
                now=None, question_type="timed",
            )

    def test_merging_the_lists_would_break_this_suite(self):
        """States the intent so the guard is not quietly weakened.

        If STAGING_CHANNEL_IDS ever contains a community channel, the
        disjointness test fails and the module-level assertion in allowlist.py
        raises at import — the failure arrives before any code runs, not at
        the moment something posts.
        """
        from virtual_context.core.engagement import allowlist

        source = pathlib.Path(allowlist.__file__).read_text()
        assert "assert not set(STAGING_CHANNEL_IDS) & set(PUBLISH_CHANNEL_IDS)" \
            in source, "the import-time separation assertion was removed"
