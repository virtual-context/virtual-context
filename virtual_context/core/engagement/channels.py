"""Which channels may be sourced from, and which may be posted to.

Two lists, never one. A channel that is a legitimate place to post is not
automatically a legitimate place to take a member's words from: the private
test channel is a valid target and must never be a source, both because it
is private and because a single operator wrote most of it, which would
dominate any naive sampling.

Membership is keyed on the immutable channel id. Labels drift — one channel
in this guild has already carried two different labels — so a label is
display text only and can never decide membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ChannelAllowlist:
    """An explicit, immutable-id-keyed allowlist for one guild."""

    source_channel_ids: frozenset[str] = frozenset()
    post_channel_ids: frozenset[str] = frozenset()
    labels: Mapping[str, str] = field(default_factory=dict)

    def may_source(self, channel_id: str) -> bool:
        """Whether a member's words may be taken from this channel."""
        return (channel_id or "") in self.source_channel_ids

    def may_post(self, channel_id: str) -> bool:
        """Whether a question may be posted to this channel."""
        return (channel_id or "") in self.post_channel_ids

    def label_for(self, channel_id: str) -> str:
        """Display label, or empty. Never used for a membership decision."""
        return self.labels.get(channel_id or "", "")


def load_channel_allowlist(config: Mapping[str, Any] | None) -> ChannelAllowlist:
    """Build an allowlist from configuration.

    Absent or empty configuration allows nothing. Failing closed matters
    here: a bug that produced an empty allowlist must stop the job, not
    silently widen it to every channel in the guild.
    """
    config = config or {}
    return ChannelAllowlist(
        source_channel_ids=frozenset(
            str(c) for c in (config.get("source_channel_ids") or []) if str(c)
        ),
        post_channel_ids=frozenset(
            str(c) for c in (config.get("post_channel_ids") or []) if str(c)
        ),
        labels={
            str(k): str(v)
            for k, v in (config.get("labels") or {}).items()
        },
    )
