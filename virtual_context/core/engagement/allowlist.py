"""The shipped channel configuration for the Men guild.

Kept as data with one entry point, so the boundary is a thing that exists
and can be asserted against, rather than an argument each caller assembles
for itself. A caller that builds its own dict can get it wrong silently; a
shipped default can be pinned by a test.

Sourcing and posting are separate lists on purpose. Every candidate is taken
from a public channel, and until live posting is approved the only
destination is the private rehearsal channel. The rehearsal channel is
deliberately absent from the source list: it is private, and its traffic is
a single operator rather than community conversation, so material taken from
it would be neither publishable nor representative.
"""

from __future__ import annotations

from .channels import ChannelAllowlist, load_channel_allowlist

GUILD_CONVERSATION_ID = "sk:agent:vast:discord:guild:1524917037191925871"

# Public biohacking channels. Sourceable, never a destination until approved.
P3PTIDES = "1524917968440524990"
GENERAL = "1524917037787250834"
FREE_T_TOTAL_T = "1524964360030785686"
RATE_MY_STACK = "1530567788949798963"
FITNESS = "1524918613008580768"

# Private rehearsal channel. A destination only, never a source.
VASTTEST = "1524946242499514418"

CHANNEL_LABELS = {
    P3PTIDES: "#p3ptides",
    GENERAL: "#general",
    FREE_T_TOTAL_T: "#free-t-total-t",
    RATE_MY_STACK: "#rate-my-stack",
    FITNESS: "#fitness",
    VASTTEST: "#vasttest",
}

SOURCE_CHANNEL_IDS = (
    P3PTIDES, GENERAL, FREE_T_TOTAL_T, RATE_MY_STACK, FITNESS,
)

# Rehearsal only. Widening this is the change that enables posting into a
# live community channel, so it is deliberately a one-line, reviewable edit
# rather than something a caller can arrive at by assembling its own config.
POST_CHANNEL_IDS = (VASTTEST,)

REHEARSAL_CONFIG = {
    "source_channel_ids": list(SOURCE_CHANNEL_IDS),
    "post_channel_ids": list(POST_CHANNEL_IDS),
    "labels": dict(CHANNEL_LABELS),
}


def rehearsal_allowlist() -> ChannelAllowlist:
    """The shipped allowlist: public sources, rehearsal-only destination."""
    return load_channel_allowlist(REHEARSAL_CONFIG)
