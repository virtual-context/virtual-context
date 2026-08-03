"""Community-engagement candidate pipeline.

Read-only. Selects and proves candidates; posts nothing and schedules
nothing. Scheduling and delivery live outside the engine.
"""

from .candidates import Candidate, Rejection, collect_candidates
from .channels import ChannelAllowlist, load_channel_allowlist
from .verify import MessageSourceRecord, verify_candidates

__all__ = [
    "Candidate",
    "ChannelAllowlist",
    "MessageSourceRecord",
    "Rejection",
    "collect_candidates",
    "load_channel_allowlist",
    "verify_candidates",
]
