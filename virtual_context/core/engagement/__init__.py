"""Community-engagement candidate pipeline.

Read-only. Selects and proves candidates; posts nothing and schedules
nothing. Scheduling and delivery live outside the engine.
"""

from .allowlist import (
    POST_CHANNEL_IDS,
    SOURCE_CHANNEL_IDS,
    REHEARSAL_CONFIG,
    rehearsal_allowlist,
)
from .history import (
    ENGAGEMENT_HISTORY_DDL,
    InMemoryPostHistory,
    PostRecord,
    check_repetition,
    fingerprint_distance,
    topic_fingerprint,
)
from .candidates import Candidate, Rejection, collect_candidates
from .compose import (
    TONE_CONSTRAINTS,
    Draft,
    DraftComposerNotConfigured,
    compose_draft,
    strip_speaker_prefix,
)
from .channels import ChannelAllowlist, load_channel_allowlist
from .fidelity import (
    ADVERSARIAL_FIDELITY_FIXTURES,
    FidelityFixture,
    FidelityGateNotConfigured,
    FidelityVerdict,
    run_fidelity_gate,
)
from .report import DryRunReport
from .select import (
    SelectionOutcome,
    apply_fidelity_outcome,
    rank_candidates,
    select_question,
)
from .timing import ThreadState, assess_thread, timed_followup_eligibility
from .verify import MessageSourceRecord, verify_candidates

__all__ = [
    "ADVERSARIAL_FIDELITY_FIXTURES",
    "POST_CHANNEL_IDS",
    "REHEARSAL_CONFIG",
    "SOURCE_CHANNEL_IDS",
    "ENGAGEMENT_HISTORY_DDL",
    "InMemoryPostHistory",
    "PostRecord",
    "check_repetition",
    "fingerprint_distance",
    "topic_fingerprint",
    "Candidate",
    "rehearsal_allowlist",
    "strip_speaker_prefix",
    "compose_draft",
    "TONE_CONSTRAINTS",
    "DraftComposerNotConfigured",
    "Draft",
    "ChannelAllowlist",
    "DryRunReport",
    "FidelityFixture",
    "FidelityGateNotConfigured",
    "FidelityVerdict",
    "MessageSourceRecord",
    "Rejection",
    "SelectionOutcome",
    "ThreadState",
    "apply_fidelity_outcome",
    "assess_thread",
    "collect_candidates",
    "load_channel_allowlist",
    "rank_candidates",
    "run_fidelity_gate",
    "select_question",
    "timed_followup_eligibility",
    "verify_candidates",
]
