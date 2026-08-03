"""Community-engagement candidate pipeline.

Read-only. Selects and proves candidates; posts nothing and schedules
nothing. Scheduling and delivery live outside the engine.
"""

from .candidates import Candidate, Rejection, collect_candidates
from .channels import ChannelAllowlist, load_channel_allowlist
from .fidelity import (
    ADVERSARIAL_FIDELITY_FIXTURES,
    FidelityFixture,
    FidelityGateNotConfigured,
    FidelityVerdict,
    run_fidelity_gate,
)
from .report import DryRunReport
from .select import SelectionOutcome, rank_candidates, select_question
from .timing import ThreadState, assess_thread, timed_followup_eligibility
from .verify import MessageSourceRecord, verify_candidates

__all__ = [
    "ADVERSARIAL_FIDELITY_FIXTURES",
    "Candidate",
    "ChannelAllowlist",
    "DryRunReport",
    "FidelityFixture",
    "FidelityGateNotConfigured",
    "FidelityVerdict",
    "MessageSourceRecord",
    "Rejection",
    "SelectionOutcome",
    "ThreadState",
    "assess_thread",
    "collect_candidates",
    "load_channel_allowlist",
    "rank_candidates",
    "run_fidelity_gate",
    "select_question",
    "timed_followup_eligibility",
    "verify_candidates",
]
