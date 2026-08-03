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
from .schedule import (
    SCHEDULE_ENABLED_BY_DEFAULT,
    ScheduleWindow,
    may_run_now,
    plan_day,
    preview_schedule,
)
from .broader import (
    BROADER_GENERATOR_GUIDANCE,
    CLAIM_CHECKER_SYSTEM_PROMPT,
    BroaderGeneratorNotConfigured,
    BroaderQuestion,
    generate_broader_question,
    validate_broader_question,
)
from .continuation import (
    CONTINUATION_HOOK_KINDS,
    HOOK_DETECTOR_SYSTEM_PROMPT,
    ContinuationHook,
    HookDetectorNotConfigured,
    find_continuation_hook,
    qualify_candidates,
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
    FIDELITY_JUDGE_SYSTEM_PROMPT,
    FidelityFixture,
    FidelityGateNotConfigured,
    FidelityVerdict,
    run_fidelity_gate,
)
from .report import DryRunReport
from .select import (
    REASON_BROADER_POOL_EMPTY,
    REASON_BROADER_POOL_NOT_CONFIGURED,
    REASON_BROADER_QUESTIONS_RECENT,
    REASON_NO_CANDIDATES,
    SelectionOutcome,
    apply_fidelity_outcome,
    rank_candidates,
    select_question,
)
from .timing import ThreadState, assess_thread, timed_followup_eligibility
from .verify import MessageSourceRecord, verify_candidates

__all__ = [
    "ADVERSARIAL_FIDELITY_FIXTURES",
    "FIDELITY_JUDGE_SYSTEM_PROMPT",
    "POST_CHANNEL_IDS",
    "REHEARSAL_CONFIG",
    "SOURCE_CHANNEL_IDS",
    "ENGAGEMENT_HISTORY_DDL",
    "InMemoryPostHistory",
    "PostRecord",
    "check_repetition",
    "fingerprint_distance",
    "topic_fingerprint",
    "SCHEDULE_ENABLED_BY_DEFAULT",
    "ScheduleWindow",
    "may_run_now",
    "plan_day",
    "preview_schedule",
    "BROADER_GENERATOR_GUIDANCE",
    "CLAIM_CHECKER_SYSTEM_PROMPT",
    "BroaderGeneratorNotConfigured",
    "BroaderQuestion",
    "generate_broader_question",
    "validate_broader_question",
    "CONTINUATION_HOOK_KINDS",
    "HOOK_DETECTOR_SYSTEM_PROMPT",
    "ContinuationHook",
    "HookDetectorNotConfigured",
    "find_continuation_hook",
    "qualify_candidates",
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
    "REASON_BROADER_POOL_EMPTY",
    "REASON_BROADER_POOL_NOT_CONFIGURED",
    "REASON_BROADER_QUESTIONS_RECENT",
    "REASON_NO_CANDIDATES",
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
