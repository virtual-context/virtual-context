"""Render a dry-run report a human can audit before anything is posted.

The report is the artifact that earns or loses trust in this pipeline, so it
shows the evidence rather than the conclusion: the exact verified quote, the
immutable attribution fields the quote was proved against, the send time
decoded from the message id, and a line for EVERY discarded candidate with
the stage and reason that discarded it. A run that found nothing and a run
that broke must not look alike.

It also states the boundary of the check it performed. Attribution here is
proved by cross-checking two independently written records; that cannot see
a message edited or deleted since ingest. A reviewer needs to know which
question the report answers before deciding what it licenses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

_EDIT_DELETE_LIMITATION = (
    "LIMITATION: attribution is proved by cross-checking the canonical row "
    "against the independently attested source record captured at ingest. "
    "This CANNOT detect a message edited or deleted on Discord since then. "
    "Live source re-fetch is required before any live posting."
)


@dataclass
class DryRunReport:
    """Everything a reviewer needs, and nothing that mutates anything."""

    generated_at: datetime
    conversation_id: str
    channel_id: str
    channel_label: str = ""
    outcome_kind: str = "skip"
    # Which of the three question types produced this, and what earned the
    # tag. A reviewer cannot judge whether a name was fairly attached
    # without knowing that a continuation rests on a specific item and which
    # one; and a label that stays inside the candidate never reaches him.
    question_type: str = ""
    hook_kind: str = ""
    question: str = ""
    quote: str = ""
    attribution: dict = field(default_factory=dict)
    thread: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    rejections: list = field(default_factory=list)
    considered: int = 0
    skip_reason: str = ""
    skip_stage: str = ""
    ladder: list = field(default_factory=list)

    def apply_outcome(self, outcome) -> "DryRunReport":
        """Take kind, reason, stage AND the selector's own rejections.

        The selector can discover a reason of its own — the fallback being
        unconfigured, for instance — and that reason has to be counted like
        every other or it exists only in a sentence. Setting the outcome
        fields by hand leaves the merge to whoever remembers; doing both
        here means a caller that reports the outcome at all cannot lose the
        rejections that came with it.
        """
        self.outcome_kind = getattr(outcome, "kind", self.outcome_kind)
        candidate = getattr(outcome, "candidate", None)
        if candidate is not None:
            self.question_type = (
                getattr(candidate, "question_type", "") or self.outcome_kind
            )
            self.hook_kind = getattr(candidate, "hook_kind", "") or ""
        elif self.outcome_kind == "broader":
            self.question_type = "broader"
        self.skip_reason = getattr(outcome, "reason", "") or ""
        self.skip_stage = getattr(outcome, "skip_stage", "") or ""
        added = list(getattr(outcome, "added_rejections", ()) or ())
        if added:
            self.rejections = list(self.rejections) + added
        return self

    def render(self) -> str:
        lines: list[str] = []
        add = lines.append
        add("=" * 72)
        add("COMMUNITY ENGAGEMENT — DRY RUN (no messages sent)")
        add("=" * 72)
        add(f"generated_at : {self.generated_at.isoformat()}")
        add(f"conversation : {self.conversation_id}")
        label = f" ({self.channel_label})" if self.channel_label else ""
        add(f"channel      : {self.channel_id}{label}")
        add(f"outcome      : {self.outcome_kind}")
        if self.question_type:
            hook = f" (hook: {self.hook_kind})" if self.hook_kind else ""
            add(f"question type: {self.question_type}{hook}")
        add(f"considered   : {self.considered} candidate(s)")
        add("")

        if self.outcome_kind == "personal":
            add("VERIFIED QUOTE")
            add("-" * 72)
            add(f"  {self.quote}")
            add("")
            add("IMMUTABLE ATTRIBUTION (cross-checked against the source record)")
            add("-" * 72)
            for key in (
                "source_message_id", "author_id", "actor_id", "handle",
                "channel_id", "guild_id", "sent_at",
            ):
                if key in self.attribution:
                    add(f"  {key:<20} {self.attribution[key]}")
            add("")
            if self.thread:
                add("LATER-CONTEXT CHECKS")
                add("-" * 72)
                for key in ("resolved", "stance", "blocks_candidate", "reason"):
                    if key in self.thread:
                        add(f"  {key:<20} {self.thread[key]}")
                if self.thread.get("resolving_text"):
                    add(f"  resolving_text       {self.thread['resolving_text']}")
                add("")
            if self.fidelity:
                add("FIDELITY GATE")
                add("-" * 72)
                add(f"  faithful             {self.fidelity.get('faithful')}")
                if self.fidelity.get("reason"):
                    add(f"  reason               {self.fidelity['reason']}")
                add("")

        if self.outcome_kind in ("personal", "broader"):
            add("PROPOSED QUESTION")
            add("-" * 72)
            add(f"  {self.question}")
            add("")
        else:
            add("SKIPPED")
            add("-" * 72)
            add(f"  stage  : {self.skip_stage}")
            add(f"  reason : {self.skip_reason}")
            add("")

        if self.ladder:
            add("SELECTION LADDER")
            add("-" * 72)
            for label, count in self.ladder:
                add(f"  {label:<20} {count}")
            add("")

        # Counts first, examples second. A truncated list makes the commonest
        # reason look like the only reason, which is how a working boundary
        # gets misread as unexplained loss.
        add(f"WHY CANDIDATES WERE REJECTED ({len(self.rejections)} total)")
        add("-" * 72)
        if not self.rejections:
            add("  (none)")
        else:
            grouped: dict[tuple[str, str], list] = {}
            for rejection in self.rejections:
                key = (
                    getattr(rejection, "stage", ""),
                    getattr(rejection, "reason", ""),
                )
                grouped.setdefault(key, []).append(rejection)
            for (stage, reason), items in sorted(
                grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]),
            ):
                add(f"  [{stage}] {reason}: {len(items)}")
                for rejection in items[:2]:
                    detail = getattr(rejection, "detail", "")
                    suffix = f" — {detail}" if detail else ""
                    add(
                        f"      e.g. "
                        f"{getattr(rejection, 'canonical_turn_id', '')}{suffix}"
                    )
                if len(items) > 2:
                    add(f"      … and {len(items) - 2} more")
        add("")
        add(_EDIT_DELETE_LIMITATION)
        add("=" * 72)
        add("NO MESSAGES WERE SENT. NO STATE WAS WRITTEN.")
        add("=" * 72)
        return "\n".join(lines)
