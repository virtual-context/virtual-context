"""Typed summarize outcomes for repairing stored segment summaries.

The compactor's own summarize path never fails loudly: a provider error
returns a source-text fallback and an unparseable response becomes the
summary. Those semantics are correct for compaction, where storing
SOMETHING beats storing nothing, and they are exactly wrong for a repair
tool, where a failure written as a "repair" strands the row as
plausible-looking garbage. This module makes one summarize call and
returns a typed outcome instead: the caller decides what a failure
means, and nothing here ever substitutes content of its own.

The request is built by ``DomainCompactor.build_segment_summary_request``
so the call is byte-identical to what compaction would send; only the
failure semantics differ.
"""

from __future__ import annotations

from typing import NamedTuple

from .compactor import DomainCompactor, SegmentSummaryRequest
from .llm_utils import parse_llm_json


class Generated(NamedTuple):
    """Parse succeeded and the summary field is a non-empty string."""

    summary: str
    usage: dict


class ProviderFailure(NamedTuple):
    """The completion call raised. No response exists; nothing was paid."""

    error: str


class Malformed(NamedTuple):
    """A response arrived but carries no usable summary.

    Parse failure, or parsed JSON whose summary is missing, non-string,
    or empty. Carries usage: the call was made and paid for, and cost
    reporting must count it.
    """

    raw_text: str
    usage: dict


SummarizeOutcome = Generated | ProviderFailure | Malformed


def summarize_segment_once(
    compactor: DomainCompactor,
    request: SegmentSummaryRequest,
) -> SummarizeOutcome:
    """One summarize call, one typed outcome, no fallback of any kind.

    The outcome set is exhaustive by construction: the call either raised
    (ProviderFailure), or returned text that parses to a non-empty string
    summary (Generated), or returned anything else (Malformed). A raw
    error page, a truncated response, and JSON with a missing or
    non-string summary are all Malformed, never Generated: the caller's
    acceptance gate must only ever see summaries the model actually
    produced as summaries.
    """
    try:
        response_text, usage = compactor.llm.complete(
            system=request.system,
            user=request.prompt,
            max_tokens=request.max_tokens,
        )
    except Exception as e:  # noqa: BLE001 - the boundary this type exists for
        return ProviderFailure(error=f"{type(e).__name__}: {e}")
    parsed = parse_llm_json(response_text)
    # parse_llm_json returns the PARSED VALUE for any valid JSON, not
    # only objects: a bare list, string, number, boolean, or null comes
    # back as itself. Every non-mapping shape is Malformed; assuming a
    # mapping here would let one odd-but-valid response crash the caller
    # instead of being counted.
    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    if isinstance(summary, str) and summary.strip():
        return Generated(summary=summary, usage=usage or {})
    return Malformed(raw_text=response_text, usage=usage or {})
