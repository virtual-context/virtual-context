"""DomainCompactor: summarizes TaggedSegments using a configurable LLM."""

from __future__ import annotations

import fnmatch
import json
import logging
import time
from typing import Callable

from ..types import (
    CompactionResult,
    CompactorConfig,
    Fact,
    FactSignal,
    LLMProvider,
    Message,
    SegmentMetadata,
    StoredSummary,
    TaggedSegment,
    TagPromptRule,
    TagSummary,
    TemporalStatus,
    get_sender_name,
)
from .llm_utils import format_code_ref, normalize_code_refs, normalize_tag, parse_llm_json
from .telemetry import TelemetryLedger

logger = logging.getLogger(__name__)

TAG_SUMMARY_ROLLUP_PROMPT = """\
You are summarizing all stored context about the tag "{tag}".
Below are {count} segment summaries that each cover a portion of conversation
where "{tag}" was discussed. Roll them up into a single coherent summary that
preserves all key decisions, action items, entities, and names.
Keep the chronological progression. The summary should be comprehensive enough
that someone could resume the conversation from it.

Preserve the emotional and conversational tone across sessions — if the user was excited about
a breakthrough, frustrated with a problem, or casually exploring, that texture should survive
the rollup. A summary that reads like a dry changelog loses the human context that makes
retrieval feel natural.

IMPORTANT: When a segment mentions something personal the user disclosed about
themselves (experiences, preferences, life events, possessions, relationships),
ALWAYS preserve that disclosure in the summary even if it is tangential to the
tag's main topic. Personal disclosures are high-value context that must survive rollup.

CRITICAL — Any text involving numbers is mandatory and absolutely essential to the summary, always include them exactly as in the source.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number.

Also provide a "description": a concise paragraph (max 80 words) capturing who is
involved, what is being discussed, key facts, and distinctive details. This will
be shown as a topic label in a compact topic list — it is the reader's primary
way to decide which topics are relevant, so include enough detail to be useful.
Any text involving numbers is mandatory and absolutely essential to the description.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number (e.g. "2 hours" must stay "2 hours", not
"about an hour"; "$45" must stay "$45", not "around $50").
Always preserve the user's role and relationship to the topic — phrases like
"I led", "my project", "solo project", "I built", "I'm responsible for" are
critical personal context. Never paraphrase these into passive voice or drop them
in favor of technical details.

Target length: {target_tokens} tokens or fewer.

Segment summaries:
{segment_summaries}

Respond with JSON:
{{
  "summary": "...",
  "description": "concise topic paragraph, max 80 words",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."]
}}"""


CODE_TAG_SUMMARY_ROLLUP_PROMPT = """\
You are rolling up engineering context for the tag "{tag}" from {count} segment summaries.
Preserve codebase state, architecture decisions, deployments, regressions, rejected approaches,
and open work. Compress investigative narration unless it led to a concrete finding.

Merge rules:
- Preserve chronology when the same artifact changed more than once.
- Deduplicate repeated findings across segments.
- Keep the final state of files, functions, services, and config values explicit.
- Prefer a stable verb vocabulary when it fits: changed/modified, has bug/broken, fixed,
  added/created, removed/deleted, deployed, reverted.
- Carry forward concrete code references into a deduplicated "code_refs" list.

Target length: {target_tokens} tokens or fewer.

Segment summaries:
{segment_summaries}

Respond with JSON:
{{
  "summary": "...",
  "description": "concise topic paragraph, max 80 words",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."],
  "code_refs": [{{"file": "...", "line": 123, "symbol": "..."}}]
}}"""


DEFAULT_SUMMARY_PROMPT = """\
SESSION DATE: {session_date}

Summarize the following conversation segment (tags: {tags}).
Preserve: key decisions, action items, entities mentioned, specific data points,
and specific feature/concept names exactly as discussed (e.g. "cook mode", "dark theme", "rate limiter" —
do NOT generalize these into broader categories like "UI features" or "infrastructure").

CRITICAL — Any text involving numbers is mandatory and absolutely essential to the summary, always include them exactly as in the conversation.
Dates, prices, any number is important and should not be modified.
Never round, approximate, or paraphrase a number (e.g. "2 hours" must stay "2 hours", not
"about an hour"; "$45" must stay "$45", not "around $50").

When the user states what they are doing, have done, or where they keep/store something,
preserve that as a direct assertion, not as a plan or intention. Conversely, when the conversation
is about a future activity the summary must clearly indicate this is planning/discussion, not a completed activity.

Capture the tone and texture of the conversation — was it casual/playful, urgent/stressed,
analytical/technical, emotional/vulnerable, collaborative/brainstorming? Preserve the user's
voice: if they were sarcastic, enthusiastic, frustrated, or reflective, that should come
through in the summary, not be flattened into neutral reportage.

Always preserve the user's role and relationship to the topic — phrases like
"I led", "my project", "solo project", "I built", "I'm responsible for" are
critical personal context. Never paraphrase these into passive voice or drop them
in favor of technical details.

When speakers are identified by name in the conversation (e.g. "Alice (14:30):"), always use their
actual name in the summary. Never replace a named speaker with "User" or "the user".
"Assistant" is a valid speaker label — do not replace it with a person's name from elsewhere.

Be concise but retain enough detail that the conversation could be resumed from this summary.
The summary should be {target_tokens} tokens or fewer.

Conversation:
{conversation_text}

Respond with JSON:
{{
  "summary": "...",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."],
  "date_references": ["..."],
  "refined_tags": ["tag1", "tag2"],
  "related_tags": ["alternate-term1", "alternate-term2"]
}}

For "related_tags", generate 3-8 alternate terms someone might use to refer to these
concepts later (e.g. if discussing "materialized views", related_tags might include
"caching", "precomputed", "feed-optimization", "query-cache").

Also extract facts from the RAW CONVERSATION TEXT above (not from your summary).
The summary may omit details — facts must capture ALL substantive information
from every speaker in the conversation, even details not included in the summary.
For each fact:
- "subject": who — use the actual name when conversation metadata identifies the sender (e.g. if metadata shows sender "Bob", the subject is "Bob", not "user"). When no name is available, use "user". For people mentioned but not speaking, use their name.
- "verb": the EXACT action verb from the conversation text (e.g. "led", "built", "prefers", "lives in", "ordered")
  VERB RULE: Use the verb that matches the actual event described.
  When someone says "we were given X", the verb is "were given" — NOT "mentioned" or "discussed".
  When someone says "I went to X", the verb is "went to" — NOT "talked about".
  Only use "mentioned"/"discussed" when the conversation is genuinely about referencing
  something without doing it (e.g. "I mentioned to my boss..." or "we talked about maybe going...").
- "object": what (specific noun phrase — preserve ALL numbers, names, dates, amounts exactly)
  For experience facts: derive the object from the user's own declarative sentence
  ("I went to X", "I visited X", "I got back from X"). A place or activity mentioned
  in a question or comparison is not a standalone experience fact — it is context.
- "status": one of: active, completed, planned, abandoned, recurring
- "fact_type": classify as "personal" (user's life, identity, preferences, plans),
  "experience" (assistant-provided info the user engaged with), or
  "world" (facts about other people, places, things in the user's world)
- "what": one full sentence capturing the complete fact with ALL specifics preserved.
  WRONG: "User has a personal best time." RIGHT: "User has a personal best 5K time of 27:12."
- "who": ALL people involved (populate when present, empty string if n/a)
  WHO RULE: Resolve pronouns. "We" in a speaker's message means the speaker + someone.
  Use preceding context to determine who. If unclear, write "speaker and companion".
- "when": the calendar date this event occurred, resolved from context.
  DATE RULES — use SESSION DATE ({session_date}) as your reference point:
  "today" / "this morning" / "just now" → use {session_date}.
  "yesterday" → the day before {session_date}.
  "last Saturday" → the most recent Saturday before {session_date}.
  "last week" → approximately 7 days before {session_date}.
  "next month" → the month after {session_date}.
  "last year" → the year before {session_date}.
  Always write the RESOLVED calendar date (e.g. "2023-05-20"), NOT the relative term.
  If truly unresolvable ("recently", "a while ago", no temporal reference) → use "".
- "where": location (populate when present, empty string if n/a)
- "why": context or significance (populate when present, empty string if n/a)
When the same event is disclosed directly in the current turn AND referenced in passing
(e.g. asking a follow-up question about it), emit ONE fact using the direct disclosure
as the primary source, enriched with any additional detail from the reference.
Extract the FACT behind the question, not the conversational act.
WRONG: "user asks about Cairo restaurants" RIGHT: "user wants to try authentic Egyptian food in Cairo"
Extract both EXPLICIT and IMPLIED facts. When someone refers to "the last photo/time
with someone", "sadly the last time together", "I'm sorry for your loss" — extract the
implied fact (e.g. the person passed away). When someone says "we were given a game" —
extract both the receiving and the implied playing.
If two signals describe the same event, emit one fact with the richest details.
Include "facts" in the JSON response.
Only extract facts with genuine substance. Skip greetings and filler."""


CODE_SUMMARY_PROMPT = """\
SESSION DATE: {session_date}

Summarize the following engineering conversation segment (tags: {tags}).

Preserve:
- What changed in the codebase: files, functions, classes, config keys, schemas, services
- Why it changed: bug, feature request, deployment need, or explicit user decision
- What was tried and rejected when it explains the final design
- Current state: what works now, what is still broken, what is deployed vs local only
- Open items: TODOs, deferred decisions, known follow-up fixes

Suppress:
- Emotional texture and conversational filler
- Investigation steps like reading files, grepping, or running commands unless they produced a finding
- Assistant-centered narration; capture the resulting artifact state instead

CRITICAL:
- Preserve all numbers exactly: dates, ports, durations, prices, versions, token counts
- Preserve file paths, symbols, config keys, env vars, and endpoint names verbatim

Extract facts about the resulting codebase state and the user's decisions.
Good subjects are files, functions, services, config keys, or "User" for explicit preferences/decisions.
Never use "Assistant" as a subject.
Prefer a stable verb vocabulary when it fits:
- changed / modified
- has bug / broken
- fixed
- added / created
- removed / deleted
- deployed
- reverted

Also extract "code_refs": a deduplicated list of the files/functions/classes materially discussed or changed.
Each ref should be an object like {{"file": "path/to/file.py", "line": 123, "symbol": "function_name"}}.
Use the best available file path even if line/symbol is unknown.

The summary should be {target_tokens} tokens or fewer.

Conversation:
{conversation_text}

Respond with JSON:
{{
  "summary": "...",
  "entities": ["..."],
  "key_decisions": ["..."],
  "action_items": ["..."],
  "date_references": ["..."],
  "refined_tags": ["tag1", "tag2"],
  "related_tags": ["alternate-term1", "alternate-term2"],
  "facts": [{{"subject": "...", "verb": "...", "object": "...", "status": "...", "fact_type": "personal|experience|world", "what": "..."}}],
  "code_refs": [{{"file": "...", "line": 123, "symbol": "..."}}]
}}

Only output valid JSON."""

CODE_FACT_EXTRACTION_PROMPT = """

CODING CONVERSATION MODE:
- Do NOT extract intermediary/investigative actions (e.g. "Assistant ran the tests", "Assistant checked the file").
- DO extract: conclusions, findings, discoveries, decisions made, user preferences, configuration values, deployment details, architectural choices, bugs found and their fixes, tool/library choices.
- DO extract: what was built, fixed, changed, or implemented as a result of the conversation. Frame these as facts about the THING, not about the assistant. E.g. "Facts endpoint now supports date_created sorting" not "Assistant added sorting to the endpoint".
- DO extract: facts about the user's projects, infrastructure, workflows, and environment.
- The subject for findings should be the THING, not "Assistant" — e.g. "Deploy target is Linode at 45.33.74.201" not "Assistant deployed to Linode".
- Prefer these verbs when they fit the actual event: changed/modified, has bug/broken, fixed, added/created, removed/deleted, deployed, reverted.
- Also emit "code_refs": a deduplicated list of concrete file/function/class refs that were materially discussed or changed.
  Each item should look like {"file": "path/to/file.py", "line": 123, "symbol": "function_name"}.
  Use line/symbol only when the conversation provides them confidently.
  Never emit more than 12 refs."""

CODE_MODE_FACT_PROMPT = CODE_FACT_EXTRACTION_PROMPT

def _collect_code_refs(*groups: object, max_refs: int = 12) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, int | None, str]] = set()
    for group in groups:
        for ref in normalize_code_refs(group, max_refs=max_refs):
            key = (
                ref.get("file", ""),
                ref.get("line") if isinstance(ref.get("line"), int) else None,
                ref.get("symbol", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(ref)
            if len(merged) >= max_refs:
                return merged
    return merged


class DomainCompactor:
    """Summarize each TaggedSegment independently using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: CompactorConfig,
        token_counter: Callable[[str], int] | None = None,
        model_name: str = "",
        tag_rules: list[TagPromptRule] | None = None,
        telemetry_ledger: TelemetryLedger | None = None,
        cost_tracker=None,  # deprecated, ignored — remove when callers updated
    ) -> None:
        self.llm = llm_provider
        self.config = config
        self.token_counter = token_counter or (lambda text: len(text) // 4)
        self.model_name = model_name
        self.tag_rules = tag_rules or []
        self._telemetry = telemetry_ledger

    def compact(
        self,
        segments: list[TaggedSegment],
        fact_signals_by_segment: dict[str, list[FactSignal]] | None = None,
        code_refs_by_segment: dict[str, list[dict]] | None = None,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[CompactionResult]:
        """Summarize each segment independently.

        Uses ThreadPoolExecutor for concurrent summarization when there are
        multiple segments. Falls back to sequential for single segments.

        *fact_signals_by_segment* maps segment.id → list of FactSignal
        collected from per-turn tagging.  Passed as hints to the compactor
        prompt for verification and consolidation.
        """
        signals = fact_signals_by_segment or {}
        code_refs = code_refs_by_segment or {}
        import sys as _sys
        import time as _time
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        logger.info(
            "Compacting %d segments (%d workers)...",
            len(segments), min(self.config.max_concurrent_summaries, len(segments)),
        )
        # Build preceding-conversation context for pronoun resolution.
        # Each segment gets the raw text of the preceding segments (up to
        # ~5000 chars) so the compactor can resolve pronouns like "we",
        # "they", "he/she" using nearby conversational context.
        _MAX_CONTEXT_CHARS = 5000
        prev_contexts: list[str] = [""]  # first segment has no predecessor
        for i in range(1, len(segments)):
            parts: list[str] = []
            total_chars = 0
            for j in range(i - 1, -1, -1):
                part = self._format_conversation(segments[j].messages)
                if total_chars + len(part) > _MAX_CONTEXT_CHARS:
                    # Add truncated tail of this segment to fill remaining budget
                    remaining = _MAX_CONTEXT_CHARS - total_chars
                    if remaining > 200:
                        parts.insert(0, part[-remaining:])
                    break
                parts.insert(0, part)
                total_chars += len(part)
            prev_contexts.append("\n---\n".join(parts))

        progress_started = _time.time()

        def _emit_progress(
            done_count: int,
            result: CompactionResult | None = None,
            *,
            heartbeat: bool = False,
            in_flight: int = 0,
        ) -> None:
            if not progress_callback or not segments:
                return
            elapsed = _time.time() - progress_started
            rate = done_count / elapsed if elapsed > 0 else 0.0
            eta_ms = int(((len(segments) - done_count) / rate) * 1000) if rate > 0 else None
            progress_callback(
                done_count,
                len(segments),
                result,
                phase="segment_compacting",
                phase_name="compactor",
                heartbeat=heartbeat,
                in_flight=in_flight,
                elapsed_ms=int(elapsed * 1000),
                eta_ms=eta_ms,
            )

        if len(segments) <= 1:
            # Sequential for single segment — no threading overhead
            _emit_progress(0, None, in_flight=len(segments))
            results: list[CompactionResult] = []
            for idx, segment in enumerate(segments, start=1):
                result = self._compact_one(
                    segment,
                    signals.get(segment.id, []),
                    code_refs.get(segment.id, []),
                    prev_context=prev_contexts[idx - 1],
                )
                results.append(result)
                _emit_progress(idx, result, in_flight=max(len(segments) - idx, 0))
            return results

        max_workers = min(self.config.max_concurrent_summaries, len(segments))

        results: list[CompactionResult] = [None] * len(segments)  # type: ignore[list-item]
        done_count = 0
        _compact_start = _time.time()
        heartbeat_interval_s = 2.0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {
                executor.submit(
                    self._compact_one,
                    segment,
                    signals.get(segment.id, []),
                    code_refs.get(segment.id, []),
                    prev_context=prev_contexts[i],
                ): i
                for i, segment in enumerate(segments)
            }
            _emit_progress(0, None, in_flight=len(pending))
            while pending:
                done, _ = wait(tuple(pending.keys()), timeout=heartbeat_interval_s, return_when=FIRST_COMPLETED)
                if not done:
                    _emit_progress(done_count, None, heartbeat=True, in_flight=len(pending))
                    continue
                for future in done:
                    idx = pending.pop(future)
                    done_count += 1
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Concurrent compaction failed for segment {idx}: {e}")
                        # Create a fallback result
                        segment = segments[idx]
                        conversation_text = self._format_conversation(segment.messages)
                        results[idx] = CompactionResult(
                            segment_id=segment.id,
                            primary_tag=segment.primary_tag,
                            tags=segment.tags,
                            summary=conversation_text[:2000],
                            summary_tokens=self.token_counter(conversation_text[:2000]),
                            original_tokens=self.token_counter(conversation_text),
                            compression_ratio=0.0,
                            full_text=conversation_text,
                            timestamp=segment.start_timestamp,
                        )
                    logger.info(
                        "  Segment %d/%d done (%s, %dt → %dt)",
                        done_count, len(segments),
                        results[idx].primary_tag,
                        results[idx].original_tokens,
                        results[idx].summary_tokens,
                    )
                    _elapsed = _time.time() - _compact_start
                    _rate = done_count / _elapsed if _elapsed > 0 else 0
                    _eta = int((len(segments) - done_count) / _rate) if _rate > 0 else 0
                    _sys.stderr.write(
                        f"\r  COMPACT: {done_count}/{len(segments)} segments | "
                        f"{_rate:.1f} seg/s | ETA {_eta}s   "
                    )
                    _sys.stderr.flush()
                    _emit_progress(done_count, results[idx], in_flight=len(pending))

        _sys.stderr.write("\n")
        _sys.stderr.flush()
        return results

    def _compact_one(
        self,
        segment: TaggedSegment,
        fact_signals: list[FactSignal] | None = None,
        code_refs: list[dict] | None = None,
        prev_context: str = "",
    ) -> CompactionResult:
        conversation_text = self._format_conversation(segment.messages)
        original_tokens = self.token_counter(conversation_text)

        # DIAGNOSTIC: log compactor input for each segment
        logger.info(
            "COMPACTOR_INPUT ref=%s tag=%s tokens=%d text_preview=\"%s\"",
            segment.id[:8], segment.primary_tag, original_tokens,
            conversation_text[:300].replace("\n", "\\n"),
        )

        target_tokens = max(
            self.config.min_summary_tokens,
            min(
                self.config.max_summary_tokens,
                int(original_tokens * self.config.summary_ratio),
            ),
        )

        # Check for custom prompt from tag rules
        custom_prompt = self._get_prompt_for_tags(segment.tags)

        # D1: build signal hints from per-turn tagger fact signals
        signals_text = ""
        if fact_signals:
            hint_lines = []
            for s in fact_signals:
                if s.subject and s.object:
                    line = f"- [{s.fact_type}] {s.subject} {s.verb} {s.object} ({s.status})"
                    if s.what:
                        line += f" — {s.what}"
                    hint_lines.append(line)
            if hint_lines:
                signals_text = (
                    "\n\nPer-turn fact signals (verify and consolidate with full context):\n"
                    + "\n".join(hint_lines)
                )

        refs_text = ""
        normalized_refs = normalize_code_refs(code_refs)
        if normalized_refs:
            ref_lines = [f"- {format_code_ref(ref)}" for ref in normalized_refs]
            refs_text = (
                "\n\nPer-turn code references (use only if they genuinely apply to this segment):\n"
                + "\n".join(ref_lines)
            )

        # Previous segment context for pronoun resolution — XML-tagged
        # to structurally separate it from the segment to summarize.
        context_block = ""
        if prev_context:
            context_block = (
                f"<context_for_pronoun_resolution_only>\n"
                f"The following is prior conversation for resolving pronouns ONLY.\n"
                f"DO NOT include any information from this block in your summary.\n"
                f"{prev_context}\n"
                f"</context_for_pronoun_resolution_only>\n\n"
            )

        # Wrap conversation text in <segment_to_summarize> tags when
        # prev_context is present, so the LLM can distinguish the two.
        # Note: the DEFAULT_SUMMARY_PROMPT template already has "Conversation:\n"
        # before {conversation_text}, so conv_block should NOT add it again.
        if prev_context:
            conv_block = (
                f"<segment_to_summarize>\n"
                f"{conversation_text}\n"
                f"</segment_to_summarize>"
            )
        else:
            conv_block = conversation_text

        if custom_prompt:
            prompt = (
                f"{custom_prompt}\n\n"
                f"Target length: {target_tokens} tokens or fewer.\n\n"
                f"{context_block}"
                f"{conv_block}"
                f"{signals_text}"
                f"{refs_text}\n\n"
                'Respond with JSON: {{"summary": "...", "entities": ["..."], '
                '"key_decisions": ["..."], "action_items": ["..."], '
                '"date_references": ["..."], "refined_tags": ["tag1", "tag2"], '
                '"facts": [...], "code_refs": [...]}}'
            )
            if getattr(self.config, "code_mode", False):
                prompt += CODE_FACT_EXTRACTION_PROMPT
        else:
            tags_str = ", ".join(segment.tags) if segment.tags else segment.primary_tag
            prompt_template = (
                CODE_SUMMARY_PROMPT
                if getattr(self.config, "code_mode", False)
                else DEFAULT_SUMMARY_PROMPT
            )
            prompt = prompt_template.format(
                tags=tags_str,
                target_tokens=target_tokens,
                conversation_text=context_block + conv_block,
                session_date=segment.session_date or "",
            )
            if signals_text:
                prompt += signals_text
            if refs_text:
                prompt += refs_text
            if getattr(self.config, "code_mode", False):
                prompt += CODE_FACT_EXTRACTION_PROMPT

        system = (
            "You are a conversation summarizer. Output valid JSON only. "
            "No markdown fences, no extra text."
        )
        if prev_context:
            system += (
                " ONLY summarize content inside <segment_to_summarize> tags."
                " NEVER include information from <context_for_pronoun_resolution_only> in your summary."
            )

        try:
            t0 = time.time()
            response_text, usage = self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_summary_tokens + self.config.llm_token_overhead,
            )
            duration_ms = (time.time() - t0) * 1000
            self._log_usage("segment_summarize", duration_ms=duration_ms, usage=usage)
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM summarization failed for segment {segment.id}: {e}")
            parsed = {
                "summary": conversation_text[:target_tokens * 4],
                "entities": [],
                "key_decisions": [],
                "action_items": [],
                "date_references": [],
                "refined_tags": segment.tags,
            }

        summary = parsed.get("summary", "")
        summary_tokens = self.token_counter(summary)

        # DIAGNOSTIC: log compactor output
        logger.info(
            "COMPACTOR_OUTPUT ref=%s tag=%s summary_preview=\"%s\"",
            segment.id[:8], segment.primary_tag,
            summary[:200].replace("\n", "\\n"),
        )

        # Always preserve original segment tags (source of truth from TurnTagIndex).
        # LLM refined_tags and related_tags can ADD new tags but never remove originals.
        refined_tags = parsed.get("refined_tags", [])
        related_tags = parsed.get("related_tags", [])
        all_tags = set(segment.tags)
        if refined_tags:
            all_tags.update(self._normalize_tag_list(refined_tags))
        if related_tags:
            all_tags.update(self._normalize_tag_list(related_tags))
        refined_tags = sorted(all_tags)

        parsed_code_refs = _collect_code_refs(parsed.get("code_refs"), normalized_refs)

        metadata = SegmentMetadata(
            entities=parsed.get("entities", []),
            key_decisions=parsed.get("key_decisions", []),
            action_items=parsed.get("action_items", []),
            date_references=parsed.get("date_references", []),
            code_refs=parsed_code_refs,
            turn_count=segment.turn_count,
            time_span=(segment.start_timestamp, segment.end_timestamp),
            session_date=segment.session_date,
        )

        messages_dicts = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in segment.messages
        ]

        # D1: Parse extracted facts
        valid_statuses = {e.value for e in TemporalStatus}
        raw_facts = parsed.get("facts", [])
        facts: list[Fact] = []

        def _str(val) -> str:
            if isinstance(val, list):
                return ", ".join(str(v) for v in val)
            return str(val) if val else ""

        if isinstance(raw_facts, list):
            for f in raw_facts:
                if not isinstance(f, dict) or not f.get("subject") or not f.get("object"):
                    continue
                status = f.get("status", "active")
                if status not in valid_statuses:
                    status = "active"

                # Resolve when_date: LLM first, then deterministic fallback
                raw_when = _str(f.get("when", ""))
                sess_date = segment.session_date or ""
                raw_what = _str(f.get("what", ""))
                from virtual_context.ingest.date_resolver import (
                    resolve_relative_date, normalize_fact_text,
                )
                # Validate raw_when looks like a real date (contains a digit).
                # LLMs sometimes echo the field name (e.g. "when") or other
                # garbage; treat those as blank so the fallback kicks in.
                raw_when_valid = raw_when and any(c.isdigit() for c in raw_when)
                if not raw_when_valid or raw_when == sess_date:
                    resolved = resolve_relative_date(raw_what, sess_date)
                    when_date = resolved or sess_date
                else:
                    when_date = raw_when
                # Always normalize relative terms in the fact text
                fact_what = normalize_fact_text(raw_what, sess_date)

                facts.append(Fact(
                    subject=_str(f.get("subject", "")),
                    verb=_str(f.get("verb", f.get("role", ""))),
                    object=_str(f.get("object", "")),
                    status=status,
                    what=fact_what,
                    who=_str(f.get("who", "")),
                    when_date=when_date,
                    where=_str(f.get("where", "")),
                    why=_str(f.get("why", "")),
                    fact_type=f.get("fact_type", "personal"),
                    tags=refined_tags,
                    session_date=sess_date,
                ))

        return CompactionResult(
            segment_id=segment.id,
            primary_tag=segment.primary_tag,
            tags=refined_tags,
            summary=summary,
            summary_tokens=summary_tokens,
            original_tokens=original_tokens,
            compression_ratio=summary_tokens / original_tokens if original_tokens > 0 else 0.0,
            metadata=metadata,
            full_text=conversation_text,
            messages=messages_dicts,
            timestamp=segment.start_timestamp,
            facts=facts,
        )

    @staticmethod
    def _normalize_tag_list(tags: list) -> list[str]:
        result = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            t = normalize_tag(tag)
            if t:
                result.append(t)
        return result

    def _get_prompt_for_tags(self, tags: list[str]) -> str | None:
        for rule in self.tag_rules:
            for tag in tags:
                if fnmatch.fnmatch(tag, rule.match) and rule.summary_prompt:
                    return rule.summary_prompt
        return None

    def _log_usage(self, event_type: str, duration_ms: float = 0.0, usage: dict | None = None) -> None:
        """Log LLM token usage to the telemetry ledger.

        *usage* should be the usage dict returned by ``complete()``.
        Falls back to ``self.llm.last_usage`` (deprecated) when not provided.
        """
        if not self._telemetry:
            return
        if usage is None:
            usage = getattr(self.llm, "last_usage", {})
        if not usage:
            return
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        if input_tokens or output_tokens:
            self._telemetry.log(
                component="compactor",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                detail=event_type,
            )

    def _format_conversation(self, messages: list[Message]) -> str:
        # Build tool_use_id -> name map for tool_result resolution
        tool_name_map: dict[str, str] = {}
        for m in messages:
            if m.raw_content:
                for block in m.raw_content:
                    if block.get("type") == "tool_use" and "id" in block:
                        tool_name_map[block["id"]] = block.get("name", "")

        lines: list[str] = []
        # Prepend session date header from the first timestamped message
        first_ts = next((m.timestamp for m in messages if m.timestamp), None)
        if first_ts:
            lines.append(f"[Session: {first_ts.strftime('%B %d, %Y %I:%M %p')}]")
        for m in messages:
            ts = ""
            if m.timestamp:
                ts = f" ({m.timestamp.strftime('%H:%M')})"
            label = get_sender_name(m.metadata) or m.role.capitalize()
            if m.raw_content:
                parts = self._render_raw_content(m.raw_content, tool_name_map)
                lines.append(f"{label}{ts}: {parts}")
            else:
                lines.append(f"{label}{ts}: {m.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _render_raw_content(blocks: list[dict], tool_name_map: dict[str, str]) -> str:
        parts: list[str] = []
        for block in blocks:
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                inputs = block.get("input", {})
                parts.append(f"Assistant called {name}({json.dumps(inputs)})")
            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                name = tool_name_map.get(tool_use_id)
                content = block.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                if name:
                    parts.append(f"Tool result for {name}: {content}")
                else:
                    parts.append(f"Tool result ({tool_use_id}): {content}")
            elif btype == "thinking":
                pass  # skip thinking blocks
            else:
                text = block.get("text", block.get("content", ""))
                if text:
                    parts.append(f"{btype}: {text}")
        return "\n".join(parts) if parts else ""

    def _parse_response(self, response: str) -> dict:
        result = parse_llm_json(response)
        if result:
            return result
        # parse_llm_json failed — raw response becomes summary fallback.
        # Safety net: if the raw text looks like our JSON schema, try
        # a lenient extraction of just the "summary" field so we don't
        # store the entire JSON blob as the summary text.
        text = response.strip()
        if '"summary"' in text:
            import re as _re
            m = _re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m:
                logger.info(
                    "Recovered summary field from unparseable JSON response (len=%d)",
                    len(text),
                )
                return {
                    "summary": m.group(1).replace('\\"', '"').replace("\\n", "\n"),
                    "entities": [],
                    "key_decisions": [],
                    "action_items": [],
                    "date_references": [],
                    "refined_tags": [],
                    "code_refs": [],
                }
        return {
            "summary": text,
            "entities": [],
            "key_decisions": [],
            "action_items": [],
            "date_references": [],
            "refined_tags": [],
            "code_refs": [],
        }

    def compact_tag_summaries(
        self,
        cover_tags: list[str],
        tag_to_summaries: dict[str, list[StoredSummary]],
        tag_to_turns: dict[str, list[int]],
        existing_tag_summaries: dict[str, TagSummary],
        max_turn: int,
    ) -> list[TagSummary]:
        """Build or update tag summaries for cover tags.

        Only rebuilds tag summaries that are stale (``covers_through_turn`` <
        ``max_turn``) or missing.  Uses ThreadPoolExecutor for concurrency.
        """
        tags_to_build: list[str] = []
        for tag in cover_tags:
            summaries = tag_to_summaries.get(tag, [])
            if not summaries:
                continue  # no segment summaries yet
            existing = existing_tag_summaries.get(tag)
            if existing is None or existing.covers_through_turn < max_turn:
                tags_to_build.append(tag)

        if not tags_to_build:
            return []

        logger.info("Building tag summaries for %d tags: %s", len(tags_to_build), tags_to_build[:10])

        if len(tags_to_build) <= 1:
            return [
                self._build_one_tag_summary(
                    tag,
                    tag_to_summaries.get(tag, []),
                    tag_to_turns.get(tag, []),
                    max_turn,
                )
                for tag in tags_to_build
            ]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        import sys as _sys
        import time as _time

        max_workers = min(self.config.max_concurrent_summaries, len(tags_to_build))
        results: list[TagSummary | None] = [None] * len(tags_to_build)
        _ts_start = _time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._build_one_tag_summary,
                    tag,
                    tag_to_summaries.get(tag, []),
                    tag_to_turns.get(tag, []),
                    max_turn,
                ): i
                for i, tag in enumerate(tags_to_build)
            }
            done_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                done_count += 1
                try:
                    results[idx] = future.result()
                    logger.info(
                        "  Tag summary %d/%d done (%s, %dt)",
                        done_count, len(tags_to_build),
                        tags_to_build[idx], results[idx].summary_tokens,
                    )
                    _elapsed = _time.time() - _ts_start
                    _rate = done_count / _elapsed if _elapsed > 0 else 0
                    _eta = int((len(tags_to_build) - done_count) / _rate) if _rate > 0 else 0
                    _sys.stderr.write(
                        f"\r  TAG_SUMMARIES: {done_count}/{len(tags_to_build)} tags | "
                        f"{_rate:.1f} tag/s | ETA {_eta}s   "
                    )
                    _sys.stderr.flush()
                except Exception as e:
                    tag = tags_to_build[idx]
                    logger.error(f"Tag summary build failed for '{tag}': {e}")
                    summaries = tag_to_summaries.get(tag, [])
                    fallback_text = "\n\n".join(s.summary for s in summaries)[:4000]
                    results[idx] = TagSummary(
                        tag=tag,
                        summary=fallback_text,
                        summary_tokens=self.token_counter(fallback_text),
                        source_segment_refs=[s.ref for s in summaries],
                        source_turn_numbers=sorted(set(tag_to_turns.get(tag, []))),
                        code_refs=_collect_code_refs(
                            *[getattr(s.metadata, "code_refs", []) for s in summaries]
                        ),
                        covers_through_turn=max_turn,
                    )

        _sys.stderr.write("\n")
        _sys.stderr.flush()
        return [r for r in results if r is not None]

    def _build_one_tag_summary(
        self,
        tag: str,
        summaries: list[StoredSummary],
        turn_numbers: list[int],
        max_turn: int,
    ) -> TagSummary:
        combined = "\n\n---\n\n".join(
            (
                f"[Segment {s.ref}, tags: {', '.join(s.tags)}]\n"
                f"{s.summary}"
                + (
                    "\nRefs: " + ", ".join(format_code_ref(ref) for ref in s.metadata.code_refs)
                    if getattr(s.metadata, "code_refs", None)
                    else ""
                )
            )
            for s in summaries
        )
        combined_tokens = self.token_counter(combined)
        target_tokens = max(
            self.config.min_summary_tokens,
            min(self.config.max_summary_tokens, int(combined_tokens * self.config.summary_ratio)),
        )

        prompt_template = (
            CODE_TAG_SUMMARY_ROLLUP_PROMPT
            if getattr(self.config, "code_mode", False)
            else TAG_SUMMARY_ROLLUP_PROMPT
        )
        prompt = prompt_template.format(
            tag=tag,
            count=len(summaries),
            target_tokens=target_tokens,
            segment_summaries=combined,
        )

        system = (
            "You are a conversation summarizer. Output valid JSON only. "
            "No markdown fences, no extra text."
        )

        try:
            t0 = time.time()
            response_text, usage = self.llm.complete(
                system=system,
                user=prompt,
                max_tokens=self.config.max_summary_tokens + self.config.llm_token_overhead,
            )
            duration_ms = (time.time() - t0) * 1000
            self._log_usage("tag_rollup", duration_ms=duration_ms, usage=usage)
            parsed = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"LLM tag summary rollup failed for '{tag}': {e}")
            parsed = {"summary": combined[:4000]}

        summary_text = parsed.get("summary", "")
        description = parsed.get("description", "")
        code_refs = _collect_code_refs(
            parsed.get("code_refs"),
            *[getattr(s.metadata, "code_refs", []) for s in summaries],
        )
        return TagSummary(
            tag=tag,
            summary=summary_text,
            description=description,
            summary_tokens=self.token_counter(summary_text),
            source_segment_refs=[s.ref for s in summaries],
            source_turn_numbers=sorted(set(turn_numbers)),
            code_refs=code_refs,
            covers_through_turn=max_turn,
        )
