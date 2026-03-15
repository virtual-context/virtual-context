"""Fact supersession checker: detect and mark contradicted facts."""

from __future__ import annotations

import json
import logging
import re
import time

from ..core.store import ContextStore
from ..core.telemetry import TelemetryLedger
from ..types import Fact, FactLink, LLMProvider, RelationType, SupersessionConfig

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "from", "with", "that", "about", "into", "over", "have", "been", "will",
    "this", "their", "there", "where", "which", "would", "could", "should",
    "after", "before", "during", "while", "other", "another", "these", "those",
    "trip", "solo", "recent", "just", "back", "today", "recently", "returned",
    "camping", "hiking", "visited", "started", "began",
})


def _parse_date_for_comparison(date_str: str):
    """Parse a date string into a comparable date object. Returns None on failure."""
    from datetime import date, datetime
    if not date_str or date_str == "(unknown)":
        return None
    # Try YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str[:10], fmt).date()
        except ValueError:
            pass
    # Try "2023/04/20 (Thu) 04:17" format
    try:
        return datetime.strptime(date_str.split("(")[0].strip(), "%Y/%m/%d").date()
    except ValueError:
        pass
    return None


def _extract_object_keyword(object_str: str) -> str | None:
    """Extract the most distinctive word from a fact's object string.

    Used to find cross-session duplicate facts via object_contains lookup.
    Prefers proper nouns (initial capital); falls back to longest word >= 5 chars.
    Returns None if no distinctive word is found.
    """
    words = re.findall(r"[A-Za-z']+", object_str)
    candidates = [w for w in words if len(w) >= 5 and w.lower() not in _STOPWORDS]
    if not candidates:
        return None
    proper = [w for w in candidates if w[0].isupper()]
    pool = proper if proper else candidates
    return max(pool, key=len)


_MERGE_SYSTEM = (
    "You are a memory consolidation assistant. Respond only with JSON."
)

_MERGE_PROMPT = """\
A fact has been superseded by a newer fact. Produce a single merged fact \
that captures the current state and what changed.

State what IS true now as durable knowledge. Do not echo the original \
conversational phrasing — restate the facts in neutral, declarative language.

SUPERSEDED (old) fact:
  {old_formatted}

CURRENT (new) fact:
  {new_formatted}

Produce a merged fact with updated fields:
- "verb": a declarative verb describing the current state (e.g. "has", "holds", "improved")
- "object": the current value with specifics preserved
- "status": the appropriate temporal status
- "what": one or two sentences of durable knowledge — current state and what changed

Reply with JSON: {{"verb": "...", "object": "...", "status": "...", "what": "..."}}\
"""


_PROMOTE_SYSTEM = "You are a memory rewrite assistant. Respond only with JSON."

_PROMOTE_PROMPT = """\
A planned event's date has passed. Rewrite this fact as a completed event.

Original fact:
  Subject: {subject}
  Verb: {verb}
  Object: {object}
  Who: {who}
  What: {what}
  Planned date: {when_date}

The date {when_date} is now in the past (current date: {ref_date}). \
Rewrite the fact as something that happened, not something that was planned. \
Preserve who was involved (the "Who" field) in the rewritten text.

Produce updated fields:
- "verb": a past-tense action verb (e.g. "played", "attended", "went to")
- "object": the object with planning language removed
- "what": one sentence stating what happened, in past tense, including who was involved

Reply with JSON: {{"verb": "...", "object": "...", "what": "..."}}\
"""


def promote_planned_facts(
    store: ContextStore,
    reference_date: str = "",
    llm_provider: LLMProvider | None = None,
    model: str = "",
) -> int:
    """Promote 'planned' facts whose when_date has passed to 'completed'.

    When a fact has status='planned' and a concrete when_date that is before
    the reference_date (or today if not provided), its status is updated to
    'completed' and verb/what are rewritten via LLM to reflect past tense.

    Falls back to a simple status flip if no LLM provider is available.

    Returns the number of facts promoted.
    """
    from datetime import date, datetime

    ref = None
    if reference_date:
        ref = _parse_date_for_comparison(reference_date)
    if ref is None:
        ref = date.today()

    planned = store.query_facts(status="planned", limit=10000)
    promoted = 0
    for fact in planned:
        fact_date = _parse_date_for_comparison(fact.when_date or "")
        if fact_date and fact_date < ref:
            verb = fact.verb
            obj = fact.object
            what = fact.what or ""

            # LLM rewrite for clean past-tense text
            if llm_provider:
                prompt = _PROMOTE_PROMPT.format(
                    subject=fact.subject, verb=fact.verb,
                    object=fact.object, who=fact.who or "",
                    what=fact.what or "",
                    when_date=fact.when_date or str(fact_date),
                    ref_date=ref.isoformat(),
                )
                try:
                    response, _ = llm_provider.complete(
                        system=_PROMOTE_SYSTEM, user=prompt, max_tokens=200,
                    )
                    cleaned = re.sub(r"<think>.*?</think>", "", response.strip(), flags=re.DOTALL).strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    data = json.loads(cleaned)
                    verb = data.get("verb", verb)
                    obj = data.get("object", obj)
                    what = data.get("what", what)
                except Exception as e:
                    logger.warning("Planned fact rewrite failed, using status-only: %s", e)

            store.update_fact_fields(fact.id, verb=verb, object=obj, status="completed", what=what)
            promoted += 1
            logger.info(
                "Promoted planned→completed: %s %s → %s %s [when: %s]",
                fact.subject, fact.verb, verb, obj[:40], fact.when_date,
            )
    if promoted:
        logger.info("Promoted %d planned facts to completed (ref date: %s)", promoted, ref.isoformat())
    return promoted


def dedup_facts(store: ContextStore) -> int:
    """Remove exact-duplicate facts from the store.

    Groups non-superseded facts by (subject, verb, object, what) and marks
    duplicates as superseded by the first occurrence.  Only considers facts
    with non-empty 'what' to avoid collapsing distinct facts that happen to
    share subject+verb but differ in meaning.

    Storage-agnostic — works with any ContextStore implementation.

    Returns the number of duplicates removed.
    """
    all_facts = store.query_facts(limit=50000)
    groups: dict[tuple, list[Fact]] = {}
    for f in all_facts:
        if f.superseded_by or not f.what:
            continue
        key = (f.subject.lower(), f.verb.lower(), f.object.lower(), f.what.lower())
        groups.setdefault(key, []).append(f)

    deduped = 0
    for key, facts in groups.items():
        if len(facts) <= 1:
            continue
        keeper = facts[0]
        for dupe in facts[1:]:
            store.set_fact_superseded(dupe.id, keeper.id)
            deduped += 1

    if deduped:
        logger.info("Deduped %d exact-duplicate facts", deduped)
    return deduped


class FactSupersessionChecker:
    """Check new facts against existing facts and mark superseded ones."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        store: ContextStore,
        config: SupersessionConfig,
        telemetry_ledger: TelemetryLedger | None = None,
        embed_fn=None,
    ):
        self.llm = llm_provider
        self.model = model
        self.store = store
        self.config = config
        self._telemetry = telemetry_ledger
        self._embed_fn = embed_fn
        self._all_facts_cache: list[Fact] | None = None

    def check_and_supersede(self, new_facts: list[Fact]) -> int:
        """For each new fact, find candidates by subject, ask LLM, mark superseded.

        Returns count of superseded facts.
        """
        if not self.config.enabled or not new_facts:
            return 0

        import sys as _sys
        import time as _time
        _ss_start = _time.time()

        superseded_count = 0
        superseded_this_run: set[str] = set()
        total = len(new_facts)
        _skipped = 0
        _llm_calls = 0
        for idx, fact in enumerate(new_facts, 1):
            if fact.id in superseded_this_run:
                logger.info("  Supersession %d/%d: skipped (already superseded this run)", idx, total)
                continue
            if not fact.subject:
                logger.info("  Supersession %d/%d: skipped (no subject)", idx, total)
                continue
            # Query existing non-superseded facts with same subject.
            # When tags are available, filter by them to avoid sending
            # unrelated facts to the LLM (reduces false supersessions).
            # Tag-based candidates (existing behaviour)
            candidates = self.store.query_facts(
                subject=fact.subject,
                tags=fact.tags if fact.tags else None,
                limit=self.config.batch_size,
            )
            # Object-similarity candidates — catches cross-session duplicates
            # whose tags don't overlap with the new fact's tags.
            # Use case-sensitive keyword to avoid false matches
            # (e.g. "Apple" the brand vs "apple" the fruit).
            keyword = _extract_object_keyword(fact.object)
            if keyword and fact.tags:  # only when tag-scoped: unfiltered query already covers all subjects
                obj_candidates = self.store.query_facts(
                    subject=fact.subject,
                    object_contains=keyword,
                    limit=self.config.batch_size,
                )
                # Filter to case-sensitive whole-word matches to avoid
                # false positives from substring/case-insensitive SQL LIKE
                kw_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')
                seen_ids = {c.id for c in candidates}
                for c in obj_candidates:
                    if c.id not in seen_ids and kw_pattern.search(c.object):
                        candidates.append(c)
                        seen_ids.add(c.id)
            # Embedding-based candidates — finds semantically similar facts
            # regardless of tag overlap or ingestion order.
            seen_ids = {c.id for c in candidates}
            embed_candidates = self._embedding_candidates(fact, seen_ids)
            candidates.extend(embed_candidates)
            candidates = [c for c in candidates if c.id != fact.id and c.id not in superseded_this_run]
            if not candidates:
                _skipped += 1
                if idx % 100 == 0 or idx == total:
                    _elapsed = _time.time() - _ss_start
                    _rate = idx / _elapsed if _elapsed > 0 else 0
                    _eta = int((total - idx) / _rate) if _rate > 0 else 0
                    _sys.stderr.write(
                        f"\r  SUPERSESSION: {idx}/{total} facts | "
                        f"{superseded_count} superseded | {_skipped} skipped | "
                        f"{_llm_calls} LLM calls | {_rate:.1f} fact/s | ETA {_eta}s   "
                    )
                    _sys.stderr.flush()
                continue

            _llm_calls += 1
            superseded_ids = self._check_batch(fact, candidates)
            if superseded_ids:
                # Hard date guard: never supersede a candidate with a newer
                # session date than the new fact.  The LLM prompt asks for
                # this but doesn't always comply.
                cand_by_id = {c.id: c for c in candidates}
                new_date = _parse_date_for_comparison(fact.when_date or fact.session_date or "")
                safe_ids = []
                for old_id in superseded_ids:
                    old_fact = cand_by_id.get(old_id)
                    if old_fact:
                        old_date = _parse_date_for_comparison(old_fact.when_date or old_fact.session_date or "")
                        if new_date and old_date and old_date > new_date:
                            logger.info("  Supersession date guard: refusing to supersede newer fact %s (%s > %s)",
                                        old_id[:8], old_date, new_date)
                            continue
                    safe_ids.append(old_id)
                for old_id in safe_ids:
                    self.store.set_fact_superseded(old_id, fact.id)
                    superseded_this_run.add(old_id)
                    superseded_count += 1
                logger.info("  Supersession %d/%d: %d superseded (total %d) — %s",
                            idx, total, len(superseded_ids), superseded_count, fact.subject[:40])
            else:
                logger.info("  Supersession %d/%d: 0 superseded — %s [%d candidates]",
                            idx, total, fact.subject[:40], len(candidates))
            # Progress on every LLM call
            _elapsed = _time.time() - _ss_start
            _rate = idx / _elapsed if _elapsed > 0 else 0
            _eta = int((total - idx) / _rate) if _rate > 0 else 0
            _sys.stderr.write(
                f"\r  SUPERSESSION: {idx}/{total} facts | "
                f"{superseded_count} superseded | {_skipped} skipped | "
                f"{_llm_calls} LLM calls | {_rate:.1f} fact/s | ETA {_eta}s   "
            )
            _sys.stderr.flush()

        _sys.stderr.write("\n")
        _sys.stderr.flush()
        return superseded_count

    def _get_all_facts(self) -> list[Fact]:
        """Cache all non-superseded facts for embedding search."""
        if self._all_facts_cache is None:
            self._all_facts_cache = self.store.query_facts(limit=10000)
        return self._all_facts_cache

    def _embedding_candidates(
        self, fact: Fact, already_seen: set[str], top_k: int = 10, threshold: float = 0.5
    ) -> list[Fact]:
        """Find semantically similar facts via embedding on the 'what' field."""
        if self._embed_fn is None:
            return []
        query_text = fact.what or f"{fact.subject} {fact.verb} {fact.object}"
        if not query_text.strip():
            return []

        all_facts = self._get_all_facts()
        # Filter to same subject, not already seen
        pool = [
            f for f in all_facts
            if f.id not in already_seen
            and f.id != fact.id
            and f.subject and f.subject.lower() == (fact.subject or "").lower()
        ]
        if not pool:
            return []

        from ..core.math_utils import cosine_similarity

        whats = [f.what or f"{f.subject} {f.verb} {f.object}" for f in pool]
        texts = [query_text] + whats
        vectors = self._embed_fn(texts)
        query_vec = vectors[0]

        scored = []
        for i, f in enumerate(pool):
            sim = cosine_similarity(query_vec, vectors[i + 1])
            if sim >= threshold:
                scored.append((sim, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:top_k]]

    def _log_usage(self, detail: str, duration_ms: float = 0.0) -> None:
        if not self._telemetry:
            return
        usage = getattr(self.llm, "last_usage", {})
        if not usage:
            return
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        if input_tokens or output_tokens:
            self._telemetry.log(
                component="supersession",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                detail=detail,
            )

    def _check_batch(self, new_fact: Fact, candidates: list[Fact]) -> list[str]:
        prompt = self._build_prompt(new_fact, candidates)
        logger.debug("  _check_batch: %d candidates, prompt %d chars", len(candidates), len(prompt))
        try:
            t0 = time.time()
            response, _usage = self.llm.complete(
                system="You are a fact comparison assistant. Respond only with a JSON array.",
                user=prompt,
                max_tokens=200,
            )
            duration_ms = (time.time() - t0) * 1000
            logger.debug("  _check_batch: LLM call %.1fms, response=%r", duration_ms, response[:100] if response else None)
            self._log_usage("check_batch", duration_ms=duration_ms)
        except Exception as e:
            logger.warning("Supersession LLM call failed (%.1fs): %s", time.time() - t0, e)
            return []
        return self._parse_response(response, candidates)

    def _build_prompt(self, new_fact: Fact, candidates: list[Fact]) -> str:
        lines = [
            "A new fact has been extracted from a conversation:",
            f"  {new_fact.format_for_prompt()}",
            "",
            "Existing facts with the same subject:",
        ]
        for i, c in enumerate(candidates):
            lines.append(f"  {c.format_for_prompt(include_index=i)}")
        lines.append("")
        lines.append(
            "Which existing facts (by index) are CONTRADICTED, SUPERSEDED, or "
            "DUPLICATED by the new fact? A fact is duplicated if it describes the "
            "same underlying event/state with different wording. When duplicates "
            "are found, mark the LESS detailed version for removal. "
            "A fact is SUPERSEDED when it describes an earlier value of the same "
            "attribute (e.g. a previous record, an old address, a former preference). "
            "Look at the underlying attribute being described, not just the verb phrasing. "
            "IMPORTANT: Only mark a candidate as superseded if its session date is OLDER "
            "than the new fact's session date, or if the dates are unknown/equal. "
            "Never supersede a candidate whose session date is newer (later) than the new fact. "
            "NEVER mark an existing fact as superseded if it is MORE specific than the new fact. "
            "A fact with concrete details (locations, items, methods) must survive over a "
            "vague fact about the same topic. "
            "CRITICAL: Sharing a keyword does NOT make two facts about the same attribute. "
            "An EVENT (something the user did) is a different attribute from a STATE "
            "(something the user has or prefers). Events and states that merely mention "
            "the same object are independent facts — neither supersedes the other. "
            "Only supersede when facts describe the SAME attribute with an updated value. "
            "Reply with a JSON array of indices, e.g. [0, 2]. "
            "Reply [] if none are superseded or duplicated."
        )
        return "\n".join(lines)

    def _merge_facts(self, winning_fact: Fact, old_fact: Fact) -> None:
        """Merge old fact's knowledge into the winning fact via LLM.

        Updates verb, object, status, and what on the winning fact to capture
        the temporal transition as durable knowledge.
        """
        prompt = _MERGE_PROMPT.format(
            old_formatted=old_fact.format_for_prompt(),
            new_formatted=winning_fact.format_for_prompt(),
        )
        try:
            response, _ = self.llm.complete(
                system=_MERGE_SYSTEM,
                user=prompt,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning("Supersession merge LLM call failed: %s", e)
            return

        if not response:
            return
        merged = self._parse_merge_response(response)
        if not merged:
            return

        verb = merged.get("verb", winning_fact.verb)
        obj = merged.get("object", winning_fact.object)
        status = merged.get("status", winning_fact.status)
        what = merged.get("what", winning_fact.what)

        self.store.update_fact_fields(winning_fact.id, verb, obj, status, what)
        # Update the in-memory object so subsequent merges see current state
        winning_fact.verb = verb
        winning_fact.object = obj
        winning_fact.status = status
        winning_fact.what = what
        logger.info(
            "Merged superseded fact into %s: verb=%r, object=%r",
            winning_fact.id[:8], verb, obj,
        )

    def _parse_merge_response(self, response: str) -> dict | None:
        text = response.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "verb" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            # Try extracting JSON object by scanning for balanced braces
            for i, ch in enumerate(text):
                if ch == '{':
                    try:
                        obj = json.loads(text[i:])
                        if isinstance(obj, dict) and "verb" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    # Try to find a balanced-brace substring
                    depth = 0
                    for j in range(i, len(text)):
                        if text[j] == '{':
                            depth += 1
                        elif text[j] == '}':
                            depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(text[i:j + 1])
                                if isinstance(obj, dict) and "verb" in obj:
                                    return obj
                            except (json.JSONDecodeError, ValueError):
                                break
        logger.warning("Failed to parse merge response: %s", text[:200])
        return None

    def _parse_response(self, response: str | None, candidates: list[Fact]) -> list[str]:
        if not response:
            return []
        text = response.strip()
        # Strip thinking tags from models like Qwen3
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to parse as JSON object first (Qwen3 returns {"updated": [0]})
        try:
            data = json.loads(text)
            if isinstance(data, list):
                indices = data
            elif isinstance(data, dict):
                # Accept various key names models may use
                for key in ("updated", "superseded", "indices",
                            "contradicted_or_updated", "result"):
                    if key in data and isinstance(data[key], list):
                        indices = data[key]
                        break
                else:
                    indices = []
            else:
                indices = []
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract bare array from response text
            match = re.search(r'\[[\d,\s]*\]', text)
            if not match:
                return []
            try:
                indices = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                return []

        return [candidates[i].id for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]


# Valid relation types for link detection
_VALID_LINK_TYPES = frozenset(rt.value for rt in RelationType)


class FactLinkChecker:
    """Extended supersession checker that also detects inter-fact relationships.

    When ``graph_links=False``, delegates to FactSupersessionChecker.check_and_supersede()
    (identical to pre-graph behaviour).

    When ``graph_links=True``, uses an expanded prompt to detect all relationship
    types (SUPERSEDES, CAUSED_BY, PART_OF, CONTRADICTS, SAME_AS, RELATED_TO)
    and stores them as FactLink objects.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        store: ContextStore,
        config: SupersessionConfig,
        graph_links: bool = False,
        telemetry_ledger: TelemetryLedger | None = None,
        embed_fn=None,
    ):
        self._supersession = FactSupersessionChecker(
            llm_provider=llm_provider,
            model=model,
            store=store,
            config=config,
            telemetry_ledger=telemetry_ledger,
            embed_fn=embed_fn,
        )
        self.store = store
        self.llm = llm_provider
        self.model = model
        self.config = config
        self.graph_links = graph_links
        self._telemetry = telemetry_ledger

    def check_and_link(self, new_facts: list[Fact]) -> tuple[int, int]:
        """Detect supersession and (optionally) inter-fact links.

        Runs a deterministic planned→completed promotion pass first,
        then LLM-based supersession/linking.

        Returns ``(links_created, facts_superseded)``.
        """
        if not self.config.enabled or not new_facts:
            return 0, 0

        # Pre-pass: promote planned facts whose date has passed (LLM rewrite)
        promote_planned_facts(self.store, llm_provider=self.llm, model=self.model)

        if not self.graph_links:
            superseded = self._supersession.check_and_supersede(new_facts)
            return 0, superseded

        # Graph mode: expanded prompt for all relationship types
        total_links = 0
        total_superseded = 0

        for fact in new_facts:
            if not fact.subject:
                continue

            candidates = self.store.query_facts(
                subject=fact.subject,
                tags=fact.tags if fact.tags else None,
                limit=self.config.batch_size,
            )
            candidates = [c for c in candidates if c.id != fact.id]
            if not candidates:
                continue

            try:
                links, superseded_ids = self._check_links(fact, candidates)
            except Exception as e:
                logger.warning("FactLinkChecker LLM call failed: %s", e)
                continue

            # Handle supersession
            for old_id in superseded_ids:
                self.store.set_fact_superseded(old_id, fact.id)
                total_superseded += 1

            # Store links
            if links:
                self.store.store_fact_links(links)
                total_links += len(links)

        return total_links, total_superseded

    def _check_links(
        self, new_fact: Fact, candidates: list[Fact],
    ) -> tuple[list[FactLink], list[str]]:
        """Ask LLM to identify relationships between new fact and candidates.

        Returns ``(fact_links, superseded_fact_ids)``.
        """
        prompt = self._build_link_prompt(new_fact, candidates)
        t0 = time.time()
        response, _usage = self.llm.complete(
            system="You are a fact relationship assistant. Respond only with JSON.",
            user=prompt,
            max_tokens=500,
        )
        duration_ms = (time.time() - t0) * 1000
        if self._telemetry:
            usage = getattr(self.llm, "last_usage", {})
            input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            if input_tokens or output_tokens:
                self._telemetry.log(
                    component="fact_link_checker",
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    detail="check_links",
                )

        return self._parse_link_response(response, new_fact, candidates)

    def _build_link_prompt(self, new_fact: Fact, candidates: list[Fact]) -> str:
        lines = [
            "A new fact has been extracted from a conversation:",
            f"  N0: {new_fact.format_for_prompt()}",
            "",
            "Existing facts:",
        ]
        for i, c in enumerate(candidates):
            lines.append(f"  E{i}: {c.format_for_prompt()}")
        lines.append("")
        lines.append(
            "Identify relationships between facts. Reply with JSON:\n"
            '{"superseded": [indices of existing facts superseded by N0],\n'
            ' "links": [{"source": "N0 or E<i>", "target": "N0 or E<i>", '
            '"relation": "<type>", "confidence": 0.0-1.0, "context": "one sentence"}]}\n\n'
            "Valid relation types: supersedes, caused_by, part_of, contradicts, same_as, related_to\n\n"
            "Rules:\n"
            "- supersedes: N0 replaces an existing fact (knowledge update)\n"
            "- caused_by: one fact happened because of another\n"
            "- part_of: one fact is a component/aspect of another\n"
            "- contradicts: facts conflict but neither replaces the other\n"
            "- same_as: facts refer to the same entity/event with different names\n"
            "- related_to: clear relationship that doesn't fit above types\n"
            "- Only create links when clear from context. Prefer no link over a weak one.\n"
            '- Reply {"superseded": [], "links": []} if no relationships found.'
        )
        return "\n".join(lines)

    def _parse_link_response(
        self, response: str | None, new_fact: Fact, candidates: list[Fact],
    ) -> tuple[list[FactLink], list[str]]:
        """Parse LLM response into FactLink objects and superseded IDs."""
        if not response:
            return [], []

        text = response.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON object in response
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not match:
                return [], []
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                return [], []

        if not isinstance(data, dict):
            return [], []

        # Parse superseded indices
        superseded_raw = data.get("superseded", [])
        superseded_ids = []
        if isinstance(superseded_raw, list):
            for idx in superseded_raw:
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    superseded_ids.append(candidates[idx].id)

        # Parse links
        links_raw = data.get("links", [])
        fact_links: list[FactLink] = []
        if isinstance(links_raw, list):
            for link_data in links_raw:
                if not isinstance(link_data, dict):
                    continue
                relation = link_data.get("relation", "").lower()
                if relation not in _VALID_LINK_TYPES:
                    continue

                source_ref = str(link_data.get("source", ""))
                target_ref = str(link_data.get("target", ""))
                source_id = self._resolve_ref(source_ref, new_fact, candidates)
                target_id = self._resolve_ref(target_ref, new_fact, candidates)

                if not source_id or not target_id or source_id == target_id:
                    continue

                fact_links.append(FactLink(
                    source_fact_id=source_id,
                    target_fact_id=target_id,
                    relation_type=relation,
                    confidence=float(link_data.get("confidence", 0.8)),
                    context=str(link_data.get("context", "")),
                    created_by="compaction",
                ))

        return fact_links, superseded_ids

    @staticmethod
    def _resolve_ref(ref: str, new_fact: Fact, candidates: list[Fact]) -> str | None:
        """Resolve 'N0' or 'E<i>' to a fact ID."""
        ref = ref.strip().upper()
        if ref == "N0":
            return new_fact.id
        if ref.startswith("E"):
            try:
                idx = int(ref[1:])
                if 0 <= idx < len(candidates):
                    return candidates[idx].id
            except ValueError:
                pass
        return None
