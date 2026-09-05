"""Fact supersession checker: detect and mark contradicted facts."""

from __future__ import annotations

import json
import logging
import re
import time

from ..core.fact_lifecycle import decide_supersession, fact_version, parse_fact_date
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


def refresh_fact_embedding(
    store: ContextStore,
    embed_fn,
    model: str,
    fact: Fact,
    *,
    operation_id: str | None = None,
    owner_worker_id: str | None = None,
    lifecycle_epoch: int | None = None,
) -> None:
    """Recompute and persist a fact's dense embedding after a successful
    ``update_fact_fields`` mutation.

    The backend has already invalidated the stale vector inside the update
    transaction, so this only re-embeds the current text. Best-effort:
    ``CompactionLeaseLost`` propagates (fail-closed) so a fenced caller
    aborts cleanly; any other failure is logged and swallowed, leaving the
    fact vector-less (never stale) until the next backfill/compaction.
    """
    from ..types import CompactionLeaseLost
    if embed_fn is None:
        return
    conv_id = fact.conversation_id
    if not conv_id:
        return
    try:
        text = fact.embed_text()
        if not text:
            return
        emb = embed_fn([text])[0]
        store.store_fact_embeddings(
            fact.id, conv_id, model, emb,
            operation_id=operation_id,
            owner_worker_id=owner_worker_id,
            lifecycle_epoch=lifecycle_epoch,
        )
    except CompactionLeaseLost:
        raise
    except Exception as e:
        logger.warning("Failed to refresh fact embedding for %s: %s", fact.id, e)


def _parse_date_for_comparison(date_str: str):
    """Compatibility name for the shared lifecycle date parser."""
    return parse_fact_date(date_str)


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


def promote_planned_facts(
    store: ContextStore,
    reference_date: str = "",
    llm_provider: LLMProvider | None = None,
    model: str = "",
    *,
    embed_fn=None,
    embedding_model: str = "all-MiniLM-L6-v2",
    operation_id: str | None = None,
    owner_worker_id: str | None = None,
    lifecycle_epoch: int | None = None,
    conversation_id: str | None = None,
) -> int:
    """Compatibility no-op: elapsed plans are not evidence of completion.

    Kept for callers of older releases. A plan retains its original status,
    text, and embedding until new canonical evidence establishes an outcome.
    The past ``when_date`` already identifies an elapsed plan to readers.
    No model calls or storage mutations are performed.
    """
    return 0


def _fact_scope(store, fact):
    resolver = getattr(store, "get_fact_admission_scope", None)
    if not callable(resolver):
        return None
    scope = resolver(fact.id)
    return scope if isinstance(scope, tuple) and len(scope) == 2 and all(isinstance(value, str) for value in scope) else None


def _proposal_snapshot(store, fact, *, cache=None):
    key = (fact.id, fact_version(fact))
    if cache is not None and key in cache:
        return cache[key]
    resolver = getattr(store, "get_fact_admission_snapshot", None)
    if not callable(resolver):
        return None
    snapshot = resolver(fact.id)
    if not isinstance(snapshot, dict) or snapshot.get("fact_version") != key[1]:
        snapshot = None
    if cache is not None:
        cache[key] = snapshot
    return snapshot


def _proposal_versions(new_snapshot, old_snapshot):
    return {
        "expected_old_version": old_snapshot["fact_version"],
        "expected_new_version": new_snapshot["fact_version"],
        "expected_source_versions": tuple(sorted(set(
            tuple(pair) for pair in (*old_snapshot.get("source_versions", ()), *new_snapshot.get("source_versions", ()))
        ))),
    }


def _admitted_snapshots(store, new, candidates, *, snapshot_cache=None):
    """Bind each fact once before model I/O; SQL revalidates these proofs by CAS.

    The cache lives only for one incoming fact's candidate selection. Embedding
    and comparison share that immutable proposal, never a cross-request proof.
    """
    cache = {} if snapshot_cache is None else snapshot_cache
    new_snapshot = _proposal_snapshot(store, new, cache=cache)
    if new_snapshot is None:
        return None, [], {}
    accepted, old_snapshots = [], {}
    for old in candidates:
        if old.id == new.id:
            continue
        snapshot = _proposal_snapshot(store, old, cache=cache)
        if snapshot is None:
            continue
        if decide_supersession(new, old, new_audience=new_snapshot.get("audience"), old_audience=snapshot.get("audience")).accepted:
            accepted.append(old)
            old_snapshots[old.id] = snapshot
    return new_snapshot, accepted, old_snapshots


def _admit_supersession_ids(
    new_fact: Fact, candidates: list[Fact], proposed_ids: list[str], *, snapshots,
) -> list[str]:
    """Restrict model IDs to the pre-admitted immutable proposal."""
    candidates_by_id = {fact.id: fact for fact in candidates}
    return [old_id for old_id in dict.fromkeys(proposed_ids)
            if old_id in candidates_by_id and decide_supersession(
                new_fact, candidates_by_id[old_id],
                new_audience=snapshots[new_fact.id].get("audience"),
                old_audience=snapshots[old_id].get("audience"),
            ).accepted]


def dedup_facts(store: ContextStore, *, conversation_id: str | None = None) -> int:
    """Consolidate exact duplicates only inside one admitted source scope."""
    kwargs = {"limit": 50000}
    if conversation_id is not None:
        kwargs["conversation_id"] = conversation_id
    all_facts = store.query_facts(**kwargs)
    groups: dict[tuple, list[Fact]] = {}
    for fact in all_facts:
        if fact.superseded_by or not fact.what or not fact.conversation_id:
            continue
        key = (fact.conversation_id, fact.author_actor_id, fact.author_source_role,
               _fact_scope(store, fact), fact.subject.casefold(), fact.verb.casefold(),
               fact.object.casefold(), fact.what.casefold())
        groups.setdefault(key, []).append(fact)
    deduped = 0
    for facts in groups.values():
        if len(facts) < 2:
            continue
        # Chronology still applies to duplicate source claims. Retain the latest
        # dated occurrence and record which historical occurrence it replaces.
        facts.sort(key=lambda fact: (str(parse_fact_date(fact.when_date or fact.session_date) or ""), str(fact.mentioned_at), fact.id), reverse=True)
        keeper = facts[0]
        for duplicate in facts[1:]:
            new_snapshot, candidates, snapshots = _admitted_snapshots(store, keeper, [duplicate])
            if candidates and store.set_fact_superseded(duplicate.id, keeper.id, **_proposal_versions(new_snapshot, snapshots[duplicate.id])) is True:
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
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.llm = llm_provider
        self.model = model
        self.store = store
        self.config = config
        self._telemetry = telemetry_ledger
        self._embed_fn = embed_fn
        self._embedding_model = embedding_model
        self._all_facts_cache: dict[str, list[Fact]] = {}

    def check_and_supersede(
        self,
        new_facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
    ) -> int:
        """For each new fact, find candidates by subject, ask LLM, mark superseded.

        When called from a compaction phase, the caller forwards the
        guard kwargs so ``set_fact_superseded`` writes through the
        active operation-id fence (fencing plan §5.6 caller-side
        propagation). Candidate queries are then scoped to each
        fact's ``conversation_id`` (which the compaction pipeline
        sets to the active op's conversation) so a candidate from a
        different conversation never reaches the fenced supersession
        write, avoiding a spurious ``CompactionLeaseLost`` from the
        both-endpoint validation. Conversation, author, audience and chronology
        constrain every caller before any model sees the candidates.

        Returns count of superseded facts.
        """
        if not self.config.enabled or not new_facts:
            return 0

        # All reads are conversation-scoped; write fences are independent.

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
            if not fact.subject or not fact.conversation_id:
                logger.info("  Supersession %d/%d: skipped (no subject)", idx, total)
                continue
            # Query existing non-superseded facts with same subject.
            # When tags are available, filter by them to avoid sending
            # unrelated facts to the LLM (reduces false supersessions).
            # Tag-based candidates (existing behaviour)
            _query_kwargs: dict[str, object] = {
                "subject": fact.subject,
                "tags": fact.tags if fact.tags else None,
                "limit": self.config.batch_size,
            }
            if fact.conversation_id:
                _query_kwargs["conversation_id"] = fact.conversation_id
            candidates = self.store.query_facts(**_query_kwargs)
            # Object-similarity candidates -- catches cross-session duplicates
            # whose tags don't overlap with the new fact's tags.
            # Use case-sensitive keyword to avoid false matches
            # (e.g. "Apple" the brand vs "apple" the fruit).
            keyword = _extract_object_keyword(fact.object)
            if keyword and fact.tags:  # only when tag-scoped: unfiltered query already covers all subjects
                _obj_query_kwargs: dict[str, object] = {
                    "subject": fact.subject,
                    "object_contains": keyword,
                    "limit": self.config.batch_size,
                }
                if fact.conversation_id:
                    _obj_query_kwargs["conversation_id"] = fact.conversation_id
                obj_candidates = self.store.query_facts(**_obj_query_kwargs)
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
            snapshot_cache = {}
            seen_ids = {c.id for c in candidates}
            embed_candidates = self._embedding_candidates(
                fact, seen_ids,
                restrict_conversation_id=(
                    fact.conversation_id
                ),
                snapshot_cache=snapshot_cache,
            )
            candidates.extend(embed_candidates)
            candidates = [c for c in candidates if c.id not in superseded_this_run]
            new_snapshot, candidates, snapshots = _admitted_snapshots(self.store, fact, candidates, snapshot_cache=snapshot_cache)
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
                safe_ids = _admit_supersession_ids(fact, candidates, superseded_ids, snapshots={**snapshots, fact.id: new_snapshot})
                for old_id in safe_ids:
                    accepted = self.store.set_fact_superseded(
                        old_id, fact.id,
                        operation_id=operation_id,
                        owner_worker_id=owner_worker_id,
                        lifecycle_epoch=lifecycle_epoch,
                        **_proposal_versions(new_snapshot, snapshots[old_id]),
                    )
                    if accepted is True:
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

    def _get_all_facts(self, conversation_id: str) -> list[Fact]:
        """Cache only this owned conversation's bounded candidate pool."""
        if conversation_id not in self._all_facts_cache:
            self._all_facts_cache[conversation_id] = self.store.query_facts(limit=10000, conversation_id=conversation_id)
        return self._all_facts_cache[conversation_id]

    def _embedding_candidates(
        self, fact: Fact, already_seen: set[str], top_k: int = 10, threshold: float = 0.5,
        *,
        restrict_conversation_id: str | None = None,
        snapshot_cache=None,
    ) -> list[Fact]:
        """Find semantically similar facts via embedding on the 'what' field.

        When ``restrict_conversation_id`` is supplied, the candidate
        pool is filtered to facts in that conversation. The compaction
        pipeline passes the active op's conversation_id to avoid
        cross-conversation candidates triggering a guarded supersession
        write rejection.
        """
        if self._embed_fn is None:
            return []
        query_text = fact.what or f"{fact.subject} {fact.verb} {fact.object}"
        if not query_text.strip():
            return []

        all_facts = self._get_all_facts(fact.conversation_id)
        # Filter to same subject, not already seen
        pool = [
            f for f in all_facts
            if f.id not in already_seen
            and f.id != fact.id
            and f.subject and f.subject.lower() == (fact.subject or "").lower()
            and (
                restrict_conversation_id is None
                or f.conversation_id == restrict_conversation_id
            )
        ]
        _, pool, _ = _admitted_snapshots(self.store, fact, pool, snapshot_cache=snapshot_cache)
        if not pool:
            return []

        from ..core.math_utils import rank_by_embedding

        whats = [f.what or f"{f.subject} {f.verb} {f.object}" for f in pool]
        scored = rank_by_embedding(
            query_text, pool, whats, self._embed_fn, threshold=threshold,
        )
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
        """Compatibility no-op: accepted source-derived claims stay unchanged.

        Supersession records the old/new relationship and source versions.
        Model-written consolidation prose is not new evidence and cannot
        rewrite the surviving fact's fields or temporal status.
        """
        return None

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

        return [candidates[i].id for i in indices if type(i) is int and 0 <= i < len(candidates)]


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
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._supersession = FactSupersessionChecker(
            llm_provider=llm_provider,
            model=model,
            store=store,
            config=config,
            telemetry_ledger=telemetry_ledger,
            embed_fn=embed_fn,
            embedding_model=embedding_model,
        )
        self.store = store
        self.llm = llm_provider
        self.model = model
        self.config = config
        self.graph_links = graph_links
        self._telemetry = telemetry_ledger
        self._embed_fn = embed_fn
        self._embedding_model = embedding_model

    def check_and_link(
        self,
        new_facts: list[Fact],
        *,
        operation_id: str | None = None,
        owner_worker_id: str | None = None,
        lifecycle_epoch: int | None = None,
        conversation_id: str | None = None,
    ) -> tuple[int, int]:
        """Detect supersession and (optionally) inter-fact links.

        A model proposes relationships; deterministic chronology admits
        supersession. Elapsed plans remain plans until new evidence arrives.

        When called from a compaction phase, the caller forwards the
        guard kwargs so ``check_and_supersede``, ``set_fact_superseded``, and
        ``store_fact_links`` all write through the active
        operation-id fence (fencing plan §5.6 caller-side propagation).
        ``conversation_id`` is required alongside the guard triple for
        ``store_fact_links`` so the active op's conversation can be
        matched against both endpoint facts.

        Returns ``(links_created, facts_superseded)``.
        """
        if not self.config.enabled or not new_facts:
            return 0, 0
        from ..types import CompactionLeaseLost

        if not self.graph_links:
            superseded = self._supersession.check_and_supersede(
                new_facts,
                operation_id=operation_id,
                owner_worker_id=owner_worker_id,
                lifecycle_epoch=lifecycle_epoch,
            )
            return 0, superseded

        # Graph mode: expanded prompt for all relationship types
        total_links = 0
        total_superseded = 0

        for fact in new_facts:
            if not fact.subject or not fact.conversation_id:
                continue

            # Constrain every candidate read before model comparison.
            _query_kwargs: dict[str, object] = {
                "subject": fact.subject,
                "tags": fact.tags if fact.tags else None,
                "limit": self.config.batch_size,
            }
            if conversation_id is not None and conversation_id != fact.conversation_id:
                continue
            _query_kwargs["conversation_id"] = fact.conversation_id
            candidates = self.store.query_facts(**_query_kwargs)
            new_snapshot, candidates, snapshots = _admitted_snapshots(self.store, fact, candidates)
            if not candidates:
                continue

            try:
                links, superseded_ids = self._check_links(fact, candidates)
            except CompactionLeaseLost:
                raise
            except Exception as e:
                logger.warning("FactLinkChecker LLM call failed: %s", e)
                continue

            snapshots[fact.id] = new_snapshot
            safe_ids = _admit_supersession_ids(fact, candidates, superseded_ids, snapshots=snapshots)
            known_facts = {candidate.id: candidate for candidate in [fact, *candidates]}
            pairs = {(old_id, fact.id) for old_id in safe_ids}
            for link in links:
                if link.relation_type == "supersedes":
                    new = known_facts.get(link.source_fact_id)
                    old = known_facts.get(link.target_fact_id)
                    if new and old and decide_supersession(
                        new, old, new_audience=snapshots[new.id].get("audience"),
                        old_audience=snapshots[old.id].get("audience"),
                    ).accepted:
                        pairs.add((old.id, new.id))
            accepted_pairs = set()
            for old_id, new_id in sorted(pairs):
                accepted = self.store.set_fact_superseded(
                    old_id, new_id, operation_id=operation_id,
                    owner_worker_id=owner_worker_id, lifecycle_epoch=lifecycle_epoch,
                    **_proposal_versions(snapshots[new_id], snapshots[old_id]),
                )
                if accepted is True:
                    accepted_pairs.add((old_id, new_id))
                    total_superseded += 1
            links = [link for link in links if link.relation_type != "supersedes"
                     or (link.target_fact_id, link.source_fact_id) in accepted_pairs]

            # Store links
            if links:
                stored_count = self.store.store_fact_links(
                    links,
                    operation_id=operation_id,
                    owner_worker_id=owner_worker_id,
                    lifecycle_epoch=lifecycle_epoch,
                    conversation_id=conversation_id or fact.conversation_id,
                )
                total_links += stored_count if type(stored_count) is int else 0

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
                if type(idx) is int and 0 <= idx < len(candidates):
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
