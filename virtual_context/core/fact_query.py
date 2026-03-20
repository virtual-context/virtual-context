"""FactQueryEngine: structured fact querying with verb expansion, semantic search, and re-ranking."""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


class FactQueryEngine:
    """Handles structured fact queries with verb expansion, semantic search, and re-ranking.

    Constructor takes:
        store:    a Store instance (SQLiteStore / FilesystemStore)
        semantic: a SemanticSearchManager instance
        config:   a VirtualContextConfig instance
    """

    # Manual verb synonym clusters.  If the query verb matches any member
    # of a cluster (case-insensitive), all other members that actually exist
    # in the fact store are included in the expansion — regardless of
    # embedding similarity.  This catches contextual synonyms that small
    # embedding models miss (e.g. "visited" ↔ "returned from").
    _VERB_CLUSTERS: list[set[str]] = [
        {
            "visited", "returned from", "got back from", "completed",
            "went on", "took", "explored", "toured", "traveled to",
            "went to", "hiked", "drove to", "flew to", "camped at",
            "stayed at",
        },
        {"bought", "purchased", "ordered", "got", "acquired", "picked up"},
        {"started", "began", "launched", "kicked off", "initiated"},
        {"finished", "completed", "wrapped up", "ended", "concluded"},
        {"likes", "enjoys", "loves", "prefers", "is fond of", "is a fan of"},
        {"dislikes", "hates", "avoids", "is not a fan of"},
        {"made", "created", "built", "crafted", "prepared", "cooked", "baked"},
        {"joined", "enrolled in", "signed up for", "registered for"},
        {"quit", "left", "resigned from", "stopped", "dropped out of"},
        {"moved to", "relocated to", "settled in", "is moving to"},
    ]

    # Pre-built lookup: lowercase verb → set of cluster synonyms
    _VERB_CLUSTER_MAP: dict[str, set[str]] = {}
    for _cluster in _VERB_CLUSTERS:
        _lower_cluster = {v.lower() for v in _cluster}
        for _v in _lower_cluster:
            _VERB_CLUSTER_MAP[_v] = _lower_cluster

    def __init__(self, *, store, semantic, config) -> None:
        self._store = store
        self._semantic = semantic
        self._config = config

    def _get_embed_fn(self):
        return self._semantic.get_embed_fn()

    def query(self, **kwargs) -> list | dict:
        """Query structured facts by filters. Expands verb semantically, then delegates to store.

        Returns list[Fact] normally.  When called with ``_return_meta=True``
        (used by the tool loop), returns a dict with ``facts``, ``expanded_verbs``,
        ``object_relaxed``, and ``semantic_note`` keys so the caller can
        annotate the response.
        """
        return_meta = kwargs.pop("_return_meta", False)
        intent_context = kwargs.pop("_intent_context", "") or ""
        expanded_verbs: list[str] | None = None
        semantic_note: str | None = None

        # Scope all store queries to this conversation
        kwargs.setdefault("conversation_id", self._config.conversation_id)

        # Save original params before verb expansion mutates kwargs
        orig_verb = kwargs.get("verb")
        orig_subject = kwargs.get("subject")
        orig_object = kwargs.get("object_contains")
        status_filter = kwargs.get("status")

        verb = kwargs.get("verb")
        if verb and not kwargs.get("verbs"):
            expanded = self._expand_verb(verb)
            if expanded and len(expanded) > 1:
                expanded_verbs = expanded
                kwargs["verbs"] = expanded
                kwargs.pop("verb", None)
                # Verb expansion widens the query significantly — bump the
                # SQL limit so rare-verb facts don't get cut off by ORDER BY
                # mentioned_at DESC.  Scale with number of expanded verbs.
                if "limit" not in kwargs:
                    kwargs["limit"] = max(50, len(expanded) * 10)

        results = self._store.query_facts(**kwargs)

        # Semantic search: find additional facts via embedding on 'what'
        # field.  Runs before auto-relax so it can provide precise results
        # and prevent the noisy fallback that drops object_contains.
        sem_all = self._semantic_fact_search(
            existing=results,
            subject=orig_subject,
            verb=orig_verb,
            object_contains=orig_object,
            intent_context=intent_context,
        )
        if sem_all:
            sem_filtered = sem_all
            if status_filter:
                sem_filtered = [f for f in sem_filtered if f.status == status_filter]
            # Respect the reader's explicit object_contains filter —
            # semantic search ignores structured constraints when building
            # candidates, so post-filter here to avoid returning facts
            # that contradict the reader's request (BUG-032).
            if orig_object:
                obj_lower = orig_object.lower()
                sem_filtered = [
                    f for f in sem_filtered
                    if obj_lower in (f.object or "").lower()
                    or obj_lower in (f.what or "").lower()
                ]
            if sem_filtered:
                results = results + sem_filtered
                semantic_note = f"semantic search added {len(sem_filtered)} fact(s)"

        # Tag-based sibling discovery: when we have some results, use their
        # tags to find related facts the structured query missed (e.g. "made
        # apple pie" when the reader searched verb="baked").
        if results and orig_subject:
            result_ids = {f.id for f in results}
            # Collect tags from current results
            seed_tags: set[str] = set()
            for f in results:
                seed_tags.update(f.tags)
            if seed_tags:
                siblings = self._store.query_facts(
                    subject=orig_subject,
                    tags=list(seed_tags),
                    limit=50,
                    conversation_id=self._config.conversation_id,
                )
                new_siblings = [s for s in siblings if s.id not in result_ids]
                if new_siblings:
                    results = results + new_siblings
                    sibling_note = f"tag siblings added {len(new_siblings)} fact(s)"
                    semantic_note = (
                        f"{semantic_note}; {sibling_note}" if semantic_note
                        else sibling_note
                    )

        # Auto-relax removed (BUG-032): dropping object_contains returns facts
        # that contradict the reader's explicit constraint, causing over-counting.
        # If 0 results, the reader should broaden its own query intentionally.

        # Relevance re-ranking: sort results by embedding similarity to the
        # reader's question so the most relevant facts appear first.  Without
        # this, results are ordered by ingestion timestamp (mentioned_at DESC)
        # which is essentially random with respect to the query.
        if results and intent_context.strip():
            results = self._rerank_facts(results, intent_context)

        # Auto-follow fact links when graph_links is enabled
        linked_facts = []
        _graph_links = getattr(self._config, "facts", None)
        if _graph_links and _graph_links.graph_links and results:
            fact_ids = [f.id for f in results]
            linked_facts = self._store.get_linked_facts(fact_ids, depth=1)
            # Deduplicate — don't include linked facts already in primary results
            primary_ids = set(fact_ids)
            linked_facts = [lf for lf in linked_facts if lf.fact.id not in primary_ids]

        if return_meta:
            meta: dict = {
                "facts": results,
                "expanded_verbs": expanded_verbs,
                "object_relaxed": False,  # auto-relax removed (BUG-032)
                "semantic_note": semantic_note,
            }
            if linked_facts:
                meta["linked_facts"] = linked_facts
            # When status was filtered, also query without status so the
            # caller can show total_all_statuses — prevents the reader from
            # splitting into per-status calls and never seeing the grand total.
            if status_filter:
                # Strip both status and object_contains — we want the
                # grand total for this verb+subject across all statuses.
                # object_contains uses strict LIKE at the store layer and
                # would return 0 when the auto-relax only happened above.
                unfiltered = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("status", "object_contains")
                }
                all_facts = self._store.query_facts(**unfiltered)
                # Merge semantic matches into the unfiltered total
                if sem_all:
                    all_ids = {f.id for f in all_facts}
                    for f in sem_all:
                        if f.id not in all_ids:
                            all_facts.append(f)
                            all_ids.add(f.id)
                status_counts: dict[str, int] = {}
                for f in all_facts:
                    s = f.status or "unknown"
                    status_counts[s] = status_counts.get(s, 0) + 1
                meta["all_statuses"] = status_counts
                meta["total_all_statuses"] = len(all_facts)
            return meta
        return results

    def _expand_verb(self, verb: str) -> list[str] | None:
        """Find semantically similar verbs in the facts DB via embedding similarity.

        Also includes manual synonym clusters for contextual synonyms that
        embeddings miss.  Returns list of matching verbs (including original)
        if expansions found, None if no expansions.
        """
        all_verbs = self._store.get_unique_fact_verbs(conversation_id=self._config.conversation_id)
        if not all_verbs:
            return None
        all_verbs_lower = {v.lower(): v for v in all_verbs}

        matches = [verb]
        seen = {verb.lower()}

        # Manual cluster expansion
        cluster = self._VERB_CLUSTER_MAP.get(verb.lower())
        if cluster:
            for synonym in cluster:
                if synonym not in seen and synonym in all_verbs_lower:
                    matches.append(all_verbs_lower[synonym])
                    seen.add(synonym)

        # Embedding-based expansion
        embed_fn = self._get_embed_fn()
        if embed_fn is not None:
            from .math_utils import rank_by_embedding

            scored = rank_by_embedding(
                verb, all_verbs, all_verbs, embed_fn, threshold=0.53,
            )
            for _, v in scored:
                if v.lower() not in seen:
                    matches.append(v)
                    seen.add(v.lower())

        return matches if len(matches) > 1 else None

    def _semantic_fact_search(
        self,
        existing: list,
        subject: str | None = None,
        verb: str | None = None,
        object_contains: str | None = None,
        intent_context: str = "",
    ) -> list:
        """Find additional facts by embedding similarity on the ``what`` field.

        Builds a natural-language query from the provided structured params
        (and optionally the user's question via *intent_context*), retrieves
        all facts for the given subject, and returns those whose ``what``
        field is semantically close to the query but that were NOT already
        in *existing*.  Returns an empty list when embeddings are unavailable
        or no new matches are found.
        """
        # Need at least a verb or object to form a meaningful query
        if not verb and not object_contains:
            return []
        embed_fn = self._get_embed_fn()
        if embed_fn is None:
            return []

        # Build query string from whatever params were provided.
        # When intent_context (the user's question) is available, use it
        # as the primary query — it carries richer semantic signal than the
        # short structured params (e.g. "How many health-related devices
        # do I use?" vs just "user use").
        if intent_context.strip():
            query_str = intent_context.strip()
            # The intent_context may be the full enriched prompt (context
            # summaries + question).  Strip the <virtual-context> block so
            # only the trailing question/instruction remains.
            for _vc_end_tag in ("</system-reminder>", "</virtual-context>"):
                if _vc_end_tag in query_str:
                    query_str = query_str.split(_vc_end_tag)[-1].strip()
            # Strip leading preamble that the benchmark wraps around the
            # context block — only keep meaningful trailing text.
            m = re.search(r"(?:^|\n)\s*Question:\s*(.+)", query_str, re.IGNORECASE)
            if m:
                query_str = m.group(1).strip()
                # Remove trailing "Answer:" marker if present
                query_str = re.sub(r"\s*Answer:\s*$", "", query_str).strip()
        else:
            parts: list[str] = []
            if subject:
                parts.append(subject)
            if verb:
                parts.append(verb)
            if object_contains:
                parts.append(object_contains)
            query_str = " ".join(parts)

        # Broad candidate set: all facts for this subject (no verb/object filter)
        cand_kwargs: dict = {"limit": 200, "conversation_id": self._config.conversation_id}
        if subject:
            cand_kwargs["subject"] = subject
        candidates = self._store.query_facts(**cand_kwargs)

        # Exclude already-found facts and facts without a 'what' description
        existing_ids = {f.id for f in existing}
        candidates = [f for f in candidates if f.id not in existing_ids and f.what]
        if not candidates:
            return []

        from .math_utils import rank_by_embedding

        scored = rank_by_embedding(
            query_str, candidates, [f.what for f in candidates],
            embed_fn, threshold=0.35,
        )
        return [fact for _, fact in scored]

    def _rerank_facts(self, facts: list, intent_context: str) -> list:
        """Re-rank *facts* by embedding similarity to the reader's question.

        Extracts the question from *intent_context*, embeds it alongside
        each fact's ``what`` field, and returns facts sorted by descending
        cosine similarity.  Falls back to the original order if embeddings
        are unavailable.
        """
        embed_fn = self._get_embed_fn()
        if embed_fn is None or not facts:
            return facts

        # Extract just the question from intent_context
        query_str = intent_context.strip()
        for _tag in ("</system-reminder>", "</virtual-context>"):
            if _tag in query_str:
                query_str = query_str.split(_tag)[-1].strip()
        m = re.search(r"(?:^|\n)\s*Question:\s*(.+)", query_str, re.IGNORECASE)
        if m:
            query_str = m.group(1).strip()
            query_str = re.sub(r"\s*Answer:\s*$", "", query_str).strip()
        if not query_str:
            return facts

        from .math_utils import rank_by_embedding

        whats = [f.what or f"{f.subject} {f.verb} {f.object}" for f in facts]
        scored = rank_by_embedding(
            query_str, facts, whats, embed_fn, threshold=-1.0,
        )
        return [fact for _, fact in scored]
