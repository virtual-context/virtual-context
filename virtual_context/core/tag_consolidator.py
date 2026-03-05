"""Tag Consolidation Engine — post-compaction semantic clustering of tags.

Scans the full tag universe in a store, identifies semantically related tags
that should be treated as equivalent during retrieval, and writes alias
mappings + backfills segment_tags so retrieval covers both canonical and
variant tags.

Usage:
    from virtual_context.core.tag_consolidator import consolidate_tags
    result = consolidate_tags(store, llm_provider)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .llm_utils import parse_llm_json
from .store import ContextStore
from ..types import LLMProvider

logger = logging.getLogger(__name__)


# ── prompt ──────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a tag taxonomy expert. Output valid JSON only. "
    "No markdown fences, no extra text."
)

_CONSOLIDATION_PROMPT = """\
Below is a list of tags from a conversation memory store.  Each tag has a
description explaining what it covers.

Your job: identify groups of tags that refer to THE SAME broad topic and
should be unified so that a search for any member of the group also finds
content stored under the other members.

Rules:
1. Only group tags that are truly about the same domain.  "model-kit" and
   "model-tanks" both refer to scale-model hobby projects — group them.
   But "model-kit" and "data-model" are unrelated — do NOT group them.
2. Each group must have exactly one "canonical" tag (the most general one)
   and one or more "aliases" (the more specific or variant tags).
3. A tag may appear in at most one group. If it doesn't belong to any group
   leave it out.
4. Only create groups where cross-referencing genuinely helps retrieval.
   Trivial morphological variants (plurals, hyphenation) are already handled
   elsewhere — focus on SEMANTIC relationships.
5. Keep the number of groups reasonable.  Quality over quantity.
6. Many tags have "(no description)".  Use the tag NAME as a semantic signal:
   "model-tanks" clearly relates to scale modeling even without a description.

Tags (tag → description):
{tag_list}

Respond with JSON:
{{
  "groups": [
    {{
      "canonical": "the-main-tag",
      "aliases": ["variant-1", "variant-2"],
      "reason": "one sentence explaining the grouping"
    }}
  ]
}}"""


# ── types ───────────────────────────────────────────────────────────────

@dataclass
class ConsolidationGroup:
    """A cluster of semantically equivalent tags."""
    canonical: str
    aliases: list[str]
    reason: str = ""


@dataclass
class ConsolidationResult:
    """Result of running tag consolidation on a store."""
    groups: list[ConsolidationGroup] = field(default_factory=list)
    aliases_written: int = 0
    segment_tags_added: int = 0


# ── core logic ──────────────────────────────────────────────────────────

def consolidate_tags(
    store: ContextStore,
    llm: LLMProvider,
    *,
    dry_run: bool = False,
    batch_size: int = 500,
) -> ConsolidationResult:
    """Run tag consolidation on *store*.

    1. Load all tags with descriptions (from tag_summaries).
    2. Send to LLM to identify semantic clusters.
    3. Write alias mappings to tag_aliases table.
    4. Backfill segment_tags: for every segment that has an alias tag,
       also add the canonical tag so retrieval covers both.

    Args:
        store: The context store to consolidate.
        llm: LLM provider for semantic clustering.
        dry_run: If True, compute groups but don't write to store.
        batch_size: Max tags per LLM call (for very large stores).

    Returns:
        ConsolidationResult with groups found and counts of writes.
    """
    # 1. Gather tags + descriptions
    all_tags = store.get_all_tags()
    tag_summaries = {ts.tag: ts for ts in store.get_all_tag_summaries()}

    # For tags without summaries, pull a snippet from their segment text.
    # This is critical — unsummarized tags are the most likely to be
    # orphaned and need consolidation the most.
    orphan_descriptions = _get_orphan_tag_descriptions(store, tag_summaries)

    tag_entries: list[tuple[str, str]] = []
    for ts in all_tags:
        summary = tag_summaries.get(ts.tag)
        desc = ""
        if summary:
            desc = summary.description or summary.summary[:150]
        else:
            desc = orphan_descriptions.get(ts.tag, "")
        tag_entries.append((ts.tag, desc))

    if not tag_entries:
        logger.info("No tags found — nothing to consolidate.")
        return ConsolidationResult()

    logger.info("Consolidating %d tags...", len(tag_entries))

    # 2. Phase 1 — batch tag entries and call LLM for each batch
    all_groups: list[ConsolidationGroup] = []
    for batch_start in range(0, len(tag_entries), batch_size):
        batch = tag_entries[batch_start:batch_start + batch_size]
        tag_list = "\n".join(
            f"- {tag}: {desc}" if desc else f"- {tag}: (no description)"
            for tag, desc in batch
        )
        prompt = _CONSOLIDATION_PROMPT.format(tag_list=tag_list)

        try:
            response, _ = llm.complete(system=_SYSTEM, user=prompt, max_tokens=4096)
            groups = _parse_response(response)
            all_groups.extend(groups)
        except Exception as e:
            logger.error("LLM consolidation call failed: %s", e)

    if not all_groups:
        logger.info("No consolidation groups identified.")
        return ConsolidationResult()

    # Phase 2 — cross-batch merge: if a tag is canonical in one group
    # and an alias in another, merge the groups transitively.
    all_groups = _merge_transitive_groups(all_groups)

    logger.info("Found %d consolidation groups (after merge).", len(all_groups))
    for g in all_groups:
        logger.info("  %s ← %s (%s)", g.canonical, g.aliases, g.reason)

    result = ConsolidationResult(groups=all_groups)

    if dry_run:
        return result

    # 3. Write aliases
    existing_aliases = store.get_tag_aliases()
    for group in all_groups:
        for alias in group.aliases:
            if alias not in existing_aliases:
                store.set_tag_alias(alias, group.canonical)
                result.aliases_written += 1

    logger.info("Wrote %d new aliases.", result.aliases_written)

    # 4. Backfill segment_tags — add canonical tag to segments that have alias tags
    result.segment_tags_added = _backfill_segment_tags(store, all_groups)
    logger.info("Backfilled %d segment_tags entries.", result.segment_tags_added)

    return result


def _get_orphan_tag_descriptions(
    store: ContextStore,
    tag_summaries: dict,
) -> dict[str, str]:
    """Get descriptions for tags that have no tag_summary entry.

    Delegates to the store's ``get_orphan_tag_snippets`` method to fetch
    one segment summary snippet per orphan tag, so the LLM has context
    for consolidation decisions.
    """
    try:
        rows = store.get_orphan_tag_snippets(limit=1000)
        return {row["tag"]: row["snippet"] for row in rows}
    except Exception as e:
        logger.warning("Failed to fetch orphan tag descriptions: %s", e)
        return {}


def _merge_transitive_groups(
    groups: list[ConsolidationGroup],
) -> list[ConsolidationGroup]:
    """Merge groups that share tags transitively.

    If group A has canonical="scale-model" aliases=["model-kit"] and group B
    has canonical="model-kit" aliases=["model-kit-assembly"], merge them into
    one group with canonical="scale-model" and all others as aliases.

    Also handles the case where a canonical from one group appears as an alias
    in another, or two groups share an alias.
    """
    # Build union-find over all tag names mentioned in groups
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Register all tags and union within each group
    for g in groups:
        parent.setdefault(g.canonical, g.canonical)
        for alias in g.aliases:
            parent.setdefault(alias, alias)
            union(alias, g.canonical)

    # Collect clusters
    clusters: dict[str, set[str]] = {}
    all_tags_in_groups = set()
    for g in groups:
        all_tags_in_groups.add(g.canonical)
        all_tags_in_groups.update(g.aliases)

    for tag in all_tags_in_groups:
        root = find(tag)
        clusters.setdefault(root, set()).add(tag)

    # Pick canonical for each cluster: prefer the tag that was canonical
    # in the most groups, breaking ties by shortest name
    canonical_counts: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for g in groups:
        canonical_counts[g.canonical] = canonical_counts.get(g.canonical, 0) + 1
        if g.reason:
            reasons.setdefault(find(g.canonical), []).append(g.reason)

    merged: list[ConsolidationGroup] = []
    for root, members in clusters.items():
        if len(members) <= 1:
            continue
        # Pick canonical: highest canonical_count, then shortest
        best = max(
            members,
            key=lambda t: (canonical_counts.get(t, 0), -len(t)),
        )
        aliases = sorted(members - {best})
        reason = (reasons.get(root, [""]))[0]
        merged.append(ConsolidationGroup(
            canonical=best,
            aliases=aliases,
            reason=reason,
        ))

    return merged


def _backfill_segment_tags(
    store: ContextStore,
    groups: list[ConsolidationGroup],
) -> int:
    """For each group, add the canonical tag to segments that only have aliases.

    This ensures segments are discoverable via the canonical tag. The
    retriever's alias ride-along handles the reverse direction at query
    time — no need to add every alias to every segment.
    """
    added = 0

    for group in groups:
        # Find segments that have any alias tag
        alias_segments = store.get_summaries_by_tags(
            tags=group.aliases,
            min_overlap=1,
            limit=1000,
        )

        # Find segments that already have the canonical tag
        canonical_segments = store.get_summaries_by_tags(
            tags=[group.canonical],
            min_overlap=1,
            limit=1000,
        )
        canonical_refs = {s.ref for s in canonical_segments}

        for summary in alias_segments:
            if summary.ref not in canonical_refs:
                segment = store.get_segment(summary.ref)
                if segment and group.canonical not in segment.tags:
                    segment.tags.append(group.canonical)
                    store.store_segment(segment)
                    added += 1
                    logger.debug(
                        "  Added '%s' to segment %s",
                        group.canonical,
                        summary.ref[:12],
                    )

    return added


def _parse_response(response: str) -> list[ConsolidationGroup]:
    """Parse LLM JSON response into ConsolidationGroup list."""
    data = parse_llm_json(response)
    if not data:
        logger.error("Failed to parse consolidation response.")
        return []

    groups = []
    for g in data.get("groups", []):
        canonical = g.get("canonical", "")
        aliases = g.get("aliases", [])
        reason = g.get("reason", "")
        if canonical and aliases:
            groups.append(ConsolidationGroup(
                canonical=canonical,
                aliases=aliases,
                reason=reason,
            ))
    return groups
