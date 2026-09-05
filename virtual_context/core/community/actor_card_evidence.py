"""ActorCardEvidenceService: explicit dependencies for community memory work."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from .canonical_sources import physical_rows_by_group, physical_rows_by_id

# Keep the existing operator log channel stable across the extraction.
logger = logging.getLogger("virtual_context.core.compaction_pipeline")


class ActorCardEvidenceService:
    def __init__(self, *, store, paired_agent_replies: Callable) -> None:
        self._store = store
        self._paired_agent_replies = paired_agent_replies

    def fingerprint_records(self, fact_sources: list, turn_sources: list):
        """Stream exact ancillary evidence, including corrected old replies.

        The input manifest must cover evidence used by admission, not just
        derived fact strings. No timestamp high-water mark can establish this:
        corrections preserve source ids and may concern an old carryover.
        """
        for (owner, group), reply in sorted(self._paired_agent_replies(turn_sources).items()):
            yield {"kind": "paired_reply", "owner": owner, "group": group, "reply": reply}
        refs = sorted(
            {
                (source.owner_conversation_id, source.fact.segment_ref)
                for source in fact_sources
                if source.fact.segment_ref
            }
        )
        for owner, ref in refs:
            segment = self._store.get_segment(ref, conversation_id=owner)
            ids = list(segment.metadata.canonical_turn_ids or ()) if segment else []
            yield {"kind": "fact_segment", "owner": owner, "ref": ref, "source_ids": ids}
            rows = physical_rows_by_id(self._store, ((owner, source_id) for source_id in ids))
            for source_id in ids:
                row = rows.get((owner, source_id))
                if row is None:
                    yield {"kind": "missing_source", "owner": owner, "id": source_id}
                    continue
                yield {
                    "kind": "fact_source",
                    "owner": owner,
                    "id": source_id,
                    "actor": row.sender_actor_id,
                    "audience": row.audience_conversation_id,
                    "channel": row.origin_channel_id,
                    "attribution_version": row.audience_attribution_version,
                    "content": row.user_content,
                    "turn": row.turn_number,
                    "timestamp": row.created_at or row.first_seen_at or "",
                }

    def paired_agent_replies(self, turn_sources: list) -> dict:
        """Map (conversation_id, turn_group_number) -> the agent's reply.

        The agent's live adjudication of a request is admission evidence:
        a refused behavior-change request must not become a preference, so
        the judge needs the paired assistant response beside each cited
        message. The assistant halves already live in canonical storage
        keyed by the same turn group; this is a read, not new state.
        """
        replies: dict[tuple[str, int], str] = {}
        groups_by_owner: dict[str, set[int]] = {}
        for source in turn_sources:
            owner = getattr(source.turn, "conversation_id", "")
            group = getattr(source.turn, "turn_group_number", None)
            if owner and type(group) is int and group >= 0:
                groups_by_owner.setdefault(owner, set()).add(group)
        for conversation_id, groups in sorted(groups_by_owner.items()):
            try:
                grouped = physical_rows_by_group(self._store, conversation_id, groups)
            except Exception:
                continue
            for group, rows in grouped.items():
                for row in rows:
                    text = (getattr(row, "assistant_content", "") or "").strip()
                    if text:
                        replies.setdefault((conversation_id, group), text)
        return replies

    def prompt_turns(
        self,
        turn_sources: list,
        *,
        max_chars: int = 96_000,
    ) -> list[dict]:
        """Render a bounded, deterministic set of actor-authored messages.

        Discord messages are normally small, but canonical ingestion is also
        used by API callers. Individual and aggregate bounds prevent one actor
        from turning card curation into an unbounded model call. A truncated
        message remains visibly marked so neither model can treat it as exact
        evidence for a dropped qualifier. Each message carries the agent's
        paired reply when one exists, bounded, so the models can apply the
        honored-versus-refused adjudication rule; a message whose group is
        unassigned gets no reply, and the fail-closed no-honored-signal
        default then governs behavior-change requests.
        """
        replies = self._paired_agent_replies(turn_sources)
        rendered: list[dict] = []
        used = 0
        for source in turn_sources:
            content = (source.turn.user_content or "").strip()
            if not content:
                continue
            truncated = len(content) > 4_000
            if truncated:
                content = (
                    content[:1_940]
                    + "\n...[middle omitted; do not infer omitted text]...\n"
                    + content[-1_940:]
                )
            item = {
                "id": source.turn.canonical_turn_id,
                "timestamp": (source.turn.created_at or source.turn.first_seen_at or ""),
                "audience_conversation_id": (source.audience_conversation_id),
                "audience_channel_id": source.audience_channel_id,
                "content": content,
                "truncated": truncated,
            }
            raw_group = getattr(source.turn, "turn_group_number", None)
            try:
                group = int(raw_group) if raw_group is not None else -1
            except (TypeError, ValueError):
                group = -1
            if group >= 0:
                reply = replies.get(
                    (source.turn.conversation_id, group),
                    "",
                )
                if reply:
                    if len(reply) > 600:
                        reply = reply[:600] + " ...[truncated]"
                    item["agent_reply"] = reply
            cost = len(json.dumps(item, separators=(",", ":")))
            if used + cost > max(0, int(max_chars)):
                break
            rendered.append(item)
            used += cost
        return rendered

    def evidence_segments(
        self,
        actor_id: str,
        audience_conversation_id: str,
        sources: list,
        candidate_fact_ids: set[str],
        *,
        required_fact_ids: set[str] | None = None,
        max_chars: int = 64_000,
    ) -> tuple[list[dict], set[tuple[str, str]]]:
        """Return bounded actor-authored turns from candidate-cited segments.

        Selection is provenance-based: canonical actor ids and segment source
        mappings decide which messages are evidence. Message text is never
        regex-classified. Uncited segments remain available to the admission
        model as compact facts, not as unrelated raw conversation text.
        """
        from ...types import AUDIENCE_ATTRIBUTION_VERSION

        source_by_id = {
            source.fact.id: source
            for source in sources
            if source.audience_conversation_id == audience_conversation_id
        }
        required_fact_ids = set(required_fact_ids or ())
        candidate_refs = {
            (
                source_by_id[fact_id].owner_conversation_id,
                source_by_id[fact_id].fact.segment_ref,
            )
            for fact_id in candidate_fact_ids
            if fact_id in source_by_id and source_by_id[fact_id].fact.segment_ref
        }
        required_refs = {
            (
                source_by_id[fact_id].owner_conversation_id,
                source_by_id[fact_id].fact.segment_ref,
            )
            for fact_id in required_fact_ids
            if fact_id in source_by_id and source_by_id[fact_id].fact.segment_ref
        }
        by_ref: dict[tuple[str, str], dict] = {}
        for source in sources:
            ref = source.fact.segment_ref
            ref_key = (source.owner_conversation_id, ref)
            if not ref or ref_key not in candidate_refs or ref_key in by_ref:
                continue
            segment = self._store.get_segment(
                ref,
                conversation_id=source.owner_conversation_id,
            )
            if segment is None:
                continue
            messages: list[dict] = []
            try:
                newest_time = float(segment.end_timestamp.timestamp())
            except (AttributeError, TypeError, ValueError, OSError):
                newest_time = float("-inf")
            source_ids = list(segment.metadata.canonical_turn_ids or [])
            # Only this cited segment's physical sources can enter the judge.
            # Reading each segment separately also bounds retained raw text.
            rows = physical_rows_by_id(
                self._store,
                ((source.owner_conversation_id, source_id) for source_id in source_ids),
            )
            for canonical_id in source_ids:
                row = rows.get((source.owner_conversation_id, canonical_id))
                content = (row.user_content or "").strip() if row else ""
                if (
                    row is None
                    or row.sender_actor_id != actor_id
                    or row.audience_conversation_id != audience_conversation_id
                    or int(row.audience_attribution_version or 0) != AUDIENCE_ATTRIBUTION_VERSION
                    or not content
                ):
                    continue
                if len(content) > 1200:
                    content = content[:580] + "\n...[middle truncated]...\n" + content[-580:]
                messages.append(
                    {
                        "turn": row.turn_number,
                        "timestamp": (row.created_at or row.first_seen_at or ""),
                        "content": content,
                    }
                )
            if messages:
                by_ref[ref_key] = {
                    "owner_conversation_id": source.owner_conversation_id,
                    "segment_ref": ref,
                    "messages": messages,
                    "_newest_time": newest_time,
                }

        ordered = sorted(
            by_ref.values(),
            key=lambda item: (
                0
                if (
                    item["owner_conversation_id"],
                    item["segment_ref"],
                )
                in required_refs
                else 1,
                -item["_newest_time"],
                item["owner_conversation_id"],
                item["segment_ref"],
            ),
        )
        admitted: list[dict] = []
        admitted_refs: set[tuple[str, str]] = set()
        used = 0
        for item in ordered:
            public = {
                "owner_conversation_id": item["owner_conversation_id"],
                "segment_ref": item["segment_ref"],
                "messages": item["messages"],
            }
            cost = len(json.dumps(public, separators=(",", ":")))
            if used + cost > max_chars:
                continue
            admitted.append(public)
            admitted_refs.add(
                (
                    item["owner_conversation_id"],
                    item["segment_ref"],
                )
            )
            used += cost
        return admitted, admitted_refs
