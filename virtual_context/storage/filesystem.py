"""FilesystemStore: markdown files with YAML frontmatter + JSON index (tag-based)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ..core.store import ContextStore
from ..types import DepthLevel, EngineStateSnapshot, SegmentMetadata, SessionStats, StoredSegment, StoredSummary, TagStats, TagSummary, TurnTagEntry, WorkingSetEntry


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _segment_to_index_entry(seg: StoredSegment) -> dict:
    return {
        "ref": seg.ref,
        "session_id": seg.session_id,
        "primary_tag": seg.primary_tag,
        "tags": seg.tags,
        "summary_tokens": seg.summary_tokens,
        "full_tokens": seg.full_tokens,
        "created_at": _dt_to_str(seg.created_at),
        "start_timestamp": _dt_to_str(seg.start_timestamp),
        "end_timestamp": _dt_to_str(seg.end_timestamp),
        "compression_ratio": seg.compression_ratio,
        "compaction_model": seg.compaction_model,
        "entities": seg.metadata.entities,
    }


def _segment_to_markdown(seg: StoredSegment) -> str:
    frontmatter = {
        "ref": seg.ref,
        "session_id": seg.session_id,
        "primary_tag": seg.primary_tag,
        "tags": seg.tags,
        "summary_tokens": seg.summary_tokens,
        "full_tokens": seg.full_tokens,
        "created_at": _dt_to_str(seg.created_at),
        "start_timestamp": _dt_to_str(seg.start_timestamp),
        "end_timestamp": _dt_to_str(seg.end_timestamp),
        "compaction_model": seg.compaction_model,
        "compression_ratio": seg.compression_ratio,
        "entities": seg.metadata.entities,
        "key_decisions": seg.metadata.key_decisions,
        "action_items": seg.metadata.action_items,
        "date_references": seg.metadata.date_references,
        "turn_count": seg.metadata.turn_count,
    }

    lines = ["---"]
    lines.append(yaml.dump(frontmatter, default_flow_style=False).strip())
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(seg.summary)
    lines.append("")
    lines.append("## Full Conversation")
    lines.append("")
    lines.append(seg.full_text)
    lines.append("")

    if seg.messages:
        lines.append("## Messages (JSON)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(seg.messages, indent=2, default=str))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _markdown_to_segment(text: str, ref: str) -> StoredSegment | None:
    """Parse a markdown file back into a StoredSegment."""
    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None

    if not fm:
        return None

    body = parts[2]

    # Extract summary
    summary = ""
    if "## Summary" in body:
        after_summary = body.split("## Summary", 1)[1]
        if "## Full Conversation" in after_summary:
            summary = after_summary.split("## Full Conversation", 1)[0].strip()
        else:
            summary = after_summary.strip()

    # Extract full text
    full_text = ""
    if "## Full Conversation" in body:
        after_full = body.split("## Full Conversation", 1)[1]
        if "## Messages (JSON)" in after_full:
            full_text = after_full.split("## Messages (JSON)", 1)[0].strip()
        else:
            full_text = after_full.strip()

    # Extract messages JSON
    messages = []
    if "## Messages (JSON)" in body:
        json_section = body.split("## Messages (JSON)", 1)[1]
        if "```json" in json_section and "```" in json_section.split("```json", 1)[1]:
            json_text = json_section.split("```json", 1)[1].split("```", 1)[0].strip()
            try:
                messages = json.loads(json_text)
            except json.JSONDecodeError:
                pass

    metadata = SegmentMetadata(
        entities=fm.get("entities", []),
        key_decisions=fm.get("key_decisions", []),
        action_items=fm.get("action_items", []),
        date_references=fm.get("date_references", []),
        turn_count=fm.get("turn_count", 0),
    )

    return StoredSegment(
        ref=fm.get("ref", ref),
        session_id=fm.get("session_id", ""),
        primary_tag=fm.get("primary_tag", "_general"),
        tags=fm.get("tags", []),
        summary=summary,
        summary_tokens=fm.get("summary_tokens", 0),
        full_text=full_text,
        full_tokens=fm.get("full_tokens", 0),
        messages=messages,
        metadata=metadata,
        created_at=_str_to_dt(fm["created_at"]) if "created_at" in fm else datetime.now(timezone.utc),
        start_timestamp=_str_to_dt(fm["start_timestamp"]) if "start_timestamp" in fm else datetime.now(timezone.utc),
        end_timestamp=_str_to_dt(fm["end_timestamp"]) if "end_timestamp" in fm else datetime.now(timezone.utc),
        compaction_model=fm.get("compaction_model", ""),
        compression_ratio=fm.get("compression_ratio", 0.0),
    )


def _segment_to_summary(seg: StoredSegment) -> StoredSummary:
    return StoredSummary(
        ref=seg.ref,
        primary_tag=seg.primary_tag,
        tags=seg.tags,
        summary=seg.summary,
        summary_tokens=seg.summary_tokens,
        full_tokens=seg.full_tokens,
        metadata=seg.metadata,
        created_at=seg.created_at,
        start_timestamp=seg.start_timestamp,
        end_timestamp=seg.end_timestamp,
    )


class FilesystemStore(ContextStore):
    """Store segments as markdown files organized by primary_tag directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._index_path = self.root / "_index.json"
        self._index: dict[str, dict] = {}
        self._aliases: dict[str, str] = {}
        self._ensure_root()
        self._load_index()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> None:
        if self._index_path.is_file():
            try:
                data = json.loads(self._index_path.read_text())
                self._index = {entry["ref"]: entry for entry in data}
            except (json.JSONDecodeError, KeyError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(list(self._index.values()), indent=2, default=str)
        )

    def _segment_path(self, primary_tag: str, ref: str) -> Path:
        tag_dir = self.root / primary_tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        return tag_dir / f"seg-{ref}.md"

    def store_segment(self, segment: StoredSegment) -> str:
        path = self._segment_path(segment.primary_tag, segment.ref)
        path.write_text(_segment_to_markdown(segment))
        self._index[segment.ref] = _segment_to_index_entry(segment)
        self._save_index()
        return segment.ref

    def get_segment(self, ref: str) -> StoredSegment | None:
        entry = self._index.get(ref)
        if not entry:
            return None
        primary_tag = entry["primary_tag"]
        path = self._segment_path(primary_tag, ref)
        if not path.is_file():
            return None
        return _markdown_to_segment(path.read_text(), ref)

    def get_summary(self, ref: str) -> StoredSummary | None:
        seg = self.get_segment(ref)
        return _segment_to_summary(seg) if seg else None

    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[StoredSummary]:
        if not tags:
            return []

        tag_set = set(tags)
        scored: list[tuple[int, dict]] = []

        for entry in self._index.values():
            entry_tags = set(entry.get("tags", []))
            overlap = len(tag_set & entry_tags)
            if overlap < min_overlap:
                continue

            created = _str_to_dt(entry["created_at"])
            if before and created >= before:
                continue
            if after and created <= after:
                continue

            scored.append((overlap, entry))

        # Sort by overlap desc, then created_at desc
        scored.sort(key=lambda x: (x[0], x[1]["created_at"]), reverse=True)
        scored = scored[:limit]

        results = []
        for _, entry in scored:
            seg = self.get_segment(entry["ref"])
            if seg:
                results.append(_segment_to_summary(seg))
        return results

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[StoredSummary]:
        """Keyword search against summary text and entities."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        tag_set = set(tags) if tags else None

        scored: list[tuple[float, dict]] = []
        for entry in self._index.values():
            if tag_set:
                entry_tags = set(entry.get("tags", []))
                if not (tag_set & entry_tags):
                    continue

            # Score: count matching terms in entities
            entities_text = " ".join(entry.get("entities", [])).lower()
            score = sum(1 for term in query_terms if term in entities_text)

            if score > 0:
                scored.append((score, entry))

        # Also check summaries for matches
        if not scored:
            for entry in self._index.values():
                if tag_set:
                    entry_tags = set(entry.get("tags", []))
                    if not (tag_set & entry_tags):
                        continue
                seg = self.get_segment(entry["ref"])
                if seg and any(term in seg.summary.lower() for term in query_terms):
                    scored.append((1.0, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, entry in scored[:limit]:
            seg = self.get_segment(entry["ref"])
            if seg:
                results.append(_segment_to_summary(seg))
        return results

    def get_all_tags(self) -> list[TagStats]:
        tag_map: dict[str, TagStats] = {}

        for entry in self._index.values():
            for tag in entry.get("tags", []):
                if tag not in tag_map:
                    tag_map[tag] = TagStats(tag=tag)

                stats = tag_map[tag]
                stats.usage_count += 1
                stats.total_full_tokens += entry.get("full_tokens", 0)
                stats.total_summary_tokens += entry.get("summary_tokens", 0)

                created = _str_to_dt(entry["created_at"])
                if stats.oldest_segment is None or created < stats.oldest_segment:
                    stats.oldest_segment = created
                if stats.newest_segment is None or created > stats.newest_segment:
                    stats.newest_segment = created

        return sorted(tag_map.values(), key=lambda s: s.usage_count, reverse=True)

    def get_session_stats(self) -> list[SessionStats]:
        session_map: dict[str, SessionStats] = {}

        for entry in self._index.values():
            sid = entry.get("session_id", "")
            if not sid:
                continue

            if sid not in session_map:
                session_map[sid] = SessionStats(session_id=sid)

            stats = session_map[sid]
            stats.segment_count += 1
            stats.total_full_tokens += entry.get("full_tokens", 0)
            stats.total_summary_tokens += entry.get("summary_tokens", 0)

            created = _str_to_dt(entry["created_at"])
            if stats.oldest_segment is None or created < stats.oldest_segment:
                stats.oldest_segment = created
            if stats.newest_segment is None or created > stats.newest_segment:
                stats.newest_segment = created

            model = entry.get("compaction_model", "")
            if model:
                stats.compaction_model = model

            for tag in entry.get("tags", []):
                if tag not in stats.distinct_tags:
                    stats.distinct_tags.append(tag)

        for stats in session_map.values():
            if stats.total_full_tokens > 0:
                stats.compression_ratio = round(
                    stats.total_summary_tokens / stats.total_full_tokens, 3
                )
            stats.distinct_tags.sort()

        return sorted(
            session_map.values(),
            key=lambda s: s.newest_segment or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    def get_tag_aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def set_tag_alias(self, alias: str, canonical: str) -> None:
        self._aliases[alias] = canonical

    def delete_segment(self, ref: str) -> bool:
        entry = self._index.pop(ref, None)
        if not entry:
            return False
        path = self._segment_path(entry["primary_tag"], ref)
        if path.is_file():
            path.unlink()
        self._save_index()
        return True

    def cleanup(
        self,
        max_age: timedelta | None = None,
        max_total_tokens: int | None = None,
    ) -> int:
        deleted = 0
        now = datetime.now(timezone.utc)

        if max_age:
            cutoff = now - max_age
            to_delete = [
                ref for ref, entry in self._index.items()
                if _str_to_dt(entry["created_at"]) < cutoff
            ]
            for ref in to_delete:
                self.delete_segment(ref)
                deleted += 1

        return deleted

    def save_tag_summary(self, tag_summary: TagSummary) -> None:
        ts_dir = self.root / "_tag_summaries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        path = ts_dir / f"{tag_summary.tag}.json"
        data = {
            "tag": tag_summary.tag,
            "summary": tag_summary.summary,
            "summary_tokens": tag_summary.summary_tokens,
            "source_segment_refs": tag_summary.source_segment_refs,
            "source_turn_numbers": tag_summary.source_turn_numbers,
            "covers_through_turn": tag_summary.covers_through_turn,
            "created_at": _dt_to_str(tag_summary.created_at),
            "updated_at": _dt_to_str(tag_summary.updated_at),
        }
        path.write_text(json.dumps(data, indent=2))

    def get_tag_summary(self, tag: str) -> TagSummary | None:
        path = self.root / "_tag_summaries" / f"{tag}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return TagSummary(
            tag=data["tag"],
            summary=data.get("summary", ""),
            summary_tokens=data.get("summary_tokens", 0),
            source_segment_refs=data.get("source_segment_refs", []),
            source_turn_numbers=data.get("source_turn_numbers", []),
            covers_through_turn=data.get("covers_through_turn", -1),
            created_at=_str_to_dt(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=_str_to_dt(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    def get_all_tag_summaries(self) -> list[TagSummary]:
        ts_dir = self.root / "_tag_summaries"
        if not ts_dir.is_dir():
            return []
        results: list[TagSummary] = []
        for path in sorted(ts_dir.glob("*.json")):
            ts = self.get_tag_summary(path.stem)
            if ts:
                results.append(ts)
        return results

    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
    ) -> list[StoredSegment]:
        if not tags:
            return []
        tag_set = set(tags)
        scored: list[tuple[int, str]] = []
        for ref, entry in self._index.items():
            entry_tags = set(entry.get("tags", []))
            overlap = len(tag_set & entry_tags)
            if overlap >= min_overlap:
                scored.append((overlap, ref))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[StoredSegment] = []
        for _, ref in scored[:limit]:
            seg = self.get_segment(ref)
            if seg:
                results.append(seg)
        return results

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        state_dir = self.root / "_engine_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{state.session_id}.json"
        data = {
            "session_id": state.session_id,
            "compacted_through": state.compacted_through,
            "turn_count": state.turn_count,
            "saved_at": _dt_to_str(state.saved_at),
            "turn_tag_entries": [
                {
                    "turn_number": e.turn_number,
                    "message_hash": e.message_hash,
                    "tags": e.tags,
                    "primary_tag": e.primary_tag,
                    "timestamp": _dt_to_str(e.timestamp),
                }
                for e in state.turn_tag_entries
            ],
            "split_processed_tags": state.split_processed_tags,
            "working_set": [
                {
                    "tag": ws.tag,
                    "depth": ws.depth.value if hasattr(ws.depth, 'value') else ws.depth,
                    "tokens": ws.tokens,
                    "last_accessed_turn": ws.last_accessed_turn,
                }
                for ws in state.working_set
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_engine_state(self, session_id: str) -> EngineStateSnapshot | None:
        path = self.root / "_engine_state" / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        entries = [
            TurnTagEntry(
                turn_number=e["turn_number"],
                message_hash=e["message_hash"],
                tags=e["tags"],
                primary_tag=e["primary_tag"],
                timestamp=_str_to_dt(e["timestamp"]),
            )
            for e in data.get("turn_tag_entries", [])
        ]
        working_set = [
            WorkingSetEntry(
                tag=ws["tag"],
                depth=DepthLevel(ws["depth"]),
                tokens=ws.get("tokens", 0),
                last_accessed_turn=ws.get("last_accessed_turn", 0),
            )
            for ws in data.get("working_set", [])
        ]
        return EngineStateSnapshot(
            session_id=data["session_id"],
            compacted_through=data.get("compacted_through", 0),
            turn_tag_entries=entries,
            turn_count=data.get("turn_count", 0),
            saved_at=_str_to_dt(data["saved_at"]) if "saved_at" in data else datetime.now(timezone.utc),
            split_processed_tags=data.get("split_processed_tags", []),
            working_set=working_set,
        )
