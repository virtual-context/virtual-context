"""FilesystemStore: markdown files with YAML frontmatter + JSON index."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ..core.store import ContextStore
from ..types import DomainStats, SegmentMetadata, StoredSegment, StoredSummary


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
        "domain": seg.domain,
        "secondary_domains": seg.secondary_domains,
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
        "domain": seg.domain,
        "secondary_domains": seg.secondary_domains,
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
        domain=fm.get("domain", "_general"),
        secondary_domains=fm.get("secondary_domains", []),
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
        domain=seg.domain,
        secondary_domains=seg.secondary_domains,
        summary=seg.summary,
        summary_tokens=seg.summary_tokens,
        full_tokens=seg.full_tokens,
        metadata=seg.metadata,
        created_at=seg.created_at,
        start_timestamp=seg.start_timestamp,
        end_timestamp=seg.end_timestamp,
    )


class FilesystemStore(ContextStore):
    """Store segments as markdown files organized by domain directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._index_path = self.root / "_index.json"
        self._index: dict[str, dict] = {}
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

    def _segment_path(self, domain: str, ref: str) -> Path:
        domain_dir = self.root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        return domain_dir / f"seg-{ref}.md"

    async def store_segment(self, segment: StoredSegment) -> str:
        path = self._segment_path(segment.domain, segment.ref)
        path.write_text(_segment_to_markdown(segment))
        self._index[segment.ref] = _segment_to_index_entry(segment)
        self._save_index()
        return segment.ref

    async def get_segment(self, ref: str) -> StoredSegment | None:
        entry = self._index.get(ref)
        if not entry:
            return None
        domain = entry["domain"]
        path = self._segment_path(domain, ref)
        if not path.is_file():
            return None
        return _markdown_to_segment(path.read_text(), ref)

    async def get_summary(self, ref: str) -> StoredSummary | None:
        seg = await self.get_segment(ref)
        return _segment_to_summary(seg) if seg else None

    async def get_summaries(
        self,
        domain: str | None = None,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> list[StoredSummary]:
        entries = list(self._index.values())

        if domain:
            entries = [e for e in entries if e["domain"] == domain]
        if before:
            entries = [e for e in entries if _str_to_dt(e["created_at"]) < before]
        if after:
            entries = [e for e in entries if _str_to_dt(e["created_at"]) > after]

        # Sort newest first
        entries.sort(key=lambda e: e["created_at"], reverse=True)
        entries = entries[:limit]

        results = []
        for entry in entries:
            seg = await self.get_segment(entry["ref"])
            if seg:
                results.append(_segment_to_summary(seg))
        return results

    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        limit: int = 5,
    ) -> list[StoredSummary]:
        """Keyword search against summary text and entities."""
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored: list[tuple[float, dict]] = []
        for entry in self._index.values():
            if domains and entry["domain"] not in domains:
                continue

            # Score: count matching terms in entities
            entities_text = " ".join(entry.get("entities", [])).lower()
            score = sum(1 for term in query_terms if term in entities_text)

            if score > 0:
                scored.append((score, entry))

        # Also check summaries for matches (load from disk)
        if not scored:
            for entry in self._index.values():
                if domains and entry["domain"] not in domains:
                    continue
                seg = await self.get_segment(entry["ref"])
                if seg and any(term in seg.summary.lower() for term in query_terms):
                    scored.append((1.0, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, entry in scored[:limit]:
            seg = await self.get_segment(entry["ref"])
            if seg:
                results.append(_segment_to_summary(seg))
        return results

    async def list_domains(self) -> list[DomainStats]:
        domain_map: dict[str, DomainStats] = {}

        for entry in self._index.values():
            domain = entry["domain"]
            if domain not in domain_map:
                domain_map[domain] = DomainStats(domain=domain)

            stats = domain_map[domain]
            stats.segment_count += 1
            stats.total_full_tokens += entry.get("full_tokens", 0)
            stats.total_summary_tokens += entry.get("summary_tokens", 0)

            created = _str_to_dt(entry["created_at"])
            if stats.oldest_segment is None or created < stats.oldest_segment:
                stats.oldest_segment = created
            if stats.newest_segment is None or created > stats.newest_segment:
                stats.newest_segment = created

        return sorted(domain_map.values(), key=lambda s: s.segment_count, reverse=True)

    async def delete_segment(self, ref: str) -> bool:
        entry = self._index.pop(ref, None)
        if not entry:
            return False
        path = self._segment_path(entry["domain"], ref)
        if path.is_file():
            path.unlink()
        self._save_index()
        return True

    async def cleanup(
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
                await self.delete_segment(ref)
                deleted += 1

        return deleted
