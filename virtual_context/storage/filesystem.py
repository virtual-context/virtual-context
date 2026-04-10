"""FilesystemStore: markdown files with YAML frontmatter + JSON index (tag-based)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ..core.store import ContextStore
from ..types import ChunkEmbedding, ConversationStats, DepthLevel, EngineStateSnapshot, QuoteResult, SegmentMetadata, StoredSegment, StoredSummary, TagStats, TagSummary, TurnTagEntry, WorkingSetEntry
from .helpers import dt_to_str as _dt_to_str, str_to_dt as _str_to_dt, extract_excerpt as _extract_excerpt


def _segment_to_index_entry(seg: StoredSegment) -> dict:
    return {
        "ref": seg.ref,
        "conversation_id": seg.conversation_id,
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
        "conversation_id": seg.conversation_id,
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
        "code_refs": getattr(seg.metadata, "code_refs", []),
        "turn_count": seg.metadata.turn_count,
        "start_turn_number": getattr(seg.metadata, "start_turn_number", -1),
        "end_turn_number": getattr(seg.metadata, "end_turn_number", -1),
        "generated_by_turn_id": getattr(seg.metadata, "generated_by_turn_id", ""),
    }
    if seg.metadata.session_date:
        frontmatter["session_date"] = seg.metadata.session_date

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
        code_refs=fm.get("code_refs", []),
        turn_count=fm.get("turn_count", 0),
        start_turn_number=fm.get("start_turn_number", -1),
        end_turn_number=fm.get("end_turn_number", -1),
        generated_by_turn_id=fm.get("generated_by_turn_id", ""),
        session_date=fm.get("session_date", ""),
    )

    return StoredSegment(
        ref=fm.get("ref", ref),
        conversation_id=fm.get("conversation_id", fm.get("session_id", "")),
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
        self._aliases_path = self.root / "_aliases.json"
        self._index: dict[str, dict] = {}
        self._aliases: dict[str, str] = {}
        self._conversation_aliases: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self.search_config = None  # set by engine after construction
        self._ensure_root()
        self._load_index()
        self._load_aliases()
        self._vcattach_aliases_path = self.root / "_vcattach_aliases.json"
        self._vcattach_aliases: dict[str, str] = {}
        self._load_vcattach_aliases()

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
        tmp_path = self._index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(list(self._index.values()), indent=2, default=str)
        )
        os.replace(str(tmp_path), str(self._index_path))

    def _load_aliases(self) -> None:
        if self._aliases_path.is_file():
            try:
                raw = json.loads(self._aliases_path.read_text())
                if (
                    isinstance(raw, dict)
                    and isinstance(raw.get("global"), dict)
                    and isinstance(raw.get("by_conversation"), dict)
                ):
                    self._aliases = {
                        str(alias): str(canonical)
                        for alias, canonical in raw.get("global", {}).items()
                    }
                    self._conversation_aliases = {
                        str(conv_id): {
                            str(alias): str(canonical)
                            for alias, canonical in aliases.items()
                        }
                        for conv_id, aliases in raw.get("by_conversation", {}).items()
                        if isinstance(aliases, dict)
                    }
                elif isinstance(raw, dict):
                    self._aliases = {
                        str(alias): str(canonical)
                        for alias, canonical in raw.items()
                    }
                    self._conversation_aliases = {}
                else:
                    self._aliases = {}
                    self._conversation_aliases = {}
            except (json.JSONDecodeError, OSError):
                self._aliases = {}
                self._conversation_aliases = {}
        else:
            self._aliases = {}
            self._conversation_aliases = {}

    def _save_aliases(self) -> None:
        tmp_path = self._aliases_path.with_suffix(".json.tmp")
        payload = {
            "global": self._aliases,
            "by_conversation": self._conversation_aliases,
        }
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(str(tmp_path), str(self._aliases_path))

    def _segment_path(self, primary_tag: str, ref: str) -> Path:
        # Sanitize tag and ref to prevent path traversal
        safe_tag = primary_tag.replace("/", "_").replace("\\", "_").replace("..", "_")
        safe_ref = ref.replace("/", "_").replace("\\", "_").replace("..", "_")
        tag_dir = self.root / safe_tag
        resolved = tag_dir.resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError(f"Path traversal detected in primary_tag: {primary_tag}")
        tag_dir.mkdir(parents=True, exist_ok=True)
        return tag_dir / f"seg-{safe_ref}.md"

    def store_segment(self, segment: StoredSegment) -> str:
        path = self._segment_path(segment.primary_tag, segment.ref)
        path.write_text(_segment_to_markdown(segment))
        with self._lock:
            self._index[segment.ref] = _segment_to_index_entry(segment)
            self._save_index()
        return segment.ref

    def get_segment(self, ref: str, *, conversation_id: str | None = None) -> StoredSegment | None:
        entry = self._index.get(ref)
        if not entry:
            return None
        if conversation_id is not None:
            if entry.get("conversation_id", "") != conversation_id:
                return None
        primary_tag = entry["primary_tag"]
        path = self._segment_path(primary_tag, ref)
        if not path.is_file():
            return None
        return _markdown_to_segment(path.read_text(), ref)

    def get_summary(self, ref: str, *, conversation_id: str | None = None) -> StoredSummary | None:
        seg = self.get_segment(ref, conversation_id=conversation_id)
        return _segment_to_summary(seg) if seg else None

    def get_summaries_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 10,
        before: datetime | None = None,
        after: datetime | None = None,
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        if not tags:
            return []

        tag_set = set(tags)
        scored: list[tuple[int, dict]] = []

        for entry in self._index.values():
            if conversation_id is not None:
                if entry.get("conversation_id", "") != conversation_id:
                    continue

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
        conversation_id: str | None = None,
    ) -> list[StoredSummary]:
        """Keyword search against summary text and entities."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        tag_set = set(tags) if tags else None

        scored: list[tuple[float, dict]] = []
        for entry in self._index.values():
            if conversation_id is not None:
                if entry.get("conversation_id", "") != conversation_id:
                    continue
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
                if conversation_id is not None:
                    if entry.get("conversation_id", "") != conversation_id:
                        continue
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

    def search_full_text(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[QuoteResult]:
        query_lower = query.lower()
        results: list[QuoteResult] = []
        for entry in self._index.values():
            if conversation_id is not None:
                if entry.get("conversation_id", "") != conversation_id:
                    continue
            seg = self.get_segment(entry["ref"])
            if seg and query_lower in seg.full_text.lower():
                _sc = self.search_config
                _ctx = _sc.excerpt_context_chars if _sc else 200
                excerpt = _extract_excerpt(seg.full_text, query, context_chars=_ctx)
                results.append(QuoteResult(
                    text=excerpt,
                    tag=seg.primary_tag,
                    segment_ref=seg.ref,
                    tags=seg.tags,
                    match_type="like",
                    session_date=seg.metadata.session_date,
                ))
                if len(results) >= limit:
                    break
        return results

    def get_all_tags(self, conversation_id: str | None = None) -> list[TagStats]:
        tag_map: dict[str, TagStats] = {}

        for entry in self._index.values():
            if conversation_id is not None:
                cid = entry.get("conversation_id", entry.get("session_id", ""))
                if cid != conversation_id:
                    continue
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

    def get_conversation_stats(self) -> list[ConversationStats]:
        conversation_map: dict[str, ConversationStats] = {}

        for entry in self._index.values():
            cid = entry.get("conversation_id", entry.get("session_id", ""))
            if not cid:
                continue

            if cid not in conversation_map:
                conversation_map[cid] = ConversationStats(conversation_id=cid)

            stats = conversation_map[cid]
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

        for stats in conversation_map.values():
            if stats.total_full_tokens > 0:
                stats.compression_ratio = round(
                    stats.total_summary_tokens / stats.total_full_tokens, 3
                )
            stats.distinct_tags.sort()

        return sorted(
            conversation_map.values(),
            key=lambda s: s.newest_segment or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    def get_all_segments(
        self,
        *,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[StoredSegment]:
        entries = list(self._index.values())
        if conversation_id is not None:
            entries = [
                entry for entry in entries
                if entry.get("conversation_id", entry.get("session_id", "")) == conversation_id
            ]
        entries.sort(key=lambda entry: entry.get("created_at", ""), reverse=True)
        if limit is not None and limit > 0:
            entries = entries[:limit]
        results: list[StoredSegment] = []
        for entry in entries:
            seg = self.get_segment(entry["ref"], conversation_id=conversation_id)
            if seg is not None:
                results.append(seg)
        return results

    def get_tag_aliases(self, conversation_id: str | None = None) -> dict[str, str]:
        aliases = dict(self._aliases)
        if conversation_id:
            aliases.update(self._conversation_aliases.get(conversation_id, {}))
        return aliases

    def set_tag_alias(
        self,
        alias: str,
        canonical: str,
        conversation_id: str = "",
    ) -> None:
        with self._lock:
            if conversation_id:
                self._conversation_aliases.setdefault(conversation_id, {})[alias] = canonical
            else:
                self._aliases[alias] = canonical
            self._save_aliases()

    def delete_tag_aliases_for_conversation(self, conversation_id: str) -> int:
        with self._lock:
            removed = len(self._conversation_aliases.get(conversation_id, {}))
            if removed:
                self._conversation_aliases.pop(conversation_id, None)
                self._save_aliases()
            return removed

    def delete_segment(self, ref: str) -> bool:
        with self._lock:
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

    def save_tag_summary(self, tag_summary: TagSummary, conversation_id: str = "") -> None:
        ts_dir = self.root / "_tag_summaries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag_summary.tag.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = ts_dir / f"{safe_tag}.json"
        data = {
            "tag": tag_summary.tag,
            "summary": tag_summary.summary,
            "description": tag_summary.description,
            "code_refs": getattr(tag_summary, "code_refs", []) or [],
            "summary_tokens": tag_summary.summary_tokens,
            "source_segment_refs": tag_summary.source_segment_refs,
            "source_turn_numbers": tag_summary.source_turn_numbers,
            "covers_through_turn": tag_summary.covers_through_turn,
            "generated_by_turn_id": getattr(tag_summary, "generated_by_turn_id", "") or "",
            "created_at": _dt_to_str(tag_summary.created_at),
            "updated_at": _dt_to_str(tag_summary.updated_at),
        }
        path.write_text(json.dumps(data, indent=2))

    def get_tag_summary(self, tag: str, conversation_id: str = "") -> TagSummary | None:
        safe_tag = tag.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = self.root / "_tag_summaries" / f"{safe_tag}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return TagSummary(
            tag=data["tag"],
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            code_refs=data.get("code_refs", []),
            summary_tokens=data.get("summary_tokens", 0),
            source_segment_refs=data.get("source_segment_refs", []),
            source_turn_numbers=data.get("source_turn_numbers", []),
            covers_through_turn=data.get("covers_through_turn", -1),
            generated_by_turn_id=data.get("generated_by_turn_id", ""),
            created_at=_str_to_dt(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=_str_to_dt(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )

    def get_all_tag_summaries(self, *, conversation_id: str | None = None) -> list[TagSummary]:
        ts_dir = self.root / "_tag_summaries"
        if not ts_dir.is_dir():
            return []
        results: list[TagSummary] = []
        for path in sorted(ts_dir.glob("*.json")):
            ts = self.get_tag_summary(path.stem)
            if ts is None:
                continue
            if conversation_id is not None and ts.source_segment_refs:
                # Exclude only when ALL resolved source segments belong to
                # a different conversation.  Unresolvable refs (not in index)
                # are treated as neutral — they don't exclude.
                resolved = [
                    self._index[ref]
                    for ref in ts.source_segment_refs
                    if ref in self._index
                ]
                if resolved and not any(
                    entry.get("conversation_id", "") == conversation_id
                    for entry in resolved
                ):
                    continue
            results.append(ts)
        return results

    def get_segments_by_tags(
        self,
        tags: list[str],
        min_overlap: int = 1,
        limit: int = 20,
        conversation_id: str | None = None,
    ) -> list[StoredSegment]:
        if not tags:
            return []
        tag_set = set(tags)
        scored: list[tuple[int, str]] = []
        for ref, entry in self._index.items():
            if conversation_id is not None:
                if entry.get("conversation_id", "") != conversation_id:
                    continue
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

    def store_chunk_embeddings(self, segment_ref: str, chunks: list[ChunkEmbedding]) -> None:
        embed_dir = self.root / "_embeddings"
        embed_dir.mkdir(parents=True, exist_ok=True)
        safe_ref = segment_ref.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = embed_dir / f"{safe_ref}.json"
        data = [
            {
                "segment_ref": c.segment_ref,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "embedding": c.embedding,
            }
            for c in chunks
        ]
        path.write_text(json.dumps(data, indent=2))

    def get_all_chunk_embeddings(self) -> list[ChunkEmbedding]:
        embed_dir = self.root / "_embeddings"
        if not embed_dir.is_dir():
            return []
        results: list[ChunkEmbedding] = []
        for path in sorted(embed_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for item in data:
                results.append(ChunkEmbedding(
                    segment_ref=item["segment_ref"],
                    chunk_index=item["chunk_index"],
                    text=item["text"],
                    embedding=item["embedding"],
                ))
        return results

    def save_engine_state(self, state: EngineStateSnapshot) -> None:
        state_dir = self.root / "_engine_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        safe_id = state.conversation_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = state_dir / f"{safe_id}.json"
        data = {
            "conversation_id": state.conversation_id,
            "compacted_through": state.compacted_through,
            "last_compacted_turn": state.last_compacted_turn,
            "last_completed_turn": state.last_completed_turn,
            "last_indexed_turn": state.last_indexed_turn,
            "checkpoint_version": state.checkpoint_version,
            "turn_count": state.turn_count,
            "saved_at": _dt_to_str(state.saved_at),
            "turn_tag_entries": [
                {
                    "turn_number": e.turn_number,
                    "message_hash": e.message_hash,
                    "tags": e.tags,
                    "primary_tag": e.primary_tag,
                    "timestamp": _dt_to_str(e.timestamp),
                    "session_date": e.session_date,
                    "sender": e.sender,
                    "code_refs": list(getattr(e, "code_refs", []) or []),
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
            "trailing_fingerprint": state.trailing_fingerprint,
            "request_captures": state.request_captures,
            "provider": state.provider,
            "flushed_through": state.flushed_through,
            "last_request_time": state.last_request_time,
            "tool_tag_counter": state.tool_tag_counter,
        }
        path.write_text(json.dumps(data, indent=2))

    def _parse_engine_state_data(self, data: dict) -> EngineStateSnapshot:
        entries = [
            TurnTagEntry(
                turn_number=e["turn_number"],
                message_hash=e["message_hash"],
                tags=e["tags"],
                primary_tag=e["primary_tag"],
                timestamp=_str_to_dt(e["timestamp"]),
                session_date=e.get("session_date", ""),
                sender=e.get("sender", ""),
                code_refs=e.get("code_refs", []) or [],
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
            conversation_id=data.get("conversation_id", data.get("session_id", "")),
            compacted_through=data.get("compacted_through", 0),
            flushed_through=data.get("flushed_through", 0),
            flushed_through_present=("flushed_through" in data),
            last_request_time=data.get("last_request_time", 0.0),
            turn_tag_entries=entries,
            turn_count=data.get("turn_count", 0),
            last_compacted_turn=data.get(
                "last_compacted_turn",
                (data.get("compacted_through", 0) // 2) - 1 if data.get("compacted_through", 0) > 0 else -1,
            ),
            last_completed_turn=data.get(
                "last_completed_turn",
                max(data.get("turn_count", 0) - 1, len(entries) - 1),
            ),
            last_indexed_turn=data.get("last_indexed_turn", len(entries) - 1),
            checkpoint_version=data.get("checkpoint_version", 0),
            conversation_generation=self.get_conversation_generation(
                data.get("conversation_id", data.get("session_id", "")),
            ),
            saved_at=_str_to_dt(data["saved_at"]) if "saved_at" in data else datetime.now(timezone.utc),
            split_processed_tags=data.get("split_processed_tags", []),
            working_set=working_set,
            trailing_fingerprint=data.get("trailing_fingerprint", ""),
            request_captures=data.get("request_captures", []),
            provider=data.get("provider", ""),
            tool_tag_counter=data.get("tool_tag_counter", 0),
        )

    def load_engine_state(self, conversation_id: str) -> EngineStateSnapshot | None:
        safe_id = conversation_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = self.root / "_engine_state" / f"{safe_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return self._parse_engine_state_data(data)

    def load_latest_engine_state(self) -> EngineStateSnapshot | None:
        state_dir = self.root / "_engine_state"
        if not state_dir.is_dir():
            return None
        # Parse all state files and pick the one with most progress
        candidates: list[tuple[int, str, dict]] = []
        for path in state_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                ct = data.get("compacted_through", 0)
                saved = data.get("saved_at", "")
                candidates.append((ct, saved, data))
            except (json.JSONDecodeError, OSError):
                continue
        if not candidates:
            return None
        # Highest compacted_through first, then most recent saved_at
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        try:
            return self._parse_engine_state_data(candidates[0][2])
        except (KeyError, ValueError):
            return None

    def list_engine_state_fingerprints(self) -> dict[str, str]:
        state_dir = self.root / "_engine_state"
        if not state_dir.is_dir():
            return {}
        result: dict[str, str] = {}
        for path in state_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                fp = data.get("trailing_fingerprint", "")
                if fp:
                    result[fp] = data.get("conversation_id", data.get("session_id", ""))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return result

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

    def _conversation_lifecycle_path(self, conversation_id: str) -> Path:
        safe_id = conversation_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.root / "_conversation_lifecycle" / f"{safe_id}.json"

    def _load_conversation_lifecycle(self, conversation_id: str) -> dict | None:
        path = self._conversation_lifecycle_path(conversation_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def activate_conversation(self, conversation_id: str) -> int:
        path = self._conversation_lifecycle_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._load_conversation_lifecycle(conversation_id) or {"generation": 0}
        data["deleted"] = False
        data["updated_at"] = _dt_to_str(datetime.now(timezone.utc))
        path.write_text(json.dumps(data, indent=2))
        return int(data.get("generation", 0) or 0)

    def begin_conversation_deletion(self, conversation_id: str) -> int:
        path = self._conversation_lifecycle_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._load_conversation_lifecycle(conversation_id) or {"generation": 0}
        generation = int(data.get("generation", 0) or 0) + 1
        data.update({
            "generation": generation,
            "deleted": True,
            "updated_at": _dt_to_str(datetime.now(timezone.utc)),
        })
        path.write_text(json.dumps(data, indent=2))
        return generation

    def get_conversation_generation(self, conversation_id: str) -> int:
        data = self._load_conversation_lifecycle(conversation_id) or {}
        return int(data.get("generation", 0) or 0)

    def is_conversation_generation_current(
        self,
        conversation_id: str,
        generation: int,
    ) -> bool:
        data = self._load_conversation_lifecycle(conversation_id)
        if data is None:
            return int(generation or 0) == 0
        return (
            int(data.get("generation", 0) or 0) == int(generation or 0)
            and not bool(data.get("deleted", False))
        )

    # ------------------------------------------------------------------
    # Turn messages
    # ------------------------------------------------------------------

    def _turn_messages_path(self, conversation_id: str) -> Path:
        safe_id = conversation_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.root / "_turn_messages" / f"{safe_id}.json"

    def save_turn_message(
        self,
        conversation_id: str,
        turn_number: int,
        user_content: str,
        assistant_content: str,
        user_raw_content: str | None = None,
        assistant_raw_content: str | None = None,
    ) -> None:
        path = self._turn_messages_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        entry = {
            "user_content": user_content,
            "assistant_content": assistant_content,
        }
        if user_raw_content is not None:
            entry["user_raw_content"] = user_raw_content
        if assistant_raw_content is not None:
            entry["assistant_raw_content"] = assistant_raw_content
        existing[str(turn_number)] = entry
        path.write_text(json.dumps(existing, indent=2))

    def get_turn_messages(
        self,
        conversation_id: str,
        turn_numbers: list[int],
    ) -> dict[int, tuple[str, str, str | None, str | None]]:
        if not turn_numbers:
            return {}
        path = self._turn_messages_path(conversation_id)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        result: dict[int, tuple[str, str, str | None, str | None]] = {}
        for tn in turn_numbers:
            entry = data.get(str(tn))
            if entry:
                result[tn] = (
                    entry.get("user_content", ""),
                    entry.get("assistant_content", ""),
                    entry.get("user_raw_content"),
                    entry.get("assistant_raw_content"),
                )
        return result

    def load_recent_turn_messages(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> list[tuple[int, str, str]]:
        path = self._turn_messages_path(conversation_id)
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        rows: list[tuple[int, str, str]] = []
        for key, entry in data.items():
            try:
                turn_number = int(key)
            except (TypeError, ValueError):
                continue
            rows.append((
                turn_number,
                entry.get("user_content", ""),
                entry.get("assistant_content", ""),
            ))
        rows.sort(key=lambda item: item[0])
        if limit > 0:
            rows = rows[-limit:]
        return rows

    def prune_turn_messages(self, conversation_id: str, keep_from_turn: int) -> int:
        path = self._turn_messages_path(conversation_id)
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0
        removed = 0
        kept: dict[str, dict] = {}
        for key, entry in data.items():
            try:
                turn_number = int(key)
            except (TypeError, ValueError):
                kept[key] = entry
                continue
            if turn_number < keep_from_turn:
                removed += 1
                continue
            kept[key] = entry
        if removed:
            path.write_text(json.dumps(kept, indent=2))
        return removed

    # ------------------------------------------------------------------
    # Conversation aliases (VCATTACH)
    # ------------------------------------------------------------------

    def _load_vcattach_aliases(self) -> None:
        if self._vcattach_aliases_path.is_file():
            try:
                self._vcattach_aliases = json.loads(
                    self._vcattach_aliases_path.read_text()
                )
            except (json.JSONDecodeError, OSError):
                self._vcattach_aliases = {}
        else:
            self._vcattach_aliases = {}

    def _save_vcattach_aliases(self) -> None:
        tmp = self._vcattach_aliases_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._vcattach_aliases, indent=2))
        os.replace(str(tmp), str(self._vcattach_aliases_path))

    def save_conversation_alias(self, alias_id: str, target_id: str) -> None:
        self._vcattach_aliases[alias_id] = target_id
        self._save_vcattach_aliases()

    def resolve_conversation_alias(self, alias_id: str) -> str | None:
        return self._vcattach_aliases.get(alias_id)

    # ------------------------------------------------------------------
    # Cross-cutting queries (stubs — FilesystemStore lacks SQL)
    # ------------------------------------------------------------------

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all segments for a conversation. Returns segment count deleted."""
        deleted = 0
        safe_id = conversation_id.replace("/", "_").replace("\\", "_").replace("..", "_")

        with self._lock:
            refs = [
                ref for ref, entry in self._index.items()
                if entry.get("conversation_id", "") == conversation_id
            ]

        for ref in refs:
            if self.delete_segment(ref):
                deleted += 1

        with self._lock:
            state_path = self.root / "_engine_state" / f"{safe_id}.json"
            if state_path.is_file():
                state_path.unlink()

            turn_messages_path = self._turn_messages_path(conversation_id)
            if turn_messages_path.is_file():
                turn_messages_path.unlink()

            captures_path = self.root / "_request_captures.json"
            if captures_path.is_file():
                try:
                    captures = json.loads(captures_path.read_text())
                except (json.JSONDecodeError, OSError):
                    captures = []
                captures = [
                    capture for capture in captures
                    if (capture.get("conversation_id", "") or "") != conversation_id
                ]
                captures_path.write_text(json.dumps(captures, indent=2))

            if conversation_id in self._conversation_aliases:
                self._conversation_aliases.pop(conversation_id, None)
                self._save_aliases()

        return deleted

    def get_orphan_tag_snippets(self, limit: int = 1000) -> list[dict]:
        """Return snippet info for orphan tags.

        FilesystemStore iterates the in-memory index to approximate
        the SQL query used by SQLiteStore.
        """
        # Collect tags that have a tag_summary on disk
        summarized_tags: set[str] = set()
        ts_dir = self.root / "_tag_summaries"
        if ts_dir.is_dir():
            for p in ts_dir.glob("*.json"):
                summarized_tags.add(p.stem)

        results: list[dict] = []
        seen_tags: set[str] = set()
        for entry in self._index.values():
            for t in entry.get("tags", []):
                if t in summarized_tags or t in seen_tags:
                    continue
                seg = self.get_segment(entry["ref"])
                if seg:
                    snippet = seg.summary[:100] if seg.summary else ""
                    results.append({"tag": t, "snippet": snippet})
                    seen_tags.add(t)
                    if len(results) >= limit:
                        return results
        return results

    # ------------------------------------------------------------------
    # Fact / tool-output stubs (FilesystemStore lacks SQL backing)
    # ------------------------------------------------------------------

    def query_facts(
        self,
        *,
        subject: str | None = None,
        verb: str | None = None,
        verbs: list[str] | None = None,
        object_contains: str | None = None,
        status: str | None = None,
        fact_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list:
        return []

    def replace_facts_for_segment(self, conversation_id: str, segment_ref: str, facts: list) -> tuple[int, int]:
        return 0, 0

    def search_facts(self, query: str, limit: int = 10, conversation_id: str | None = None) -> list:
        return []

    def search_tool_outputs(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list:
        return []

    def link_turn_tool_output(self, conversation_id: str, turn_number: int, tool_output_ref: str) -> None:
        pass

    def get_tool_outputs_for_turn(self, conversation_id: str, turn_number: int) -> list[str]:
        return []

    def link_segment_tool_output(self, conversation_id: str, segment_ref: str, tool_output_ref: str) -> None:
        pass

    def get_tool_outputs_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        return []

    def get_tool_output_refs_for_turn(self, conversation_id: str, turn: int) -> list[str]:
        return []

    def get_tool_output_by_ref(self, conversation_id: str, ref: str) -> str | None:
        return None

    def store_chain_snapshot(self, ref: str, conversation_id: str, turn_number: int, chain_json: str, message_count: int, tool_output_refs: str = "") -> None:
        pass

    def get_chain_snapshot(self, conversation_id: str, ref: str) -> dict | None:
        return None

    def get_chain_snapshots_for_conversation(self, conversation_id: str, min_turn: int = 0) -> list[dict]:
        return []

    def get_tool_names_for_refs(self, refs: list[str]) -> list[str]:
        return []

    def get_tool_names_for_segment(self, conversation_id: str, segment_ref: str) -> list[str]:
        return []

    def get_fact_count_by_tags(self, *, conversation_id: str | None = None) -> dict[str, int]:
        return {}

    def get_unique_fact_verbs(self, *, conversation_id: str | None = None) -> list[str]:
        return []

    def get_superseded_facts(self, fact_ids: list[str]) -> list[dict]:
        return []

    def save_request_capture(self, capture: dict) -> None:
        captures_path = self.root / "_request_captures.json"
        with self._lock:
            existing = []
            if captures_path.exists():
                try:
                    existing = json.loads(captures_path.read_text())
                except (json.JSONDecodeError, OSError):
                    existing = []
            conv_id = capture.get("conversation_id", "") or ""
            turn_id = capture.get("turn_id", "") or ""

            def _capture_key(item: dict) -> tuple[str, int | str, str]:
                return (
                    (item.get("conversation_id", "") or ""),
                    item.get("turn"),
                    (item.get("turn_id", "") or ""),
                )

            # Upsert by (conversation_id, turn, turn_id)
            existing = [
                c for c in existing
                if _capture_key(c) != (conv_id, capture.get("turn"), turn_id)
            ]
            existing.append(capture)
            if conv_id:
                scoped = [
                    c for c in existing
                    if (c.get("conversation_id", "") or "") == conv_id
                ]
                scoped.sort(key=lambda c: c.get("ts", ""))
                keep = {
                    _capture_key(c)
                    for c in scoped[-50:]
                }
                existing = [
                    c for c in existing
                    if _capture_key(c) in keep
                    or (c.get("conversation_id", "") or "") != conv_id
                ]
            captures_path.write_text(json.dumps(existing, indent=2))

    def load_request_captures(
        self,
        limit: int = 50,
        conversation_id: str | None = None,
    ) -> list[dict]:
        captures_path = self.root / "_request_captures.json"
        with self._lock:
            if not captures_path.exists():
                return []
            try:
                data = json.loads(captures_path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
            if conversation_id is not None:
                data = [
                    c for c in data
                    if (c.get("conversation_id", "") or "") == conversation_id
                ]
            data.sort(key=lambda c: c.get("ts", ""))
            return data[-limit:]
