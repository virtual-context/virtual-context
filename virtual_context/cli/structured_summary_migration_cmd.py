"""Safely migrate stored segment and tag summaries to structured claims.

The migration is deliberately narrower than the ordinary compaction path:

* the default dry run opens only a server-enforced read-only psycopg
  connection (constructing a store or engine can run schema/bootstrap writes);
* model input is rebuilt exclusively from exact canonical rows named by a
  segment's proved source mapping -- legacy ``full_text``, ``messages_json``
  and ``summary`` text are never selected (the database emits only an opaque
  checksum of the old retrieval synopsis for the journal);
* writes add only the deterministic ``metadata_json.structured_summary``
  envelope; existing retrieval synopsis bytes and their indexes are preserved;
* a crash-durable write-ahead journal makes every committed write auditable;
* tag rollups select an ordered bounded subset of strict segment claims and
  copy those claim objects exactly without model-authored factual prose;
  existing tag synopsis/description/embedding bytes remain untouched while
  the source coordinates and structured envelope are revalidated under locks.

The command is Postgres-only because ``xmin`` is the row-version fence used by
the online store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Iterable, Mapping

from ..core.structured_summary import (
    apply_tag_claim_safety_floor,
    event_time_is_supported,
    infer_modality,
    infer_temporal_status,
    is_safety_critical_personal_evidence,
    structured_claim_fingerprint,
    structured_source_digest,
    structured_source_provenance_digest,
    structured_tag_claim_digest,
    validate_tag_rollup_inputs,
)
from ..core.compactor import build_deterministic_structured_summary
from ..core.summary_identity import structured_claims_contain_internal_identity
from ..types import (
    AUDIENCE_ATTRIBUTION_VERSION,
    AUTHOR_ROLE_REQUESTER,
    ActorRoster,
    FactLane,
    SOURCE_CANONICAL_TURN_IDS_KEY,
    STRUCTURED_SUMMARY_MAX_CLAIMS,
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
    STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS,
    Message,
    SegmentMetadata,
    StoredSummary,
    StructuredSummary,
    TagSummary,
    TaggedSegment,
    strict_structured_summary,
    structured_summary_to_dict,
)


_SEGMENT_COLUMNS = """\
s.ref, s.conversation_id, s.primary_tag, s.full_tokens, s.created_at,
s.start_timestamp, s.end_timestamp, s.metadata_json,
s.xmin::text AS row_version, md5(s.summary) AS old_retrieval_synopsis_md5,
length(s.summary) AS retrieval_synopsis_chars,
(SELECT COALESCE(array_agg(st.tag ORDER BY st.tag), ARRAY[]::text[])
   FROM segment_tags st WHERE st.segment_ref = s.ref) AS tags
"""


def _selection_sql(after_ref: str | None) -> str:
    """Return the inventory query.

    The absence of all three legacy prose values is a safety property, not a
    performance optimization. The old summary contributes only an in-database
    checksum for journaling; its value is unavailable to the apply path.
    """
    after = " AND s.ref > %(after_ref)s" if after_ref else ""
    return (
        f"SELECT {_SEGMENT_COLUMNS} FROM segments s "
        "WHERE s.conversation_id = %(conversation_id)s"
        f"{after} ORDER BY s.ref ASC"
    )


_CANONICAL_COLUMNS = """\
canonical_turn_id::text AS canonical_turn_id, conversation_id,
turn_group_number, sort_key, user_content, assistant_content,
primary_tag, tags_json, session_date, sender, origin_channel_id,
origin_channel_label, sender_actor_id, source_message_id,
reply_target_message_id, reply_subject_actor_id, reply_subject_label,
reply_target_body, reply_attribution_version, audience_conversation_id,
audience_attribution_version, fact_signals_json, code_refs_json,
first_seen_at, last_seen_at, created_at, updated_at,
origin_conversation_id
"""


_CAS_UPDATE_SQL = """\
UPDATE segments SET metadata_json = jsonb_set(metadata_json::jsonb,
'{structured_summary}', %s::jsonb, true)::text
WHERE ref = %s AND conversation_id = %s AND xmin::text = %s
AND EXISTS (SELECT 1 FROM conversations c
WHERE c.conversation_id = %s AND c.tenant_id = %s
AND c.lifecycle_epoch = %s AND c.phase = 'active'
AND c.pending_raw_payload_entries = 0)
AND EXISTS (SELECT 1 FROM conversation_lifecycle cl
WHERE cl.conversation_id = %s AND cl.generation = %s
AND cl.deleted = FALSE)
AND NOT EXISTS (SELECT 1 FROM ingestion_episode ie
WHERE ie.conversation_id = %s AND ie.lifecycle_epoch = %s
AND ie.status = 'running')
AND NOT EXISTS (SELECT 1 FROM compaction_operation co
WHERE co.conversation_id = %s AND co.lifecycle_epoch = %s
AND co.status IN ('queued','running'))
"""

_DETERMINISTIC_GENERATION_MODEL = "deterministic-extractive-migration-v1"
_MIGRATION_MODE = "deterministic_metadata_only_v1"


class MigrationError(RuntimeError):
    """A fail-closed scope or data-validation error."""


class _TagCASConflict(RuntimeError):
    """A tag/embedding race that must roll back both writes."""


@dataclass(frozen=True)
class _Scope:
    tenant_id: str
    conversation_id: str
    lifecycle_epoch: int
    lifecycle_generation: int
    phase: str
    direct_source_alias_count: int


@dataclass
class _Inventory:
    scanned: int
    candidates: list[dict]
    current_rows: list[dict]
    skipped_rows: list[dict]
    current: int
    skipped_reason_counts: Counter[str]
    candidate_reason_counts: Counter[str]
    affected_tags: set[str]


@dataclass
class _TagInventory:
    scanned: int
    candidates: list[dict]
    blocked: list[dict]
    current: int
    reason_counts: Counter[str]
    source_segment_count: int


class _ResumeCursor:
    """A cursor that never advances past a row with an undecided write."""

    def __init__(self, start: str | None):
        self.ref = start
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def on_decided(self, ref: str) -> None:
        if not self.frozen:
            self.ref = ref


def configure_parser(admin_sub, positive_int) -> None:
    """Register the dedicated admin subcommand on an argparse subparser."""
    parser = admin_sub.add_parser(
        "migrate-structured-summaries",
        help="Migrate proved canonical segment sources to structured claims",
    )
    parser.add_argument("conversation_id")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write selected segment/tag layers; default is read-only",
    )
    parser.add_argument(
        "--limit", type=positive_int, default=None,
        help="Cap attempted candidates per selected phase",
    )
    parser.add_argument(
        "--after-ref", help="Resume strictly after this segment ref",
    )
    parser.add_argument(
        "--after-tag", help="Resume the tag phase strictly after this tag",
    )
    parser.add_argument(
        "--phase", choices=("segments", "tags", "all"), default="all",
        help="Migration layer to run (default: all)",
    )
    parser.add_argument(
        "--journal",
        help="Crash-durable append-only JSONL journal path",
    )
    parser.add_argument("--postgres-dsn")


def _resolve_dsn(args) -> str | None:
    return getattr(args, "postgres_dsn", None) or os.environ.get("DATABASE_URL")


def _connect(dsn: str, *, read_only: bool):
    import psycopg
    from psycopg.rows import dict_row

    options = "-c default_transaction_read_only=on" if read_only else None
    return psycopg.connect(
        dsn, row_factory=dict_row, autocommit=True, options=options,
    )


def _fail(stage: str, error: str, args) -> None:
    print(json.dumps({
        "status": "error",
        "stage": stage,
        "conversation_id": args.conversation_id,
        "error": error,
    }))
    raise SystemExit(1)


def _mapping_value(row: object, key: str, default=None):
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _metadata_object(raw: object) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_ids(metadata: Mapping[str, object]) -> list[str] | None:
    raw = metadata.get("canonical_turn_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            type(value) is not str
            or not value
            or value != value.strip()
            for value in raw
        )
        or len(set(raw)) != len(raw)
    ):
        return None
    return list(raw)


def _alias_terminal(graph: Mapping[str, str], conversation_id: str) -> str | None:
    """Resolve one alias graph without guessing across a missing edge/cycle."""
    current = conversation_id
    seen: set[str] = set()
    for _ in range(64):
        target = graph.get(current)
        if target is None:
            return current
        if current in seen or not target:
            return None
        seen.add(current)
        current = target
    return None


def _source_validation_reasons(
    canonical_ids: list[str],
    rows_by_id: Mapping[str, object],
    *,
    owner_conversation_id: str,
    alias_graph: Mapping[str, str],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for canonical_id in canonical_ids:
        row = rows_by_id.get(canonical_id)
        if row is None:
            reasons.append("canonical_row_missing")
            continue
        if str(_mapping_value(row, "conversation_id", "") or "") != owner_conversation_id:
            reasons.append("canonical_owner_mismatch")
        user = str(_mapping_value(row, "user_content", "") or "")
        assistant = str(_mapping_value(row, "assistant_content", "") or "")
        if not user.strip() and not assistant.strip():
            reasons.append("canonical_row_empty")
        # Schema v1 admits requester evidence only. Legacy assistant-only rows
        # remain part of mapping completeness, but their missing historical
        # audience annotation cannot authorize or contaminate a requester
        # claim because they are absent from both the claim catalog and digest.
        if user.strip():
            audience = str(
                _mapping_value(row, "audience_conversation_id", "") or "",
            ).strip()
            version = _mapping_value(row, "audience_attribution_version", 0)
            if (
                not audience
                or type(version) is not int
                or version != AUDIENCE_ATTRIBUTION_VERSION
            ):
                reasons.append("canonical_audience_unproved")
            elif _alias_terminal(alias_graph, audience) != owner_conversation_id:
                reasons.append("canonical_audience_not_owned")
    return tuple(dict.fromkeys(reasons))


def _admissible_requester_records(
    canonical_ids: Iterable[str],
    rows_by_id: Mapping[str, object],
    *,
    session_date: str,
) -> list[dict[str, object]]:
    """Return the exact requester records admitted by the compactor.

    Only independently hydrated requester lanes participate. Historical
    assistant output is prior model text and is excluded from the v1 trust
    boundary. Reply-target copies are deliberately absent, matching the normal
    structured-summary boundary.
    """
    from ..core.summary_identity import (
        human_label_collision_key,
        is_safe_human_label,
    )
    _ = session_date

    candidates: list[tuple[object, str, str, str]] = []
    actors_by_label: dict[str, set[str]] = {}
    seen: set[str] = set()
    for canonical_id in canonical_ids:
        row = rows_by_id.get(canonical_id)
        if row is None:
            continue
        user = str(_mapping_value(row, "user_content", "") or "")
        if user.strip() and canonical_id not in seen:
            seen.add(canonical_id)
            actor = str(_mapping_value(row, "sender_actor_id", "") or "").strip()
            label = str(_mapping_value(row, "sender", "") or "").strip()
            if actor and is_safe_human_label(label, actor):
                actors_by_label.setdefault(
                    human_label_collision_key(label), set(),
                ).add(actor)
                candidates.append((row, "requester", label, actor))

    colliding = {
        key for key, actors in actors_by_label.items() if len(actors) > 1
    }
    records: list[dict[str, object]] = []
    for row, role, label, actor in candidates:
        if human_label_collision_key(label) in colliding:
            continue
        records.append({
            "canonical_turn_id": str(
                _mapping_value(row, "canonical_turn_id", "") or "",
            ),
            "source_role": role,
            "actor_id": actor,
            "speaker_label": label,
            "content": str(
                _mapping_value(row, "user_content", "") or "",
            ).strip(),
            "session_date": str(
                _mapping_value(row, "session_date", "") or "",
            ).strip(),
            "audience_conversation_id": str(
                _mapping_value(row, "audience_conversation_id", "") or "",
            ),
            "origin_channel_id": str(
                _mapping_value(row, "origin_channel_id", "") or "",
            ),
            "audience_attribution_version": _mapping_value(
                row, "audience_attribution_version", 0,
            ),
        })
    return records


def _expected_source_digest(
    canonical_ids: Iterable[str],
    rows_by_id: Mapping[str, object],
    *,
    session_date: str,
) -> str:
    """Mirror the compactor's public segment digest from physical rows."""
    records = _admissible_requester_records(
        canonical_ids, rows_by_id, session_date=session_date,
    )
    return structured_source_digest(records) if records else ""


def _update_reason_counts(counter: Counter[str], reasons: Iterable[str]) -> None:
    """Increment each distinct reason once for one inventory row."""
    counter.update(dict.fromkeys(reasons, 1))


def _has_bounded_claim_source(records: Iterable[Mapping[str, object]]) -> bool:
    """Whether strict generation can produce at least one v1 claim.

    The aggregate segment digest intentionally binds every admitted requester
    lane, including long ones. Claim evidence is separately bounded by the v1
    schema. A segment containing only over-limit lanes is safe canonical
    fallback, not a migration candidate that should block the tag phase
    forever.
    """
    return any(
        0 < len(str(record.get("content", "") or "").strip())
        <= STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS
        for record in records
    )


def _structured_source_block_reason(
    records: Iterable[Mapping[str, object]],
) -> str | None:
    """Return why deterministic v1 cannot represent this source snapshot.

    The aggregate digest can bind an overlong ordinary lane while the compact
    layer omits it. A safety-critical correction cannot be omitted, however,
    and no claim may be substring-truncated. Likewise, exceeding the schema's
    atomic claim capacity must use canonical fallback rather than persisting a
    partial envelope that could hide one source lane.
    """
    bounded_count = 0
    for record in records:
        content = str(record.get("content", "") or "").strip()
        if not content:
            continue
        if len(content) <= STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS:
            bounded_count += 1
            continue
        if is_safety_critical_personal_evidence(content):
            return "critical_source_over_excerpt_limit"
    if bounded_count > STRUCTURED_SUMMARY_MAX_CLAIMS:
        return "bounded_claim_overflow"
    return None


def _segment_migration_block_reason(
    records: Iterable[Mapping[str, object]],
    *,
    retrieval_synopsis_chars: int,
) -> str | None:
    """Return a source/retrieval defect metadata-only repair cannot fix."""
    materialized = tuple(records)
    if structured_claims_contain_internal_identity(
        (
            SimpleNamespace(sources=(SimpleNamespace(
                evidence_excerpt=str(record.get("content", "") or ""),
            ),))
            for record in materialized
        ),
        admitted_actor_ids=(
            record.get("actor_id", "") for record in materialized
        ),
    ):
        return "internal_identity_in_source"
    source_reason = _structured_source_block_reason(materialized)
    if source_reason is not None:
        return source_reason
    if not _has_bounded_claim_source(materialized):
        return "no_bounded_structured_claim_source"
    if retrieval_synopsis_chars < 1:
        # This command deliberately never invents or rewrites retrieval prose.
        # Leaving this as a candidate would produce an endless apply loop and
        # let tag preflight mistake an unavailable lower layer for repairable
        # work. Keep it explicit and on canonical fallback instead.
        return "retrieval_synopsis_missing"
    return None


def _envelope_candidate_reason(
    raw_envelope: object,
    envelope: StructuredSummary,
    expected_digest: str,
) -> str | None:
    """Why an otherwise eligible segment still needs migration."""
    if envelope.schema_version != STRUCTURED_SUMMARY_SCHEMA_VERSION:
        return (
            "legacy_or_missing_schema"
            if not isinstance(raw_envelope, dict)
            or raw_envelope.get("schema_version") in (None, 0)
            else "invalid_or_unknown_schema"
        )
    if envelope.source_digest != expected_digest:
        return "source_digest_stale"
    if not envelope.generation_model.strip():
        return "generation_model_missing"
    if not envelope.claims:
        # Online compaction may persist a v1/digest envelope with no claims
        # after its availability-oriented source fallback. That is not a
        # completed migration: the strict migration seam must produce at least
        # one independently admitted claim.
        return "zero_claim_envelope"
    if any(len(claim.sources) != 1 for claim in envelope.claims):
        return "claim_source_cardinality_invalid"
    if any(
        claim.sources[0].source_role != "requester"
        for claim in envelope.claims
    ):
        return "claim_source_role_invalid"
    return None


def _load_alias_graph(conn, *, lock: bool = False) -> dict[str, str]:
    suffix = " FOR SHARE" if lock else ""
    rows = conn.execute(
        "SELECT alias_id, target_id FROM conversation_aliases" + suffix,
    ).fetchall()
    return {
        str(row["alias_id"]): str(row["target_id"])
        for row in rows
        if row.get("alias_id") and row.get("target_id")
    }


def _load_source_alias_chains_locked(
    conn,
    rows_by_id: Mapping[str, object],
    canonical_ids: Iterable[str],
    *,
    owner_conversation_id: str,
) -> dict[str, str]:
    """Lock only alias rows that the selected canonical sources traverse."""
    graph: dict[str, str] = {}
    for canonical_id in canonical_ids:
        row = rows_by_id.get(canonical_id)
        audience = str(
            _mapping_value(row, "audience_conversation_id", "") or "",
        ).strip()
        current = audience
        seen: set[str] = set()
        while current and current != owner_conversation_id and current not in seen:
            seen.add(current)
            alias = conn.execute(
                "SELECT target_id FROM conversation_aliases "
                "WHERE alias_id = %s FOR SHARE",
                (current,),
            ).fetchone()
            if not alias or not alias.get("target_id"):
                break
            target = str(alias["target_id"])
            graph[current] = target
            current = target
    return graph


def _load_source_rows(
    conn,
    conversation_id: str,
    canonical_ids: list[str],
    *,
    lock: bool = False,
) -> list[dict]:
    if not canonical_ids:
        return []
    suffix = " FOR SHARE" if lock else ""
    return list(conn.execute(
        f"SELECT {_CANONICAL_COLUMNS} FROM canonical_turns "
        "WHERE conversation_id = %s "
        "AND canonical_turn_id::text = ANY(%s) "
        "ORDER BY sort_key, canonical_turn_id" + suffix,
        (conversation_id, canonical_ids),
    ).fetchall())


def _preflight(conn, args, *, lock: bool = False) -> _Scope:
    """Prove exact owner, tenant and quiescent lifecycle scope."""
    required = (
        "segments", "segment_tags", "canonical_turns", "conversations",
        "conversation_lifecycle", "conversation_aliases",
        "ingestion_episode", "compaction_operation",
        "tag_summaries", "tag_summary_embeddings",
    )
    missing = []
    for name in required:
        row = conn.execute("SELECT to_regclass(%s) AS rel", (name,)).fetchone()
        if not row or row.get("rel") is None:
            missing.append(name)
    if missing:
        raise MigrationError("required schema missing: " + ", ".join(missing))

    lock_sql = " FOR SHARE" if lock else ""
    alias = conn.execute(
        "SELECT target_id FROM conversation_aliases WHERE alias_id = %s"
        + lock_sql,
        (args.conversation_id,),
    ).fetchone()
    if alias:
        raise MigrationError(
            "conversation_id is a source alias; rerun with its active owner "
            + str(alias["target_id"]),
        )

    conversation = conn.execute(
        "SELECT tenant_id, lifecycle_epoch, phase, pending_raw_payload_entries "
        "FROM conversations WHERE conversation_id = %s" + lock_sql,
        (args.conversation_id,),
    ).fetchone()
    if not conversation:
        raise MigrationError("conversation owner does not exist")
    if str(conversation["tenant_id"]) != args.tenant_id:
        raise MigrationError("tenant_id does not own conversation_id")
    if str(conversation["phase"]) != "active":
        raise MigrationError(
            f"conversation phase must be active, got {conversation['phase']}",
        )
    if int(conversation["pending_raw_payload_entries"] or 0) != 0:
        raise MigrationError("conversation has pending raw payload entries")

    lifecycle = conn.execute(
        "SELECT generation, deleted FROM conversation_lifecycle "
        "WHERE conversation_id = %s" + lock_sql,
        (args.conversation_id,),
    ).fetchone()
    if not lifecycle or bool(lifecycle["deleted"]):
        raise MigrationError("conversation lifecycle is missing or deleted")

    active_ingestion = conn.execute(
        "SELECT count(*) AS n FROM ingestion_episode "
        "WHERE conversation_id = %s AND lifecycle_epoch = %s "
        "AND status = 'running'",
        (args.conversation_id, conversation["lifecycle_epoch"]),
    ).fetchone()["n"]
    active_compaction = conn.execute(
        "SELECT count(*) AS n FROM compaction_operation "
        "WHERE conversation_id = %s AND lifecycle_epoch = %s "
        "AND status IN ('queued','running')",
        (args.conversation_id, conversation["lifecycle_epoch"]),
    ).fetchone()["n"]
    if int(active_ingestion or 0) or int(active_compaction or 0):
        raise MigrationError("conversation has active ingestion or compaction")

    alias_count = conn.execute(
        "SELECT count(*) AS n FROM conversation_aliases WHERE target_id = %s",
        (args.conversation_id,),
    ).fetchone()["n"]
    return _Scope(
        tenant_id=args.tenant_id,
        conversation_id=args.conversation_id,
        lifecycle_epoch=int(conversation["lifecycle_epoch"]),
        lifecycle_generation=int(lifecycle["generation"]),
        phase="active",
        direct_source_alias_count=int(alias_count or 0),
    )


def _checksums(conn, conversation_id: str) -> dict:
    """Hash every table/field the dry-run inventory consumes, without prose."""
    segments = conn.execute(
        "SELECT count(*) AS n, "
        "md5(COALESCE(string_agg(concat_ws(chr(31), ref, metadata_json, "
        "primary_tag, full_tokens::text, start_timestamp, end_timestamp), "
        "'' ORDER BY ref), '')) AS digest "
        "FROM segments WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    canonical = conn.execute(
        "SELECT count(*) AS n, "
        "md5(COALESCE(string_agg(concat_ws(chr(31), canonical_turn_id::text, "
        "md5(COALESCE(user_content,'')), md5(COALESCE(assistant_content,'')), "
        "sender, sender_actor_id, audience_conversation_id, origin_channel_id, "
        "audience_attribution_version::text), '' ORDER BY sort_key, "
        "canonical_turn_id), '')) AS digest "
        "FROM canonical_turns WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    tags = conn.execute(
        "SELECT count(*) AS n, "
        "md5(COALESCE(string_agg(concat_ws(chr(31), st.segment_ref, st.tag), "
        "'' ORDER BY st.segment_ref, st.tag), '')) AS digest "
        "FROM segment_tags st JOIN segments s ON s.ref = st.segment_ref "
        "WHERE s.conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    scope = conn.execute(
        "SELECT c.lifecycle_epoch, c.phase, c.pending_raw_payload_entries, "
        "cl.generation, cl.deleted, c.updated_at::text "
        "FROM conversations c JOIN conversation_lifecycle cl USING "
        "(conversation_id) WHERE c.conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    tag_summaries = conn.execute(
        "SELECT count(*) AS n, md5(COALESCE(string_agg(concat_ws(chr(31), "
        "tag, md5(summary), md5(description), source_segment_refs, "
        "structured_summary_json, updated_at), '' ORDER BY tag), '')) AS digest "
        "FROM tag_summaries WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    tag_embeddings = conn.execute(
        "SELECT count(*) AS n, md5(COALESCE(string_agg(concat_ws(chr(31), "
        "tag, md5(embedding_json)), '' ORDER BY tag), '')) AS digest "
        "FROM tag_summary_embeddings WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    return {
        "segments": {"count": segments["n"], "md5": segments["digest"]},
        "canonical_turns": {
            "count": canonical["n"], "md5": canonical["digest"],
        },
        "segment_tags": {"count": tags["n"], "md5": tags["digest"]},
        "tag_summaries": {
            "count": tag_summaries["n"], "md5": tag_summaries["digest"],
        },
        "tag_summary_embeddings": {
            "count": tag_embeddings["n"], "md5": tag_embeddings["digest"],
        },
        "scope": dict(scope or {}),
    }


def _inventory(conn, args) -> tuple[_Scope, _Inventory, dict[str, str]]:
    scope = _preflight(conn, args)
    rows = list(conn.execute(
        _selection_sql(getattr(args, "after_ref", None)),
        {
            "conversation_id": args.conversation_id,
            "after_ref": getattr(args, "after_ref", None),
        },
    ).fetchall())

    metadata_by_ref: dict[str, dict] = {}
    all_ids: list[str] = []
    initial_reasons: dict[str, list[str]] = {}
    for row in rows:
        ref = str(row["ref"])
        metadata = _metadata_object(row.get("metadata_json"))
        reasons: list[str] = []
        if metadata is None:
            reasons.append("metadata_invalid")
        elif metadata.get("source_mapping_complete") is not True:
            reasons.append("source_mapping_incomplete")
        ids = _canonical_ids(metadata or {})
        if not ids:
            reasons.append("canonical_turn_ids_missing_or_invalid")
        else:
            all_ids.extend(ids)
        if metadata is not None:
            metadata_by_ref[ref] = metadata
        initial_reasons[ref] = reasons

    unique_ids = list(dict.fromkeys(all_ids))
    source_rows = _load_source_rows(
        conn, args.conversation_id, unique_ids,
    )
    source_by_id = {
        str(row["canonical_turn_id"]): row for row in source_rows
    }
    alias_graph = _load_alias_graph(conn)

    candidates: list[dict] = []
    current_rows: list[dict] = []
    skipped_rows: list[dict] = []
    skipped: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    current = 0
    affected_tags: set[str] = set()
    for row in rows:
        ref = str(row["ref"])
        reasons = list(initial_reasons[ref])
        metadata = metadata_by_ref.get(ref, {})
        ids = _canonical_ids(metadata)
        session_date_raw = metadata.get("session_date", "")
        session_date = (
            session_date_raw if type(session_date_raw) is str else ""
        )
        if ids:
            reasons.extend(_source_validation_reasons(
                ids, source_by_id,
                owner_conversation_id=args.conversation_id,
                alias_graph=alias_graph,
            ))
        if reasons:
            _update_reason_counts(skipped, reasons)
            copied = dict(row)
            copied["_skip_reasons"] = tuple(dict.fromkeys(reasons))
            skipped_rows.append(copied)
            affected_tags.update(_row_tags(row))
            continue

        expected_digest = _expected_source_digest(
            ids or (), source_by_id, session_date=session_date,
        )
        if not expected_digest:
            skipped["no_admissible_structured_sources"] += 1
            copied = dict(row)
            copied["_skip_reasons"] = ("no_admissible_structured_sources",)
            skipped_rows.append(copied)
            affected_tags.update(_row_tags(row))
            continue
        admissible_records = _admissible_requester_records(
            ids or (), source_by_id, session_date=session_date,
        )
        migration_block_reason = _segment_migration_block_reason(
            admissible_records,
            retrieval_synopsis_chars=int(
                row.get("retrieval_synopsis_chars") or 0,
            ),
        )
        if migration_block_reason is not None:
            skipped[migration_block_reason] += 1
            copied = dict(row)
            copied["_skip_reasons"] = (migration_block_reason,)
            skipped_rows.append(copied)
            affected_tags.update(_row_tags(row))
            continue

        raw_envelope = metadata.get("structured_summary")
        envelope = strict_structured_summary(raw_envelope)
        reason = _envelope_candidate_reason(
            raw_envelope, envelope, expected_digest,
        )
        if reason is None:
            stored_rejection = _validate_envelope(
                envelope,
                expected_digest=expected_digest,
                expected_model="",
                canonical_ids=ids or [],
                rows_by_id=source_by_id,
                session_date=session_date,
            )
            if stored_rejection is not None:
                reason = "stored_" + stored_rejection
        if reason is None:
            copied = dict(row)
            copied["_metadata"] = metadata
            copied["_canonical_ids"] = ids
            copied["_expected_source_digest"] = expected_digest
            current_rows.append(copied)
            current += 1
            continue

        copied = dict(row)
        copied["_metadata"] = metadata
        copied["_canonical_ids"] = ids
        copied["_expected_source_digest"] = expected_digest
        copied["_candidate_reason"] = reason
        candidates.append(copied)
        candidate_reasons[reason] += 1
        tags = {
            str(tag) for tag in (row.get("tags") or []) if str(tag).strip()
        }
        if not tags and row.get("primary_tag"):
            tags.add(str(row["primary_tag"]))
        affected_tags.update(tags)

    return scope, _Inventory(
        scanned=len(rows),
        candidates=candidates,
        current_rows=current_rows,
        skipped_rows=skipped_rows,
        current=current,
        skipped_reason_counts=skipped,
        candidate_reason_counts=candidate_reasons,
        affected_tags=affected_tags,
    ), alias_graph


def _row_tags(row: Mapping[str, object]) -> set[str]:
    tags = {
        str(tag).strip() for tag in (row.get("tags") or [])
        if str(tag).strip()
    }
    primary = str(row.get("primary_tag") or "").strip()
    if primary:
        tags.add(primary)
    return tags


def _tag_source_sort_key(row: Mapping[str, object]) -> tuple[float, float, str]:
    created = _parse_timestamp(row.get("created_at"))
    started = _parse_timestamp(row.get("start_timestamp"))
    return (
        created.timestamp() if created is not None else float("-inf"),
        started.timestamp() if started is not None else float("-inf"),
        str(row.get("ref") or ""),
    )


def _ordered_tag_sources(
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Newest segment claims win the bounded tag presentation window."""
    return sorted(rows, key=_tag_source_sort_key, reverse=True)


def _tag_source_canonical_ids(
    ordered_rows: Iterable[Mapping[str, object]],
) -> list[str]:
    """Exact deduplicated source-id order used by tag persistence."""
    source_ids: list[str] = []
    seen: set[str] = set()
    for row in ordered_rows:
        metadata = row.get("_metadata")
        if not isinstance(metadata, dict):
            metadata = _metadata_object(row.get("metadata_json")) or {}
        for canonical_id in _canonical_ids(metadata) or []:
            if canonical_id not in seen:
                seen.add(canonical_id)
                source_ids.append(canonical_id)
    return source_ids


def _deterministic_tag_envelope(
    source_rows: Iterable[Mapping[str, object]],
    *,
    generation_model: str,
) -> StructuredSummary:
    """Build the exact newest-first pool from which the compactor may select."""
    claims = []
    seen: set[str] = set()
    ordered_rows = _ordered_tag_sources(source_rows)
    source_ids = _tag_source_canonical_ids(ordered_rows)
    for row in ordered_rows:
        metadata = row.get("_metadata")
        if not isinstance(metadata, dict):
            metadata = _metadata_object(row.get("metadata_json")) or {}
        structured = strict_structured_summary(metadata.get("structured_summary"))
        if (
            structured.schema_version != STRUCTURED_SUMMARY_SCHEMA_VERSION
            or not structured.claims
        ):
            continue
        # Segment v1 already authenticates the mandatory safety prefix. Do not
        # let mutable metadata id order rearrange this claim tuple.
        for claim in structured.claims:
            if (
                len(claim.sources) != 1
                or claim.sources[0].source_role != "requester"
            ):
                raise MigrationError(
                    "tag source segment contains a non-v1 claim source shape",
                )
            fingerprint = structured_claim_fingerprint(claim)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            claims.append(claim)
            if len(claims) >= 256:
                break
        if len(claims) >= 256:
            break
    digest = structured_tag_claim_digest(claims, source_ids)
    return StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=tuple(claims),
        source_digest=digest,
        generation_model=generation_model,
    )


def _deterministic_tag_selection(
    expected_pool: StructuredSummary,
    source_canonical_turn_ids: Iterable[str],
) -> StructuredSummary:
    """Choose a bounded exact tag layer without provider-authored content."""
    selected = tuple(apply_tag_claim_safety_floor(
        expected_pool.claims,
        expected_pool.claims[:_MAX_TAG_SELECTED_CLAIMS],
        limit=_MAX_TAG_SELECTED_CLAIMS,
    ))
    source_ids = tuple(source_canonical_turn_ids)
    return StructuredSummary(
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        claims=selected,
        source_digest=structured_tag_claim_digest(selected, source_ids),
        generation_model=_DETERMINISTIC_GENERATION_MODEL,
    )


_MAX_TAG_SELECTED_CLAIMS = 16


def _tag_claim_selection_digest(
    claims: Iterable[object],
    source_canonical_turn_ids: Iterable[object],
) -> str:
    """Bind selected claims to exact ordered full tag source coverage."""
    return structured_tag_claim_digest(claims, source_canonical_turn_ids)


def _tag_claim_subset_reason(
    selected: Iterable[object],
    expected_pool: Iterable[object],
) -> str | None:
    """Prove every selected claim is an exact, non-duplicated pool member."""
    selected_claims = tuple(selected)
    expected_claims = tuple(expected_pool)
    if not selected_claims:
        return "tag_zero_claims"
    if len(selected_claims) > _MAX_TAG_SELECTED_CLAIMS:
        return "tag_claim_count_exceeded"

    expected_by_fingerprint = {
        structured_claim_fingerprint(claim): claim for claim in expected_claims
    }
    seen: set[str] = set()
    for claim in selected_claims:
        fingerprint = structured_claim_fingerprint(claim)
        if fingerprint in seen:
            return "tag_claim_duplicate"
        seen.add(fingerprint)
        if expected_by_fingerprint.get(fingerprint) != claim:
            return "tag_claim_copy_mismatch"
    if tuple(apply_tag_claim_safety_floor(
        expected_claims,
        selected_claims,
        limit=_MAX_TAG_SELECTED_CLAIMS,
    )) != selected_claims:
        return "tag_claim_safety_floor_mismatch"
    return None


def _json_string_list(raw: object) -> list[str] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if (
        not isinstance(raw, list)
        or any(
            type(value) is not str
            or not value
            or value != value.strip()
            for value in raw
        )
        or len(set(raw)) != len(raw)
    ):
        return None
    return list(raw)


def _load_existing_tag_rows(
    conn,
    conversation_id: str,
    tags: list[str],
) -> dict[str, dict]:
    if not tags:
        return {}
    rows = conn.execute(
        "SELECT ts.tag, ts.source_segment_refs, ts.source_turn_numbers, "
        "ts.source_canonical_turn_ids, ts.structured_summary_json, "
        "ts.covers_through_turn, ts.covers_through_canonical_turn_id, "
        "ts.xmin::text AS row_version, "
        "md5(ts.summary) AS old_summary_md5, "
        "md5(ts.description) AS old_description_md5, "
        "md5(jsonb_build_array(ts.summary, ts.description, ts.summary_tokens, "
        "ts.code_refs, ts.generated_by_turn_id, ts.created_at, "
        "ts.updated_at)::text) AS old_retrieval_artifacts_md5, "
        "e.xmin::text AS embedding_row_version, "
        "md5(e.embedding_json) AS old_embedding_md5 "
        "FROM tag_summaries ts LEFT JOIN tag_summary_embeddings e "
        "ON e.tag = ts.tag AND e.conversation_id = ts.conversation_id "
        "WHERE ts.conversation_id = %s AND ts.tag = ANY(%s)",
        (conversation_id, tags),
    ).fetchall()
    result = {str(row["tag"]): dict(row) for row in rows}
    embedding_rows = conn.execute(
        "SELECT tag, xmin::text AS embedding_row_version, "
        "md5(embedding_json) AS old_embedding_md5 "
        "FROM tag_summary_embeddings WHERE conversation_id = %s "
        "AND tag = ANY(%s)",
        (conversation_id, tags),
    ).fetchall()
    for row in embedding_rows:
        tag = str(row["tag"])
        target = result.setdefault(tag, {
            "tag": tag,
            "row_version": None,
            "source_segment_refs": None,
            "source_turn_numbers": None,
            "source_canonical_turn_ids": None,
            "structured_summary_json": None,
            "covers_through_turn": None,
            "covers_through_canonical_turn_id": None,
            "old_summary_md5": "",
            "old_description_md5": "",
            "old_retrieval_artifacts_md5": "",
        })
        target["embedding_row_version"] = row.get("embedding_row_version")
        target["old_embedding_md5"] = row.get("old_embedding_md5")
    return result


def _json_int_list(raw: object) -> list[int] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, list) or any(type(value) is not int for value in raw):
        return None
    return list(raw)


def _tag_inventory(
    conn,
    args,
    segment_inventory: _Inventory,
) -> _TagInventory:
    current_by_ref = {
        str(row["ref"]): row for row in segment_inventory.current_rows
    }
    candidate_by_ref = {
        str(row["ref"]): row for row in segment_inventory.candidates
    }
    skipped_by_ref = {
        str(row["ref"]): row for row in segment_inventory.skipped_rows
    }
    tag_sources: dict[str, list[dict]] = {}
    tag_pending: dict[str, list[str]] = {}
    tag_ineligible: dict[str, list[str]] = {}
    for row in segment_inventory.current_rows:
        for tag in _row_tags(row):
            tag_sources.setdefault(tag, []).append(row)
    for row in segment_inventory.candidates:
        for tag in _row_tags(row):
            tag_pending.setdefault(tag, []).append(str(row["ref"]))
    for row in segment_inventory.skipped_rows:
        for tag in _row_tags(row):
            tag_ineligible.setdefault(tag, []).append(str(row["ref"]))

    all_tags = sorted(set(tag_sources) | set(tag_pending) | set(tag_ineligible))
    after_tag = str(getattr(args, "after_tag", None) or "")
    if after_tag:
        all_tags = [tag for tag in all_tags if tag > after_tag]
    existing = _load_existing_tag_rows(conn, args.conversation_id, all_tags)
    all_source_ids = list(dict.fromkeys(
        canonical_id
        for row in segment_inventory.current_rows
        for canonical_id in (
            _canonical_ids(
                row.get("_metadata")
                if isinstance(row.get("_metadata"), dict)
                else (_metadata_object(row.get("metadata_json")) or {}),
            ) or []
        )
    ))
    physical_by_id = {
        str(row["canonical_turn_id"]): row
        for row in _load_source_rows(
            conn, args.conversation_id, all_source_ids, lock=False,
        )
    }

    candidates: list[dict] = []
    blocked: list[dict] = []
    reasons: Counter[str] = Counter()
    current = 0
    for tag in all_tags:
        ineligible_refs = sorted(tag_ineligible.get(tag, []))
        if ineligible_refs:
            blocked.append({
                "tag": tag,
                "reason": "source_segments_ineligible",
                "ineligible_source_count": len(ineligible_refs),
            })
            reasons["source_segments_ineligible"] += 1
            continue
        pending_refs = sorted(tag_pending.get(tag, []))
        if pending_refs:
            blocked.append({
                "tag": tag,
                "reason": "source_segments_pending",
                "pending_source_count": len(pending_refs),
            })
            reasons["source_segments_pending"] += 1
            continue
        sources = list(_ordered_tag_sources(tag_sources.get(tag, [])))
        if not sources:
            blocked.append({"tag": tag, "reason": "no_current_segment_claims"})
            reasons["no_current_segment_claims"] += 1
            continue
        expected = _deterministic_tag_envelope(
            sources, generation_model="inventory",
        )
        if not expected.claims or not expected.source_digest:
            blocked.append({"tag": tag, "reason": "no_current_segment_claims"})
            reasons["no_current_segment_claims"] += 1
            continue
        deterministic = _deterministic_tag_selection(
            expected, _tag_source_canonical_ids(sources),
        )
        if not deterministic.claims:
            blocked.append({
                "tag": tag,
                "reason": "tag_claim_safety_floor_overflow",
            })
            reasons["tag_claim_safety_floor_overflow"] += 1
            continue
        old = existing.get(tag)
        reason = ""
        if old is None or old.get("row_version") is None:
            blocked.append({
                "tag": tag,
                "reason": "tag_summary_missing",
            })
            reasons["tag_summary_missing"] += 1
            continue
        else:
            structured_raw = old.get("structured_summary_json")
            if isinstance(structured_raw, str):
                try:
                    structured_raw = json.loads(structured_raw)
                except (TypeError, ValueError):
                    structured_raw = None
            structured = strict_structured_summary(structured_raw)
            refs = _json_string_list(old.get("source_segment_refs"))
            turns = _json_int_list(old.get("source_turn_numbers"))
            source_ids = _json_string_list(
                old.get("source_canonical_turn_ids"),
            )
            expected_refs = [str(row["ref"]) for row in sources]
            expected_turns, expected_source_ids, expected_max_turn = (
                _tag_source_coordinates(
                    [_stored_summary_from_row(row) for row in sources],
                    physical_by_id,
                )
            )
            expected_cover_id = (
                expected_source_ids[-1] if expected_source_ids else ""
            )
            if structured.schema_version != STRUCTURED_SUMMARY_SCHEMA_VERSION:
                reason = "tag_structured_schema_missing_or_invalid"
            elif not structured.claims:
                reason = "tag_zero_claim_envelope"
            elif source_ids is None:
                reason = "tag_source_ids_missing_or_invalid"
            elif (
                structured.source_digest
                != _tag_claim_selection_digest(
                    structured.claims, source_ids,
                )
            ):
                reason = "tag_source_digest_stale"
            elif _tag_claim_subset_reason(
                structured.claims, expected.claims,
            ) is not None:
                reason = "tag_claim_selection_stale"
            elif not structured.generation_model.strip():
                reason = "tag_generation_model_missing"
            elif refs != expected_refs:
                reason = "tag_source_segment_set_stale"
            elif source_ids != expected_source_ids:
                reason = "tag_source_id_set_stale"
            elif turns != expected_turns:
                reason = "tag_source_turn_set_stale"
            elif old.get("covers_through_turn") != expected_max_turn:
                reason = "tag_covers_through_turn_stale"
            elif (
                old.get("covers_through_canonical_turn_id")
                != expected_cover_id
            ):
                reason = "tag_covers_through_id_stale"
            else:
                current += 1
                continue

        candidates.append({
            "tag": tag,
            "_candidate_reason": reason,
            "_source_rows": sources,
            "_source_refs": [str(row["ref"]) for row in sources],
            "_expected_structured": expected,
            "_deterministic_structured": deterministic,
            "_existing": old,
        })
        reasons[reason] += 1

    return _TagInventory(
        scanned=len(all_tags),
        candidates=candidates,
        blocked=blocked,
        current=current,
        reason_counts=reasons,
        source_segment_count=len(
            set(current_by_ref) | set(candidate_by_ref) | set(skipped_by_ref)
        ),
    )


def _full_segment_inventory(conn, args) -> tuple[_Scope, _Inventory, dict[str, str]]:
    """Tag gates always scan every segment, never a segment resume suffix."""
    values = dict(vars(args))
    values["after_ref"] = None
    return _inventory(conn, SimpleNamespace(**values))


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _source_timestamp(row: object) -> datetime | None:
    for key in ("first_seen_at", "last_seen_at", "created_at", "updated_at"):
        parsed = _parse_timestamp(_mapping_value(row, key))
        if parsed is not None:
            return parsed
    return None


def _reconstruct_segment(row: Mapping[str, object], rows_by_id: Mapping[str, object]) -> TaggedSegment:
    """Build model input exclusively from exact physical canonical rows."""
    metadata = row.get("_metadata")
    if not isinstance(metadata, dict):
        metadata = _metadata_object(row.get("metadata_json")) or {}
    ids = row.get("_canonical_ids")
    if not isinstance(ids, list):
        ids = _canonical_ids(metadata) or []
    messages: list[Message] = []
    for canonical_id in ids:
        source = rows_by_id[canonical_id]
        source_meta = {SOURCE_CANONICAL_TURN_IDS_KEY: [canonical_id]}
        raw_group = _mapping_value(source, "turn_group_number", -1)
        common = {
            "timestamp": _source_timestamp(source),
            "source_logical_turn_number": int(
                raw_group if raw_group is not None else -1,
            ),
            "source_audience_conversation_id": str(
                _mapping_value(source, "audience_conversation_id", "") or "",
            ),
            "source_origin_channel_id": str(
                _mapping_value(source, "origin_channel_id", "") or "",
            ),
            "source_audience_attribution_version": int(
                _mapping_value(source, "audience_attribution_version", 0) or 0,
            ),
        }
        user = str(_mapping_value(source, "user_content", "") or "")
        if user.strip():
            user_meta = dict(source_meta)
            sender = str(_mapping_value(source, "sender", "") or "").strip()
            if sender:
                user_meta["sender"] = {"name": sender}
            messages.append(Message(
                role="user",
                content=user,
                metadata=user_meta,
                source_actor_id=str(
                    _mapping_value(source, "sender_actor_id", "") or "",
                ).strip(),
                **common,
            ))
        assistant = str(_mapping_value(source, "assistant_content", "") or "")
        if assistant.strip():
            messages.append(Message(
                role="assistant",
                content=assistant,
                metadata=dict(source_meta),
                **common,
            ))

    tags = [str(tag) for tag in (row.get("tags") or []) if str(tag).strip()]
    primary = str(row.get("primary_tag") or "_general")
    if not tags:
        tags = [primary]
    start = _parse_timestamp(row.get("start_timestamp")) or datetime.now(timezone.utc)
    end = _parse_timestamp(row.get("end_timestamp")) or start
    turn_count_raw = metadata.get("turn_count", 0)
    turn_count = turn_count_raw if type(turn_count_raw) is int else 0
    session_date_raw = metadata.get("session_date", "")
    session_date = session_date_raw if type(session_date_raw) is str else ""
    return TaggedSegment(
        id=str(row["ref"]),
        primary_tag=primary,
        tags=tags,
        messages=messages,
        token_count=int(row.get("full_tokens") or 0),
        start_timestamp=start,
        end_timestamp=end,
        turn_count=turn_count,
        session_date=session_date,
    )


def _build_migration_actor_roster(
    canonical_ids: Iterable[str],
    rows_by_id: Mapping[str, object],
) -> ActorRoster:
    """Build the requester-only physical roster needed by the pure writer.

    The migration has already proved mapping completeness and source scope.
    Rebuilding this small DTO directly avoids constructing an Engine, store,
    summarization provider, or embedding provider during historical repair.
    """
    ids = list(canonical_ids)
    roster = ActorRoster(complete=bool(ids))
    for canonical_id in ids:
        row = rows_by_id.get(canonical_id)
        if row is None:
            roster.complete = False
            continue
        content = str(_mapping_value(row, "user_content", "") or "").strip()
        if not content:
            continue
        actor = str(
            _mapping_value(row, "sender_actor_id", "") or "",
        ).strip()
        label = str(_mapping_value(row, "sender", "") or "").strip()
        if actor:
            roster.actor_ids.add(actor)
            if label:
                roster.labels.setdefault(label.casefold(), set()).add(actor)
        else:
            roster.has_unidentified_user_row = True
        roster.lanes.append(FactLane(
            role=AUTHOR_ROLE_REQUESTER,
            text=content,
            actor_id=actor,
            source_message_id=str(
                _mapping_value(row, "source_message_id", "") or "",
            ),
            canonical_turn_id=str(canonical_id),
            speaker_label=label,
            session_date=str(
                _mapping_value(row, "session_date", "") or "",
            ).strip(),
            audience_conversation_id=str(
                _mapping_value(row, "audience_conversation_id", "") or "",
            ).strip(),
            origin_channel_id=str(
                _mapping_value(row, "origin_channel_id", "") or "",
            ).strip(),
            audience_attribution_version=(
                _mapping_value(row, "audience_attribution_version", 0)
                if type(_mapping_value(
                    row, "audience_attribution_version", 0,
                )) is int
                else 0
            ),
        ))
    return roster


def _load_summary_rows_by_refs(
    conn,
    conversation_id: str,
    refs: list[str],
    *,
    lock: bool = False,
) -> list[dict]:
    """Load only the already-validated segment summary layer for tag rollup."""
    if not refs:
        return []
    suffix = " FOR SHARE OF s" if lock else ""
    return list(conn.execute(
        "SELECT s.ref, s.conversation_id, s.primary_tag, s.summary, "
        "s.summary_tokens, s.full_tokens, s.metadata_json, s.created_at, "
        "s.start_timestamp, s.end_timestamp, s.xmin::text AS row_version, "
        "(SELECT COALESCE(array_agg(st.tag ORDER BY st.tag), ARRAY[]::text[]) "
        "FROM segment_tags st WHERE st.segment_ref = s.ref) AS tags "
        "FROM segments s WHERE s.conversation_id = %s AND s.ref = ANY(%s) "
        "ORDER BY s.created_at DESC, s.start_timestamp DESC, s.ref DESC" + suffix,
        (conversation_id, refs),
    ).fetchall())


def _stored_summary_from_row(row: Mapping[str, object]) -> StoredSummary:
    metadata = _metadata_object(row.get("metadata_json")) or {}
    structured = strict_structured_summary(metadata.get("structured_summary"))
    canonical_ids = _canonical_ids(metadata) or []
    labels = metadata.get("source_speaker_labels")
    if not isinstance(labels, list) or any(type(value) is not str for value in labels):
        labels = []
    code_refs = metadata.get("code_refs")
    if not isinstance(code_refs, list):
        code_refs = []
    return StoredSummary(
        ref=str(row["ref"]),
        primary_tag=str(row.get("primary_tag") or "_general"),
        tags=[str(tag) for tag in (row.get("tags") or []) if str(tag).strip()],
        summary=str(row.get("summary") or ""),
        summary_tokens=int(row.get("summary_tokens") or 0),
        full_tokens=int(row.get("full_tokens") or 0),
        metadata=SegmentMetadata(
            entities=(metadata.get("entities") if isinstance(
                metadata.get("entities"), list,
            ) else []),
            key_decisions=(metadata.get("key_decisions") if isinstance(
                metadata.get("key_decisions"), list,
            ) else []),
            action_items=(metadata.get("action_items") if isinstance(
                metadata.get("action_items"), list,
            ) else []),
            date_references=(metadata.get("date_references") if isinstance(
                metadata.get("date_references"), list,
            ) else []),
            code_refs=code_refs,
            turn_count=(
                metadata.get("turn_count", 0)
                if type(metadata.get("turn_count", 0)) is int else 0
            ),
            canonical_turn_ids=canonical_ids,
            source_speaker_labels=list(labels),
            source_speaker_identity_count=(
                metadata.get("source_speaker_identity_count", 0)
                if type(metadata.get("source_speaker_identity_count", 0)) is int
                else 0
            ),
            source_speaker_identity_fingerprint=str(
                metadata.get("source_speaker_identity_fingerprint", "") or "",
            ),
            source_audience_fingerprint=str(
                metadata.get("source_audience_fingerprint", "") or "",
            ),
            session_date=(
                metadata.get("session_date", "")
                if type(metadata.get("session_date", "")) is str else ""
            ),
            source_mapping_complete=metadata.get("source_mapping_complete") is True,
            structured_summary=structured,
        ),
        created_at=_parse_timestamp(row.get("created_at")) or datetime.now(timezone.utc),
        start_timestamp=(
            _parse_timestamp(row.get("start_timestamp"))
            or datetime.now(timezone.utc)
        ),
        end_timestamp=(
            _parse_timestamp(row.get("end_timestamp"))
            or datetime.now(timezone.utc)
        ),
    )


def _tag_source_coordinates(
    summaries: Iterable[StoredSummary],
    physical_by_id: Mapping[str, object],
) -> tuple[list[int], list[str], int]:
    canonical_ids: list[str] = []
    seen: set[str] = set()
    turns: set[int] = set()
    for summary in summaries:
        for canonical_id in summary.metadata.canonical_turn_ids:
            if canonical_id not in seen:
                seen.add(canonical_id)
                canonical_ids.append(canonical_id)
            row = physical_by_id.get(canonical_id)
            raw_turn = _mapping_value(row, "turn_group_number", -1)
            if type(raw_turn) is int and raw_turn >= 0:
                turns.add(raw_turn)
    turn_numbers = sorted(turns)
    return turn_numbers, canonical_ids, max(turn_numbers, default=-1)


def _validate_tag_result(
    result: object,
    *,
    tag: str,
    expected: StructuredSummary,
    expected_model: str,
    source_refs: list[str],
    turn_numbers: list[int],
    canonical_ids: list[str],
) -> str | None:
    from ..types import TagSummary

    if not isinstance(result, TagSummary):
        return "not_tag_summary"
    if result.tag != tag:
        return "tag_mismatch"
    if not isinstance(result.summary, str) or not result.summary.strip():
        return "tag_synopsis_missing"
    if not isinstance(result.description, str):
        return "tag_description_invalid"
    structured = result.structured_summary
    if strict_structured_summary(structured_summary_to_dict(structured)) != structured:
        return "tag_schema_round_trip_failed"
    subset_reason = _tag_claim_subset_reason(
        structured.claims, expected.claims,
    )
    if subset_reason is not None:
        return subset_reason
    if any(
        len(claim.sources) != 1
        or claim.sources[0].source_role != "requester"
        for claim in structured.claims
    ):
        return "tag_claim_source_cardinality"
    if (
        structured.source_digest
        != _tag_claim_selection_digest(
            structured.claims, result.source_canonical_turn_ids,
        )
    ):
        return "tag_source_digest_mismatch"
    if not structured.generation_model.strip():
        return "tag_generation_model_missing"
    if expected_model and structured.generation_model != expected_model:
        return "tag_generation_model_mismatch"
    if result.source_segment_refs != source_refs:
        return "tag_source_refs_mismatch"
    if result.source_turn_numbers != turn_numbers:
        return "tag_turn_numbers_mismatch"
    if result.source_canonical_turn_ids != canonical_ids:
        return "tag_canonical_ids_mismatch"
    return None


def _tag_source_set_digest(rows: Iterable[Mapping[str, object]]) -> str:
    payload = [
        {
            "ref": str(row["ref"]),
            "row_version": str(row["row_version"]),
            "source_digest": strict_structured_summary(
                (_metadata_object(row.get("metadata_json")) or {}).get(
                    "structured_summary",
                ),
            ).source_digest,
        }
        for row in _ordered_tag_sources(rows)
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"vc-tag-source-set-v1\0" + encoded).hexdigest()


def _validate_envelope(
    envelope: object,
    *,
    expected_digest: str,
    expected_model: str,
    canonical_ids: list[str],
    rows_by_id: Mapping[str, object],
    session_date: str,
) -> str | None:
    """Return a rejection reason, or ``None`` for a persistable envelope."""
    if not isinstance(envelope, StructuredSummary):
        return "not_structured_summary"
    encoded = structured_summary_to_dict(envelope)
    if strict_structured_summary(encoded) != envelope:
        return "schema_round_trip_failed"
    if envelope.schema_version != STRUCTURED_SUMMARY_SCHEMA_VERSION:
        return "wrong_schema_version"
    if envelope.source_digest != expected_digest:
        return "source_digest_mismatch"
    if not envelope.generation_model.strip():
        return "generation_model_missing"
    if expected_model and envelope.generation_model != expected_model:
        return "generation_model_mismatch"
    if not envelope.claims:
        # A provider fallback and a genuinely claim-free response are not
        # distinguishable after compaction.  Never strand either as current.
        return "zero_claims"

    allowed = set(canonical_ids)
    admissible_by_id = {
        str(record["canonical_turn_id"]): record
        for record in _admissible_requester_records(
            canonical_ids, rows_by_id, session_date=session_date,
        )
    }
    if structured_claims_contain_internal_identity(
        envelope.claims,
        admitted_actor_ids=(
            record.get("actor_id", "") for record in admissible_by_id.values()
        ),
    ):
        return "internal_identity_in_claim"
    claimed_source_ids = {
        source.canonical_turn_id
        for claim in envelope.claims
        for source in claim.sources
    }
    for canonical_id, record in admissible_by_id.items():
        content = str(record.get("content", "") or "").strip()
        if (
            0 < len(content) <= STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS
            and canonical_id not in claimed_source_ids
        ):
            return "bounded_source_claim_missing"
    for claim in envelope.claims:
        requester_actors: set[str] = set()
        if claim.claim_type != "conversation":
            return "claim_type_not_evidenced"
        if len(claim.sources) != 1:
            return "claim_source_cardinality"
        for source in claim.sources:
            if source.canonical_turn_id not in allowed:
                return "source_id_outside_segment"
            row = rows_by_id.get(source.canonical_turn_id)
            if row is None:
                return "source_row_missing"
            if source.canonical_turn_id not in admissible_by_id:
                return "requester_source_not_admissible"
            if source.source_role != "requester":
                return "assistant_source_forbidden"
            content = str(_mapping_value(row, "user_content", "") or "")
            label = str(_mapping_value(row, "sender", "") or "").strip()
            actor = str(
                _mapping_value(row, "sender_actor_id", "") or "",
            ).strip()
            requester_actors.add(actor)
            if not actor or source.speaker_label != label:
                return "requester_identity_mismatch"
            # V1 presents a complete trimmed human-authored lane. Substring
            # evidence is not sufficient: selecting around a negation can
            # invert the meaning of an otherwise exact source row.
            if source.evidence_excerpt != content.strip():
                return "evidence_excerpt_mismatch"
            physical_session_date = str(
                _mapping_value(row, "session_date", "") or "",
            )
            if source.session_date != physical_session_date:
                return "source_session_date_mismatch"
            expected_provenance = structured_source_provenance_digest({
                "canonical_turn_id": source.canonical_turn_id,
                "source_role": "requester",
                "actor_id": actor,
                "speaker_label": label,
                "content": content.strip(),
                "session_date": physical_session_date,
                "audience_conversation_id": str(
                    _mapping_value(row, "audience_conversation_id", "") or "",
                ),
                "origin_channel_id": str(
                    _mapping_value(row, "origin_channel_id", "") or "",
                ),
                "audience_attribution_version": _mapping_value(
                    row, "audience_attribution_version", 0,
                ),
            })
            if source.source_provenance_digest != expected_provenance:
                return "source_provenance_digest_mismatch"
        if len(requester_actors) != 1 or "" in requester_actors:
            return "requester_actor_ambiguous"
        evidence = claim.sources[0].evidence_excerpt
        if claim.temporal_status != infer_temporal_status(evidence):
            return "temporal_status_not_evidenced"
        if claim.modality != infer_modality(evidence):
            return "modality_not_evidenced"
        if not event_time_is_supported(
            claim.event_time, [evidence], claim.sources[0].session_date,
        ):
            return "event_time_not_evidenced"
    bounded_records = [
        record for record in admissible_by_id.values()
        if 0 < len(str(record.get("content", "") or "").strip())
        <= STRUCTURED_SUMMARY_MAX_EXCERPT_CHARS
    ]
    critical_ids = [
        str(record["canonical_turn_id"])
        for record in reversed(bounded_records)
        if is_safety_critical_personal_evidence(
            str(record.get("content", "") or "").strip(),
        )
    ]
    ordinary_ids = [
        str(record["canonical_turn_id"])
        for record in bounded_records
        if not is_safety_critical_personal_evidence(
            str(record.get("content", "") or "").strip(),
        )
    ]
    actual_claim_ids = [
        claim.sources[0].canonical_turn_id
        for claim in envelope.claims
    ]
    if actual_claim_ids != critical_ids + ordinary_ids:
        return "claim_order_mismatch"
    return None


def _fsync_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "/"
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _journal_append(handle, payload: Mapping[str, object]) -> None:
    record = dict(payload)
    record.setdefault("migration_mode", _MIGRATION_MODE)
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _journal_path(args) -> str:
    if getattr(args, "journal", None):
        return args.journal
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.conversation_id).strip("_")
    return f"structured-summary-migration-{safe or 'conversation'}.jsonl"


def _same_scope(left: _Scope, right: _Scope) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.conversation_id == right.conversation_id
        and left.lifecycle_epoch == right.lifecycle_epoch
        and left.lifecycle_generation == right.lifecycle_generation
        and right.phase == "active"
    )


def _cas_persist(
    conn,
    args,
    *,
    initial_scope: _Scope,
    row: Mapping[str, object],
    envelope: StructuredSummary,
    journal,
    run_id: str,
) -> str:
    """Revalidate source/lifecycle and add only the structured layer.

    The existing free-form segment synopsis remains the retrieval index.  It
    is deliberately neither selected into this process nor rewritten by the
    historical repair.
    """
    metadata = row["_metadata"]
    ids = list(row["_canonical_ids"])
    expected_digest = str(row["_expected_source_digest"])
    with conn.transaction():
        try:
            current_scope = _preflight(conn, args, lock=True)
        except MigrationError:
            return "lifecycle_changed"
        if not _same_scope(initial_scope, current_scope):
            return "lifecycle_changed"

        current_segment = conn.execute(
            "SELECT xmin::text AS row_version FROM segments "
            "WHERE ref = %s AND conversation_id = %s FOR UPDATE",
            (row["ref"], args.conversation_id),
        ).fetchone()
        if (
            not current_segment
            or str(current_segment["row_version"]) != str(row["row_version"])
        ):
            return "segment_changed"

        source_rows = _load_source_rows(
            conn, args.conversation_id, ids, lock=True,
        )
        source_by_id = {
            str(source["canonical_turn_id"]): source for source in source_rows
        }
        alias_graph = _load_source_alias_chains_locked(
            conn, source_by_id, ids,
            owner_conversation_id=args.conversation_id,
        )
        source_reasons = _source_validation_reasons(
            ids, source_by_id,
            owner_conversation_id=args.conversation_id,
            alias_graph=alias_graph,
        )
        if source_reasons:
            return "canonical_source_changed"
        session_date = metadata.get("session_date", "")
        if type(session_date) is not str:
            session_date = ""
        locked_digest = _expected_source_digest(
            ids, source_by_id, session_date=session_date,
        )
        if locked_digest != expected_digest or envelope.source_digest != locked_digest:
            return "canonical_source_changed"
        if _validate_envelope(
            envelope,
            expected_digest=locked_digest,
            expected_model=envelope.generation_model,
            canonical_ids=ids,
            rows_by_id=source_by_id,
            session_date=session_date,
        ) is not None:
            return "canonical_source_changed"

        encoded = structured_summary_to_dict(envelope)
        encoded_json = json.dumps(
            encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        prepared = {
            "event": "prepared",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "tenant_id": args.tenant_id,
            "conversation_id": args.conversation_id,
            "lifecycle_epoch": initial_scope.lifecycle_epoch,
            "lifecycle_generation": initial_scope.lifecycle_generation,
            "ref": row["ref"],
            "expected_row_version": str(row["row_version"]),
            "source_digest": locked_digest,
            "structured_summary_sha256": hashlib.sha256(
                encoded_json.encode("utf-8"),
            ).hexdigest(),
            "old_retrieval_synopsis_md5": str(
                row.get("old_retrieval_synopsis_md5") or "",
            ),
            "new_retrieval_synopsis_md5": str(
                row.get("old_retrieval_synopsis_md5") or "",
            ),
            "retrieval_synopsis_preserved": True,
            "generation_model": envelope.generation_model,
            "claim_count": len(envelope.claims),
        }
        # Write-ahead is durable before the SQL mutation. A crash may leave a
        # harmless prepared-only entry, but never an unjournaled commit.
        _journal_append(journal, prepared)

        updated = conn.execute(
            _CAS_UPDATE_SQL,
            (
                encoded_json,
                row["ref"], args.conversation_id,
                str(row["row_version"]), args.conversation_id, args.tenant_id,
                initial_scope.lifecycle_epoch, args.conversation_id,
                initial_scope.lifecycle_generation, args.conversation_id,
                initial_scope.lifecycle_epoch, args.conversation_id,
                initial_scope.lifecycle_epoch,
            ),
        ).rowcount
        if updated != 1:
            return "cas_failed"
    return "accepted"


def _load_linked_tag_segments_locked(conn, conversation_id: str, tag: str) -> list[dict]:
    # Lock the existing link rows as well as segments. Conversation lifecycle
    # locking in the caller prevents a normal compaction from adding a new
    # source set while this transaction commits.
    conn.execute(
        "SELECT st.segment_ref FROM segment_tags st JOIN segments s "
        "ON s.ref = st.segment_ref WHERE st.tag = %s "
        "AND s.conversation_id = %s FOR SHARE OF st",
        (tag, conversation_id),
    ).fetchall()
    return list(conn.execute(
        "SELECT s.ref, s.conversation_id, s.primary_tag, s.summary, "
        "s.summary_tokens, s.full_tokens, s.metadata_json, s.created_at, "
        "s.start_timestamp, s.end_timestamp, s.xmin::text AS row_version, "
        "(SELECT COALESCE(array_agg(st.tag ORDER BY st.tag), ARRAY[]::text[]) "
        "FROM segment_tags st WHERE st.segment_ref = s.ref) AS tags "
        "FROM segments s WHERE s.conversation_id = %s AND "
        "(s.primary_tag = %s OR EXISTS (SELECT 1 FROM segment_tags link "
        "WHERE link.segment_ref = s.ref AND link.tag = %s)) "
        "ORDER BY s.created_at DESC, s.start_timestamp DESC, s.ref DESC "
        "FOR SHARE OF s",
        (conversation_id, tag, tag),
    ).fetchall())


def _locked_current_tag_sources(
    conn,
    args,
    *,
    tag: str,
    expected_rows: list[Mapping[str, object]],
) -> tuple[str | None, list[dict]]:
    linked = _load_linked_tag_segments_locked(
        conn, args.conversation_id, tag,
    )
    metadata_by_ref: dict[str, dict] = {}
    ids_by_ref: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for row in linked:
        metadata = _metadata_object(row.get("metadata_json"))
        if metadata is None or metadata.get("source_mapping_complete") is not True:
            continue
        ids = _canonical_ids(metadata)
        if not ids:
            continue
        metadata_by_ref[str(row["ref"])] = metadata
        ids_by_ref[str(row["ref"])] = ids
        all_ids.extend(ids)
    source_rows = _load_source_rows(
        conn, args.conversation_id, list(dict.fromkeys(all_ids)), lock=True,
    )
    canonical_by_id = {
        str(row["canonical_turn_id"]): row for row in source_rows
    }
    alias_graph = _load_source_alias_chains_locked(
        conn, canonical_by_id, list(dict.fromkeys(all_ids)),
        owner_conversation_id=args.conversation_id,
    )

    current: list[dict] = []
    for row in linked:
        ref = str(row["ref"])
        metadata = metadata_by_ref.get(ref)
        ids = ids_by_ref.get(ref)
        if metadata is None or not ids:
            continue
        reasons = _source_validation_reasons(
            ids, canonical_by_id,
            owner_conversation_id=args.conversation_id,
            alias_graph=alias_graph,
        )
        if reasons:
            continue
        session_date = metadata.get("session_date", "")
        if type(session_date) is not str:
            session_date = ""
        expected_digest = _expected_source_digest(
            ids, canonical_by_id, session_date=session_date,
        )
        if not expected_digest:
            continue
        structured = strict_structured_summary(metadata.get("structured_summary"))
        if (
            structured.schema_version != STRUCTURED_SUMMARY_SCHEMA_VERSION
            or not structured.claims
            or structured.source_digest != expected_digest
            or not structured.generation_model.strip()
            or not str(row.get("summary") or "").strip()
        ):
            return "source_segment_not_current", []
        if _validate_envelope(
            structured,
            expected_digest=expected_digest,
            expected_model="",
            canonical_ids=ids,
            rows_by_id=canonical_by_id,
            session_date=session_date,
        ) is not None:
            return "source_segment_not_current", []
        copied = dict(row)
        copied["_metadata"] = metadata
        copied["_canonical_ids"] = ids
        copied["_expected_source_digest"] = expected_digest
        current.append(copied)

    current = list(_ordered_tag_sources(current))
    expected_sorted = list(_ordered_tag_sources(expected_rows))
    if [row["ref"] for row in current] != [row["ref"] for row in expected_sorted]:
        return "tag_source_set_changed", []
    for observed, expected in zip(current, expected_sorted, strict=True):
        if str(observed["row_version"]) != str(expected["row_version"]):
            return "tag_source_segment_changed", []
    return None, current


def _tag_metadata_write_values(result: TagSummary) -> dict[str, str]:
    """Encode only the source-bound fields this repair is allowed to change."""
    structured_json = json.dumps(
        structured_summary_to_dict(result.structured_summary),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return {
        "structured_json": structured_json,
        "source_refs_json": json.dumps(
            result.source_segment_refs, separators=(",", ":"),
        ),
        "turns_json": json.dumps(
            result.source_turn_numbers, separators=(",", ":"),
        ),
        "canonical_ids_json": json.dumps(
            result.source_canonical_turn_ids, separators=(",", ":"),
        ),
    }


def _cas_tag_persist(
    conn,
    args,
    *,
    initial_scope: _Scope,
    candidate: Mapping[str, object],
    result: TagSummary,
    journal,
    run_id: str,
) -> str:
    """CAS only tag provenance metadata; retrieval artifacts stay byte-identical."""
    tag = str(candidate["tag"])
    old = candidate.get("_existing")
    old = old if isinstance(old, Mapping) else {}
    values = _tag_metadata_write_values(result)
    try:
        with conn.transaction():
            try:
                current_scope = _preflight(conn, args, lock=True)
            except MigrationError:
                return "lifecycle_changed"
            if not _same_scope(initial_scope, current_scope):
                return "lifecycle_changed"
            source_error, locked_sources = _locked_current_tag_sources(
                conn, args, tag=tag,
                expected_rows=list(candidate["_source_rows"]),
            )
            if source_error:
                return source_error
            locked_canonical_ids = list(dict.fromkeys(
                canonical_id
                for source_row in locked_sources
                for canonical_id in (
                    _canonical_ids(
                        _metadata_object(source_row.get("metadata_json")) or {},
                    ) or []
                )
            ))
            locked_physical_rows = _load_source_rows(
                conn,
                args.conversation_id,
                locked_canonical_ids,
                lock=True,
            )
            locked_physical_by_id = {
                str(row["canonical_turn_id"]): row
                for row in locked_physical_rows
            }
            locked_summaries = [
                _stored_summary_from_row(row) for row in locked_sources
            ]
            locked_turns, locked_ids, locked_max_turn = _tag_source_coordinates(
                locked_summaries, locked_physical_by_id,
            )
            locked_refs = [summary.ref for summary in locked_summaries]
            locked_cover_id = locked_ids[-1] if locked_ids else ""
            if (
                result.source_segment_refs != locked_refs
                or result.source_turn_numbers != locked_turns
                or result.source_canonical_turn_ids != locked_ids
                or result.covers_through_turn != locked_max_turn
                or result.covers_through_canonical_turn_id != locked_cover_id
            ):
                return "tag_source_coordinates_changed"
            locked_expected = _deterministic_tag_envelope(
                locked_sources,
                generation_model=result.structured_summary.generation_model,
            )
            if _tag_claim_subset_reason(
                result.structured_summary.claims, locked_expected.claims,
            ) is not None:
                return "tag_structured_source_changed"
            if (
                result.structured_summary.source_digest
                != _tag_claim_selection_digest(
                    result.structured_summary.claims, locked_ids,
                )
            ):
                return "tag_structured_source_changed"
            if (
                result.structured_summary.generation_model
                != _DETERMINISTIC_GENERATION_MODEL
                or strict_structured_summary(structured_summary_to_dict(
                    result.structured_summary,
                )) != result.structured_summary
            ):
                return "tag_structured_source_changed"

            current_tag = conn.execute(
                "SELECT xmin::text AS row_version, "
                "md5(summary) AS summary_md5, "
                "md5(description) AS description_md5, "
                "md5(jsonb_build_array(summary, description, summary_tokens, "
                "code_refs, generated_by_turn_id, created_at, updated_at)::text) "
                "AS retrieval_artifacts_md5 FROM tag_summaries "
                "WHERE tag = %s AND conversation_id = %s FOR UPDATE",
                (tag, args.conversation_id),
            ).fetchone()
            current_embedding = conn.execute(
                "SELECT xmin::text AS row_version, "
                "md5(embedding_json) AS embedding_md5 "
                "FROM tag_summary_embeddings "
                "WHERE tag = %s AND conversation_id = %s FOR UPDATE",
                (tag, args.conversation_id),
            ).fetchone()
            expected_tag_version = old.get("row_version")
            expected_embedding_version = old.get("embedding_row_version")
            old_summary_md5 = str(old.get("old_summary_md5") or "")
            old_description_md5 = str(old.get("old_description_md5") or "")
            old_retrieval_md5 = str(
                old.get("old_retrieval_artifacts_md5") or "",
            )
            old_embedding_md5 = str(old.get("old_embedding_md5") or "")
            if (
                current_tag is None
                or expected_tag_version is None
                or str(current_tag["row_version"]) != str(expected_tag_version)
                or str(current_tag.get("summary_md5") or "") != old_summary_md5
                or (
                    str(current_tag.get("description_md5") or "")
                    != old_description_md5
                )
                or (
                    str(current_tag.get("retrieval_artifacts_md5") or "")
                    != old_retrieval_md5
                )
                or (current_embedding is None) != (expected_embedding_version is None)
                or (
                    current_embedding is not None
                    and str(current_embedding["row_version"])
                    != str(expected_embedding_version)
                )
                or (
                    str((
                        current_embedding.get("embedding_md5")
                        if current_embedding is not None else ""
                    ) or "")
                    != old_embedding_md5
                )
            ):
                return "tag_or_embedding_changed"

            prepared = {
                "event": "tag_prepared",
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "tenant_id": args.tenant_id,
                "conversation_id": args.conversation_id,
                "lifecycle_epoch": initial_scope.lifecycle_epoch,
                "lifecycle_generation": initial_scope.lifecycle_generation,
                "tag": tag,
                "migration_mode": _MIGRATION_MODE,
                "provider_calls": 0,
                "source_set_digest": _tag_source_set_digest(locked_sources),
                "source_coordinates_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "refs": locked_refs,
                            "turns": locked_turns,
                            "canonical_ids": locked_ids,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                ).hexdigest(),
                "old_summary_md5": old_summary_md5,
                "new_summary_md5": old_summary_md5,
                "old_description_md5": old_description_md5,
                "new_description_md5": old_description_md5,
                "old_retrieval_artifacts_md5": old_retrieval_md5,
                "new_retrieval_artifacts_md5": old_retrieval_md5,
                "old_embedding_md5": old_embedding_md5,
                "new_embedding_md5": old_embedding_md5,
                "retrieval_artifacts_preserved": True,
                "embedding_preserved": True,
                "structured_summary_sha256": hashlib.sha256(
                    str(values["structured_json"]).encode("utf-8"),
                ).hexdigest(),
                "claim_count": len(result.structured_summary.claims),
            }
            _journal_append(journal, prepared)

            tag_write = conn.execute(
                "UPDATE tag_summaries SET source_segment_refs = %s, "
                "source_turn_numbers = %s, source_canonical_turn_ids = %s, "
                "structured_summary_json = %s, covers_through_turn = %s, "
                "covers_through_canonical_turn_id = %s "
                "WHERE tag = %s AND conversation_id = %s AND xmin::text = %s "
                "RETURNING md5(summary) AS summary_md5, "
                "md5(description) AS description_md5, "
                "md5(jsonb_build_array(summary, description, summary_tokens, "
                "code_refs, generated_by_turn_id, created_at, updated_at)::text) "
                "AS retrieval_artifacts_md5",
                (
                    values["source_refs_json"], values["turns_json"],
                    values["canonical_ids_json"], values["structured_json"],
                    result.covers_through_turn,
                    result.covers_through_canonical_turn_id,
                    tag, args.conversation_id, str(expected_tag_version),
                ),
            ).fetchone()
            if tag_write is None:
                raise _TagCASConflict("tag summary CAS failed")
            if (
                str(tag_write.get("summary_md5") or "") != old_summary_md5
                or str(tag_write.get("description_md5") or "")
                != old_description_md5
                or str(tag_write.get("retrieval_artifacts_md5") or "")
                != old_retrieval_md5
            ):
                raise _TagCASConflict("tag retrieval artifacts changed")
    except _TagCASConflict:
        return "tag_or_embedding_changed"
    return "accepted"


def _dry_run(args) -> None:
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)", args)
    try:
        with _connect(dsn, read_only=True) as conn:
            mode = conn.execute("SHOW transaction_read_only").fetchone()
            read_only = str(next(iter(mode.values()))).lower() == "on"
            if not read_only:
                raise MigrationError("server did not enforce read-only mode")
            before = _checksums(conn, args.conversation_id)
            scope, inventory, _aliases = _inventory(conn, args)
            after = _checksums(conn, args.conversation_id)
    except Exception as exc:  # noqa: BLE001
        _fail("dry_run", str(exc), args)

    planned = inventory.candidates
    if args.limit is not None:
        planned = planned[:args.limit]
    print(json.dumps({
        "status": "dry_run",
        "phase": "segments",
        "conversation_id": args.conversation_id,
        "tenant_id": args.tenant_id,
        "lifecycle_epoch": scope.lifecycle_epoch,
        "lifecycle_generation": scope.lifecycle_generation,
        "source_aliases_verified": scope.direct_source_alias_count,
        "server_enforced_read_only": True,
        "scanned": inventory.scanned,
        "current": inventory.current,
        "candidate_total": len(inventory.candidates),
        "selected": len(planned),
        "candidate_reason_counts": dict(sorted(
            inventory.candidate_reason_counts.items(),
        )),
        "skipped_reason_counts": dict(sorted(
            inventory.skipped_reason_counts.items(),
        )),
        "first_ref": planned[0]["ref"] if planned else None,
        "last_ref": planned[-1]["ref"] if planned else None,
        "segments": [
            {"ref": row["ref"], "reason": row["_candidate_reason"]}
            for row in planned
        ],
        "provider_call_estimate": {
            "minimum": 0,
            "maximum_with_retry": 0,
        },
        "affected_tag_count": len(inventory.affected_tags),
        "checksums_before": before,
        "checksums_after": after,
        "checksums_stable": before == after,
        "tag_phase": "runs_after_segments_when_phase_all",
        "note": (
            "Apply re-reads canonical rows, rebuilds the ActorRoster, and "
            "revalidates xmin, lifecycle, aliases, and source_digest before "
            "atomically adding metadata_json.structured_summary. Existing "
            "retrieval synopsis bytes and indexes are preserved."
        ),
    }, indent=2))


def _cache_action(conversation_id: str) -> dict[str, object]:
    escaped_conversation_id = re.sub(
        r"([\\*?\[\]])", r"\\\1", conversation_id,
    )
    return {
        "required": True,
        "redis_delete_keys": [
            f"vc:tag_summary_embeddings:{conversation_id}",
            f"vc:tag_stats:{conversation_id}",
        ],
        "redis_delete_glob": f"vc:context_hint:{escaped_conversation_id}:*",
        "worker_action": (
            "recycle every serving worker after Redis invalidation to clear "
            "process-local tag-summary, tag-stat, and context-hint caches"
        ),
        "completion_rule": (
            "tag migration is not serving-complete until Redis keys are gone "
            "and every worker start time is later than the migration"
        ),
    }


def _dry_run_tags(args) -> None:
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)", args)
    try:
        with _connect(dsn, read_only=True) as conn:
            mode = conn.execute("SHOW transaction_read_only").fetchone()
            if str(next(iter(mode.values()))).lower() != "on":
                raise MigrationError("server did not enforce read-only mode")
            before = _checksums(conn, args.conversation_id)
            scope, segment_inventory, _aliases = _full_segment_inventory(conn, args)
            tag_inventory = _tag_inventory(conn, args, segment_inventory)
            after = _checksums(conn, args.conversation_id)
    except Exception as exc:  # noqa: BLE001
        _fail("tag_dry_run", str(exc), args)

    planned = tag_inventory.candidates
    if args.limit is not None:
        planned = planned[:args.limit]
    segment_gate = len(segment_inventory.candidates)
    print(json.dumps({
        "status": "dry_run",
        "phase": "tags",
        "conversation_id": args.conversation_id,
        "tenant_id": args.tenant_id,
        "lifecycle_epoch": scope.lifecycle_epoch,
        "lifecycle_generation": scope.lifecycle_generation,
        "server_enforced_read_only": True,
        "segment_gate": {
            "pending_eligible_segments": segment_gate,
            "ready": segment_gate == 0,
            "rule": (
                "tag apply refuses until every eligible segment has a "
                "non-empty current structured envelope"
            ),
        },
        "scanned": tag_inventory.scanned,
        "current": tag_inventory.current,
        "candidate_total": len(tag_inventory.candidates),
        "selected": len(planned),
        "blocked": len(tag_inventory.blocked),
        "reason_counts": dict(sorted(tag_inventory.reason_counts.items())),
        "provider_call_estimate": {
            "minimum": 0,
            "maximum_with_retry": 0,
        },
        "migration_mode": _MIGRATION_MODE,
        "first_tag": planned[0]["tag"] if planned else None,
        "last_tag": planned[-1]["tag"] if planned else None,
        "tags": [
            {"tag": row["tag"], "reason": row["_candidate_reason"]}
            for row in planned
        ],
        "blocked_tags": tag_inventory.blocked,
        "checksums_before": before,
        "checksums_after": after,
        "checksums_stable": before == after,
        "cache_action_after_apply": _cache_action(args.conversation_id),
        "apply_ready": segment_gate == 0 and before == after,
        "note": (
            "Tag apply copies deterministic exact segment claims and changes "
            "only source coordinates plus structured_summary_json. Existing "
            "tag synopsis, description, tokens, code refs, timestamps, and "
            "embedding bytes are preserved without provider calls."
        ),
    }, indent=2))


def _apply(args) -> None:
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)", args)

    # Inventory and generation use only direct database rows. Historical
    # repair must not construct an Engine or any LLM/embedding provider.
    try:
        with _connect(dsn, read_only=True) as conn:
            scope, inventory, alias_graph = _inventory(conn, args)
    except Exception as exc:  # noqa: BLE001
        _fail("apply_preflight", str(exc), args)

    selected = inventory.candidates
    if args.limit is not None:
        selected = selected[:args.limit]
    journal_path = _journal_path(args)
    run_id = str(uuid.uuid4())
    if not selected:
        print(json.dumps({
            "status": "completed",
            "phase": "segments",
            "conversation_id": args.conversation_id,
            "selected": 0,
            "accepted": 0,
            "migration_mode": _MIGRATION_MODE,
            "provider_calls": 0,
            "usage": {"calls": 0, "tokens_in": 0, "tokens_out": 0},
            "tag_phase": "runs_after_segments_when_phase_all",
        }, indent=2))
        return

    selected_source_ids = list(dict.fromkeys(
        canonical_id
        for row in selected
        for canonical_id in list(row["_canonical_ids"])
    ))
    try:
        with _connect(dsn, read_only=True) as source_conn:
            physical_rows = _load_source_rows(
                source_conn,
                args.conversation_id,
                selected_source_ids,
                lock=False,
            )
    except Exception as exc:  # noqa: BLE001
        _fail("load_canonical_sources", str(exc), args)
    physical_by_id = {
        str(row["canonical_turn_id"]): row for row in physical_rows
    }

    counts = Counter({
        "accepted": 0,
        "skipped_concurrent": 0,
        "rejected": 0,
    })
    rejected: Counter[str] = Counter()
    usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    affected_tags: set[str] = set()
    cursor = _ResumeCursor(args.after_ref)
    status = "completed"
    last_attempted_ref = None

    try:
        with open(journal_path, "a", encoding="utf-8") as journal:
            _fsync_parent_dir(journal_path)
            _journal_append(journal, {
                "event": "run_started",
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "tenant_id": args.tenant_id,
                "conversation_id": args.conversation_id,
                "lifecycle_epoch": scope.lifecycle_epoch,
                "lifecycle_generation": scope.lifecycle_generation,
                "after_ref": args.after_ref,
                "selected": len(selected),
            })

            with _connect(dsn, read_only=False) as write_conn:
                for row in selected:
                    last_attempted_ref = str(row["ref"])
                    ids = list(row["_canonical_ids"])
                    source_reasons = _source_validation_reasons(
                        ids, physical_by_id,
                        owner_conversation_id=args.conversation_id,
                        alias_graph=alias_graph,
                    )
                    metadata = row["_metadata"]
                    session_date = metadata.get("session_date", "")
                    if type(session_date) is not str:
                        session_date = ""
                    current_digest = _expected_source_digest(
                        ids, physical_by_id, session_date=session_date,
                    )
                    if source_reasons or current_digest != row["_expected_source_digest"]:
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "ref": row["ref"],
                            "reason": "canonical_source_changed_before_generation",
                        })
                        continue

                    segment = _reconstruct_segment(row, physical_by_id)
                    roster = _build_migration_actor_roster(
                        ids, physical_by_id,
                    )
                    if not getattr(roster, "complete", False):
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "ref": row["ref"],
                            "reason": "actor_roster_incomplete",
                        })
                        continue

                    envelope = build_deterministic_structured_summary(
                        roster=roster,
                        segment=segment,
                        generation_model=_DETERMINISTIC_GENERATION_MODEL,
                    )
                    rejection = _validate_envelope(
                        envelope,
                        expected_digest=str(row["_expected_source_digest"]),
                        expected_model=_DETERMINISTIC_GENERATION_MODEL,
                        canonical_ids=ids,
                        rows_by_id=physical_by_id,
                        session_date=session_date,
                    )
                    if rejection is not None:
                        counts["rejected"] += 1
                        rejected[rejection] += 1
                        cursor.on_decided(str(row["ref"]))
                        _journal_append(journal, {
                            "event": "rejected",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "ref": row["ref"],
                            "reason": rejection,
                        })
                        continue

                    outcome = _cas_persist(
                        write_conn, args,
                        initial_scope=scope,
                        row=row,
                        envelope=envelope,
                        journal=journal,
                        run_id=run_id,
                    )
                    if outcome == "accepted":
                        counts["accepted"] += 1
                        cursor.on_decided(str(row["ref"]))
                        affected_tags.update(segment.tags)
                        _journal_append(journal, {
                            "event": "committed",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "ref": row["ref"],
                            "source_digest": envelope.source_digest,
                            "old_retrieval_synopsis_md5": str(
                                row.get("old_retrieval_synopsis_md5") or "",
                            ),
                            "new_retrieval_synopsis_md5": str(
                                row.get("old_retrieval_synopsis_md5") or "",
                            ),
                            "retrieval_synopsis_preserved": True,
                            "structured_summary_sha256": hashlib.sha256(
                                json.dumps(
                                    structured_summary_to_dict(envelope),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8"),
                            ).hexdigest(),
                        })
                    else:
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "ref": row["ref"],
                            "reason": outcome,
                        })
    except Exception as exc:  # noqa: BLE001
        _fail("apply", str(exc), args)

    print(json.dumps({
        "status": status,
        "phase": "segments",
        "conversation_id": args.conversation_id,
        "tenant_id": args.tenant_id,
        "lifecycle_epoch": scope.lifecycle_epoch,
        "lifecycle_generation": scope.lifecycle_generation,
        "selected": len(selected),
        "counts": dict(counts),
        "rejected": dict(sorted(rejected.items())),
        "usage": usage,
        "migration_mode": _MIGRATION_MODE,
        "provider_calls": 0,
        "journal": journal_path,
        "last_attempted_ref": last_attempted_ref,
        "resume_after_ref": cursor.ref,
        "resume_cursor_frozen": cursor.frozen,
        "affected_tag_count": len(affected_tags),
        "tag_phase": "runs_after_segments_when_phase_all",
        "note": (
            "If the cursor froze, resume with --after-ref equal to the reported "
            "resume_after_ref. Finish with one fresh run without --after-ref; "
            "current digest-matching rows are skipped without a model call."
        ),
    }, indent=2))


def _apply_tags(args) -> None:
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)", args)
    try:
        with _connect(dsn, read_only=True) as conn:
            scope, segment_inventory, _aliases = _full_segment_inventory(conn, args)
            if segment_inventory.candidates:
                raise MigrationError(
                    f"{len(segment_inventory.candidates)} eligible source segment(s) "
                    "are still legacy, empty, or stale; finish segment phase first"
                )
            tag_inventory = _tag_inventory(conn, args, segment_inventory)
    except Exception as exc:  # noqa: BLE001
        _fail("tag_apply_preflight", str(exc), args)

    selected = tag_inventory.candidates
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        print(json.dumps({
            "status": "completed",
            "phase": "tags",
            "conversation_id": args.conversation_id,
            "selected": 0,
            "accepted": 0,
            "cache_action_required": False,
            "migration_mode": _MIGRATION_MODE,
            "provider_calls": 0,
            "usage": {"calls": 0, "tokens_in": 0, "tokens_out": 0},
        }, indent=2))
        return

    journal_path = _journal_path(args)
    run_id = str(uuid.uuid4())
    counts = Counter({
        "accepted": 0,
        "skipped_concurrent": 0,
        "rejected": 0,
    })
    rejected: Counter[str] = Counter()
    usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    cursor = _ResumeCursor(args.after_tag)
    status = "completed"
    last_attempted_tag = None

    try:
        with open(journal_path, "a", encoding="utf-8") as journal:
            _fsync_parent_dir(journal_path)
            _journal_append(journal, {
                "event": "tag_run_started",
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "tenant_id": args.tenant_id,
                "conversation_id": args.conversation_id,
                "lifecycle_epoch": scope.lifecycle_epoch,
                "lifecycle_generation": scope.lifecycle_generation,
                "after_tag": args.after_tag,
                "selected": len(selected),
                "migration_mode": _MIGRATION_MODE,
                "provider_calls": 0,
            })
            with _connect(dsn, read_only=False) as write_conn:
                for candidate in selected:
                    tag = str(candidate["tag"])
                    last_attempted_tag = tag
                    source_refs = list(candidate["_source_refs"])
                    loaded = _load_summary_rows_by_refs(
                        write_conn, args.conversation_id, source_refs,
                    )
                    loaded_by_ref = {str(row["ref"]): row for row in loaded}
                    expected_rows = list(candidate["_source_rows"])
                    if (
                        set(loaded_by_ref) != set(source_refs)
                        or any(
                            str(loaded_by_ref[str(row["ref"])]["row_version"])
                            != str(row["row_version"])
                            for row in expected_rows
                        )
                    ):
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "tag_skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "tag": tag,
                            "reason": "source_segment_changed_before_generation",
                        })
                        continue
                    ordered_rows = list(_ordered_tag_sources(loaded))
                    expected = _deterministic_tag_envelope(
                        ordered_rows,
                        generation_model=_DETERMINISTIC_GENERATION_MODEL,
                    )
                    candidate_expected = candidate["_expected_structured"]
                    if (
                        expected.source_digest != candidate_expected.source_digest
                        or expected.claims != candidate_expected.claims
                    ):
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        continue
                    summaries = [_stored_summary_from_row(row) for row in ordered_rows]
                    physical_ids = list(dict.fromkeys(
                        canonical_id
                        for summary in summaries
                        for canonical_id in summary.metadata.canonical_turn_ids
                    ))
                    physical_rows = _load_source_rows(
                        write_conn, args.conversation_id, physical_ids, lock=False,
                    )
                    physical_by_id = {
                        str(row["canonical_turn_id"]): row
                        for row in physical_rows
                    }
                    turn_numbers, canonical_ids, max_turn = _tag_source_coordinates(
                        summaries, physical_by_id,
                    )
                    validated_tag_rollup_inputs = validate_tag_rollup_inputs(
                        summaries,
                        physical_by_id,
                        conversation_id=args.conversation_id,
                    )
                    if validated_tag_rollup_inputs.segment_refs != set(source_refs):
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "tag_skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "tag": tag,
                            "reason": "source_segment_failed_physical_validation",
                        })
                        continue
                    structured = _deterministic_tag_selection(
                        expected, canonical_ids,
                    )
                    result = TagSummary(
                        tag=tag,
                        source_segment_refs=[summary.ref for summary in summaries],
                        source_turn_numbers=turn_numbers,
                        source_canonical_turn_ids=canonical_ids,
                        covers_through_turn=max_turn,
                        covers_through_canonical_turn_id=(
                            canonical_ids[-1] if canonical_ids else ""
                        ),
                        structured_summary=structured,
                    )
                    rejection = _tag_claim_subset_reason(
                        structured.claims, expected.claims,
                    )
                    if rejection is None and (
                        not structured.claims
                        or structured.source_digest
                        != _tag_claim_selection_digest(
                            structured.claims, canonical_ids,
                        )
                    ):
                        rejection = "tag_structured_source_changed"
                    if rejection is not None:
                        counts["rejected"] += 1
                        rejected[rejection] += 1
                        cursor.on_decided(tag)
                        _journal_append(journal, {
                            "event": "tag_rejected",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "tag": tag,
                            "reason": rejection,
                        })
                        continue

                    outcome = _cas_tag_persist(
                        write_conn, args,
                        initial_scope=scope,
                        candidate=candidate,
                        result=result,
                        journal=journal,
                        run_id=run_id,
                    )
                    if outcome == "accepted":
                        counts["accepted"] += 1
                        cursor.on_decided(tag)
                        values = _tag_metadata_write_values(result)
                        old = candidate.get("_existing")
                        old = old if isinstance(old, Mapping) else {}
                        _journal_append(journal, {
                            "event": "tag_committed",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "tag": tag,
                            "migration_mode": _MIGRATION_MODE,
                            "provider_calls": 0,
                            "source_digest": result.structured_summary.source_digest,
                            "old_summary_md5": str(
                                old.get("old_summary_md5") or "",
                            ),
                            "new_summary_md5": str(
                                old.get("old_summary_md5") or "",
                            ),
                            "old_description_md5": str(
                                old.get("old_description_md5") or "",
                            ),
                            "new_description_md5": str(
                                old.get("old_description_md5") or "",
                            ),
                            "old_retrieval_artifacts_md5": str(
                                old.get("old_retrieval_artifacts_md5") or "",
                            ),
                            "new_retrieval_artifacts_md5": str(
                                old.get("old_retrieval_artifacts_md5") or "",
                            ),
                            "old_embedding_md5": str(
                                old.get("old_embedding_md5") or "",
                            ),
                            "new_embedding_md5": str(
                                old.get("old_embedding_md5") or "",
                            ),
                            "retrieval_artifacts_preserved": True,
                            "embedding_preserved": True,
                            "structured_summary_sha256": hashlib.sha256(
                                str(values["structured_json"]).encode("utf-8"),
                            ).hexdigest(),
                        })
                    else:
                        counts["skipped_concurrent"] += 1
                        cursor.freeze()
                        _journal_append(journal, {
                            "event": "tag_skipped_concurrent",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "tag": tag,
                            "reason": outcome,
                        })
    except Exception as exc:  # noqa: BLE001
        accepted_before_failure = int(counts["accepted"])
        print(json.dumps({
            "status": "error",
            "stage": "tag_apply",
            "conversation_id": args.conversation_id,
            "error": str(exc),
            "accepted_before_failure": accepted_before_failure,
            "journal": journal_path,
            "cache_action": (
                _cache_action(args.conversation_id)
                if accepted_before_failure else {"required": False}
            ),
        }))
        raise SystemExit(1) from exc

    accepted = int(counts["accepted"])
    print(json.dumps({
        "status": status,
        "phase": "tags",
        "conversation_id": args.conversation_id,
        "tenant_id": args.tenant_id,
        "lifecycle_epoch": scope.lifecycle_epoch,
        "lifecycle_generation": scope.lifecycle_generation,
        "selected": len(selected),
        "counts": dict(counts),
        "rejected": dict(sorted(rejected.items())),
        "usage": usage,
        "migration_mode": _MIGRATION_MODE,
        "provider_calls": 0,
        "journal": journal_path,
        "last_attempted_tag": last_attempted_tag,
        "resume_after_tag": cursor.ref,
        "resume_cursor_frozen": cursor.frozen,
        "cache_action": (
            _cache_action(args.conversation_id)
            if accepted else {"required": False}
        ),
        "note": (
            "After any accepted tag write, complete the reported Redis "
            "invalidation and worker recycle before considering serving "
            "verification complete. Finish with a fresh tag run without "
            "--after-tag."
        ),
    }, indent=2))


def cmd_admin_migrate_structured_summaries(args) -> None:
    phase = getattr(args, "phase", "all")
    if getattr(args, "apply", False):
        if phase in ("segments", "all"):
            _apply(args)
        if phase in ("tags", "all"):
            _apply_tags(args)
    else:
        if phase in ("segments", "all"):
            _dry_run(args)
        if phase in ("tags", "all"):
            _dry_run_tags(args)
