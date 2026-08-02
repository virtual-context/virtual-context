#!/usr/bin/env python3
"""Snapshot or atomically apply an externally attested canonical repair.

The default ``apply`` behavior is a dry-run.  A live swap additionally needs
``--apply`` and a snapshot manifest that records a verified full database dump.
The transaction locks the exact conversation lifecycle, rechecks the current
canonical snapshot hash, refuses active ingestion/compaction/merge work, clears
only rebuildable memory for that owner, replaces its canonical rows, installs
the immutable Discord source ledger, and validates the committed candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from virtual_context.core.canonical_turns import (
    HASH_VERSION,
    compute_turn_hash_from_raw,
)


OWNER = "sk:agent:vast:discord:guild:1524917037191925871"
GUILD_ID = "1524917037191925871"
PLATFORM = "discord"
AGENT_SCOPE_ID = "vast"
ACCOUNT_ID = "vast"

# These values pin the independently reviewed, frozen Men repair artifact.
# They are deliberately not inferred from the artifact itself: changing the
# source corpus or a disposition requires a new reviewed artifact and an
# intentional update of this attestation block before it can be applied.
EXPECTED_ARTIFACT_VERSION = 4
EXPECTED_SNAPSHOT_VERSION = 5
EXPECTED_CANONICAL_ROWS = 3188
EXPECTED_GROUPS = 1594
EXPECTED_SOURCE_MEMBERSHIPS = 1594
EXPECTED_ACTORS = 18
EXPECTED_OLD_GROUPS = 1894
EXPECTED_DISPOSITION = {
    "sha256": "bff6c25c08b66b1254412c7c2b47aa82efa830730c00d24cbbce648972ac12b8",
    "raw_vast_deliveries": 2204,
    "canonical_deliveries": 1961,
    "excluded_deliveries": 237,
    "quarantined_deliveries": 6,
    "canonical_groups": 1594,
}
EXPECTED_TRANSCRIPT = {
    "sha256": "2f90cfd63159ba2428cb4c0ec7de067b6d7e2785bcef4a1461a7b6ae402d4eb6",
    "rows": 11667,
}
EXPECTED_RECOVERY_BUNDLE = {
    "sha256": "ccb0ffd142caf902afe5dc1793fe3b8c1d0a711b84d9f76cca9a506b56a95cdb",
    "files": 14,
    "evidence_records": 533,
}

CANONICAL_COLUMNS = (
    "canonical_turn_id",
    "conversation_id",
    "turn_group_number",
    "sort_key",
    "turn_hash",
    "hash_version",
    "normalized_user_text",
    "normalized_assistant_text",
    "user_content",
    "assistant_content",
    "user_raw_content",
    "assistant_raw_content",
    "primary_tag",
    "tags_json",
    "session_date",
    "sender",
    "fact_signals_json",
    "code_refs_json",
    "tagged_at",
    "compacted_at",
    "first_seen_at",
    "last_seen_at",
    "source_batch_id",
    "created_at",
    "updated_at",
    "covered_ingestible_entries",
    "compaction_operation_id",
    "origin_conversation_id",
    "origin_channel_id",
    "origin_channel_label",
    "sender_actor_id",
    "source_message_id",
    "reply_target_message_id",
    "reply_subject_actor_id",
    "reply_subject_label",
    "reply_target_body",
    "reply_attribution_version",
    "audience_conversation_id",
    "audience_attribution_version",
)

SOURCE_COLUMNS = (
    "tenant_id",
    "agent_scope_id",
    "platform",
    "account_id",
    "message_id",
    "canonical_turn_id",
    "assistant_canonical_turn_id",
    "assistant_turn_hash",
    "turn_group_number",
    "pair_version",
    "audience_conversation_id",
    "channel_id",
    "guild_id",
    "author_id",
    "source_actor_id",
    "transport_body_sha256",
    "canonical_body_sha256",
    "projection_version",
    "canonical_turn_hash",
    "reply_target_message_id",
    "observed_at",
    "created_at",
)

EVIDENCE_EXPORTS = {
    "conversation": (
        "conversation.jsonl",
        "SELECT * FROM conversations WHERE conversation_id=%s",
        lambda owner, tenant_id: (owner,),
    ),
    "conversation_lifecycle": (
        "conversation_lifecycle.jsonl",
        "SELECT * FROM conversation_lifecycle WHERE conversation_id=%s",
        lambda owner, tenant_id: (owner,),
    ),
    "engine_state": (
        "engine_state.jsonl",
        "SELECT * FROM engine_state WHERE conversation_id=%s",
        lambda owner, tenant_id: (owner,),
    ),
    "canonical_message_sources": (
        "canonical_message_sources.jsonl",
        """SELECT * FROM canonical_message_sources
             WHERE tenant_id=%s AND (
                   audience_conversation_id=%s
                   OR canonical_turn_id IN (
                       SELECT canonical_turn_id FROM canonical_turns
                        WHERE conversation_id=%s
                   )
                   OR assistant_canonical_turn_id IN (
                       SELECT canonical_turn_id FROM canonical_turns
                        WHERE conversation_id=%s
                   )
             )
             ORDER BY agent_scope_id, platform, account_id, message_id""",
        lambda owner, tenant_id: (tenant_id, owner, owner, owner),
    ),
    "canonical_turn_chunks": (
        "canonical_turn_chunks.jsonl",
        """SELECT * FROM canonical_turn_chunks WHERE conversation_id=%s
             ORDER BY canonical_turn_id, side, chunk_index""",
        lambda owner, tenant_id: (owner,),
    ),
    "canonical_turn_anchors": (
        "canonical_turn_anchors.jsonl",
        """SELECT * FROM canonical_turn_anchors WHERE conversation_id=%s
             ORDER BY window_size, anchor_hash, start_turn_id""",
        lambda owner, tenant_id: (owner,),
    ),
    "segments": (
        "segments.jsonl",
        "SELECT * FROM segments WHERE conversation_id=%s ORDER BY ref",
        lambda owner, tenant_id: (owner,),
    ),
    "segment_chunks": (
        "segment_chunks.jsonl",
        """SELECT sc.* FROM segment_chunks sc
             JOIN segments s ON s.ref=sc.segment_ref
            WHERE s.conversation_id=%s
            ORDER BY sc.segment_ref, sc.chunk_index""",
        lambda owner, tenant_id: (owner,),
    ),
    "segment_tags": (
        "segment_tags.jsonl",
        """SELECT st.* FROM segment_tags st
             JOIN segments s ON s.ref=st.segment_ref
            WHERE s.conversation_id=%s
            ORDER BY st.segment_ref, st.tag""",
        lambda owner, tenant_id: (owner,),
    ),
    "facts": (
        "facts.jsonl",
        "SELECT * FROM facts WHERE conversation_id=%s ORDER BY id",
        lambda owner, tenant_id: (owner,),
    ),
    "fact_tags": (
        "fact_tags.jsonl",
        """SELECT ft.* FROM fact_tags ft
             JOIN facts f ON f.id=ft.fact_id
            WHERE f.conversation_id=%s
            ORDER BY ft.fact_id, ft.tag""",
        lambda owner, tenant_id: (owner,),
    ),
    "fact_links": (
        "fact_links.jsonl",
        """SELECT * FROM fact_links
             WHERE source_fact_id IN (
                       SELECT id FROM facts WHERE conversation_id=%s
                   )
                OR target_fact_id IN (
                       SELECT id FROM facts WHERE conversation_id=%s
                   )
             ORDER BY id""",
        lambda owner, tenant_id: (owner, owner),
    ),
    "fact_embeddings": (
        "fact_embeddings.jsonl",
        """SELECT * FROM fact_embeddings WHERE conversation_id=%s
             ORDER BY fact_id, model""",
        lambda owner, tenant_id: (owner,),
    ),
    "tag_summaries": (
        "tag_summaries.jsonl",
        """SELECT * FROM tag_summaries WHERE conversation_id=%s
             ORDER BY tag""",
        lambda owner, tenant_id: (owner,),
    ),
    "tag_summary_embeddings": (
        "tag_summary_embeddings.jsonl",
        """SELECT * FROM tag_summary_embeddings WHERE conversation_id=%s
             ORDER BY tag""",
        lambda owner, tenant_id: (owner,),
    ),
    "tool_outputs": (
        "tool_outputs.jsonl",
        "SELECT * FROM tool_outputs WHERE conversation_id=%s ORDER BY ref",
        lambda owner, tenant_id: (owner,),
    ),
    "turn_tool_outputs": (
        "turn_tool_outputs.jsonl",
        """SELECT * FROM turn_tool_outputs WHERE conversation_id=%s
             ORDER BY turn_number, tool_output_ref""",
        lambda owner, tenant_id: (owner,),
    ),
    "chain_snapshots": (
        "chain_snapshots.jsonl",
        """SELECT * FROM chain_snapshots WHERE conversation_id=%s
             ORDER BY ref""",
        lambda owner, tenant_id: (owner,),
    ),
    "segment_tool_outputs": (
        "segment_tool_outputs.jsonl",
        """SELECT * FROM segment_tool_outputs WHERE conversation_id=%s
             ORDER BY segment_ref, tool_output_ref""",
        lambda owner, tenant_id: (owner,),
    ),
    "speaker_handles": (
        "speaker_handles.jsonl",
        """SELECT * FROM speaker_handles
             WHERE tenant_id=%s AND audience_conversation_id=%s
             ORDER BY actor_id""",
        lambda owner, tenant_id: (tenant_id, owner),
    ),
    # Full tenant-scoped actor-card state is rollback evidence. A single card
    # entry can cite both Men and DM facts/turns, and canonical DELETE triggers
    # can otherwise cascade away the unrelated source links invisibly.
    "actor_profiles": (
        "actor_profiles.jsonl",
        "SELECT * FROM actor_profiles WHERE tenant_id=%s ORDER BY actor_id",
        lambda owner, tenant_id: (tenant_id,),
    ),
    "actor_card_entries": (
        "actor_card_entries.jsonl",
        """SELECT * FROM actor_card_entries WHERE tenant_id=%s
             ORDER BY actor_id, id""",
        lambda owner, tenant_id: (tenant_id,),
    ),
    "actor_card_entry_sources": (
        "actor_card_entry_sources.jsonl",
        """SELECT * FROM actor_card_entry_sources WHERE tenant_id=%s
             ORDER BY entry_id, fact_id""",
        lambda owner, tenant_id: (tenant_id,),
    ),
    "actor_card_turn_sources": (
        "actor_card_turn_sources.jsonl",
        """SELECT * FROM actor_card_turn_sources WHERE tenant_id=%s
             ORDER BY entry_id, canonical_turn_id""",
        lambda owner, tenant_id: (tenant_id,),
    ),
    "actor_card_rebuild_status": (
        "actor_card_rebuild_status.jsonl",
        """SELECT * FROM actor_card_rebuild_status WHERE tenant_id=%s
             ORDER BY actor_id""",
        lambda owner, tenant_id: (tenant_id,),
    ),
}


class RepairError(RuntimeError):
    """A precondition or postcondition failed; the transaction must abort."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (UUID, datetime, date, Decimal)):
        return str(value)
    return value


def _encode_row(row: dict[str, Any]) -> bytes:
    normalized = {key: _json_value(value) for key, value in row.items()}
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RepairError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RepairError(f"non-object JSONL {path}:{line_number}")
            rows.append(value)
    return rows


def _hash_rows(rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_encode_row(row))
        count += 1
    return digest.hexdigest(), count


def _evidence_rows(conn, owner: str, tenant_id: str) -> dict[str, list[dict[str, Any]]]:
    return {
        name: list(conn.execute(sql, params(owner, tenant_id)).fetchall())
        for name, (_filename, sql, params) in EVIDENCE_EXPORTS.items()
    }


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            encoded = _encode_row(row)
            handle.write(encoded)
            digest.update(encoded)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count, digest.hexdigest()


def _write_text_durable(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_backup_file(path_value: object, sha_value: object) -> dict[str, Any]:
    """Prove that the rollback artifact named by a manifest exists now.

    A path and a caller-supplied digest are not backup attestation.  The
    repair process must be able to read a non-empty regular file and must
    independently reproduce the recorded digest before it snapshots or
    deletes any production data.
    """
    path_text = str(path_value or "").strip()
    expected_sha = str(sha_value or "").strip()
    if not path_text or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RepairError("database backup path or SHA-256 is invalid")
    try:
        path = Path(path_text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepairError(f"database backup is not readable: {path_text}") from exc
    if not path.is_file():
        raise RepairError(f"database backup is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RepairError(f"database backup is empty: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise RepairError(
            "database backup checksum mismatch: "
            f"expected={expected_sha} actual={actual_sha}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": size,
    }


def _postgres_tool_env(dsn: str) -> dict[str, str]:
    """Translate a libpq DSN into process env without exposing passwords."""
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError as exc:
        raise RepairError("psycopg is required") from exc
    info = conninfo_to_dict(dsn)
    if not str(info.get("dbname") or "").strip():
        raise RepairError("database DSN has no database name")
    env = dict(os.environ)
    mapping = {
        "host": "PGHOST",
        "hostaddr": "PGHOSTADDR",
        "port": "PGPORT",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "dbname": "PGDATABASE",
        "sslmode": "PGSSLMODE",
        "sslrootcert": "PGSSLROOTCERT",
        "sslcert": "PGSSLCERT",
        "sslkey": "PGSSLKEY",
        "passfile": "PGPASSFILE",
    }
    for pg_env in mapping.values():
        env.pop(pg_env, None)
    for key, pg_env in mapping.items():
        value = str(info.get(key) or "").strip()
        if value:
            env[pg_env] = value
    return env


def _run_postgres_tool(
    executable: str,
    args: list[str],
    *,
    dsn: str,
) -> None:
    path = shutil.which(executable)
    if path is None:
        raise RepairError(f"{executable} is required but was not found")
    completed = subprocess.run(
        [path, *args],
        env=_postgres_tool_env(dsn),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise RepairError(
            f"{executable} failed with exit={completed.returncode}: {detail}"
        )


def _create_exported_snapshot_dump(
    *,
    dsn: str,
    exported_snapshot: str,
    backup_path: Path,
) -> dict[str, Any]:
    """Create a custom dump of the exact still-open exported snapshot."""
    resolved = backup_path.expanduser().resolve(strict=False)
    if resolved.exists():
        raise RepairError(f"refusing to overwrite database backup: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _run_postgres_tool(
        "pg_dump",
        [
            "--format=custom",
            f"--snapshot={exported_snapshot}",
            f"--file={resolved}",
        ],
        dsn=dsn,
    )
    with resolved.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(resolved.parent)
    digest = _sha256_file(resolved)
    verified = _verify_backup_file(resolved, digest)
    return {
        **verified,
        "format": "postgres-custom",
        "exported_snapshot": exported_snapshot,
    }


def _connect(dsn_env: str):
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RepairError("psycopg is required") from exc
    dsn = os.environ.get(dsn_env, "").strip()
    if not dsn:
        raise RepairError(f"environment variable {dsn_env} is empty")
    return psycopg.connect(dsn, row_factory=dict_row)


def _current_canonical_rows(conn, owner: str) -> list[dict[str, Any]]:
    columns = ", ".join(CANONICAL_COLUMNS)
    return list(conn.execute(
        f"SELECT {columns} FROM canonical_turns "
        "WHERE conversation_id = %s ORDER BY sort_key, canonical_turn_id",
        (owner,),
    ).fetchall())


def _conversation_row(conn, owner: str) -> dict[str, Any]:
    row = conn.execute(
        """SELECT tenant_id, lifecycle_epoch, phase, deleted_at,
                  pending_raw_payload_entries
             FROM conversations WHERE conversation_id = %s""",
        (owner,),
    ).fetchone()
    if row is None:
        raise RepairError("conversation does not exist")
    return dict(row)


def _relation_columns(conn, relation: str) -> set[str]:
    return {
        str(row["column_name"])
        for row in conn.execute(
            """SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s""",
            (relation,),
        ).fetchall()
    }


def _assert_schema(conn) -> None:
    canonical = _relation_columns(conn, "canonical_turns")
    sources = _relation_columns(conn, "canonical_message_sources")
    missing_canonical = set(CANONICAL_COLUMNS) - canonical
    missing_sources = set(SOURCE_COLUMNS) - sources
    if missing_canonical or missing_sources:
        raise RepairError(
            "repair schema incomplete: canonical="
            f"{sorted(missing_canonical)} sources={sorted(missing_sources)}"
        )
    trigger = conn.execute(
        """SELECT 1 FROM pg_trigger
            WHERE tgrelid = 'canonical_turns'::regclass
              AND tgname = 'trg_guard_attested_canonical_turn_update'
              AND NOT tgisinternal"""
    ).fetchone()
    if trigger is None:
        raise RepairError("source immutability trigger is missing")


def _derived_counts(conn, owner: str, tenant_id: str) -> dict[str, int]:
    queries = {
        "canonical_turns": (
            "SELECT COUNT(*) AS n FROM canonical_turns WHERE conversation_id=%s",
            (owner,),
        ),
        "canonical_turn_chunks": (
            "SELECT COUNT(*) AS n FROM canonical_turn_chunks WHERE conversation_id=%s",
            (owner,),
        ),
        "canonical_turn_anchors": (
            "SELECT COUNT(*) AS n FROM canonical_turn_anchors WHERE conversation_id=%s",
            (owner,),
        ),
        "segments": (
            "SELECT COUNT(*) AS n FROM segments WHERE conversation_id=%s",
            (owner,),
        ),
        "facts": (
            "SELECT COUNT(*) AS n FROM facts WHERE conversation_id=%s",
            (owner,),
        ),
        "tag_summaries": (
            "SELECT COUNT(*) AS n FROM tag_summaries WHERE conversation_id=%s",
            (owner,),
        ),
        "speaker_handles": (
            """SELECT COUNT(*) AS n FROM speaker_handles
                WHERE tenant_id=%s AND audience_conversation_id=%s""",
            (tenant_id, owner),
        ),
        "card_fact_sources": (
            """SELECT COUNT(*) AS n
                 FROM actor_card_entry_sources s
                 JOIN facts f ON f.id=s.fact_id
                WHERE s.tenant_id=%s AND f.conversation_id=%s""",
            (tenant_id, owner),
        ),
        "card_turn_sources": (
            """SELECT COUNT(*) AS n
                 FROM actor_card_turn_sources s
                 JOIN canonical_turns ct
                   ON ct.canonical_turn_id=s.canonical_turn_id
                WHERE s.tenant_id=%s AND ct.conversation_id=%s""",
            (tenant_id, owner),
        ),
    }
    return {
        name: int(conn.execute(sql, params).fetchone()["n"])
        for name, (sql, params) in queries.items()
    }


def snapshot(
    *,
    dsn_env: str,
    owner: str,
    tenant_id: str,
    lifecycle_epoch: int,
    output_dir: Path,
    backup_path: str,
    maintenance_freeze_confirmed: bool,
) -> dict[str, Any]:
    if not maintenance_freeze_confirmed:
        raise RepairError(
            "snapshot requires a confirmed gateway, compaction, and actor-card "
            "worker maintenance freeze"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    dsn = os.environ.get(dsn_env, "").strip()
    if not dsn:
        raise RepairError(f"environment variable {dsn_env} is empty")
    with _connect(dsn_env) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        database_snapshot = conn.execute(
            "SELECT pg_export_snapshot()::text AS exported_snapshot, "
            "txid_current_snapshot()::text AS tx_snapshot, "
            "pg_current_wal_lsn()::text AS wal_lsn, "
            "clock_timestamp()::text AS observed_at, "
            "current_database()::text AS database_name"
        ).fetchone()
        verified_backup = _create_exported_snapshot_dump(
            dsn=dsn,
            exported_snapshot=str(database_snapshot["exported_snapshot"]),
            backup_path=Path(backup_path),
        )
        _assert_schema(conn)
        conversation = _conversation_row(conn, owner)
        if (
            conversation["tenant_id"] != tenant_id
            or int(conversation["lifecycle_epoch"]) != lifecycle_epoch
            or conversation["phase"] != "active"
            or conversation["deleted_at"] is not None
        ):
            raise RepairError("conversation tenant/lifecycle/phase mismatch")
        if int(conversation["pending_raw_payload_entries"] or 0) != 0:
            raise RepairError("conversation still has pending raw payload entries")
        _assert_quiescent(conn, owner, tenant_id, lifecycle_epoch)
        rows = _current_canonical_rows(conn, owner)
        canonical_path = output_dir / "canonical_rows.jsonl"
        canonical_count, canonical_sha = _write_rows(canonical_path, rows)
        evidence_files: dict[str, dict[str, Any]] = {}
        evidence = _evidence_rows(conn, owner, tenant_id)
        for name, rows_for_table in evidence.items():
            filename = EVIDENCE_EXPORTS[name][0]
            count, digest = _write_rows(output_dir / filename, rows_for_table)
            evidence_files[name] = {
                "filename": filename,
                "rows": count,
                "sha256": digest,
            }
        manifest = {
            "version": EXPECTED_SNAPSHOT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner_conversation_id": owner,
            "tenant_id": tenant_id,
            "lifecycle_epoch": lifecycle_epoch,
            "canonical_rows": canonical_count,
            "canonical_rows_sha256": canonical_sha,
            "database_snapshot": dict(database_snapshot),
            "derived_counts": _derived_counts(conn, owner, tenant_id),
            "evidence_files": evidence_files,
            "database_backup": verified_backup,
            "maintenance_freeze_confirmed": True,
        }
    manifest_path = output_dir / "manifest.json"
    _write_text_durable(
        manifest_path,
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
    )
    checksum_lines = [f"{canonical_sha}  canonical_rows.jsonl"]
    checksum_lines.extend(
        f"{meta['sha256']}  {meta['filename']}"
        for _name, meta in sorted(evidence_files.items())
    )
    checksum_lines.append(f"{_sha256_file(manifest_path)}  manifest.json")
    _write_text_durable(
        output_dir / "SHA256SUMS",
        "\n".join(checksum_lines) + "\n",
    )
    _fsync_directory(output_dir)
    return manifest


def _load_artifact(
    artifact_dir: Path,
    *,
    tenant_id: str,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, int | None],
]:
    manifest_path = artifact_dir / "manifest.json"
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or ""):
        raise RepairError("artifact manifest checksum is not a SHA-256 digest")
    observed_manifest_sha256 = _sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise RepairError(
            "artifact manifest does not match the independently reviewed checksum"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != EXPECTED_ARTIFACT_VERSION
        or
        manifest.get("owner_conversation_id") != OWNER
        or manifest.get("guild_id") != GUILD_ID
        or manifest.get("agent_scope_id") != AGENT_SCOPE_ID
        or manifest.get("account_id") != ACCOUNT_ID
    ):
        raise RepairError("artifact identity does not match the Men guild repair")
    expected_files = {
        "canonical_rows.jsonl",
        "group_evidence.jsonl",
        "old_group_to_new_group.jsonl",
        "old_to_new.jsonl",
        "source_memberships.jsonl",
    }
    if set(manifest.get("files", {})) != expected_files:
        raise RepairError("artifact file inventory is not the reviewed inventory")
    for filename, meta in manifest.get("files", {}).items():
        path = artifact_dir / filename
        if not path.is_file() or _sha256_file(path) != meta.get("sha256"):
            raise RepairError(f"artifact checksum mismatch: {filename}")
    rows = _read_jsonl(artifact_dir / "canonical_rows.jsonl")
    sources = _read_jsonl(artifact_dir / "source_memberships.jsonl")
    old_group_rows = _read_jsonl(
        artifact_dir / "old_group_to_new_group.jsonl",
    )
    stats = manifest.get("stats") or {}
    disposition = manifest.get("all_delivery_disposition") or {}
    transcript = (manifest.get("inputs") or {}).get("discord_transcript") or {}
    recovery = (manifest.get("inputs") or {}).get("recovery_session_bundle") or {}
    if any(disposition.get(key) != value for key, value in EXPECTED_DISPOSITION.items()):
        raise RepairError("artifact delivery disposition is not the reviewed disposition")
    if any(transcript.get(key) != value for key, value in EXPECTED_TRANSCRIPT.items()):
        raise RepairError("artifact Discord transcript is not the reviewed transcript")
    if (
        recovery.get("sha256") != EXPECTED_RECOVERY_BUNDLE["sha256"]
        or len(recovery.get("files") or []) != EXPECTED_RECOVERY_BUNDLE["files"]
        or recovery.get("evidence_records")
        != EXPECTED_RECOVERY_BUNDLE["evidence_records"]
    ):
        raise RepairError("artifact recovery bundle is not the reviewed bundle")
    if (
        stats.get("canonical_rows") != EXPECTED_CANONICAL_ROWS
        or stats.get("groups") != EXPECTED_GROUPS
        or stats.get("source_memberships") != EXPECTED_SOURCE_MEMBERSHIPS
    ):
        raise RepairError("artifact cardinalities are not the reviewed cardinalities")
    if len(rows) != int(stats["canonical_rows"]):
        raise RepairError("artifact canonical count mismatch")
    if len(sources) != int(manifest["stats"]["source_memberships"]):
        raise RepairError("artifact source count mismatch")

    known_ids: set[str] = set()
    known_source_ids: set[str] = set()
    users_by_id: dict[str, dict[str, Any]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sort_keys: set[float] = set()
    for row in rows:
        missing = set(CANONICAL_COLUMNS) - row.keys()
        if missing:
            raise RepairError(f"canonical artifact row missing {sorted(missing)}")
        canonical_id = str(row["canonical_turn_id"])
        if canonical_id in known_ids:
            raise RepairError("duplicate canonical id in artifact")
        known_ids.add(canonical_id)
        rows_by_id[canonical_id] = row
        if row["conversation_id"] != OWNER:
            raise RepairError("artifact row belongs to another conversation")
        sort_key = float(row["sort_key"])
        if sort_key in sort_keys:
            raise RepairError("duplicate sort key in artifact")
        sort_keys.add(sort_key)
        groups[int(row["turn_group_number"])].append(row)
        expected_hash, expected_user, expected_assistant = compute_turn_hash_from_raw(
            str(row["user_content"] or ""),
            str(row["assistant_content"] or ""),
            version=HASH_VERSION,
        )
        if (
            row["turn_hash"] != expected_hash
            or row["normalized_user_text"] != expected_user
            or row["normalized_assistant_text"] != expected_assistant
        ):
            raise RepairError(f"canonical hash mismatch for {canonical_id}")
        if row["user_content"]:
            source_id = str(row["source_message_id"] or "")
            if (
                not source_id
                or source_id in known_source_ids
                or not str(row["sender_actor_id"] or "").startswith("actor:discord:")
            ):
                raise RepairError("user row lacks unique source identity")
            known_source_ids.add(source_id)
            users_by_id[canonical_id] = row
        elif not row["assistant_content"]:
            raise RepairError("canonical row has no role-local content")
    if any(
        len(group_rows) != 2
        or sum(bool(row["user_content"]) for row in group_rows) != 1
        or sum(bool(row["assistant_content"]) for row in group_rows) != 1
        for group_rows in groups.values()
    ):
        raise RepairError("artifact group is not exactly one user plus one assistant")

    source_keys: set[tuple[str, ...]] = set()
    source_turn_ids: set[str] = set()
    for source in sources:
        missing = set(SOURCE_COLUMNS) - source.keys()
        if missing:
            raise RepairError(f"source artifact row missing {sorted(missing)}")
        source["tenant_id"] = tenant_id
        source_key = tuple(str(source[key]) for key in (
            "tenant_id", "agent_scope_id", "platform", "account_id", "message_id",
        ))
        canonical_id = str(source["canonical_turn_id"])
        user = users_by_id.get(canonical_id)
        assistant_id = str(source["assistant_canonical_turn_id"] or "")
        assistant = rows_by_id.get(assistant_id)
        if (
            source_key in source_keys
            or canonical_id in source_turn_ids
            or user is None
            or assistant is None
            or assistant_id == canonical_id
            or str(assistant["conversation_id"]) != OWNER
            or bool(str(assistant["user_content"] or "").strip())
            or not str(assistant["assistant_content"] or "").strip()
            or int(assistant["turn_group_number"])
               != int(user["turn_group_number"])
            or int(source["turn_group_number"]) != -1
            or int(source["pair_version"] or 0) != 1
            or source["assistant_turn_hash"] != assistant["turn_hash"]
            or source["agent_scope_id"] != AGENT_SCOPE_ID
            or source["platform"] != PLATFORM
            or source["account_id"] != ACCOUNT_ID
            or source["guild_id"] != GUILD_ID
            or source["message_id"] != user["source_message_id"]
            or source["source_actor_id"] != user["sender_actor_id"]
            or source["channel_id"] != user["origin_channel_id"]
            or source["audience_conversation_id"] != user["audience_conversation_id"]
            or source["canonical_turn_hash"] != user["turn_hash"]
            or source["canonical_body_sha256"]
               != hashlib.sha256(user["user_content"].encode("utf-8")).hexdigest()
        ):
            raise RepairError("source membership does not exactly fence its user row")
        source_keys.add(source_key)
        source_turn_ids.add(canonical_id)
    if source_turn_ids != set(users_by_id):
        raise RepairError("not every user row has exactly one source membership")
    known_new_groups = set(groups)
    old_group_map: dict[int, int | None] = {}
    for mapping in old_group_rows:
        if "old_turn_group_number" not in mapping:
            raise RepairError("old-group mapping lacks its source ordinal")
        old_group = int(mapping["old_turn_group_number"])
        if old_group in old_group_map:
            raise RepairError("duplicate old turn group in artifact mapping")
        raw_new_group = mapping.get("new_turn_group_number")
        new_group = int(raw_new_group) if raw_new_group is not None else None
        if new_group is not None and new_group not in known_new_groups:
            raise RepairError("old-group mapping targets a missing candidate group")
        classification = str(mapping.get("classification") or "")
        if (new_group is None) != classification.startswith("quarantined_"):
            raise RepairError("old-group mapping disposition contradicts its target")
        old_group_map[old_group] = new_group
    if len(old_group_map) != len(old_group_rows):
        raise RepairError("old-group mapping anti-vacuity check failed")
    actor_ids = {
        str(row["sender_actor_id"])
        for row in rows
        if str(row.get("sender_actor_id") or "")
    }
    if (
        len(groups) != EXPECTED_GROUPS
        or len(sources) != EXPECTED_SOURCE_MEMBERSHIPS
        or len(actor_ids) != EXPECTED_ACTORS
        or len(old_group_map) != EXPECTED_OLD_GROUPS
    ):
        raise RepairError("artifact anti-vacuity cardinalities failed")
    return manifest, rows, sources, old_group_map


def _load_snapshot_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    backup = manifest.get("database_backup") or {}
    if (
        manifest.get("version") != EXPECTED_SNAPSHOT_VERSION
        or
        manifest.get("owner_conversation_id") != OWNER
        or not manifest.get("tenant_id")
        or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("canonical_rows_sha256") or ""))
        or not str(backup.get("path") or "").strip()
        or not re.fullmatch(r"[0-9a-f]{64}", str(backup.get("sha256") or ""))
        or not isinstance(manifest.get("canonical_rows"), int)
        or not isinstance(backup.get("bytes"), int)
        or int(backup.get("bytes") or 0) <= 0
        or backup.get("format") != "postgres-custom"
        or manifest.get("maintenance_freeze_confirmed") is not True
    ):
        raise RepairError("snapshot manifest is incomplete or not backup-attested")
    database_snapshot = manifest.get("database_snapshot") or {}
    if not all(str(database_snapshot.get(key) or "").strip() for key in (
        "exported_snapshot", "tx_snapshot", "wal_lsn", "observed_at",
        "database_name",
    )):
        raise RepairError("snapshot lacks a repeatable-read database high-water")
    if (
        str(backup.get("exported_snapshot") or "")
        != str(database_snapshot["exported_snapshot"])
    ):
        raise RepairError("database dump is not bound to the exported snapshot")
    canonical_path = path.parent / "canonical_rows.jsonl"
    if (
        not canonical_path.is_file()
        or _sha256_file(canonical_path) != manifest["canonical_rows_sha256"]
    ):
        raise RepairError("snapshot canonical file does not match its manifest")
    verified_backup = _verify_backup_file(backup["path"], backup["sha256"])
    if verified_backup["bytes"] != int(backup["bytes"]):
        raise RepairError("database backup size changed after snapshot")
    evidence_files = manifest.get("evidence_files")
    if not isinstance(evidence_files, dict) or set(evidence_files) != set(EVIDENCE_EXPORTS):
        raise RepairError("snapshot lacks the complete turn-evidence export set")
    evidence_rows: dict[str, list[dict[str, Any]]] = {}
    for name, (expected_filename, _sql, _params) in EVIDENCE_EXPORTS.items():
        meta = evidence_files.get(name)
        if (
            not isinstance(meta, dict)
            or meta.get("filename") != expected_filename
            or not isinstance(meta.get("rows"), int)
            or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256") or ""))
        ):
            raise RepairError(f"snapshot evidence metadata is invalid: {name}")
        evidence_path = path.parent / expected_filename
        if not evidence_path.is_file() or _sha256_file(evidence_path) != meta["sha256"]:
            raise RepairError(f"snapshot evidence checksum mismatch: {name}")
        rows = _read_jsonl(evidence_path)
        if len(rows) != int(meta["rows"]):
            raise RepairError(f"snapshot evidence count mismatch: {name}")
        evidence_rows[name] = rows
    manifest["database_backup"] = {**backup, **verified_backup}
    manifest["_evidence_rows"] = evidence_rows
    return manifest


def verify_restore(
    *,
    snapshot_manifest_path: Path,
    restore_dsn_env: str,
    proof_path: Path,
) -> dict[str, Any]:
    """Restore the bound dump into an empty disposable DB and hash-check it."""
    snapshot_manifest = _load_snapshot_manifest(snapshot_manifest_path)
    target_dsn = os.environ.get(restore_dsn_env, "").strip()
    if not target_dsn:
        raise RepairError(
            f"environment variable {restore_dsn_env} is empty"
        )
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError as exc:
        raise RepairError("psycopg is required") from exc
    target_name = str(
        conninfo_to_dict(target_dsn).get("dbname") or ""
    ).strip()
    source_name = str(
        (snapshot_manifest.get("database_snapshot") or {}).get(
            "database_name"
        ) or ""
    ).strip()
    if (
        not target_name.startswith(("vc_restore_", "vc_disposable_"))
        or target_name == source_name
    ):
        raise RepairError(
            "restore proof requires a different vc_restore_* or "
            "vc_disposable_* database"
        )
    with _connect(restore_dsn_env) as target:
        existing = int(target.execute(
            """SELECT COUNT(*) AS n
                 FROM information_schema.tables
                WHERE table_schema NOT IN
                      ('pg_catalog', 'information_schema')"""
        ).fetchone()["n"])
        if existing:
            raise RepairError(
                "restore target is not empty; refusing destructive restore"
            )
    backup = snapshot_manifest["database_backup"]
    _run_postgres_tool(
        "pg_restore",
        [
            "--exit-on-error",
            "--no-owner",
            "--no-privileges",
            f"--dbname={target_name}",
            backup["path"],
        ],
        dsn=target_dsn,
    )
    with _connect(restore_dsn_env) as restored:
        restored_rows = _current_canonical_rows(restored, OWNER)
        canonical_sha, canonical_count = _hash_rows(restored_rows)
        if (
            canonical_sha != snapshot_manifest["canonical_rows_sha256"]
            or canonical_count != int(snapshot_manifest["canonical_rows"])
        ):
            raise RepairError(
                "restored canonical preimage differs from snapshot"
            )
        restored_evidence = _evidence_rows(
            restored, OWNER, snapshot_manifest["tenant_id"],
        )
        evidence_proof: dict[str, dict[str, Any]] = {}
        for name, rows_for_table in restored_evidence.items():
            digest, count = _hash_rows(rows_for_table)
            expected = snapshot_manifest["evidence_files"][name]
            if (
                digest != expected["sha256"]
                or count != int(expected["rows"])
            ):
                raise RepairError(
                    f"restored {name} differs from snapshot"
                )
            evidence_proof[name] = {
                "rows": count,
                "sha256": digest,
            }
        derived = _derived_counts(
            restored, OWNER, snapshot_manifest["tenant_id"],
        )
        if derived != snapshot_manifest["derived_counts"]:
            raise RepairError("restored derived counts differ from snapshot")
    resolved_proof = proof_path.expanduser().resolve(strict=False)
    if resolved_proof.exists():
        raise RepairError(f"refusing to overwrite restore proof: {resolved_proof}")
    proof = {
        "version": 1,
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_manifest_path": str(snapshot_manifest_path.resolve()),
        "snapshot_manifest_sha256": _sha256_file(snapshot_manifest_path),
        "backup_sha256": backup["sha256"],
        "backup_bytes": backup["bytes"],
        "backup_exported_snapshot": backup["exported_snapshot"],
        "restore_database": target_name,
        "canonical_rows": canonical_count,
        "canonical_rows_sha256": canonical_sha,
        "evidence_files": evidence_proof,
        "derived_counts": derived,
    }
    resolved_proof.parent.mkdir(parents=True, exist_ok=True)
    _write_text_durable(
        resolved_proof,
        json.dumps(proof, sort_keys=True, indent=2) + "\n",
    )
    _fsync_directory(resolved_proof.parent)
    return proof


def _load_restore_proof(
    path: Path,
    *,
    snapshot_manifest_path: Path,
    snapshot_manifest: dict[str, Any],
) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    backup = snapshot_manifest["database_backup"]
    if (
        proof.get("version") != 1
        or proof.get("status") != "verified"
        or proof.get("snapshot_manifest_sha256")
           != _sha256_file(snapshot_manifest_path)
        or proof.get("backup_sha256") != backup["sha256"]
        or type(proof.get("backup_bytes")) is not int
        or proof["backup_bytes"] != int(backup["bytes"])
        or proof.get("backup_exported_snapshot")
           != backup["exported_snapshot"]
        or type(proof.get("canonical_rows")) is not int
        or proof["canonical_rows"] != int(snapshot_manifest["canonical_rows"])
        or proof.get("canonical_rows_sha256")
           != snapshot_manifest["canonical_rows_sha256"]
        or not str(proof.get("restore_database") or "").startswith(
            ("vc_restore_", "vc_disposable_")
        )
    ):
        raise RepairError("restore proof is missing or not snapshot-bound")
    expected_evidence = snapshot_manifest["evidence_files"]
    restored_evidence = proof.get("evidence_files") or {}
    if set(restored_evidence) != set(expected_evidence):
        raise RepairError("restore proof lacks the complete evidence set")
    for name, expected in expected_evidence.items():
        actual = restored_evidence[name]
        if (
            type(actual.get("rows")) is not int
            or actual["rows"] != int(expected["rows"])
            or actual.get("sha256") != expected["sha256"]
        ):
            raise RepairError(f"restore proof mismatch: {name}")
    if proof.get("derived_counts") != snapshot_manifest["derived_counts"]:
        raise RepairError("restore proof derived counts mismatch")
    return proof


def _assert_quiescent(conn, owner: str, tenant_id: str, epoch: int) -> None:
    active_compaction = conn.execute(
        """SELECT COUNT(*) AS n FROM compaction_operation
            WHERE conversation_id=%s AND lifecycle_epoch=%s
              AND status IN ('queued','running')""",
        (owner, epoch),
    ).fetchone()["n"]
    active_ingest = conn.execute(
        """SELECT COUNT(*) AS n FROM ingestion_episode
            WHERE conversation_id=%s AND lifecycle_epoch=%s AND status='running'""",
        (owner, epoch),
    ).fetchone()["n"]
    active_merge = conn.execute(
        """SELECT COUNT(*) AS n FROM merge_audit
            WHERE tenant_id=%s AND status='in_progress'
              AND (target_conversation_id=%s OR source_conversation_id=%s)""",
        (tenant_id, owner, owner),
    ).fetchone()["n"]
    if active_compaction or active_ingest or active_merge:
        raise RepairError(
            "conversation is not quiescent: "
            f"compaction={active_compaction} ingest={active_ingest} merge={active_merge}"
        )


def _invalidate_actor_cards(
    conn,
    owner: str,
    tenant_id: str,
    candidate_actor_ids: set[str],
) -> tuple[dict[str, int], set[str]]:
    cross_tenant = conn.execute(
        """SELECT
             (SELECT COUNT(*)
                FROM actor_card_turn_sources s
                JOIN canonical_turns ct
                  ON ct.canonical_turn_id=s.canonical_turn_id
               WHERE ct.conversation_id=%s AND s.tenant_id<>%s) AS turns,
             (SELECT COUNT(*)
                FROM actor_card_entry_sources s
                JOIN facts f ON f.id=s.fact_id
               WHERE f.conversation_id=%s AND s.tenant_id<>%s) AS facts""",
        (owner, tenant_id, owner, tenant_id),
    ).fetchone()
    if int(cross_tenant["turns"]) or int(cross_tenant["facts"]):
        # The canonical DELETE trigger is intentionally global by source id.
        # A cross-tenant citation would therefore mutate data outside this
        # repair's reviewed tenant snapshot. Refuse instead of silently
        # detaching or cascading another tenant's evidence.
        raise RepairError(
            "cross-tenant actor-card evidence cites Men history"
        )
    affected = conn.execute(
        """SELECT DISTINCT s.tenant_id, e.actor_id, s.entry_id
             FROM (
                   SELECT s.entry_id, s.tenant_id
                     FROM actor_card_entry_sources s
                     JOIN facts f ON f.id=s.fact_id
                    WHERE s.tenant_id=%s AND f.conversation_id=%s
                   UNION ALL
                   SELECT s.entry_id, s.tenant_id
                     FROM actor_card_turn_sources s
                     JOIN canonical_turns ct
                       ON ct.canonical_turn_id=s.canonical_turn_id
                    WHERE s.tenant_id=%s AND ct.conversation_id=%s
             ) s
             JOIN actor_card_entries e
               ON e.id=s.entry_id AND e.tenant_id=s.tenant_id
            WHERE s.tenant_id=%s""",
        (tenant_id, owner, tenant_id, owner, tenant_id),
    ).fetchall()
    if candidate_actor_ids:
        affected.extend(conn.execute(
            """SELECT tenant_id, actor_id, id AS entry_id
                 FROM actor_card_entries
                WHERE tenant_id=%s AND actor_id=ANY(%s)""",
            (tenant_id, sorted(candidate_actor_ids)),
        ).fetchall())
    entry_ids = sorted({str(row["entry_id"]) for row in affected})
    profiles = {
        (str(row["tenant_id"]), str(row["actor_id"]))
        for row in affected
    }
    actor_ids = candidate_actor_ids | {
        actor_id for profile_tenant, actor_id in profiles
        if profile_tenant == tenant_id
    }
    non_men_turn_sources_before: set[tuple[str, str]] = set()
    non_men_fact_sources_before: set[tuple[str, str]] = set()
    if entry_ids:
        non_men_turn_sources_before = {
            (str(row["entry_id"]), str(row["canonical_turn_id"]))
            for row in conn.execute(
                """SELECT s.entry_id, s.canonical_turn_id
                     FROM actor_card_turn_sources s
                     JOIN canonical_turns ct
                       ON ct.canonical_turn_id=s.canonical_turn_id
                    WHERE s.tenant_id=%s AND s.entry_id=ANY(%s)
                      AND ct.conversation_id<>%s
                    ORDER BY s.entry_id, s.canonical_turn_id""",
                (tenant_id, entry_ids, owner),
            ).fetchall()
        }
        non_men_fact_sources_before = {
            (str(row["entry_id"]), str(row["fact_id"]))
            for row in conn.execute(
                """SELECT s.entry_id, s.fact_id
                     FROM actor_card_entry_sources s
                     JOIN facts f ON f.id=s.fact_id
                    WHERE s.tenant_id=%s AND s.entry_id=ANY(%s)
                      AND f.conversation_id<>%s
                    ORDER BY s.entry_id, s.fact_id""",
                (tenant_id, entry_ids, owner),
            ).fetchall()
        }
    # Detach every source whose foreign-key target is about to be deleted.
    # The denormalized owner/audience columns are not authority: historical
    # attribution corruption can make them disagree with the cited turn/fact.
    # PostgreSQL
    # otherwise fires vc_invalidate_actor_card_turn_source BEFORE DELETE and
    # deletes the entire entry, cascading unrelated DM sources. The entry and
    # every non-Men source remain as rollback/curation evidence while the card
    # is invalid and the offline whole-card rebuild is pending.
    removed_turn_sources = int(conn.execute(
        """DELETE FROM actor_card_turn_sources s
              USING canonical_turns ct
              WHERE s.tenant_id=%s
                AND ct.canonical_turn_id=s.canonical_turn_id
                AND ct.conversation_id=%s""",
        (tenant_id, owner),
    ).rowcount or 0)
    removed_fact_sources = int(conn.execute(
        """DELETE FROM actor_card_entry_sources s
              USING facts f
              WHERE s.tenant_id=%s
                AND f.id=s.fact_id
                AND f.conversation_id=%s""",
        (tenant_id, owner),
    ).rowcount or 0)
    residual = conn.execute(
        """SELECT
             (SELECT COUNT(*)
                FROM actor_card_turn_sources s
                JOIN canonical_turns ct
                  ON ct.canonical_turn_id=s.canonical_turn_id
               WHERE ct.conversation_id=%s) AS turns,
             (SELECT COUNT(*)
                FROM actor_card_entry_sources s
                JOIN facts f ON f.id=s.fact_id
               WHERE f.conversation_id=%s) AS facts""",
        (owner, owner),
    ).fetchone()
    if int(residual["turns"]) or int(residual["facts"]):
        raise RepairError("Men-backed actor-card sources remain attached")
    if entry_ids:
        non_men_turn_sources_after = {
            (str(row["entry_id"]), str(row["canonical_turn_id"]))
            for row in conn.execute(
                """SELECT s.entry_id, s.canonical_turn_id
                     FROM actor_card_turn_sources s
                     JOIN canonical_turns ct
                       ON ct.canonical_turn_id=s.canonical_turn_id
                    WHERE s.tenant_id=%s AND s.entry_id=ANY(%s)
                      AND ct.conversation_id<>%s
                    ORDER BY s.entry_id, s.canonical_turn_id""",
                (tenant_id, entry_ids, owner),
            ).fetchall()
        }
        non_men_fact_sources_after = {
            (str(row["entry_id"]), str(row["fact_id"]))
            for row in conn.execute(
                """SELECT s.entry_id, s.fact_id
                     FROM actor_card_entry_sources s
                     JOIN facts f ON f.id=s.fact_id
                    WHERE s.tenant_id=%s AND s.entry_id=ANY(%s)
                      AND f.conversation_id<>%s
                    ORDER BY s.entry_id, s.fact_id""",
                (tenant_id, entry_ids, owner),
            ).fetchall()
        }
        if (
            non_men_turn_sources_after != non_men_turn_sources_before
            or non_men_fact_sources_after != non_men_fact_sources_before
        ):
            raise RepairError(
                "detaching Men actor-card evidence changed non-Men sources"
            )
    for profile_tenant, actor_id in sorted(
        profiles | {(tenant_id, actor_id) for actor_id in candidate_actor_ids}
    ):
        conn.execute(
            """UPDATE actor_profiles SET card_dirty=1, card_invalid=1,
                      card_build_marker=''
                WHERE tenant_id=%s AND actor_id=%s""",
            (profile_tenant, actor_id),
        )
    return {
        "entries_preserved_pending_rebuild": len(entry_ids),
        "men_turn_sources_detached": removed_turn_sources,
        "men_fact_sources_detached": removed_fact_sources,
        "non_men_turn_sources_preserved": len(non_men_turn_sources_before),
        "non_men_fact_sources_preserved": len(non_men_fact_sources_before),
    }, actor_ids


def _clear_rebuildable(
    conn,
    owner: str,
    tenant_id: str,
    candidate_actor_ids: set[str],
) -> dict[str, int]:
    card_state, affected_actors = _invalidate_actor_cards(
        conn,
        owner,
        tenant_id,
        candidate_actor_ids,
    )
    deleted: dict[str, int] = {
        "actor_card_entries_deleted": 0,
        "actor_card_entries_preserved_pending_rebuild": card_state[
            "entries_preserved_pending_rebuild"
        ],
        "actor_card_men_turn_sources_detached": card_state[
            "men_turn_sources_detached"
        ],
        "actor_card_men_fact_sources_detached": card_state[
            "men_fact_sources_detached"
        ],
        "actor_card_non_men_turn_sources_preserved": card_state[
            "non_men_turn_sources_preserved"
        ],
        "actor_card_non_men_fact_sources_preserved": card_state[
            "non_men_fact_sources_preserved"
        ],
    }

    def run(name: str, sql: str, params: tuple[Any, ...]) -> None:
        deleted[name] = int(conn.execute(sql, params).rowcount or 0)

    run(
        "fact_links",
        """DELETE FROM fact_links WHERE source_fact_id IN
               (SELECT id FROM facts WHERE conversation_id=%s)
             OR target_fact_id IN
               (SELECT id FROM facts WHERE conversation_id=%s)""",
        (owner, owner),
    )
    run(
        "fact_tags",
        "DELETE FROM fact_tags WHERE fact_id IN (SELECT id FROM facts WHERE conversation_id=%s)",
        (owner,),
    )
    run("fact_embeddings", "DELETE FROM fact_embeddings WHERE conversation_id=%s", (owner,))
    run("facts", "DELETE FROM facts WHERE conversation_id=%s", (owner,))
    # Segment associations are derived from segment membership and are rebuilt
    # later.  Durable tool outputs, per-turn associations, and chain snapshots
    # are source evidence: they are remapped or quarantined by exact old-group
    # disposition after this derived-data clear and are never discarded.
    run("segment_tool_outputs", "DELETE FROM segment_tool_outputs WHERE conversation_id=%s", (owner,))
    run(
        "segment_chunks",
        "DELETE FROM segment_chunks WHERE segment_ref IN (SELECT ref FROM segments WHERE conversation_id=%s)",
        (owner,),
    )
    run(
        "segment_tags",
        "DELETE FROM segment_tags WHERE segment_ref IN (SELECT ref FROM segments WHERE conversation_id=%s)",
        (owner,),
    )
    run("segments", "DELETE FROM segments WHERE conversation_id=%s", (owner,))
    run("tag_summary_embeddings", "DELETE FROM tag_summary_embeddings WHERE conversation_id=%s", (owner,))
    run("tag_summaries", "DELETE FROM tag_summaries WHERE conversation_id=%s", (owner,))
    # Tag aliases can predate the corrupt history and carry no creation time;
    # preserve them for later evidence-based review instead of blanket loss.
    run("canonical_turn_chunks", "DELETE FROM canonical_turn_chunks WHERE conversation_id=%s", (owner,))
    run("canonical_turn_anchors", "DELETE FROM canonical_turn_anchors WHERE conversation_id=%s", (owner,))
    run("engine_state", "DELETE FROM engine_state WHERE conversation_id=%s", (owner,))
    run(
        "speaker_handles",
        "DELETE FROM speaker_handles WHERE tenant_id=%s AND audience_conversation_id=%s",
        (tenant_id, owner),
    )
    if affected_actors:
        run(
            "actor_card_rebuild_status",
            "DELETE FROM actor_card_rebuild_status WHERE tenant_id=%s AND actor_id=ANY(%s)",
            (tenant_id, sorted(affected_actors)),
        )
    return deleted


def _actor_ranges_from_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    ranges: dict[str, dict[str, str]] = {}
    for row in rows:
        actor_id = str(row.get("sender_actor_id") or "")
        if not actor_id:
            continue
        timestamp = str(row.get("created_at") or "")
        current = ranges.setdefault(actor_id, {
            "name": str(row.get("sender") or ""),
            "first": timestamp,
            "last": timestamp,
        })
        current["first"] = min(current["first"], timestamp)
        current["last"] = max(current["last"], timestamp)
        if row.get("sender"):
            current["name"] = str(row["sender"])
    return ranges


def _evidence_remap_plan(
    *,
    snapshot_rows: dict[str, list[dict[str, Any]]],
    old_group_map: dict[int, int | None],
) -> dict[str, Any]:
    """Partition ordinal evidence into an exact live map or quarantine."""

    def destination(value: Any) -> int | None:
        return old_group_map.get(int(value))

    candidate_tools: dict[str, tuple[dict[str, Any], int]] = {}
    quarantined: dict[str, list[dict[str, Any]]] = {
        "tool_outputs": [],
        "turn_tool_outputs": [],
        "chain_snapshots": [],
    }
    for row in snapshot_rows["tool_outputs"]:
        origin = str(row.get("origin_conversation_id") or "")
        if origin and origin != OWNER:
            quarantined["tool_outputs"].append({
                "reason": "foreign_origin_ordinal_namespace",
                "original": row,
            })
            continue
        target = destination(row["turn"])
        if target is None:
            quarantined["tool_outputs"].append({
                "reason": "old_turn_has_no_unique_external_group",
                "original": row,
            })
        else:
            ref = str(row["ref"])
            if ref in candidate_tools:
                raise RepairError(f"duplicate tool output ref in snapshot: {ref}")
            candidate_tools[ref] = (row, int(target))

    # A link must agree with the tool output's own mapped target. Any
    # disagreement quarantines the whole ref and every one of its links;
    # retaining two individually mappable but contradictory ordinals would
    # recreate a cross-turn tool association.
    corrupt_tool_refs: set[str] = set()
    for row in snapshot_rows["turn_tool_outputs"]:
        ref = str(row["tool_output_ref"])
        origin = str(row.get("origin_conversation_id") or "")
        target = (
            destination(row["turn_number"])
            if not origin or origin == OWNER
            else None
        )
        tool = candidate_tools.get(ref)
        if tool is None or target is None or int(target) != tool[1]:
            if tool is not None:
                corrupt_tool_refs.add(ref)
    for ref in sorted(corrupt_tool_refs):
        row, _target = candidate_tools.pop(ref)
        quarantined["tool_outputs"].append({
            "reason": "tool_output_and_link_targets_disagree",
            "original": row,
        })

    mapped_tools = list(candidate_tools.values())
    mapped_tool_refs = set(candidate_tools)

    mapped_links: set[tuple[str, int, str, str]] = set()
    for row in snapshot_rows["turn_tool_outputs"]:
        origin = str(row.get("origin_conversation_id") or "")
        target = (
            destination(row["turn_number"])
            if not origin or origin == OWNER
            else None
        )
        ref = str(row["tool_output_ref"])
        reason = None
        if ref in corrupt_tool_refs:
            reason = "tool_output_and_link_targets_disagree"
        elif origin and origin != OWNER:
            reason = "foreign_origin_ordinal_namespace"
        elif target is None:
            reason = "old_turn_has_no_unique_external_group"
        elif ref not in mapped_tool_refs:
            reason = "tool_output_was_quarantined"
        elif int(target) != candidate_tools[ref][1]:
            # Defensive duplicate of the pre-pass invariant.
            reason = "tool_output_and_link_targets_disagree"
        if reason:
            quarantined["turn_tool_outputs"].append({
                "reason": reason,
                "original": row,
            })
        else:
            mapped_links.add((OWNER, int(target), ref, origin))

    mapped_chains: list[tuple[dict[str, Any], int]] = []
    for row in snapshot_rows["chain_snapshots"]:
        # chain_json contains the full pre-filter user/tool/assistant chain.
        # The old database's central defect is source-B/old-user-A/assistant-B
        # grafting, so an ordinal match cannot attest those embedded messages.
        # Preserve every chain externally and serve none until a future chain
        # format carries immutable per-message source membership.
        quarantined["chain_snapshots"].append({
            "reason": "pre_repair_chain_lacks_per_message_source_attestation",
            "original": row,
        })

    return {
        "mapped_tools": mapped_tools,
        "mapped_links": sorted(mapped_links),
        "mapped_chains": mapped_chains,
        "quarantined": quarantined,
        "turn_tool_outputs_deduplicated": (
            len(snapshot_rows["turn_tool_outputs"])
            - len(mapped_links)
            - len(quarantined["turn_tool_outputs"])
        ),
    }


def _write_quarantine_artifact(
    output_dir: Path,
    *,
    plan: dict[str, Any],
    snapshot_manifest: dict[str, Any],
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Durably preserve every row removed from serving storage."""

    output_dir.mkdir(parents=True, exist_ok=False)
    file_metadata: dict[str, dict[str, Any]] = {}
    for table, rows in sorted(plan["quarantined"].items()):
        filename = f"{table}.jsonl"
        count, digest = _write_rows(output_dir / filename, rows)
        file_metadata[table] = {
            "filename": filename,
            "rows": count,
            "sha256": digest,
        }
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "owner_conversation_id": OWNER,
        "tenant_id": snapshot_manifest["tenant_id"],
        "lifecycle_epoch": snapshot_manifest["lifecycle_epoch"],
        "repair_batch_id": artifact_manifest["repair_batch_id"],
        "snapshot_canonical_rows_sha256": snapshot_manifest[
            "canonical_rows_sha256"
        ],
        "database_backup": snapshot_manifest["database_backup"],
        "files": file_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    _write_text_durable(
        manifest_path,
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
    )
    checksums = [
        f"{meta['sha256']}  {meta['filename']}"
        for _table, meta in sorted(file_metadata.items())
    ]
    manifest_sha = _sha256_file(manifest_path)
    checksums.append(f"{manifest_sha}  manifest.json")
    _write_text_durable(
        output_dir / "SHA256SUMS",
        "\n".join(checksums) + "\n",
    )
    _fsync_directory(output_dir)
    return {
        "path": str(output_dir),
        "manifest_sha256": manifest_sha,
        "rows": sum(meta["rows"] for meta in file_metadata.values()),
        "files": file_metadata,
    }


def _remap_preserved_turn_evidence(
    conn,
    *,
    owner: str,
    plan: dict[str, Any],
) -> dict[str, int]:
    """Apply the exact map and delete externally quarantined live evidence."""

    report = Counter()
    # Remove links first so quarantined tool outputs can be deleted without a
    # foreign-key dependency. Only exact, retained links are reinserted.
    report["turn_tool_outputs_original"] = int(conn.execute(
        "DELETE FROM turn_tool_outputs WHERE conversation_id=%s",
        (owner,),
    ).rowcount or 0)

    for row, target in plan["mapped_tools"]:
        updated = conn.execute(
            """UPDATE tool_outputs SET turn=%s
                 WHERE conversation_id=%s AND ref=%s""",
            (target, owner, row["ref"]),
        )
        if updated.rowcount != 1:
            raise RepairError(f"tool output disappeared during remap: {row['ref']}")
        report["tool_outputs_mapped"] += 1
    for item in plan["quarantined"]["tool_outputs"]:
        row = item["original"]
        deleted = conn.execute(
            "DELETE FROM tool_outputs WHERE conversation_id=%s AND ref=%s",
            (owner, row["ref"]),
        )
        if deleted.rowcount != 1:
            raise RepairError(f"quarantined tool output disappeared: {row['ref']}")
        report["tool_outputs_quarantined_deleted"] += 1

    for conversation_id, turn_number, ref, origin in plan["mapped_links"]:
        conn.execute(
            """INSERT INTO turn_tool_outputs
                   (conversation_id, turn_number, tool_output_ref,
                    origin_conversation_id)
               VALUES (%s,%s,%s,%s)""",
            (conversation_id, turn_number, ref, origin),
        )
        report["turn_tool_outputs_mapped"] += 1
    report["turn_tool_outputs_quarantined_deleted"] = len(
        plan["quarantined"]["turn_tool_outputs"]
    )
    report["turn_tool_outputs_deduplicated"] = plan[
        "turn_tool_outputs_deduplicated"
    ]

    for row, target in plan["mapped_chains"]:
        updated = conn.execute(
            """UPDATE chain_snapshots SET turn_number=%s
                 WHERE conversation_id=%s AND ref=%s""",
            (target, owner, row["ref"]),
        )
        if updated.rowcount != 1:
            raise RepairError(f"chain snapshot disappeared during remap: {row['ref']}")
        report["chain_snapshots_mapped"] += 1
    for item in plan["quarantined"]["chain_snapshots"]:
        row = item["original"]
        deleted = conn.execute(
            "DELETE FROM chain_snapshots WHERE conversation_id=%s AND ref=%s",
            (owner, row["ref"]),
        )
        if deleted.rowcount != 1:
            raise RepairError(f"quarantined chain snapshot disappeared: {row['ref']}")
        report["chain_snapshots_quarantined_deleted"] += 1

    # Only evidence with an independently mapped group may remain live.
    live_tool_refs = {
        str(row["ref"])
        for row in conn.execute(
            "SELECT ref FROM tool_outputs WHERE conversation_id=%s",
            (owner,),
        ).fetchall()
    }
    retained_tool_refs = {
        str(row["ref"]) for row, _target in plan["mapped_tools"]
    }
    if live_tool_refs != retained_tool_refs:
        raise RepairError("live tool outputs differ from the exact remap plan")
    live_tool_shape = {
        (
            str(row["ref"]),
            int(row["turn"]),
            str(row.get("origin_conversation_id") or ""),
        )
        for row in conn.execute(
            """SELECT ref, turn, origin_conversation_id
                 FROM tool_outputs WHERE conversation_id=%s""",
            (owner,),
        ).fetchall()
    }
    planned_tool_shape = {
        (
            str(row["ref"]),
            int(target),
            str(row.get("origin_conversation_id") or ""),
        )
        for row, target in plan["mapped_tools"]
    }
    if live_tool_shape != planned_tool_shape:
        raise RepairError("live tool output ordinals differ from the remap plan")
    live_link_shape = {
        (
            str(row["conversation_id"]),
            int(row["turn_number"]),
            str(row["tool_output_ref"]),
            str(row.get("origin_conversation_id") or ""),
        )
        for row in conn.execute(
            """SELECT conversation_id, turn_number, tool_output_ref,
                      origin_conversation_id
                 FROM turn_tool_outputs WHERE conversation_id=%s""",
            (owner,),
        ).fetchall()
    }
    if live_link_shape != set(plan["mapped_links"]):
        raise RepairError("live tool links differ from the exact remap plan")
    live_chain_refs = {
        str(row["ref"])
        for row in conn.execute(
            "SELECT ref FROM chain_snapshots WHERE conversation_id=%s",
            (owner,),
        ).fetchall()
    }
    retained_chain_refs = {
        str(row["ref"]) for row, _target in plan["mapped_chains"]
    }
    if live_chain_refs != retained_chain_refs:
        raise RepairError("live chain snapshots differ from the exact remap plan")
    negative_ordinals = conn.execute(
        """SELECT
             (SELECT COUNT(*) FROM tool_outputs
               WHERE conversation_id=%s AND turn < 0) AS tools,
             (SELECT COUNT(*) FROM turn_tool_outputs
               WHERE conversation_id=%s AND turn_number < 0) AS links,
             (SELECT COUNT(*) FROM chain_snapshots
               WHERE conversation_id=%s AND turn_number < 0) AS chains""",
        (owner, owner, owner),
    ).fetchone()
    if any(int(negative_ordinals[key]) for key in ("tools", "links", "chains")):
        raise RepairError("negative serving ordinals survived the evidence remap")
    return dict(sorted(report.items()))


def apply_repair(
    *,
    dsn_env: str,
    artifact_dir: Path,
    artifact_manifest_sha256: str,
    snapshot_manifest_path: Path,
    restore_proof_path: Path,
    tenant_id: str,
    lifecycle_epoch: int,
    quarantine_dir: Path | None,
    apply: bool,
    maintenance_freeze_confirmed: bool,
) -> dict[str, Any]:
    if not maintenance_freeze_confirmed:
        raise RepairError(
            "apply requires a confirmed gateway, compaction, and actor-card "
            "worker maintenance freeze"
        )
    artifact_manifest, rows, sources, old_group_map = _load_artifact(
        artifact_dir,
        tenant_id=tenant_id,
        expected_manifest_sha256=artifact_manifest_sha256,
    )
    actor_ranges = _actor_ranges_from_rows(rows)
    candidate_actor_ids = set(actor_ranges)
    snapshot_manifest = _load_snapshot_manifest(snapshot_manifest_path)
    restore_proof = _load_restore_proof(
        restore_proof_path,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_manifest=snapshot_manifest,
    )
    if (
        snapshot_manifest["tenant_id"] != tenant_id
        or int(snapshot_manifest["lifecycle_epoch"]) != lifecycle_epoch
    ):
        raise RepairError("snapshot tenant or lifecycle epoch mismatch")
    old_canonical = (artifact_manifest.get("inputs") or {}).get(
        "old_canonical"
    ) or {}
    if (
        old_canonical.get("sha256")
        != snapshot_manifest["canonical_rows_sha256"]
        or type(old_canonical.get("owner_rows")) is not int
        or old_canonical["owner_rows"]
        != int(snapshot_manifest["canonical_rows"])
        or type(old_canonical.get("rows")) is not int
        or old_canonical["rows"] != int(snapshot_manifest["canonical_rows"])
    ):
        raise RepairError(
            "artifact old-canonical input is not the checksummed repair snapshot"
        )
    evidence_plan = _evidence_remap_plan(
        snapshot_rows=snapshot_manifest["_evidence_rows"],
        old_group_map=old_group_map,
    )

    with _connect(dsn_env) as conn:
        _assert_schema(conn)
        conversation = _conversation_row(conn, OWNER)
        current_rows = _current_canonical_rows(conn, OWNER)
        current_sha, current_count = _hash_rows(current_rows)
        current_old_groups = {
            int(row["turn_group_number"])
            for row in current_rows
        }
        if set(old_group_map) != current_old_groups:
            raise RepairError(
                "artifact old-group mapping does not exhaust the live canonical groups"
            )
        current_evidence = _evidence_rows(conn, OWNER, tenant_id)
        for name, rows_for_table in current_evidence.items():
            current_evidence_sha, current_evidence_count = _hash_rows(rows_for_table)
            expected = snapshot_manifest["evidence_files"][name]
            if (
                current_evidence_count != int(expected["rows"])
                or current_evidence_sha != expected["sha256"]
            ):
                raise RepairError(
                    f"live {name} changed after the checksummed repair snapshot"
                )
        report: dict[str, Any] = {
            "status": "dry_run" if not apply else "pending",
            "owner_conversation_id": OWNER,
            "tenant_id": tenant_id,
            "lifecycle_epoch": lifecycle_epoch,
            "backup": snapshot_manifest["database_backup"],
            "restore_proof": {
                "path": str(restore_proof_path),
                "sha256": _sha256_file(restore_proof_path),
                "restore_database": restore_proof["restore_database"],
                "verified_at": restore_proof["verified_at"],
            },
            "before": {
                "canonical_rows": current_count,
                "canonical_rows_sha256": current_sha,
                "derived": _derived_counts(conn, OWNER, tenant_id),
            },
            "candidate": {
                "canonical_rows": len(rows),
                "source_memberships": len(sources),
                "actors": len({row["sender_actor_id"] for row in rows if row["user_content"]}),
                "artifact_repair_batch_id": artifact_manifest["repair_batch_id"],
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "old_groups_mapped": sum(
                    target is not None for target in old_group_map.values()
                ),
                "old_groups_quarantined": sum(
                    target is None for target in old_group_map.values()
                ),
                "evidence_quarantine_rows": sum(
                    len(rows)
                    for rows in evidence_plan["quarantined"].values()
                ),
            },
        }
        if current_sha != snapshot_manifest["canonical_rows_sha256"]:
            raise RepairError(
                "live canonical snapshot changed after backup snapshot: "
                f"expected={snapshot_manifest['canonical_rows_sha256']} actual={current_sha}"
            )
        if not apply:
            return report
        if quarantine_dir is None:
            raise RepairError("--quarantine-dir is required for a live apply")
        quarantine = _write_quarantine_artifact(
            quarantine_dir,
            plan=evidence_plan,
            snapshot_manifest=snapshot_manifest,
            artifact_manifest=artifact_manifest,
        )
        report["quarantine"] = quarantine

        with conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("virtual-context:men-canonical-history-repair-20260802",),
            )
            # Legacy conversations can predate conversation_lifecycle even
            # though their canonical history is otherwise valid. Establish
            # the same serialization row used by normal reconciles, without
            # reviving a deleted row, before taking the repair lock.
            lifecycle_insert = conn.execute(
                """INSERT INTO conversation_lifecycle
                       (conversation_id, generation, deleted, updated_at)
                   VALUES (%s, 0, FALSE, %s)
                   ON CONFLICT (conversation_id) DO NOTHING""",
                (OWNER, datetime.now(timezone.utc).isoformat()),
            )
            locked_lifecycle = conn.execute(
                "SELECT * FROM conversation_lifecycle WHERE conversation_id=%s FOR UPDATE",
                (OWNER,),
            ).fetchone()
            expected_lifecycle = snapshot_manifest["_evidence_rows"][
                "conversation_lifecycle"
            ]
            if len(expected_lifecycle) > 1 or locked_lifecycle is None:
                raise RepairError(
                    "conversation_lifecycle snapshot is structurally invalid"
                )
            if not expected_lifecycle:
                # A legacy snapshot legitimately has no lifecycle row. Only
                # the row inserted by this transaction may fill that gap. If
                # ON CONFLICT won, another writer changed the frozen preimage
                # between validation and lock acquisition.
                if lifecycle_insert.rowcount != 1:
                    raise RepairError(
                        "conversation_lifecycle appeared while acquiring "
                        "the repair lock"
                    )
                if (
                    str(locked_lifecycle["conversation_id"]) != OWNER
                    or int(locked_lifecycle["generation"]) != 0
                    or bool(locked_lifecycle["deleted"])
                ):
                    raise RepairError(
                        "new conversation_lifecycle lock row is invalid"
                    )
            else:
                if lifecycle_insert.rowcount != 0:
                    raise RepairError(
                        "conversation_lifecycle disappeared while acquiring "
                        "the repair lock"
                    )
                lifecycle_sha, lifecycle_count = _hash_rows([
                    dict(locked_lifecycle)
                ])
                expected_lifecycle_sha, expected_lifecycle_count = _hash_rows(
                    expected_lifecycle
                )
                if (
                    lifecycle_count != expected_lifecycle_count
                    or lifecycle_sha != expected_lifecycle_sha
                ):
                    raise RepairError(
                        "conversation_lifecycle changed while acquiring "
                        "the repair lock"
                    )
            locked = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=%s FOR UPDATE",
                (OWNER,),
            ).fetchone()
            if (
                locked is None
                or locked["tenant_id"] != tenant_id
                or int(locked["lifecycle_epoch"]) != lifecycle_epoch
                or locked["phase"] != "active"
                or locked["deleted_at"] is not None
                or int(locked["pending_raw_payload_entries"] or 0) != 0
            ):
                raise RepairError("locked conversation failed lifecycle preconditions")
            _assert_quiescent(conn, OWNER, tenant_id, lifecycle_epoch)
            locked_rows = _current_canonical_rows(conn, OWNER)
            locked_sha, _ = _hash_rows(locked_rows)
            if locked_sha != snapshot_manifest["canonical_rows_sha256"]:
                raise RepairError("canonical rows changed while acquiring the repair lock")
            locked_evidence = _evidence_rows(conn, OWNER, tenant_id)
            for name, rows_for_table in locked_evidence.items():
                # The legacy no-row case was validated and filled under the
                # lifecycle lock above. Every other evidence surface must
                # still byte-match the frozen snapshot at this point.
                if name == "conversation_lifecycle":
                    continue
                digest, count = _hash_rows(rows_for_table)
                expected = snapshot_manifest["evidence_files"][name]
                if count != int(expected["rows"]) or digest != expected["sha256"]:
                    raise RepairError(
                        f"{name} changed while acquiring the repair lock"
                    )

            # Re-hash the rollback artifact while the same lifecycle lock that
            # every canonical reconciler takes is held.  This closes the gap
            # between manifest validation and the first destructive write.
            locked_backup = _verify_backup_file(
                snapshot_manifest["database_backup"]["path"],
                snapshot_manifest["database_backup"]["sha256"],
            )
            if locked_backup["bytes"] != snapshot_manifest["database_backup"]["bytes"]:
                raise RepairError("database backup size changed before apply")

            # This is a delete-and-rebuild lifecycle boundary.  Advancing both
            # fences means an in-memory worker from the old history cannot
            # resume after this transaction and persist a stale canonical row
            # or checkpoint.  The conversation_lifecycle row lock also blocks
            # every normal canonical reconciler for the duration of the swap.
            next_epoch = lifecycle_epoch + 1
            epoch_update = conn.execute(
                """UPDATE conversations
                      SET lifecycle_epoch=%s, updated_at=%s
                    WHERE conversation_id=%s AND lifecycle_epoch=%s""",
                (
                    next_epoch,
                    datetime.now(timezone.utc),
                    OWNER,
                    lifecycle_epoch,
                ),
            )
            if epoch_update.rowcount != 1:
                raise RepairError("failed to advance conversation lifecycle epoch")
            generation_row = conn.execute(
                """UPDATE conversation_lifecycle
                      SET generation=generation+1, updated_at=%s
                    WHERE conversation_id=%s AND deleted=FALSE
                RETURNING generation""",
                (datetime.now(timezone.utc).isoformat(), OWNER),
            ).fetchone()
            if generation_row is None:
                raise RepairError("failed to advance conversation generation")
            next_generation = int(generation_row["generation"])

            linked_actor_rows = conn.execute(
                """SELECT DISTINCT e.actor_id
                     FROM (
                           SELECT s.entry_id, s.tenant_id
                             FROM actor_card_entry_sources s
                             JOIN facts f ON f.id=s.fact_id
                            WHERE s.tenant_id=%s
                              AND f.conversation_id=%s
                           UNION ALL
                           SELECT s.entry_id, s.tenant_id
                             FROM actor_card_turn_sources s
                             JOIN canonical_turns ct
                               ON ct.canonical_turn_id=s.canonical_turn_id
                            WHERE s.tenant_id=%s
                              AND ct.conversation_id=%s
                     ) s
                     JOIN actor_card_entries e
                       ON e.id=s.entry_id AND e.tenant_id=s.tenant_id
                    WHERE s.tenant_id=%s""",
                (tenant_id, OWNER, tenant_id, OWNER, tenant_id),
            ).fetchall()
            affected_actor_ids = candidate_actor_ids | {
                str(row["actor_id"]) for row in linked_actor_rows
            }
            # Ensure every candidate has a row that can be locked. Existing
            # cards for the same actor may mix Men and DM evidence; locking the
            # tenant profile serializes their CAS/replace path with this swap.
            for actor_id, values in actor_ranges.items():
                conn.execute(
                    """INSERT INTO actor_profiles (
                           tenant_id, actor_id, platform, display_name,
                           first_seen_at, last_seen_at, card_built_at,
                           card_dirty, card_invalid, card_input_hash,
                           card_build_marker
                       ) VALUES (%s,%s,%s,%s,%s,%s,NULL,1,1,'','')
                       ON CONFLICT (tenant_id, actor_id) DO UPDATE SET
                           platform=EXCLUDED.platform,
                           display_name=CASE
                               WHEN BTRIM(COALESCE(
                                   actor_profiles.display_name, ''
                               )) = ''
                               THEN EXCLUDED.display_name
                               ELSE actor_profiles.display_name
                           END,
                           first_seen_at=LEAST(
                               actor_profiles.first_seen_at,
                               EXCLUDED.first_seen_at
                           ),
                           last_seen_at=GREATEST(
                               actor_profiles.last_seen_at,
                               EXCLUDED.last_seen_at
                           ),
                           card_dirty=1, card_invalid=1,
                           card_build_marker=''""",
                    (
                        tenant_id, actor_id, PLATFORM, values["name"],
                        values["first"], values["last"],
                    ),
                )
            if affected_actor_ids:
                locked_actors = conn.execute(
                    """SELECT actor_id FROM actor_profiles
                        WHERE tenant_id=%s AND actor_id=ANY(%s)
                        ORDER BY actor_id FOR UPDATE""",
                    (tenant_id, sorted(affected_actor_ids)),
                ).fetchall()
                if {str(row["actor_id"]) for row in locked_actors} != affected_actor_ids:
                    raise RepairError(
                        "failed to lock every affected actor profile"
                    )
            deleted = _clear_rebuildable(
                conn,
                OWNER,
                tenant_id,
                candidate_actor_ids,
            )
            evidence_remap = _remap_preserved_turn_evidence(
                conn,
                owner=OWNER,
                plan=evidence_plan,
            )
            old_rows_deleted = int(conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id=%s",
                (OWNER,),
            ).rowcount or 0)
            if old_rows_deleted != current_count:
                raise RepairError("canonical delete count drifted inside transaction")

            canonical_sql = (
                "INSERT INTO canonical_turns ("
                + ",".join(CANONICAL_COLUMNS)
                + ") VALUES ("
                + ",".join(["%s"] * len(CANONICAL_COLUMNS))
                + ")"
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    canonical_sql,
                    [tuple(row[column] for column in CANONICAL_COLUMNS) for row in rows],
                )
            source_sql = (
                "INSERT INTO canonical_message_sources ("
                + ",".join(SOURCE_COLUMNS)
                + ") VALUES ("
                + ",".join(["%s"] * len(SOURCE_COLUMNS))
                + ")"
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    source_sql,
                    [tuple(source[column] for column in SOURCE_COLUMNS) for source in sources],
                )

            if actor_ranges:
                conn.execute(
                    "DELETE FROM actor_card_rebuild_status WHERE tenant_id=%s AND actor_id=ANY(%s)",
                    (tenant_id, sorted(actor_ranges)),
                )

            post_rows = _current_canonical_rows(conn, OWNER)
            post_sha, post_count = _hash_rows(post_rows)
            expected_sha, expected_count = _hash_rows(rows)
            if post_count != expected_count or post_sha != expected_sha:
                raise RepairError("post-insert canonical rows differ from the artifact")
            joined = conn.execute(
                """SELECT COUNT(*) AS n FROM canonical_message_sources cms
                    JOIN canonical_turns ct
                      ON ct.canonical_turn_id=cms.canonical_turn_id
                    JOIN canonical_turns paired
                      ON paired.canonical_turn_id=
                         cms.assistant_canonical_turn_id
                   WHERE ct.conversation_id=%s
                     AND paired.conversation_id=ct.conversation_id
                     AND paired.turn_group_number=ct.turn_group_number
                     AND cms.pair_version=1
                     AND cms.assistant_turn_hash=paired.turn_hash
                     AND BTRIM(paired.assistant_content) <> ''
                     AND BTRIM(paired.user_content) = ''
                     AND cms.tenant_id=%s
                     AND cms.agent_scope_id=%s
                     AND cms.platform=%s
                     AND cms.account_id=%s
                     AND cms.guild_id=%s
                     AND cms.message_id=ct.source_message_id
                     AND cms.source_actor_id=ct.sender_actor_id
                     AND cms.channel_id=ct.origin_channel_id
                     AND cms.canonical_turn_hash=ct.turn_hash""",
                (OWNER, tenant_id, AGENT_SCOPE_ID, PLATFORM, ACCOUNT_ID, GUILD_ID),
            ).fetchone()["n"]
            if int(joined) != len(sources):
                raise RepairError("source-ledger postcondition failed")

            # Anti-vacuity: prove the database trigger, not just Python, blocks
            # a source-backed body rewrite. The nested transaction is a
            # savepoint, so the expected violation does not poison the swap.
            protected_id = str(sources[0]["canonical_turn_id"])
            blocked = False
            try:
                with conn.transaction():
                    conn.execute(
                        "UPDATE canonical_turns SET user_content=user_content || 'x' "
                        "WHERE canonical_turn_id=%s",
                        (protected_id,),
                    )
            except Exception as exc:  # psycopg raises IntegrityError subclass
                if "attested canonical source fields are immutable" not in str(exc):
                    raise
                blocked = True
            if not blocked:
                raise RepairError("database immutability trigger did not reject a rewrite")

            report.update({
                "status": "canonical_swapped_rebuild_required",
                "deleted": deleted,
                "preserved_turn_evidence": evidence_remap,
                "old_canonical_rows_deleted": old_rows_deleted,
                "after": {
                    "canonical_rows": post_count,
                    "canonical_rows_sha256": post_sha,
                    "source_memberships": int(joined),
                    "actor_profiles_dirty_invalid": len(actor_ranges),
                    "database_immutability_probe": "blocked",
                    "lifecycle_epoch": next_epoch,
                    "conversation_generation": next_generation,
                },
                "committed_at": datetime.now(timezone.utc).isoformat(),
            })
        return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", default="DATABASE_URL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--owner", default=OWNER)
    snapshot_parser.add_argument("--tenant-id", required=True)
    snapshot_parser.add_argument("--lifecycle-epoch", type=int, required=True)
    snapshot_parser.add_argument("--output-dir", type=Path, required=True)
    snapshot_parser.add_argument("--backup-path", required=True)
    snapshot_parser.add_argument(
        "--maintenance-freeze-confirmed", action="store_true", required=True,
    )

    restore_parser = subparsers.add_parser("verify-restore")
    restore_parser.add_argument(
        "--snapshot-manifest", type=Path, required=True,
    )
    restore_parser.add_argument(
        "--restore-dsn-env", default="VC_RESTORE_DATABASE_URL",
    )
    restore_parser.add_argument("--proof", type=Path, required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--artifact-dir", type=Path, required=True)
    apply_parser.add_argument("--artifact-manifest-sha256", required=True)
    apply_parser.add_argument("--snapshot-manifest", type=Path, required=True)
    apply_parser.add_argument("--restore-proof", type=Path, required=True)
    apply_parser.add_argument("--tenant-id", required=True)
    apply_parser.add_argument("--lifecycle-epoch", type=int, required=True)
    apply_parser.add_argument("--quarantine-dir", type=Path)
    apply_parser.add_argument("--apply", action="store_true")
    apply_parser.add_argument("--report", type=Path)
    apply_parser.add_argument(
        "--maintenance-freeze-confirmed", action="store_true", required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "snapshot":
            result = snapshot(
                dsn_env=args.dsn_env,
                owner=args.owner,
                tenant_id=args.tenant_id,
                lifecycle_epoch=args.lifecycle_epoch,
                output_dir=args.output_dir.resolve(),
                backup_path=args.backup_path,
                maintenance_freeze_confirmed=bool(
                    args.maintenance_freeze_confirmed
                ),
            )
        elif args.command == "verify-restore":
            result = verify_restore(
                snapshot_manifest_path=args.snapshot_manifest.resolve(),
                restore_dsn_env=args.restore_dsn_env,
                proof_path=args.proof.resolve(),
            )
        else:
            result = apply_repair(
                dsn_env=args.dsn_env,
                artifact_dir=args.artifact_dir.resolve(),
                artifact_manifest_sha256=args.artifact_manifest_sha256,
                snapshot_manifest_path=args.snapshot_manifest.resolve(),
                restore_proof_path=args.restore_proof.resolve(),
                tenant_id=args.tenant_id,
                lifecycle_epoch=args.lifecycle_epoch,
                quarantine_dir=(
                    args.quarantine_dir.resolve()
                    if args.quarantine_dir is not None
                    else None
                ),
                apply=bool(args.apply),
                maintenance_freeze_confirmed=bool(
                    args.maintenance_freeze_confirmed
                ),
            )
            if args.report:
                args.report.write_text(
                    json.dumps(result, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
    except (RepairError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
