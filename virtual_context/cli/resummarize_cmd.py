"""Repair source-text fallback summaries in place.

Selects segments whose stored summary is a STRICT byte prefix of their
own full_text (the signature of a summarize failure stored as content),
regenerates each summary with the exact request compaction would issue,
and writes only summaries that pass the same validator compaction uses.
Every write is guarded by a row-version compare-and-set on ``xmin`` and
preceded by a durable journal append, so a concurrent recompaction wins
every race and a killed run never loses track of what it changed.

The dry run opens ONE plain psycopg connection with server-enforced
read-only and touches nothing else: no engine, no store construction
(store init runs schema DDL; engine init upserts a conversations row).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone

from ..core.canonical_turns import STRIP_WHITESPACE

# The selection predicate. STRICT prefix: equality rows are intentional
# passthrough stubs (summary == full_text means nothing was lost and
# there is nothing to recover), so ``<`` is load-bearing, and the
# equality-overlap probe below exists to notice if it ever stops being
# strict. The short-source split uses STRIPPED length with the same
# whitespace set Python's str.strip() uses, bound as a parameter (an
# escaped SQL literal for this set famously drops the vertical tab and
# gains the letter v).
_DAMAGE_PREDICATE = (
    "summary <> '' "
    "AND length(summary) < length(full_text) "
    "AND left(full_text, length(summary)) = summary"
)

_SHORT_SPLIT = "length(btrim(full_text, %(strip_ws)s)) >= 256"

_SELECT_COLUMNS = (
    "ref, conversation_id, primary_tag, summary, full_text, full_tokens, "
    "created_at, start_timestamp, end_timestamp, metadata_json, "
    "xmin::text AS row_version, "
    "(SELECT COALESCE(array_agg(tag ORDER BY tag), ARRAY[]::text[]) "
    " FROM segment_tags st WHERE st.segment_ref = segments.ref) AS tags"
)


def _selection_sql(include_short: bool, since: str | None, until: str | None,
                   after_ref: str | None) -> str:
    clauses = ["conversation_id = %(conversation_id)s", _DAMAGE_PREDICATE]
    if not include_short:
        clauses.append(_SHORT_SPLIT)
    if since:
        clauses.append("created_at::timestamptz >= %(since)s::timestamptz")
    if until:
        clauses.append("created_at::timestamptz < %(until)s::timestamptz")
    if after_ref:
        clauses.append("ref > %(after_ref)s")
    return (
        f"SELECT {_SELECT_COLUMNS} FROM segments "
        f"WHERE {' AND '.join(clauses)} ORDER BY ref ASC"
    )


def _params(args) -> dict:
    return {
        "conversation_id": args.conversation_id,
        "strip_ws": STRIP_WHITESPACE,
        "since": getattr(args, "since", None),
        "until": getattr(args, "until", None),
        "after_ref": getattr(args, "after_ref", None),
    }


def classify_generated(summary: str, full_text: str, token_counter,
                       max_summary_tokens: int) -> str | None:
    """Why a Generated summary must not be written, or None to accept.

    Uses the compactor's own validator so the repair gate and the
    compaction gate can never drift apart, then adds the two repair-only
    postconditions: the repair must DESTROY the selection predicate (a
    summary that is still a prefix of full_text carries no information
    and would be re-selected forever), and must fit the summary token
    bound compaction operates under.
    """
    from ..core.compactor import DomainCompactor

    reason = DomainCompactor._unusable_reason(summary, full_text)
    if reason is not None:
        return f"validator_{reason}"
    if full_text.startswith(summary):
        return "still_prefix"
    if token_counter(summary) > max_summary_tokens:
        return "overlong"
    return None


def _connect(dsn: str, *, read_only: bool):
    import psycopg
    from psycopg.rows import dict_row

    options = "-c default_transaction_read_only=on" if read_only else None
    return psycopg.connect(
        dsn, row_factory=dict_row, autocommit=True, options=options,
    )


def _resolve_dsn(args) -> str | None:
    return getattr(args, "postgres_dsn", None) or os.environ.get("DATABASE_URL")


def _fail(stage: str, error: str, conversation_id: str) -> None:
    print(json.dumps({
        "status": "error", "stage": stage,
        "conversation_id": conversation_id, "error": error,
    }))
    sys.exit(1)


def _sql_str(value: str) -> str:
    """A safely quoted SQL string literal for the printed runbook.

    The runbook is copy/paste material, so every interpolated value is
    quoted as data: conversation ids are caller-supplied free text and
    must not be able to terminate the literal.
    """
    return "'" + value.replace("'", "''") + "'"


def _redis_glob_escape(value: str) -> str:
    """Escape Redis glob metacharacters so an id only matches itself."""
    return re.sub(r"([\\*?\[\]])", r"\\\1", value)


def _fsync_parent_dir(path: str) -> None:
    """Make a newly created file's directory entry crash-durable."""
    parent = os.path.dirname(os.path.abspath(path)) or "/"
    dirfd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _checksums(conn, conversation_id: str) -> dict:
    seg = conn.execute(
        "SELECT count(*) AS n, "
        "md5(COALESCE(string_agg(ref || summary, '' ORDER BY ref), '')) AS digest "
        "FROM segments WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    conv = conn.execute(
        "SELECT count(*) AS n, max(updated_at)::text AS max_updated "
        "FROM conversations WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    return {
        "segments": {"count": seg["n"], "md5": seg["digest"]},
        "conversations": {"count": conv["n"], "max_updated": conv["max_updated"]},
    }


def _dry_run(args) -> None:
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)",
              args.conversation_id)
    sql = _selection_sql(
        args.include_short, args.since, args.until, args.after_ref,
    )
    params = _params(args)
    try:
        with _connect(dsn, read_only=True) as conn:
            before = _checksums(conn, args.conversation_id)
            rows = conn.execute(sql, params).fetchall()
            # Equality-overlap probe: run the SAME selection SQL with the
            # equality condition appended. Structurally zero while the
            # predicate is strict; nonzero means a future edit broke
            # strictness and would paraphrase intentional passthrough
            # stubs. Probing with the identical SQL string is the point:
            # a probe with its own predicate could pass while the real
            # one regressed.
            overlap = conn.execute(
                f"SELECT count(*) AS n FROM ({sql}) sel "
                "WHERE sel.summary = sel.full_text",
                params,
            ).fetchone()["n"]
            after = _checksums(conn, args.conversation_id)
    except Exception as exc:  # noqa: BLE001
        _fail("dry_run", repr(exc), args.conversation_id)
    print(json.dumps({
        "status": "dry_run",
        "conversation_id": args.conversation_id,
        "selected": len(rows),
        "selection_equality_overlap": overlap,
        "include_short": bool(args.include_short),
        "first_ref": rows[0]["ref"] if rows else None,
        "last_ref": rows[-1]["ref"] if rows else None,
        "checksums_before": before,
        "checksums_after": after,
        "checksums_stable": before == after,
        "note": "write-time filter: --since/--until bound created_at, the "
                "row's last WRITE, not its content time; do not use them "
                "to target a damage epoch",
    }, indent=2))


def _usage_totals(totals: dict, usage: dict) -> None:
    for key in ("input_tokens", "prompt_tokens"):
        if isinstance(usage.get(key), (int, float)):
            totals["tokens_in"] += int(usage[key])
            break
    for key in ("output_tokens", "completion_tokens"):
        if isinstance(usage.get(key), (int, float)):
            totals["tokens_out"] += int(usage[key])
            break


def _print_cascade_runbook(conversation_id: str, tags: list[str]) -> None:
    cid = _sql_str(conversation_id)
    tag_list = ", ".join(_sql_str(t) for t in sorted(tags))
    cid_arg = shlex.quote(conversation_id)
    emb_key = shlex.quote(f"vc:tag_summary_embeddings:{conversation_id}")
    stats_key = shlex.quote(f"vc:tag_stats:{conversation_id}")
    hint_glob = shlex.quote(
        f"vc:context_hint:{_redis_glob_escape(conversation_id)}:*",
    )
    print("\n=== CASCADE RUNBOOK (not executed; run each step, then its check) ===")
    print(f"# affected tags: {sorted(tags)}")
    print("\n# 1. Two-table targeted delete (tag summaries AND their embeddings):")
    print(f"DELETE FROM tag_summary_embeddings WHERE conversation_id = {cid} AND tag IN ({tag_list});")
    print(f"DELETE FROM tag_summaries WHERE conversation_id = {cid} AND tag IN ({tag_list});")
    print("#    VERIFY (expect 0 and 0):")
    print(f"SELECT count(*) FROM tag_summaries WHERE conversation_id = {cid} AND tag IN ({tag_list});")
    print(f"SELECT count(*) FROM tag_summary_embeddings WHERE conversation_id = {cid} AND tag IN ({tag_list});")
    print("\n# 2. Backfill (skip-existing regenerates exactly the deleted tags):")
    print(f"virtual-context admin backfill-tag-summaries {cid_arg} --tenant-id <tenant>")
    print("#    VERIFY (expect one fresh row per affected tag):")
    print(f"SELECT tag, updated_at FROM tag_summaries WHERE conversation_id = {cid} AND tag IN ({tag_list}) ORDER BY tag;")
    print("\n# 3. Redis invalidation (embedding snapshot, context hints, tag stats):")
    print(f"redis-cli DEL {emb_key} {stats_key}")
    print(f"redis-cli --scan --pattern {hint_glob} | xargs -r redis-cli DEL")
    print("#    VERIFY (expect 0, 0, and no keys):")
    print(f"redis-cli EXISTS {emb_key}")
    print(f"redis-cli EXISTS {stats_key}")
    print(f"redis-cli --scan --pattern {hint_glob}")
    print("\n# 4. Worker recycle (process-local caches have no expiry):")
    print("#    recycle the serving workers, then VERIFY start times are post-recycle:")
    print("ps -o pid,lstart,command -C python | grep -i uvicorn")
    print("=== END RUNBOOK: staleness persists until ALL FOUR steps verify ===")


def cmd_admin_resummarize_segments(args) -> None:
    if not args.apply:
        _dry_run(args)
        return

    from .main import _apply_storage_overrides, load_config
    from ..core.segment_repair import (
        Malformed,
        ProviderFailure,
        summarize_segment_once,
    )
    from ..types import TaggedSegment as _Seg

    conversation_id = args.conversation_id
    dsn = _resolve_dsn(args)
    if not dsn:
        _fail("resolve_dsn", "no Postgres DSN (--postgres-dsn or DATABASE_URL)",
              conversation_id)

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        _fail("load_config", repr(exc), conversation_id)
    config.conversation_id = conversation_id
    if getattr(args, "tenant_id", ""):
        config.tenant_id = args.tenant_id
    _apply_storage_overrides(config, args)
    # The engine and the repair connection MUST target the same store.
    # The override helper deliberately ignores DATABASE_URL when a -c
    # config was supplied, so a config carrying its own storage section
    # could otherwise build the engine against one database while the
    # CAS writes land in another. This command is Postgres-only by
    # construction (xmin), and its store is the resolved DSN, full stop.
    config.storage.backend = "postgres"
    config.storage.postgres_dsn = dsn

    from ..engine import VirtualContextEngine

    try:
        engine = VirtualContextEngine(config=config)
    except Exception as exc:  # noqa: BLE001
        _fail("engine_construct", repr(exc), conversation_id)
    compactor = getattr(engine, "_compactor", None)
    if compactor is None or getattr(compactor, "llm", None) is None:
        _fail("provider_check", "no summarization provider configured",
              conversation_id)
    max_summary_tokens = compactor.config.max_summary_tokens

    journal_path = args.journal or (
        f"resummarize-journal-{conversation_id.replace('/', '_').replace(':', '_')}.jsonl"
    )
    breaker_limit = args.max_consecutive_provider_failures

    counts = {
        "accepted": 0, "skipped_concurrent": 0, "provider_failure": 0,
        "malformed": 0,
    }
    rejected: dict[str, int] = {}
    totals = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    affected_tags: set[str] = set()
    consecutive_provider_failures = 0
    status = "completed"
    last_ref = None
    # The resume cursor advances only past rows that received a provider
    # RESPONSE. A row whose call failed (including the one that trips
    # the breaker) stays ahead of the cursor, so a resume retries it
    # instead of silently skipping unrepaired damage.
    resume_after_ref = args.after_ref

    try:
        with _connect(dsn, read_only=False) as conn, \
                open(journal_path, "a", encoding="utf-8") as journal:
            _fsync_parent_dir(journal_path)
            sql = _selection_sql(
                args.include_short, args.since, args.until, args.after_ref,
            )
            rows = conn.execute(sql, _params(args)).fetchall()
            for row in rows:
                if args.limit is not None and counts["accepted"] >= args.limit:
                    status = "limit_reached"
                    break
                last_ref = row["ref"]
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except ValueError:
                    metadata = {}
                segment = _Seg(
                    id=row["ref"],
                    primary_tag=row["primary_tag"],
                    tags=list(row["tags"] or [row["primary_tag"]]),
                    messages=[],
                    token_count=row["full_tokens"],
                    turn_count=int(metadata.get("turn_count") or 0),
                    session_date=str(metadata.get("session_date") or ""),
                )
                # No prev_context: repair has no neighboring-segment window,
                # and the pronoun-resolution block imports exactly the
                # content the validator's overshoot check guards against.
                request = compactor.build_segment_summary_request(
                    segment, conversation_text=row["full_text"],
                )
                outcome = summarize_segment_once(compactor, request)

                if isinstance(outcome, ProviderFailure):
                    counts["provider_failure"] += 1
                    consecutive_provider_failures += 1
                    if consecutive_provider_failures >= breaker_limit:
                        status = "aborted_provider_down"
                        break
                    continue
                consecutive_provider_failures = 0
                totals["calls"] += 1
                resume_after_ref = row["ref"]
                if isinstance(outcome, Malformed):
                    counts["malformed"] += 1
                    _usage_totals(totals, outcome.usage)
                    continue
                _usage_totals(totals, outcome.usage)
                reject = classify_generated(
                    outcome.summary, row["full_text"],
                    compactor.token_counter, max_summary_tokens,
                )
                if reject is not None:
                    rejected[reject] = rejected.get(reject, 0) + 1
                    continue

                new_tokens = compactor.token_counter(outcome.summary)
                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "conversation_id": conversation_id,
                    "ref": row["ref"],
                    "row_version": row["row_version"],
                    "old_summary_sha256": hashlib.sha256(
                        row["summary"].encode()).hexdigest(),
                    "new_summary_sha256": hashlib.sha256(
                        outcome.summary.encode()).hexdigest(),
                    "new_summary_tokens": new_tokens,
                    "tags": sorted(segment.tags),
                }
                # Write-ahead: the journal entry is durable BEFORE the CAS,
                # so committed repairs are always a subset of journal
                # entries; a crash can only leave harmless over-invalidation.
                journal.write(json.dumps(entry) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

                updated = conn.execute(
                    "UPDATE segments SET summary = %s, summary_tokens = %s, "
                    "compression_ratio = CASE WHEN full_tokens > 0 "
                    "THEN %s::real / full_tokens ELSE 0.0 END "
                    "WHERE ref = %s AND conversation_id = %s "
                    "AND xmin::text = %s",
                    (outcome.summary, new_tokens, float(new_tokens),
                     row["ref"], conversation_id, row["row_version"]),
                ).rowcount
                if updated == 1:
                    counts["accepted"] += 1
                    affected_tags.update(segment.tags)
                elif updated == 0:
                    counts["skipped_concurrent"] += 1
                else:  # pragma: no cover - PK guarantees <= 1
                    raise RuntimeError(
                        f"CAS touched {updated} rows for ref {row['ref']}",
                    )
    except Exception as exc:  # noqa: BLE001
        _fail("apply", repr(exc), conversation_id)
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass

    print(json.dumps({
        "status": status,
        "conversation_id": conversation_id,
        "selected": len(rows),
        "counts": counts,
        "rejected": rejected,
        "usage": totals,
        "journal": journal_path,
        "last_attempted_ref": last_ref,
        "resume_after_ref": resume_after_ref,
        "note": ("skipped_concurrent is NORMAL on an active conversation: "
                 "live compaction rewrites rows mid-run; re-running is the "
                 "intended completion path and is safe by idempotency"),
    }, indent=2))
    if counts["accepted"]:
        _print_cascade_runbook(conversation_id, sorted(affected_tags))
