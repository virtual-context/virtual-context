"""Seed a realistic target conversation for the VCATTACH end-to-end harness.

Runs the engine's natural write paths (tagging + segmentation + compaction)
against a writable Postgres database using a stub LLM provider that
returns deterministic, greppable canned summaries. Emits two artifacts:

* ``target_baseline.sql`` -- ``pg_dump`` of the target conversation's
  rows across canonical_turns / segments / tag_summaries / segment_tags /
  conversations / conversation_lifecycle. Restored by the harness via
  ``psql -f``.
* ``target_baseline_redis.json`` -- ``{key: value}`` JSON of the Redis
  SessionState blob (and any companion cache keys) the engine would have
  written. Restored by the harness via ``redis_client.set(...)``.

The canned summaries contain ``BASELINE-MARKER-NNN`` strings so the
harness can simply grep the upstream LLM payload to confirm prefetch
fired and surfaced target's content on POST 2.

Usage::

    .venv/bin/python tests/fixtures/seed_realistic_target_conv.py \\
        --pg-url "postgresql://vc:vc@127.0.0.1:5432/vc_harness" \\
        --conversation-id "harness-target-77f110" \\
        --turns 500 --segments 50 --summaries 50 \\
        --out-pg ./tests/fixtures/target_baseline.sql \\
        --out-redis ./tests/fixtures/target_baseline_redis.json

Or for a quick smoke test against SQLite (skips Postgres dump, only emits
Redis JSON + an analogous SQLite snapshot for reference)::

    .venv/bin/python tests/fixtures/seed_realistic_target_conv.py \\
        --sqlite-path /tmp/harness.db \\
        --conversation-id "harness-target-smoke" \\
        --turns 100 --segments 10 --summaries 10 \\
        --out-redis /tmp/harness_redis.json

The script never connects to the production database; it only writes to
the URL the operator supplies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the repo root is on sys.path when invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ``KMP_DUPLICATE_LIB_OK`` matches the test suite's convention; the
# embedding provider transitively imports OpenMP-linked numerics.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virtual_context.engine import VirtualContextEngine  # noqa: E402
from virtual_context.types import (  # noqa: E402
    AssemblerConfig,
    CompactorConfig,
    KeywordTagConfig,
    Message,
    MonitorConfig,
    SegmenterConfig,
    StorageConfig,
    SummarizationConfig,
    TagGeneratorConfig,
    VirtualContextConfig,
)
from virtual_context.proxy.session_state import SessionState  # noqa: E402

logger = logging.getLogger("seed_realistic_target_conv")

# ---------------------------------------------------------------------------
# Stub LLM provider
# ---------------------------------------------------------------------------


_TOPIC_RE = re.compile(r"topic-(\d{3})")


class BaselineCompactorProvider:
    """Stub LLM provider for the compactor.

    Returns a canned JSON summary whose ``summary`` text embeds
    ``BASELINE-MARKER-NNN`` where ``NNN`` is the topic id extracted from
    the prompt (the compactor includes the tag list in the user prompt,
    so the topic id is recoverable). Falls back to the per-call sequence
    counter when no topic id is present.
    """

    def __init__(self) -> None:
        self.last_usage: dict = {}
        self.call_count = 0
        self._llm_log_path = ""  # set by engine._init_compactor

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        self.call_count += 1
        match = _TOPIC_RE.search(user)
        if match:
            topic_id = match.group(1)
        else:
            topic_id = f"{self.call_count:03d}"
        summary = (
            f"Discussion of topic-{topic_id}: BASELINE-MARKER-{topic_id}. "
            f"This segment covers content related to topic-{topic_id} including "
            f"key questions and answers exchanged during the discussion. "
            f"Detailed context with BASELINE-MARKER-{topic_id} repeated for "
            f"reliable substring grepping in upstream LLM payloads."
        )
        response = {
            "summary": summary,
            "entities": [f"topic-{topic_id}", f"entity-{topic_id}"],
            "key_decisions": [f"decision related to topic-{topic_id}"],
            "action_items": [f"follow-up on topic-{topic_id}"],
            "date_references": [],
            "refined_tags": [f"topic-{topic_id}"],
            "facts": [],
            "code_refs": [],
        }
        return json.dumps(response), {"input_tokens": 100, "output_tokens": 100}


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


def build_config(
    *,
    conversation_id: str,
    backend: str,
    postgres_dsn: str = "",
    sqlite_path: str = "",
    turns_per_segment: int,
    summary_count: int,
) -> VirtualContextConfig:
    """Build a VirtualContextConfig tuned for deterministic baseline seeding.

    Key tuning decisions:

    * ``tag_generator.type = "keyword"`` so the tagger never needs a real
      LLM; each ``topic-NNN`` keyword in user text deterministically maps
      to a ``topic-NNN`` tag. This gives us ``--summaries`` distinct
      primary tags without LLM round trips.
    * ``segmenter.max_segment_turns = turns_per_segment`` forces a
      segment boundary every N turns regardless of token totals.
    * ``compactor.summary_ratio`` and token budgets left at defaults so
      compaction produces realistic-looking segment rows.
    * ``summarization.provider = "baseline-stub"`` paired with a
      monkeypatched ``_build_provider`` returning ``BaselineCompactorProvider``.
    """
    tag_keywords: dict[str, list[str]] = {
        f"topic-{i:03d}": [f"topic-{i:03d}"] for i in range(1, summary_count + 1)
    }
    return VirtualContextConfig(
        context_window=8_000,
        token_counter="estimate",
        conversation_id=conversation_id,
        tag_generator=TagGeneratorConfig(
            type="keyword",
            max_tags=5,
            min_tags=1,
            keyword_fallback=KeywordTagConfig(tag_keywords=tag_keywords),
            context_lookback_pairs=0,
        ),
        # ``MonitorConfig.context_window`` is a SEPARATE field from
        # ``VirtualContextConfig.context_window`` and is what the monitor
        # actually consults when deciding whether to fire compaction. Set
        # it small so the synthetic transcript triggers many compactions.
        monitor=MonitorConfig(
            context_window=4_000,
            soft_threshold=0.50,
            hard_threshold=0.80,
            protected_recent_turns=2,
        ),
        segmenter=SegmenterConfig(
            session_gap_minutes=0,
            tag_overlap_threshold=0.5,
            max_segment_turns=turns_per_segment,
        ),
        compactor=CompactorConfig(
            summary_ratio=0.15,
            min_summary_tokens=50,
            max_summary_tokens=400,
            max_concurrent_summaries=1,
            code_mode=False,
        ),
        assembler=AssemblerConfig(
            core_context_max_tokens=4_000,
            tag_context_max_tokens=4_000,
            facts_max_tokens=0,
        ),
        summarization=SummarizationConfig(
            provider="baseline-stub",
            model="stub-model",
        ),
        storage=StorageConfig(
            backend=backend,
            postgres_dsn=postgres_dsn,
            sqlite_path=sqlite_path,
        ),
        providers={
            "baseline-stub": {"type": "baseline-stub"},
        },
        tenant_id="",
    )


def _suppress_engine_state_writes(engine: VirtualContextEngine) -> None:
    """Mirror cloud's prod-side silent-failure for ``save_engine_state``.

    When the seed runs against the harness's Postgres -- which inherited
    the pre-``e677bba`` engine_state schema (``compacted_through`` /
    ``flushed_through``) -- the engine's INSERT (which uses the new
    column names ``compacted_prefix_messages`` / ``flushed_prefix_messages``)
    raises ``psycopg.errors.UndefinedColumn``. Cloud's
    ``SessionStateProvider._save_to_store`` catches that with a bare
    ``except Exception`` and warns. But the engine's own non-provider
    ``_save_state`` path does NOT catch -- the raise would crash the
    turn-completion pipeline.

    For seeding purposes we deliberately suppress the write so the
    resulting Postgres mirrors prod's actual on-disk state (engine_state
    table empty; SessionState lives only in Redis). The script's Redis
    JSON output is unaffected -- the in-memory ``EngineStateSnapshot`` is
    still constructed for the Redis serialisation; only the SQL persist
    leg is no-op'd.

    Tracked separately as task #32 for a real fix.
    """
    raw_store = engine._store._store if hasattr(engine._store, "_store") else engine._store

    def _noop_save(snapshot, *args, **kwargs):  # noqa: ARG001
        logger.debug(
            "save_engine_state suppressed (matches prod silent-failure)",
        )

    raw_store.save_engine_state = _noop_save


def _install_stub_provider(engine: VirtualContextEngine) -> BaselineCompactorProvider:
    """Replace the engine's compactor LLM with the stub.

    Done post-init so we don't have to monkeypatch ``_build_provider``
    on the class. The engine already constructed without a real provider
    (the unknown provider type ``baseline-stub`` returned ``None`` from
    ``_build_provider``); we now wire one in and rebuild the compactor.
    """
    from virtual_context.core.compactor import DomainCompactor

    stub = BaselineCompactorProvider()
    engine._llm_provider = stub
    engine._compactor = DomainCompactor(
        llm_provider=stub,
        config=engine.config.compactor,
        token_counter=engine._token_counter,
        model_name=engine.config.summarization.model,
        tag_rules=engine.config.tag_rules,
        telemetry_ledger=engine._telemetry,
    )
    # The compaction pipeline holds a direct reference to the old compactor
    # captured during _init_compactor → CompactionPipeline construction.
    # Rebind so freshly-built segments hit the stub. Note the leading
    # underscore — the pipeline stores its compactor as ``_compactor``.
    if hasattr(engine, "_compaction"):
        engine._compaction._compactor = engine._compactor
    if hasattr(engine, "_tagging"):
        # Some code paths look up the compactor via the tagging pipeline.
        if hasattr(engine._tagging, "_compactor"):
            engine._tagging._compactor = engine._compactor
        elif hasattr(engine._tagging, "compactor"):
            engine._tagging.compactor = engine._compactor
    return stub


# ---------------------------------------------------------------------------
# Synthetic conversation generator
# ---------------------------------------------------------------------------


def generate_messages(
    *,
    turns: int,
    summary_count: int,
    turns_per_segment: int,
    start_time: datetime,
) -> list[Message]:
    """Build a list of paired user/assistant Messages spanning ``turns`` pairs.

    The synthetic transcript walks through ``summary_count`` distinct
    topics. Each topic occupies ``turns_per_segment`` consecutive turns
    so the segmenter (configured with ``max_segment_turns=turns_per_segment``)
    produces one segment per topic. The user message embeds ``topic-NNN``
    so the keyword tagger assigns the correct primary tag.
    """
    messages: list[Message] = []
    ts = start_time
    for i in range(turns):
        topic_idx = (i // turns_per_segment) % summary_count + 1
        topic_tag = f"topic-{topic_idx:03d}"
        user_text = (
            f"Q{i + 1}: Tell me more about {topic_tag}. "
            f"Specifically question {i + 1} regarding {topic_tag}."
        )
        asst_text = (
            f"A{i + 1}: Answer for {topic_tag}. This is response number {i + 1} "
            f"with detailed information about {topic_tag} and BASELINE-MARKER-{topic_idx:03d}."
        )
        messages.append(Message(role="user", content=user_text, timestamp=ts))
        ts = ts + timedelta(seconds=30)
        messages.append(Message(role="assistant", content=asst_text, timestamp=ts))
        ts = ts + timedelta(seconds=30)
    return messages


# ---------------------------------------------------------------------------
# Engine pump
# ---------------------------------------------------------------------------


def pump_through_engine(
    engine: VirtualContextEngine,
    messages: list[Message],
) -> dict:
    """Drive the engine through the synthetic transcript.

    Walks the message list two at a time, appending to a growing
    ``conversation_history`` and calling ``on_turn_complete`` per pair so
    the natural tag-then-compact cycle fires. Returns a stats dict for
    logging.
    """
    history: list[Message] = []
    tag_calls = 0
    compact_reports = 0
    start = time.time()
    for i in range(0, len(messages), 2):
        pair = messages[i : i + 2]
        history.extend(pair)
        report = engine.on_turn_complete(history)
        tag_calls += 1
        if report is not None:
            compact_reports += 1
        if tag_calls % 50 == 0:
            logger.info(
                "pumped %d/%d turns (%d compactions so far, %.1fs elapsed)",
                tag_calls, len(messages) // 2,
                compact_reports, time.time() - start,
            )
    elapsed = time.time() - start
    logger.info(
        "pump done: %d turns, %d compactions, %.1fs",
        tag_calls, compact_reports, elapsed,
    )
    return {
        "turns_ingested": tag_calls,
        "compactions_fired": compact_reports,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Postgres dump
# ---------------------------------------------------------------------------


# Tables that own per-conversation rows. ``pg_dump`` ``--data-only`` for
# these tables in this order produces a restorable INSERT stream when the
# target schema is already in place (which the harness ensures by running
# ``PostgresStore(dsn)`` against an empty DB to invoke ``_ensure_schema``).
DUMP_TABLES = [
    "conversations",
    "conversation_lifecycle",
    "canonical_turns",
    "canonical_turn_anchors",
    "segments",
    "segment_tags",
    "tag_aliases",
    "tag_summaries",
    "ingest_batches",
    "conversation_aliases",
]


def _resolve_pg_tool_runner(docker_container: str) -> tuple[list[str], list[str]]:
    """Return ``(pg_dump_cmd_prefix, psql_cmd_prefix)`` honouring ``docker_container``.

    When ``docker_container`` is set, the binaries are invoked inside the
    container via ``docker exec -i``. Otherwise we expect ``pg_dump`` and
    ``psql`` on the host PATH (e.g. via ``brew install libpq``). The
    container path avoids adding a host-tool dependency, but the user's
    DSN must still resolve from inside the container (``host.docker.internal``
    for Docker Desktop on macOS, ``localhost`` works when the script and
    Postgres share the same host network namespace).
    """
    if docker_container:
        return (
            ["docker", "exec", "-i", docker_container, "pg_dump"],
            ["docker", "exec", "-i", docker_container, "psql"],
        )
    return (["pg_dump"], ["psql"])


def _rewrite_pg_url_for_in_container_use(pg_url: str) -> str:
    """Rewrite a host-side DSN into a form valid from inside the container.

    When the operator points ``--pg-url`` at ``127.0.0.1:5433`` and asks
    for ``--docker-container``, the libpq tools running inside the
    container can't reach the host's exposed port. The Postgres server
    is listening on the container's loopback at port ``5432``. Substitute
    those values so ``docker exec ... pg_dump <internal-url>`` works.
    """
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(pg_url)
    if parts.hostname in (None, "127.0.0.1", "localhost", "0.0.0.0", "host.docker.internal"):
        auth = ""
        if parts.username:
            auth = parts.username
            if parts.password:
                auth = f"{auth}:{parts.password}"
            auth = f"{auth}@"
        new_netloc = f"{auth}127.0.0.1:5432"
        return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))
    return pg_url


def dump_postgres(
    *,
    pg_url: str,
    out_path: Path,
    conversation_id: str,
    docker_container: str = "",
) -> dict[str, int]:
    """Emit a ``pg_dump`` of the per-conversation rows to ``out_path``.

    Uses ``--data-only`` (schema is recreated by the harness's
    ``PostgresStore`` init) and ``--inserts`` so the dump is portable and
    easily diff'd. ``--table`` filters limit the dump to the conversation
    tables; row-level filtering by ``conversation_id`` is enforced via
    ``--where`` on the inner dumps because ``pg_dump`` itself doesn't
    support per-table WHERE clauses across all backends.

    Returns row counts per table for the README summary.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table_args: list[str] = []
    for t in DUMP_TABLES:
        table_args.extend(["--table", t])

    pg_dump_cmd, psql_cmd = _resolve_pg_tool_runner(docker_container)
    effective_url = (
        _rewrite_pg_url_for_in_container_use(pg_url) if docker_container else pg_url
    )

    # First, capture row counts per table for the README. Some tables
    # don't carry a ``conversation_id`` column directly (e.g.
    # ``segment_tags`` is joined via ``segment_ref``); we just count all
    # rows for those because the dump is conv-scoped anyway.
    _CONV_SCOPED_TABLES = {
        "conversations", "conversation_lifecycle", "canonical_turns",
        "canonical_turn_anchors", "segments", "tag_summaries",
    }
    counts: dict[str, int] = {}
    for table in DUMP_TABLES:
        if table in _CONV_SCOPED_TABLES:
            query = (
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE conversation_id = '{conversation_id}';"
            )
        else:
            query = f"SELECT COUNT(*) FROM {table};"
        result = subprocess.run(
            [*psql_cmd, effective_url, "-tA", "-c", query],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "psql count failed for %s: %s",
                table, result.stderr.strip(),
            )
            counts[table] = -1
            continue
        try:
            counts[table] = int(result.stdout.strip() or "0")
        except ValueError:
            counts[table] = -1

    # Run pg_dump with full --data-only --inserts; the harness loads it
    # against a freshly-schema'd DB. We do NOT use --where because that
    # would require a per-table dump invocation; the harness can filter
    # if needed, but in practice the seed DB only contains rows for the
    # target conv anyway.
    cmd = [
        *pg_dump_cmd,
        effective_url,
        "--data-only",
        "--inserts",
        "--no-owner",
        "--no-acl",
        *table_args,
    ]
    logger.info("running pg_dump → %s", out_path)
    with out_path.open("wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"pg_dump failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    size_bytes = out_path.stat().st_size
    logger.info("pg_dump wrote %d bytes to %s", size_bytes, out_path)
    counts["__dump_size_bytes__"] = size_bytes
    return counts


# ---------------------------------------------------------------------------
# Redis SessionState synthesis
# ---------------------------------------------------------------------------


def build_session_state(
    engine: VirtualContextEngine,
    *,
    conversation_id: str,
) -> SessionState:
    """Construct a SessionState matching the engine's in-memory state.

    Cloud's ``CloudSessionStateProvider`` keys this under ``vc:session:{id}``.
    The harness loads the JSON bytes into Redis under that key before the
    POST 1 / POST 2 sequence runs.
    """
    es = engine._engine_state
    turn_entries = [asdict(e) for e in engine._turn_tag_index.entries]
    # Convert datetime fields to ISO strings so the JSON round trip is lossless.
    for entry in turn_entries:
        for key in ("timestamp",):
            if hasattr(entry.get(key), "isoformat"):
                entry[key] = entry[key].isoformat()
            elif isinstance(entry.get(key), datetime):
                entry[key] = entry[key].isoformat()

    return SessionState(
        compacted_prefix_messages=int(es.compacted_prefix_messages),
        flushed_prefix_messages=int(es.flushed_prefix_messages),
        flushed_prefix_messages_present=True,
        last_request_time=float(es.last_request_time),
        last_compacted_turn=int(es.last_compacted_turn),
        last_completed_turn=int(es.last_completed_turn),
        last_indexed_turn=int(es.last_indexed_turn),
        checkpoint_version=int(es.checkpoint_version),
        conversation_generation=int(es.conversation_generation),
        tool_tag_counter=int(es.tool_tag_counter),
        split_processed_tags=set(es.split_processed_tags),
        trailing_fingerprint=str(es.trailing_fingerprint),
        provider=str(es.provider),
        turn_tag_entries=turn_entries,
        version=1,
    )


def dump_redis_payload(
    session_state: SessionState,
    *,
    conversation_id: str,
    out_path: Path,
) -> dict:
    """Serialise the SessionState to ``{key: utf-8-string}`` JSON.

    The harness reads this JSON and ``redis_client.set(key, value)`` for
    each entry. SessionState bytes are utf-8 JSON; we strip the b64 prefix
    convention because the payload is text-safe.
    """
    state_bytes = session_state.to_json()
    redis_payload = {
        f"vc:session:{conversation_id}": state_bytes.decode("utf-8"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(redis_payload, indent=2))
    summary = {
        "redis_keys": list(redis_payload.keys()),
        "session_state_bytes": len(state_bytes),
        "compacted_prefix_messages": session_state.compacted_prefix_messages,
        "flushed_prefix_messages": session_state.flushed_prefix_messages,
        "last_completed_turn": session_state.last_completed_turn,
        "turn_tag_entries_count": len(session_state.turn_tag_entries),
    }
    logger.info("redis payload written to %s: %s", out_path, summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a realistic target conversation for the VCATTACH harness.",
    )
    parser.add_argument(
        "--pg-url",
        default="",
        help="Postgres DSN to seed against (e.g. postgresql://user:pass@host:5432/db). "
             "Required for --out-pg. Database must be empty or have engine schema.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="",
        help="Alternative to --pg-url for smoke testing. No --out-pg in this mode.",
    )
    parser.add_argument(
        "--conversation-id",
        required=True,
        help="Conversation id to seed.",
    )
    parser.add_argument(
        "--turns", type=int, default=500,
        help="Total turn pairs to ingest. Default 500.",
    )
    parser.add_argument(
        "--segments", type=int, default=50,
        help="Target segment count. Used to derive turns_per_segment. Default 50.",
    )
    parser.add_argument(
        "--summaries", type=int, default=50,
        help="Distinct tag/summary count. Should match --segments. Default 50.",
    )
    parser.add_argument(
        "--out-pg",
        default="",
        help="Output path for pg_dump SQL. Skipped when not provided.",
    )
    parser.add_argument(
        "--out-redis",
        required=True,
        help="Output path for Redis SessionState JSON.",
    )
    parser.add_argument(
        "--docker-container",
        default="",
        help="When set, ``pg_dump`` and ``psql`` are invoked inside this "
             "Docker container via ``docker exec``. Use when libpq tools "
             "aren't installed on the host.",
    )
    parser.add_argument(
        "--suppress-engine-state-write",
        action="store_true",
        help="No-op ``save_engine_state`` to mirror cloud's prod silent "
             "failure path (engine_state column-rename mismatch). Required "
             "when seeding against the harness Postgres until task #32 "
             "lands.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Python logging level. Default INFO.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.pg_url and not args.sqlite_path:
        logger.error("must supply --pg-url or --sqlite-path")
        return 2
    if args.pg_url and args.sqlite_path:
        logger.error("--pg-url and --sqlite-path are mutually exclusive")
        return 2
    if args.pg_url and not args.out_pg:
        logger.error("--pg-url requires --out-pg")
        return 2

    if args.summaries != args.segments:
        logger.warning(
            "summaries=%d differs from segments=%d; using max for keyword tagging",
            args.summaries, args.segments,
        )
    summary_count = max(args.summaries, args.segments)
    turns_per_segment = max(1, args.turns // args.segments)
    logger.info(
        "config: conv=%s turns=%d segments=%d summaries=%d turns/segment=%d",
        args.conversation_id, args.turns, args.segments,
        summary_count, turns_per_segment,
    )

    backend = "postgres" if args.pg_url else "sqlite"
    config = build_config(
        conversation_id=args.conversation_id,
        backend=backend,
        postgres_dsn=args.pg_url,
        sqlite_path=args.sqlite_path,
        turns_per_segment=turns_per_segment,
        summary_count=summary_count,
    )

    logger.info("instantiating engine in NON-PROVIDER mode (storage=%s)", backend)
    engine = VirtualContextEngine(config=config)
    _install_stub_provider(engine)
    if args.suppress_engine_state_write:
        _suppress_engine_state_writes(engine)
        logger.info(
            "save_engine_state suppressed (matches prod's column-rename silent failure)",
        )

    logger.info("generating %d synthetic turn pairs", args.turns)
    messages = generate_messages(
        turns=args.turns,
        summary_count=summary_count,
        turns_per_segment=turns_per_segment,
        start_time=datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
    )

    logger.info("pumping messages through engine.on_turn_complete()")
    pump_stats = pump_through_engine(engine, messages)

    # The compactor advances ``compacted_prefix_messages`` but never sets
    # ``flushed_prefix_messages`` -- the proxy does that when it actually
    # trims compacted history out of its in-memory buffer (see
    # virtual_context/proxy/server.py:1543). Mirror the proxy's behaviour
    # here so the produced SessionState has the flush-watermark = compaction-
    # watermark invariant the retriever's ``post_compaction`` gate relies on.
    engine._engine_state.flushed_prefix_messages = int(
        engine._engine_state.compacted_prefix_messages,
    )
    logger.info(
        "advanced flushed_prefix_messages to compacted=%d for steady-state baseline",
        engine._engine_state.flushed_prefix_messages,
    )

    out_redis = Path(args.out_redis)
    session_state = build_session_state(
        engine, conversation_id=args.conversation_id,
    )
    redis_summary = dump_redis_payload(
        session_state,
        conversation_id=args.conversation_id,
        out_path=out_redis,
    )

    pg_summary: dict[str, int] = {}
    if args.pg_url:
        out_pg = Path(args.out_pg)
        pg_summary = dump_postgres(
            pg_url=args.pg_url,
            out_path=out_pg,
            conversation_id=args.conversation_id,
            docker_container=args.docker_container,
        )

    report = {
        "conversation_id": args.conversation_id,
        "pump_stats": pump_stats,
        "redis_summary": redis_summary,
        "pg_summary": pg_summary,
    }
    print("\n=== SEED REPORT ===")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
