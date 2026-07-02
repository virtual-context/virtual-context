"""admin retag-canonical-turns — re-tag fallback-tagged canonical rows.

Rows tagged during degraded windows (dead LLM tagger, row-sweep
fallback) carry ``_general`` primary tags even for substantive content,
because the row sweep tags rows in isolation with no conversational
lookback. This command re-tags them the way the healthy pair path
does: batched into logical turns, with preceding-pair context and the
bleed gate, through the configured tag generator.

Semantics pinned here:
* pair context — the generator receives preceding-pair context turns
* never downgrade — a fallback/empty generator result leaves the row as-is
* selection — ``--only-general`` + half-open [since, until) window on
  ``created_at``; space-separated ISO input is normalized to the
  T-separated TEXT format the rows use
* idempotent — a second run selects nothing once rows carry real tags
* dry-run — full report, zero writes
* no index/Redis writes — cross-worker coherence is composed via the
  existing backfill-session-state-markers pass
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sqlite3
from pathlib import Path

import pytest

from virtual_context.types import TagResult


def _make_engine(tmp_path: Path, conversation_id: str = "c"):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine
    cfg = load_config(config_dict={
        "context_window": 10000,
        "conversation_id": conversation_id,
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(tmp_path / f"{conversation_id}.db")},
        },
        "tag_generator": {
            "type": "keyword",
            "keyword_fallback": {
                "tag_keywords": {
                    "alpine-gardening": ["saxifrage", "cushion", "alpine"],
                },
            },
        },
    })
    return VirtualContextEngine(config=cfg)


def _seed_general_rows(engine, tmp_path: Path, n_pairs: int = 3,
                       created_at: str | None = None):
    """Persist n_pairs of rows and stamp them all _general (the degraded
    row-sweep outcome)."""
    from virtual_context.proxy.formats import detect_format
    body = {"messages": []}
    for i in range(n_pairs):
        body["messages"] += [
            {"role": "user", "content": f"tell me about saxifrage cushion topic {i}"},
            {"role": "assistant", "content": f"saxifrage answer number {i} with detail"},
        ]
    engine._ingest_reconciler.ingest_batch(
        engine.config.conversation_id, body=body, fmt=detect_format(body),
        expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
    )
    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        conn.execute(
            "UPDATE canonical_turns SET primary_tag = '_general', "
            "tags_json = '[\"_general\"]', tagged_at = '2026-06-20T00:00:00+00:00'"
            + (f", created_at = '{created_at}'" if created_at else "")
            + " WHERE conversation_id = ?",
            (engine.config.conversation_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _rows(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "c.db")
    try:
        return [
            {"content": r[0] or r[1], "primary": r[2], "tagged_at": r[3],
             "created_at": r[4]}
            for r in conn.execute(
                "SELECT user_content, assistant_content, primary_tag, "
                "tagged_at, created_at FROM canonical_turns "
                "WHERE conversation_id = 'c' ORDER BY sort_key"
            )
        ]
    finally:
        conn.close()


def test_retag_replaces_general_with_real_tags(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
        report = engine.retag_canonical_turns()
        assert report["retagged_pairs"] == 3
        assert report["rows_updated"] == 6
        for row in _rows(tmp_path):
            assert row["primary"] != "_general", row
    finally:
        engine.close()


def test_retag_passes_context_to_generator(tmp_path):
    """The whole point: pairs after the first must be tagged WITH
    preceding-pair context, unlike the row sweep."""
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
        contexts = []
        original = engine._tag_generator.generate_tags

        def _spy(text, store_tags, context_turns=None, **kwargs):
            contexts.append(context_turns)
            return original(text, store_tags, context_turns=context_turns, **kwargs)

        engine._tag_generator.generate_tags = _spy
        engine.retag_canonical_turns()
        assert contexts, "generator never invoked"
        later = [ctx for ctx in contexts[1:] if ctx]
        assert later, (
            "pairs after the first must receive preceding-pair context; "
            f"got contexts: {[bool(c) for c in contexts]}"
        )
    finally:
        engine.close()


def test_never_downgrades_on_fallback_result(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)

        def _dead_tagger(text, store_tags, context_turns=None, **kwargs):
            return TagResult(tags=["_general"], primary="_general", source="fallback")

        engine._tag_generator.generate_tags = _dead_tagger
        report = engine.retag_canonical_turns()
        assert report["retagged_pairs"] == 0
        assert report["skipped_low_quality"] == 3
        for row in _rows(tmp_path):
            assert row["primary"] == "_general"
            assert row["tagged_at"] == "2026-06-20T00:00:00+00:00", (
                "a skipped row must not be touched at all"
            )
    finally:
        engine.close()


def test_only_general_leaves_real_tags_untouched(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
        # Give one pair real tags already.
        conn = sqlite3.connect(tmp_path / "c.db")
        conn.execute(
            "UPDATE canonical_turns SET primary_tag = 'alpine-gardening', "
            "tags_json = '[\"alpine-gardening\"]' "
            "WHERE conversation_id = 'c' AND sort_key <= 2000"
        )
        conn.commit()
        conn.close()
        report = engine.retag_canonical_turns(only_general=True)
        assert report["retagged_pairs"] == 2
        rows = _rows(tmp_path)
        assert rows[0]["primary"] == "alpine-gardening"
        assert rows[1]["primary"] == "alpine-gardening"
    finally:
        engine.close()


def test_window_selection_half_open_and_normalized(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path, created_at="2026-06-20T12:00:00+00:00")
        # Window that EXCLUDES the rows (until == before created_at),
        # supplied with a space separator — must be normalized, not
        # silently mis-compared.
        report = engine.retag_canonical_turns(
            since="2026-06-01 00:00:00", until="2026-06-10 00:00:00",
        )
        assert report["retagged_pairs"] == 0
        assert all(r["primary"] == "_general" for r in _rows(tmp_path))
        # Window that includes them.
        report = engine.retag_canonical_turns(
            since="2026-06-17 17:00:00", until="2026-07-03 00:00:00",
        )
        assert report["retagged_pairs"] == 3
    finally:
        engine.close()


def test_second_run_is_idempotent(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
        first = engine.retag_canonical_turns()
        assert first["retagged_pairs"] == 3
        second = engine.retag_canonical_turns()
        assert second["selected_pairs"] == 0
        assert second["retagged_pairs"] == 0
    finally:
        engine.close()


def test_dry_run_reports_without_writing(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
        report = engine.retag_canonical_turns(dry_run=True)
        assert report["dry_run"] is True
        assert report["retagged_pairs"] == 3
        assert all(r["primary"] == "_general" for r in _rows(tmp_path))
    finally:
        engine.close()


def test_cli_command_wired(tmp_path):
    """The admin subcommand parses and emits a JSON result line."""
    import json
    import subprocess
    import sys
    engine = _make_engine(tmp_path)
    try:
        _seed_general_rows(engine, tmp_path)
    finally:
        engine.close()
    proc = subprocess.run(
        [
            sys.executable, "-m", "virtual_context.cli.main",
            "admin", "retag-canonical-turns", "c",
            "--sqlite-path", str(tmp_path / "c.db"),
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    # The bare CLI config has no keyword vocabulary, so the quality
    # gate skips every pair — the wiring assertion is the selection
    # count plus the never-downgrade accounting.
    assert payload["selected_pairs"] == 3
    assert payload["retagged_pairs"] + payload["skipped_low_quality"] == 3
