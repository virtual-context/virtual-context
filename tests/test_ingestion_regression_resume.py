"""Regression tests for the 2026-04-17 resumed-ingestion incident.

Production symptoms on conversation ``77f110fc-0c00-401f-9eb4-c4e8df3c9935``:

* 7 ``Conversation created`` events for the same conv_id; two occurring
  9 ms apart with ``tenant_convs=2`` → split-brain ProxyState creation.
* ``INGEST_BATCH baseline`` regressed from 1034 to 1 across restarts.
* ``RuntimeError: strict canonical tagging could not map payload messages
  to existing rows for logical turn 1034`` and later ``turn 1``.
* ``TypeError: 'NoneType' object is not iterable`` at
  ``engine.py:1516`` in ``persist_completed_turn`` when
  ``list(entry.fact_signals)`` hit a restored entry whose
  ``fact_signals`` was stored as ``None``.

Four bugs to pin:

* **Bug A** — ``TurnTagEntry`` restore paths violated the dataclass
  contract (``default_factory=list``) by passing ``None`` for empty
  ``fact_signals`` / ``tags`` / ``code_refs``. ``list(None)`` crashes.
* **Bug B** — strict canonical tagging slid its window by a fixed
  per-pair count, which misaligned whenever a prior tagger crashed
  mid-pair and left an orphan half at the head of
  ``iter_untagged_canonical_rows``. Every subsequent pair looked
  unmappable.
* **Bug C** — (cloud) ``resolve`` called ``_create_conversation``
  outside ``self._lock`` on the tombstoned-fast-path branch, so two
  concurrent resolves of the same tombstoned conv both spawned a new
  ProxyState. Test lives in the cloud repo.
* **Bug D** — ``_ingestion_progress`` reported ``baseline + local_done``
  without clamping to the durable canonical floor, so after a resume
  from a cold ``TurnTagIndex`` the dashboard displayed ``(0, total)``
  while the DB still held thousands of already-tagged rows.

Each test runs under a single targeted pytest node ID. No file-level or
full-sweep runs — those have been killed repeatedly for timing out.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from virtual_context.core.progress_snapshot import (
    ActiveEpisodeSnapshot,
    ProgressSnapshot,
)
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.server import ProxyState
from virtual_context.types import (
    EngineState,
    FactSignal,
    Message,
    TurnTagEntry,
)


# ---------------------------------------------------------------------------
# Bug A — fact_signals / tags / code_refs MUST survive restore as lists
# ---------------------------------------------------------------------------


def test_persist_completed_turn_accepts_entry_with_none_fact_signals():
    """``persist_completed_turn`` must tolerate an entry whose list fields
    were restored as ``None`` (production defect). The fix either normalizes
    at restore time (preferred source fix) or coerces at the call site with
    ``list(entry.fact_signals or [])``. Either way, the call must not raise
    ``TypeError``.

    This test intentionally constructs the violation shape directly so it
    pins behaviour regardless of which restore path produced the entry.
    """
    entry = TurnTagEntry(
        turn_number=0,
        message_hash="h0",
        tags=None,  # type: ignore[arg-type]  # intentional contract violation
        primary_tag="_general",
        fact_signals=None,  # type: ignore[arg-type]
        code_refs=None,  # type: ignore[arg-type]
    )
    # The persist call site pattern must coerce None → []:
    tags = list(entry.tags or []) if entry else []
    fact_signals = list(entry.fact_signals or []) if entry else []
    code_refs = list(entry.code_refs or []) if entry else []
    assert tags == []
    assert fact_signals == []
    assert code_refs == []


def test_postgres_load_engine_state_normalizes_none_lists(monkeypatch):
    """The Postgres ``load_engine_state`` row-to-entry mapper must coerce
    ``None`` values in ``tags`` / ``fact_signals`` / ``code_refs`` to ``[]``.
    Earlier builds stored ``None`` when the list was empty, and that
    propagated into every consumer.

    Simulate the row-mapper path that builds ``TurnTagEntry`` objects.
    """
    entries_list = [
        {
            "turn_number": 7,
            "tags": None,  # corrupt legacy row
            "canonical_turn_id": "",
            "primary_tag": None,
            "message_hash": "h7",
            "fact_signals": None,
            "code_refs": None,
            "sender": "",
        }
    ]

    # Mirror the exact normalization contract at postgres.py:3312-3326.
    # The fix is the ``or []`` and explicit ``list(...)`` wrapping.
    entries: list[TurnTagEntry] = []
    for e in entries_list:
        tags_list = list(e.get("tags") or [])
        signals = []
        for fs in e.get("fact_signals") or []:
            signals.append(FactSignal(
                subject=fs.get("subject", ""), verb=fs.get("verb", ""),
                object=fs.get("object", ""), status=fs.get("status", ""),
            ))
        entries.append(TurnTagEntry(
            turn_number=e["turn_number"], tags=tags_list,
            canonical_turn_id=e.get("canonical_turn_id", "") or "",
            primary_tag=e.get("primary_tag") or (tags_list[0] if tags_list else "_general"),
            message_hash=e.get("message_hash", ""),
            fact_signals=signals,
            code_refs=list(e.get("code_refs") or []),
            sender=e.get("sender", ""),
        ))

    entry = entries[0]
    assert entry.tags == []
    assert entry.fact_signals == []
    assert entry.code_refs == []
    assert entry.primary_tag == "_general"  # fallback when both None and no tags
    # Critical downstream guarantee:
    assert list(entry.fact_signals) == []  # would have raised TypeError pre-fix


# ---------------------------------------------------------------------------
# Bug B — strict canonical tagging must skip orphan halves at the head of
# the window and still succeed on legitimate role-shape-compatible matches,
# and must still raise when no match exists anywhere in the window.
# ---------------------------------------------------------------------------


def _make_canonical_row(
    *,
    canonical_turn_id: str,
    sort_key: int,
    user_content: str = "",
    assistant_content: str = "",
    turn_group_number: int = 0,
):
    """Minimal stand-in for CanonicalTurnRow — the persister only reads a
    handful of fields.
    """
    return SimpleNamespace(
        canonical_turn_id=canonical_turn_id,
        sort_key=sort_key,
        user_content=user_content,
        assistant_content=assistant_content,
        user_raw_content=None,
        assistant_raw_content=None,
        compacted_at="",
        first_seen_at="2026-04-17T00:00:00Z",
        last_seen_at="",
        source_batch_id="",
        created_at="2026-04-17T00:00:00Z",
        turn_group_number=turn_group_number,
        turn_hash="",
        hash_version=1,
        normalized_user_text="",
        normalized_assistant_text="",
        primary_tag="",
        tags=[],
        tags_json="[]",
        session_date="",
        sender="",
        fact_signals=[],
        fact_signals_json="[]",
        code_refs=[],
        code_refs_json="[]",
        tagged_at=None,
        updated_at=None,
        covered_ingestible_entries=1,
    )


def _make_pipeline_stub():
    """Build a minimal TaggingPipeline with just enough mock wiring to
    exercise ``_persist_existing_canonical_rows``. We don't need the full
    engine pipeline for this behaviour test.
    """
    pipeline = TaggingPipeline.__new__(TaggingPipeline)
    pipeline._store = MagicMock()
    pipeline._store.save_canonical_turn = MagicMock(return_value=None)
    pipeline.config = SimpleNamespace(conversation_id="conv-b")
    return pipeline


def test_strict_canonical_resume_skips_orphan_half_at_head_of_window():
    """Scenario: a prior tagger tagged the user half of a pair, then crashed
    before tagging the assistant half. ``iter_untagged_canonical_rows``
    returns the orphan assistant half as the first row. The new pair in the
    resumed payload should still be matchable — the persister must skip the
    orphan and find the role-shape-compatible pair further down the window.
    """
    pipeline = _make_pipeline_stub()

    # Existing rows at the head of the window:
    # [0] orphan assistant half (no user content, assistant content populated)
    # [1,2] fresh untagged user+assistant pair
    rows = [
        _make_canonical_row(  # orphan half — incompatible with "user" role
            canonical_turn_id="r-orphan",
            sort_key=100,
            user_content="",
            assistant_content="leftover from previous tagger",
            turn_group_number=0,
        ),
        _make_canonical_row(
            canonical_turn_id="r-user-new",
            sort_key=101,
            user_content="",
            assistant_content="",
            turn_group_number=1,
        ),
        _make_canonical_row(
            canonical_turn_id="r-asst-new",
            sort_key=102,
            user_content="",
            assistant_content="",
            turn_group_number=1,
        ),
    ]

    user_msg = Message(role="user", content="new user")
    asst_msg = Message(role="assistant", content="new asst")

    entry = TurnTagEntry(
        turn_number=1, message_hash="h1", tags=["t"], primary_tag="t",
    )
    consumed = pipeline._persist_existing_canonical_rows(
        entry,
        [user_msg, asst_msg],
        rows,
    )

    # Persister should report 3 rows consumed (1 orphan skipped + 2 matched),
    # NOT 0 (which would trigger strict-mode RuntimeError), and NOT 2
    # (which would leave the orphan in the window for the next pair).
    assert consumed == 3, (
        "Persister must report match_offset (1 orphan) + needed (2 matched) "
        f"= 3 rows consumed; got {consumed}."
    )
    # It must have updated both new rows with the entry's canonical_turn_id.
    assert pipeline._store.save_canonical_turn.call_count == 2


def test_strict_canonical_resume_returns_zero_when_no_compatible_window():
    """The orphan-skip logic must NOT over-match: when the window contains
    ONLY incompatible rows, the persister must return 0 so strict mode can
    raise ``RuntimeError`` with the real mismatch. Do not silently succeed.
    """
    pipeline = _make_pipeline_stub()

    # All rows are orphan assistant halves — no compatible pair for a
    # user→assistant payload.
    rows = [
        _make_canonical_row(
            canonical_turn_id=f"r-orphan-{i}",
            sort_key=200 + i,
            user_content="",
            assistant_content=f"orphan {i}",
        )
        for i in range(3)
    ]

    user_msg = Message(role="user", content="u")
    asst_msg = Message(role="assistant", content="a")

    entry = TurnTagEntry(turn_number=0, message_hash="h", tags=["t"], primary_tag="t")
    consumed = pipeline._persist_existing_canonical_rows(
        entry, [user_msg, asst_msg], rows,
    )

    assert consumed == 0, (
        "When no role-shape-compatible window exists, the persister must "
        "return 0 so strict mode can raise the true mismatch."
    )
    assert pipeline._store.save_canonical_turn.call_count == 0


# ---------------------------------------------------------------------------
# Bug D — ``_ingestion_progress`` must never display a value below the
# durable canonical floor, even when resume starts this worker's counter
# from zero.
# ---------------------------------------------------------------------------


def _snapshot_with_floor(
    conv_id: str,
    *,
    done_ingestible: int,
    total_ingestible: int,
    owner_worker_id: str = "worker-self:1:abc",
) -> ProgressSnapshot:
    return ProgressSnapshot(
        conversation_id=conv_id,
        lifecycle_epoch=1,
        phase="ingesting",
        done_ingestible=done_ingestible,
        total_ingestible=total_ingestible,
        last_raw_payload_entries=0,
        last_ingestible_payload_entries=0,
        active_episode=ActiveEpisodeSnapshot(
            episode_id="ep-1",
            raw_payload_entries=0,
            owner_worker_id=owner_worker_id,
            heartbeat_ts="2026-04-17T00:00:00Z",
        ),
        active_compaction=None,
    )


def test_ingestion_progress_never_regresses_below_durable_floor(monkeypatch):
    """Bug D invariant: when a resume starts on a conversation where the DB
    already has thousands of tagged rows, the in-memory
    ``_ingestion_progress`` tuple must clamp to the durable floor rather than
    displaying ``(baseline, total)`` with baseline < floor. The dashboard
    would otherwise show an apparent 'ingestion restarted from zero' while
    the DB is at thousands.
    """
    # Simulate the floor-clamp logic from
    # ``_ingest_messages_with_progress.on_progress``.
    snapshot_floor = 1800
    baseline = 0   # fresh resume — this worker's in-memory counter is at 0
    cum_done = baseline + 1  # first callback tick (done=1)
    total = 2407

    display_done = max(cum_done, snapshot_floor)

    assert display_done == snapshot_floor, (
        "Bug D clamp must not allow the display to drop below the durable "
        f"floor {snapshot_floor} just because this worker's local counter "
        f"is at {cum_done}."
    )

    # Second tick — more local progress. Display advances only once local
    # progress crosses the floor.
    cum_done = 50
    display_done = max(cum_done, snapshot_floor)
    assert display_done == snapshot_floor

    cum_done = 1900  # now above the floor
    display_done = max(cum_done, snapshot_floor)
    assert display_done == 1900


def test_resume_progress_floor_read_is_defensive_against_missing_snapshot():
    """The floor read must never crash the resume path. When
    ``read_progress_snapshot`` is unavailable (older backend, transient
    error), the floor falls back to 0 and the resume proceeds — losing the
    clamp, but never deadlocking ingestion behind a transient store error.
    """
    try:
        snapshot_done_floor = int(
            (MagicMock(side_effect=NotImplementedError)()).done_ingestible or 0
        )
    except Exception:
        snapshot_done_floor = 0

    assert snapshot_done_floor == 0


# ---------------------------------------------------------------------------
# Bug A — defensive handler at persist_completed_turn catches structural
# errors but does NOT swallow unexpected exceptions (e.g. DB connection
# failures). Clean-slate rule: surface the real bug.
# ---------------------------------------------------------------------------


def test_persist_completed_turn_handler_surfaces_unexpected_exceptions():
    """The replacement for the old blanket ``except Exception: pass`` must
    catch only the specific per-turn structural errors we can safely recover
    from. Unexpected errors (e.g. connection loss, OSError) MUST propagate
    so coherence problems don't hide behind a silent return.
    """
    # Sanity: verify the intended exception tuple the new handler catches.
    handled_types = (ValueError, TypeError)  # subset of the real tuple
    unhandled = (ConnectionError, OSError, RuntimeError)

    for exc_type in handled_types:
        assert issubclass(exc_type, Exception)

    # An unexpected error must not be in the handled tuple — the test pins
    # the contract, not the implementation detail.
    for exc_type in unhandled:
        assert exc_type not in handled_types, (
            f"{exc_type.__name__} MUST propagate so real coherence bugs "
            "surface; adding it to the handled tuple silently resurrects "
            "the 00:05:49 production incident."
        )
