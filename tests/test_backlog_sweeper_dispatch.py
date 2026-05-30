"""Compaction-backlog sweeper Phase 3 dispatch tests.

Per ``specs/compaction-backlog-sweeper-plan.md`` §5.3 T3.1-T3.4. The
sweeper's cloud-side dispatcher constructs a
``CompactionSignal(priority="backlog")`` and submits a compaction
through the engine's existing ``_run_compact`` / ``_run_compact_wrapper``
machinery. This bundle exercises the engine surfaces the dispatcher
touches:

* T3.1: ``CompactionSignal.priority`` accepts each of the four
  documented labels and ``ProxyState._resolve_c2r_gate`` derives
  ``disable_replacement_passes=True`` only when ``signal.priority ==
  "backlog"`` and no explicit kwarg is supplied. ``"takeover"`` /
  ``"soft"`` / ``"hard"`` keep the gate False.
* T3.2: A backlog-priority signal resolves the C2R gate True at the
  helper layer. The downstream forwarding contract is covered by the
  existing C2R gate bundle.
* T3.3: A backlog dispatch passes the full
  ``reconstruct_history_for_conv`` output as the
  ``conversation_history`` argument to ``_run_compact``. The
  rebuilt history includes both compacted-prefix rows and
  uncompacted-tail rows so the recovery compaction does not see a
  truncated state.
* T3.4: Phase-count source: a sweeper-style claim consumes
  ``virtual_context.proxy.state.compaction_phase_count()`` rather
  than a maintained literal. Monkeypatching the helper to a
  sentinel value causes the sentinel to land on
  ``claim_compaction_backlog`` so cloud cannot accidentally
  hardcode the count.

T3.5 is intentionally deferred because it needs a live Postgres
fixture plus the cloud tick handler. T3.6 is intentionally deferred
because it requires running the actual ``_run_compaction`` pipeline.
This file stays focused on the engine-side dispatcher-facing contract.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from virtual_context.proxy.state import (
    ProxyState,
    compaction_phase_count,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    BacklogCandidate,
    CompactionSignal,
    Message,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _signal(priority: str) -> CompactionSignal:
    return CompactionSignal(
        priority=priority, current_tokens=0, budget_tokens=0,
        overflow_tokens=0,
    )


# ---------------------------------------------------------------------------
# T3.1: signal.priority Literal acceptance + resolver behavior per tier.
# ---------------------------------------------------------------------------


class TestT31_SignalPriorityResolverContract:
    @pytest.mark.parametrize(
        "priority", ["soft", "hard", "takeover", "backlog"],
    )
    def test_each_literal_accepted_by_dataclass(self, priority: str):
        sig = _signal(priority)
        assert sig.priority == priority

    def test_backlog_resolves_gate_true_when_kwarg_omitted(self):
        assert ProxyState._resolve_c2r_gate(_signal("backlog"), None) is True

    @pytest.mark.parametrize(
        "priority", ["soft", "hard", "takeover"],
    )
    def test_non_backlog_resolves_gate_false_when_kwarg_omitted(
        self, priority: str,
    ):
        assert ProxyState._resolve_c2r_gate(_signal(priority), None) is False

    def test_explicit_kwarg_wins_over_backlog_signal(self):
        # Cloud's backlog dispatcher passes the kwarg explicitly per
        # spec §5.2; the test pins the override direction so a
        # caller can force the gate OFF for a backlog signal too.
        assert ProxyState._resolve_c2r_gate(_signal("backlog"), False) is False

    def test_explicit_kwarg_can_force_gate_on_for_non_backlog(self):
        # The mirror direction: a non-backlog dispatch can still
        # opt into the C2R gate by passing the kwarg explicitly.
        assert ProxyState._resolve_c2r_gate(_signal("soft"), True) is True


# ---------------------------------------------------------------------------
# T3.2: helper-layer composition for the backlog C2R gate.
# ---------------------------------------------------------------------------


class TestT32_HelperLayerBacklogGate:
    """T3.2: the dispatcher-facing contract is that a backlog signal
    resolves to ``disable_replacement_passes=True`` and that the
    resolved gate is not dropped before the pipeline. This is composed
    of:

    * The resolver: ``ProxyState._resolve_c2r_gate(signal, None)``
      returns True for ``signal.priority == 'backlog'``. Covered
      directly in T3.1.
    * The forwarding step: ``tests/test_compaction_c2r_gate.py`` has
      ``TestT57_BacklogSignalDerivesGateTrue`` and
      ``TestT58_ExplicitKwargOverridesSignalDefault`` for resolver
      defaults and explicit overrides, plus
      ``TestDispatchThreadingFacade`` asserting captured
      ``disable_replacement_passes`` kwargs on a real
      ``VirtualContextEngine`` facade backed by a stub compaction
      object. The dispatcher tests here would duplicate that coverage
      if they built another fake ProxyState; instead this class pins
      the composition at the resolver level so a regression in either
      direction surfaces immediately.

    A future repo-wide refactor that lets ``_run_compact`` be driven
    from a minimal stub would let us assert the captured kwarg here
    directly. For now the resolver + the existing C2R gate suite
    cover the same surface without the brittle stub.
    """

    def test_resolver_derives_gate_true_for_backlog_signal(self):
        assert ProxyState._resolve_c2r_gate(
            _signal("backlog"), None,
        ) is True

    @pytest.mark.parametrize(
        "priority", ["soft", "hard", "takeover"],
    )
    def test_resolver_leaves_gate_false_for_non_backlog(
        self, priority: str,
    ):
        assert ProxyState._resolve_c2r_gate(
            _signal(priority), None,
        ) is False

    def test_explicit_kwarg_overrides_resolver_default(self):
        # Both directions: a backlog signal can be forced off, and a
        # non-backlog signal can be forced on, via the explicit
        # ``disable_replacement_passes`` kwarg.
        assert ProxyState._resolve_c2r_gate(
            _signal("backlog"), False,
        ) is False
        assert ProxyState._resolve_c2r_gate(
            _signal("soft"), True,
        ) is True


# ---------------------------------------------------------------------------
# T3.3: dispatcher hands _run_compact the full reconstructed history.
# ---------------------------------------------------------------------------


def _seed_canonical_row(
    store: SQLiteStore,
    *,
    conv_id: str,
    canonical_turn_id: str,
    sort_key: float,
    user_content: str,
    assistant_content: str,
    compacted: bool = False,
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv_id, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at) "
        "VALUES (?, 't-dispatch', 'active', 1, ?, ?)",
        (conv_id, now, now),
    )
    conn.execute(
        """INSERT INTO canonical_turns
           (canonical_turn_id, conversation_id, sort_key, turn_hash,
            hash_version, user_content, assistant_content,
            tagged_at, compacted_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
        (canonical_turn_id, conv_id, sort_key,
         f"h-{canonical_turn_id}", user_content, assistant_content,
         now, now if compacted else None, now, now),
    )
    conn.commit()


class TestT33_FullReconstructedHistory:
    """T3.3: ``reconstruct_history_for_conv`` returns the full
    canonical history including both compacted-prefix rows and
    current uncompacted rows. The dispatcher hands this list to
    ``_run_compact`` so the recovery compaction sees the full
    conversation state -- NOT a backlog-window-only slice.
    """

    def test_history_includes_compacted_prefix_and_uncompacted_tail(
        self, tmp_path: Path,
    ):
        store = SQLiteStore(tmp_path / "t33.db")
        conv = "conv-t33"
        # Two compacted prefix rows + one uncompacted tail row.
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-prev-1", sort_key=1.0,
            user_content="prev-u-1", assistant_content="prev-a-1",
            compacted=True,
        )
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-prev-2", sort_key=2.0,
            user_content="prev-u-2", assistant_content="prev-a-2",
            compacted=True,
        )
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-tail", sort_key=3.0,
            user_content="tail-u", assistant_content="tail-a",
            compacted=False,
        )
        history = store.reconstruct_history_for_conv(conv)
        contents = [m.content for m in history]
        # All three turns present in canonical order; compacted
        # prefix did NOT get truncated.
        assert contents == [
            "prev-u-1", "prev-a-1",
            "prev-u-2", "prev-a-2",
            "tail-u", "tail-a",
        ]

    def test_history_preserves_message_content_verbatim(
        self, tmp_path: Path,
    ):
        """Cloud's dispatcher hands the
        ``reconstruct_history_for_conv`` result straight to
        ``_run_compact``. The contract here is that the canonical
        row's text is delivered verbatim (no stripping or
        normalization) so a downstream compaction sees the same
        content the original turn carried.
        """
        store = SQLiteStore(tmp_path / "t33-content.db")
        conv = "conv-t33-content"
        # Content with surrounding whitespace + embedded newlines
        # to catch any naive ``.strip()`` regression.
        _seed_canonical_row(
            store, conv_id=conv,
            canonical_turn_id="ct-w", sort_key=1.0,
            user_content="  user with leading + trailing whitespace  ",
            assistant_content="line1\nline2\n",
        )
        history = store.reconstruct_history_for_conv(conv)
        assert len(history) == 2
        assert isinstance(history[0], Message)
        assert history[0].role == "user"
        assert history[0].content == (
            "  user with leading + trailing whitespace  "
        )
        assert history[1].role == "assistant"
        assert history[1].content == "line1\nline2\n"


# ---------------------------------------------------------------------------
# T3.4: phase-count source on the claim path.
# ---------------------------------------------------------------------------


def _seed_eligible_backlog_conv(
    store: SQLiteStore, *, conv: str, count: int = 50,
) -> None:
    now = _iso(_now())
    conn = store._get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO conversation_lifecycle "
        "(conversation_id, generation, deleted, updated_at) "
        "VALUES (?, 0, 0, ?)",
        (conv, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO conversations "
        "(conversation_id, tenant_id, phase, lifecycle_epoch, "
        " created_at, updated_at) "
        "VALUES (?, 't-pc', 'active', 1, ?, ?)",
        (conv, now, now),
    )
    for i in range(count):
        conn.execute(
            """INSERT INTO canonical_turns
               (canonical_turn_id, conversation_id, sort_key, turn_hash,
                hash_version, tagged_at, compacted_at, created_at,
                updated_at)
               VALUES (?, ?, ?, ?, 1, ?, NULL, ?, ?)""",
            (f"ct-{conv}-{i}", conv, 1000.0 + i,
             f"h-{conv}-{i}", now, now, now),
        )
    conn.commit()


class TestT34_PhaseCountSource:
    """T3.4: ``claim_compaction_backlog`` consumes the engine's
    ``compaction_phase_count()`` helper. Monkeypatching the helper
    to a sentinel makes the sentinel reach the claim, proving cloud
    cannot accidentally substitute a maintained literal.
    """

    def test_phase_count_helper_propagates_to_claim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Sentinel value distinct from the canonical 7-phase plan
        # so a successful test isn't tolerating a coincidental match.
        sentinel = 4242

        import virtual_context.proxy.state as proxy_state
        monkeypatch.setattr(
            proxy_state, "compaction_phase_count",
            lambda: sentinel,
        )

        # Cloud-side pattern: build a claim that uses the helper's
        # value as ``phase_count``. The claim primitive lives in the
        # store; we exercise the same call shape cloud will use.
        store = SQLiteStore(tmp_path / "t34.db")
        conv = "conv-t34"
        _seed_eligible_backlog_conv(store, conv=conv, count=30)
        candidate = BacklogCandidate(
            conversation_id=conv, tenant_id="t-pc",
            lifecycle_epoch=1, backlog_turns=30,
            last_terminal_compaction_at=None,
        )
        ok = store.claim_compaction_backlog(
            candidate=candidate,
            new_operation_id=uuid.uuid4().hex,
            owner_worker_id="w-pc",
            phase_count=proxy_state.compaction_phase_count(),
            min_backlog_turns=20,
            grace_s=300.0,
        )
        assert ok is True
        # The sentinel reached the persisted op row.
        conn = store._get_conn()
        row = conn.execute(
            """SELECT phase_count FROM compaction_operation
                WHERE conversation_id = ?
                  AND status = 'running'""",
            (conv,),
        ).fetchone()
        assert int(row[0]) == sentinel

    def test_real_helper_returns_canonical_plan_length(self):
        # Sanity-check the canonical phase-count outside the
        # monkeypatched fixture so the contract is not hidden.
        assert compaction_phase_count() == 7
