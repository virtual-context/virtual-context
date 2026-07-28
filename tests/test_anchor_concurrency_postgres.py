"""Does the reconcile lock actually serialize the anchor refresh?

Incremental anchor writes ship disabled because the write is a
read-modify-write and a review concluded nothing serializes it across
workers, so two workers could interleave and leave an anchor set
belonging to neither. That conclusion rests on the merge lock being
process-local, which is true of the SQLite implementation: `BEGIN
IMMEDIATE` plus a thread-local depth counter.

**Postgres is different.** `PostgresStore.conversation_reconcile` opens a
transaction, takes `SELECT ... FOR UPDATE` on the conversation's
`conversation_lifecycle` row, and yields inside that transaction. If the
anchor refresh runs inside that block, a second worker cannot enter at
all, and the interleaving the review described is impossible on the
backend production actually runs.

That is a code reading, and code readings about this have been wrong
twice. These tests are the thing that should decide it.

**Threads rather than processes, deliberately.** `FOR UPDATE` locks a row
for the duration of a *transaction*, and each thread checks out its own
connection from the pool, so two threads hold two transactions exactly as
two processes would. The lock is enforced by Postgres, not by Python, so
threads exercise the same mechanism. If that assumption is wrong these
tests prove nothing, so it is stated here rather than buried.

Run serially (`-n0`) per the fleet rule: these tests share one database
and deliberately hold row locks.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import threading
import time
import uuid

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn

pytestmark = pytest.mark.skipif(
    not pg_dsn(), reason="requires a Postgres DSN",
)

# How long the holder stays inside the lock. Long enough that a blocked
# waiter is unambiguous, short enough not to slow the fleet.
_HOLD_SECONDS = 1.0
# A waiter that genuinely blocked cannot return much faster than the hold.
# Half the hold separates "blocked" from "walked straight in" without
# making the assertion sensitive to scheduling jitter.
_BLOCKED_THRESHOLD = _HOLD_SECONDS * 0.5
_JOIN_TIMEOUT = 30.0


def _store():
    from virtual_context.storage.postgres import PostgresStore
    return PostgresStore(pg_dsn())


@pytest.fixture()
def conv():
    conversation_id = f"anchorconc-{uuid.uuid4()}"
    store = _store()
    store.upsert_conversation(tenant_id="t", conversation_id=conversation_id)
    yield conversation_id, store
    with pg_test_conn() as conn:
        for table in (
            "canonical_turn_anchors",
            "canonical_turns",
            "ingest_batches",
            "conversation_lifecycle",
            "conversations",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE conversation_id = %s",
                (conversation_id,),
            )


def _run_concurrently(*targets):
    """Run targets in threads, collecting exceptions rather than losing them."""
    errors: list[BaseException] = []

    def _wrap(fn):
        def _inner():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
        return _inner

    threads = [threading.Thread(target=_wrap(t), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_JOIN_TIMEOUT)
    stuck = [t for t in threads if t.is_alive()]
    assert not stuck, f"{len(stuck)} thread(s) did not finish within {_JOIN_TIMEOUT}s"
    assert not errors, f"worker raised: {errors!r}"


@pytest.mark.regression("BUG-047")
def test_reconcile_lock_blocks_a_second_worker(conv):
    """A second worker must wait for the first to leave the reconcile.

    This is the whole premise. If it fails, the merge lock does not
    serialize anything on Postgres either, the review's conclusion stands
    unchanged, and the incremental anchor write needs a real fix rather
    than a re-reading.
    """
    conversation_id, store = conv
    holder_inside = threading.Event()
    waited: list[float] = []

    def holder():
        with store.conversation_reconcile(conversation_id):
            holder_inside.set()
            time.sleep(_HOLD_SECONDS)

    def waiter():
        assert holder_inside.wait(timeout=_JOIN_TIMEOUT), "holder never entered"
        started = time.monotonic()
        with store.conversation_reconcile(conversation_id):
            waited.append(time.monotonic() - started)

    _run_concurrently(holder, waiter)

    assert waited, "waiter never acquired the lock"
    assert waited[0] >= _BLOCKED_THRESHOLD, (
        f"second worker entered after only {waited[0]:.3f}s while the first "
        f"held the lock for {_HOLD_SECONDS}s: the reconcile is NOT excluding "
        f"concurrent workers"
    )


@pytest.mark.regression("BUG-047")
def test_a_different_conversation_is_not_blocked(conv):
    """Exclusion must be per conversation, not global.

    A lock that serialized every conversation against every other would
    also pass the test above, while being a throughput disaster. This
    separates "correct row lock" from "accidental global lock".
    """
    conversation_id, store = conv
    other_id = f"anchorconc-other-{uuid.uuid4()}"
    store.upsert_conversation(tenant_id="t", conversation_id=other_id)
    holder_inside = threading.Event()
    waited: list[float] = []

    def holder():
        with store.conversation_reconcile(conversation_id):
            holder_inside.set()
            time.sleep(_HOLD_SECONDS)

    def other():
        assert holder_inside.wait(timeout=_JOIN_TIMEOUT), "holder never entered"
        started = time.monotonic()
        with store.conversation_reconcile(other_id):
            waited.append(time.monotonic() - started)

    try:
        _run_concurrently(holder, other)
        assert waited, "second conversation never acquired"
        assert waited[0] < _BLOCKED_THRESHOLD, (
            f"an unrelated conversation waited {waited[0]:.3f}s: the lock is "
            f"global rather than per conversation"
        )
    finally:
        with pg_test_conn() as conn:
            for table in ("conversation_lifecycle", "conversations"):
                conn.execute(
                    f"DELETE FROM {table} WHERE conversation_id = %s", (other_id,),
                )


@pytest.mark.regression("BUG-047")
def test_anchor_refresh_runs_inside_the_reconcile_lock(conv, monkeypatch):
    """The refresh being inside the lock is what makes the delta safe.

    The lock existing is not enough. If ``_refresh_persisted_anchors`` ran
    outside it, two workers could still interleave their anchor
    read-modify-write while each held the lock only for the row writes.
    This pins the refresh to the inside of the critical section by
    pausing within it and showing a second worker cannot get in.
    """
    from virtual_context.core import ingest_reconciler as ir

    conversation_id, store = conv
    inside_refresh = threading.Event()
    waited: list[float] = []
    original = ir.IngestReconciler._refresh_persisted_anchors

    def _slow_refresh(self, conv_id):
        result = original(self, conv_id)
        inside_refresh.set()
        time.sleep(_HOLD_SECONDS)
        return result

    monkeypatch.setattr(
        ir.IngestReconciler, "_refresh_persisted_anchors", _slow_refresh,
    )

    def ingester():
        _ingest(store, conversation_id, _pairs(3))

    def waiter():
        assert inside_refresh.wait(timeout=_JOIN_TIMEOUT), "refresh never ran"
        started = time.monotonic()
        with store.conversation_reconcile(conversation_id):
            waited.append(time.monotonic() - started)

    _run_concurrently(ingester, waiter)

    assert waited, "waiter never acquired the lock"
    assert waited[0] >= _BLOCKED_THRESHOLD, (
        f"a second worker entered the reconcile after {waited[0]:.3f}s while "
        f"the anchor refresh was still running: the refresh is OUTSIDE the "
        f"lock and the delta can interleave"
    )


@pytest.mark.regression("BUG-047")
def test_concurrent_ingests_converge_on_the_rebuilt_anchor_set(conv, monkeypatch):
    """End to end, with the delta ENABLED, under real contention.

    Two workers ingest different payloads into one conversation at the
    same time. Whatever order they land in, the persisted anchor set must
    equal what a full rebuild of the final rows produces. This is the
    shape the review said breaks: two workers whose deltas differ, where
    a cardinality check cannot see the divergence.
    """
    from virtual_context.core import ingest_reconciler as ir
    from virtual_context.core.ingest_reconciler import _build_anchor_rows

    monkeypatch.setattr(ir, "_INCREMENTAL_ANCHOR_WRITES", True)
    conversation_id, store = conv

    _ingest(store, conversation_id, _pairs(4))

    start = threading.Barrier(2, timeout=_JOIN_TIMEOUT)

    def worker(tag: str, count: int):
        def _run():
            start.wait()
            _ingest(store, conversation_id, _pairs(count, tag=tag))
        return _run

    _run_concurrently(worker("a", 6), worker("b", 5))

    rows = store.get_canonical_turn_reconcile_rows(conversation_id)
    expected = {tuple(a) for a in _build_anchor_rows(rows)}
    stored = store.get_canonical_turn_anchors(conversation_id)

    assert set(stored) == expected, (
        "persisted anchors diverged from a rebuild of the final rows"
    )
    assert len(stored) == len(expected), (
        f"duplicate anchor rows survived: {len(stored)} stored vs "
        f"{len(expected)} distinct"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pairs(n: int, tag: str = "") -> dict:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"user {tag}{i}"})
        msgs.append({"role": "assistant", "content": f"assistant {tag}{i}"})
    return {"messages": msgs}


def _ingest(store, conversation_id: str, body: dict) -> None:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.core.ingest_reconciler import IngestReconciler
    from virtual_context.core.semantic_search import SemanticSearchManager
    from virtual_context.proxy.formats import detect_format
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id=conversation_id,
        storage=StorageConfig(backend="postgres", postgres_dsn=pg_dsn()),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    reconciler = IngestReconciler(store=store, semantic=semantic)
    reconciler.ingest_batch(
        conversation_id,
        body=body,
        fmt=detect_format({"messages": []}),
        expected_lifecycle_epoch=store.get_lifecycle_epoch(conversation_id),
    )


@pytest.mark.regression("BUG-047")
def test_the_block_is_enforced_by_postgres_not_by_python(conv):
    """Ask the database whether it is the one doing the blocking.

    Every other test here uses threads as a stand-in for workers, on the
    argument that the lock lives in a database transaction and each
    thread holds its own connection. That argument is sound but it is
    still an argument. This replaces it with an observation: while one
    worker holds the reconcile and another waits, Postgres itself must
    report an ungranted lock.

    If the exclusion were an artifact of Python — a shared connection, a
    thread-local, a coincidence of timing — ``pg_locks`` would show
    nothing ungranted and this fails. That closes the last gap between
    "threads" and "processes", because a server-side lock wait does not
    care which OS process asked.
    """
    conversation_id, store = conv
    holder_inside = threading.Event()
    waiter_started = threading.Event()
    ungranted_seen: list[int] = []
    release = threading.Event()

    def holder():
        with store.conversation_reconcile(conversation_id):
            holder_inside.set()
            release.wait(timeout=_JOIN_TIMEOUT)

    def waiter():
        assert holder_inside.wait(timeout=_JOIN_TIMEOUT), "holder never entered"
        waiter_started.set()
        with store.conversation_reconcile(conversation_id):
            pass

    def observer():
        assert waiter_started.wait(timeout=_JOIN_TIMEOUT), "waiter never started"
        # Give the waiter time to reach the lock and block on it.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with pg_test_conn() as conn:
                row = conn.execute(
                    "SELECT count(*) AS n FROM pg_locks WHERE NOT granted",
                ).fetchone()
            count = int(row["n"] if hasattr(row, "keys") else row[0])
            if count > 0:
                ungranted_seen.append(count)
                break
            time.sleep(0.1)
        release.set()

    _run_concurrently(holder, waiter, observer)

    assert ungranted_seen, (
        "Postgres never reported an ungranted lock while one worker held "
        "the reconcile and another waited: the exclusion is not being "
        "enforced server-side, so these tests are not measuring what they "
        "claim to measure"
    )
