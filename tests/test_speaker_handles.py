"""Durable speaker-handle allocation, lifecycle fencing, and VCMERGE separation.

The worst failure this relation can have is a repointed handle: a selection
string that quietly starts naming a different human. These tests pin the
storage invariants structurally — two unique keys, one fenced transaction,
deterministic suffixing, immutability across rename/merge/delete — so no
caller discipline is load-bearing.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.exceptions import LifecycleEpochMismatch
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import (
    RESERVED_SPEAKER_HANDLES,
    SPEAKER_HANDLE_MAX_LENGTH,
    SpeakerHandleCandidate,
    is_valid_speaker_handle,
    normalize_speaker_handle_base,
    speaker_handle_for_rank,
)

ALEX = "actor:discord:alex"
BLAKE = "actor:discord:blake"
CASEY = "actor:discord:casey"


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "handles.db"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _conversation(store, cid, tenant="t1", phase="active", epoch=1):
    conn = store._get_conn()
    now = _now()
    conn.execute(
        """INSERT INTO conversations
               (conversation_id, tenant_id, lifecycle_epoch, phase,
                created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (cid, tenant, epoch, phase, now, now),
    )
    conn.commit()


def _cand(actor, base, sort_key=1.0):
    return SpeakerHandleCandidate(
        actor_id=actor, normalized_base=base, first_seen_sort_key=sort_key,
    )


def _alloc(store, cands, *, tenant="t1", audience="guild", owner="guild",
           epoch=1):
    return store.allocate_speaker_handles(
        tenant, audience, cands,
        owner_conversation_id=owner, expected_lifecycle_epoch=epoch,
    )


def _rows(store, audience):
    conn = store._get_conn()
    return conn.execute(
        """SELECT actor_id, handle, normalized_base, lifecycle_epoch
             FROM speaker_handles
            WHERE audience_conversation_id = ?
            ORDER BY first_seen_sort_key, actor_id""",
        (audience,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Grammar constants
# ---------------------------------------------------------------------------

def test_handle_grammar_is_bounded_ascii():
    assert is_valid_speaker_handle("alex")
    assert is_valid_speaker_handle("alex.2")
    assert is_valid_speaker_handle("sania_khan")
    assert not is_valid_speaker_handle("Alex")          # uppercase
    assert not is_valid_speaker_handle("a b")           # whitespace
    assert not is_valid_speaker_handle("a\nb")          # control
    assert not is_valid_speaker_handle("2alex")         # digit-led
    assert not is_valid_speaker_handle("")              # empty
    assert not is_valid_speaker_handle('a"],["evil')    # injection chars
    assert not is_valid_speaker_handle("x" * (SPEAKER_HANDLE_MAX_LENGTH + 1))


def test_normalize_base_is_deterministic_and_never_reserved():
    assert normalize_speaker_handle_base("Sania Khan") == "sania_khan"
    assert normalize_speaker_handle_base("  Alex  ") == "alex"
    # Reserved engine identity falls back rather than colliding.
    for reserved in RESERVED_SPEAKER_HANDLES:
        assert normalize_speaker_handle_base(reserved) == "user"
        assert normalize_speaker_handle_base(reserved.upper()) == "user"
    # No usable ASCII at all falls back.
    assert normalize_speaker_handle_base("馬克") == "user"
    assert normalize_speaker_handle_base("") == "user"
    assert normalize_speaker_handle_base('"],"evil"') == "evil"
    # Leading digits are stripped: handles are letter-led.
    assert normalize_speaker_handle_base("9lives") == "lives"
    # Output is always a valid, non-reserved base.
    for name in ("A" * 500, "___", "1234", "a-b.c d", "\x00\x07alex"):
        base = normalize_speaker_handle_base(name)
        assert is_valid_speaker_handle(base)
        assert base not in RESERVED_SPEAKER_HANDLES


def test_suffixed_handles_always_fit_the_protocol_max():
    base = normalize_speaker_handle_base("a" * 500)
    for rank in (1, 2, 99, 12345):
        handle = speaker_handle_for_rank(base, rank)
        assert len(handle) <= SPEAKER_HANDLE_MAX_LENGTH
        assert is_valid_speaker_handle(handle)


# ---------------------------------------------------------------------------
# Allocation semantics
# ---------------------------------------------------------------------------

def test_collision_suffixes_advance_in_first_seen_order(store):
    _conversation(store, "guild")
    # Candidate order is deliberately shuffled; allocation order must follow
    # (first_seen_sort_key, actor_id), so the earliest-seen actor gets the
    # bare base.
    out = _alloc(store, [
        _cand(CASEY, "Alex", 3.0),
        _cand(ALEX, "alex", 1.0),
        _cand(BLAKE, "Alex", 2.0),
    ])
    by_actor = {a.actor_id: a.handle for a in out}
    assert by_actor == {ALEX: "alex", BLAKE: "alex.2", CASEY: "alex.3"}


def test_equal_sort_keys_break_ties_by_actor_id(store):
    _conversation(store, "guild")
    out = _alloc(store, [
        _cand(BLAKE, "sam", 1.0),
        _cand(ALEX, "sam", 1.0),
    ])
    by_actor = {a.actor_id: a.handle for a in out}
    assert by_actor == {ALEX: "sam", BLAKE: "sam.2"}


def test_rename_never_repoints_an_existing_handle(store):
    _conversation(store, "guild")
    first = _alloc(store, [_cand(ALEX, "alex", 1.0)])
    assert first[0].handle == "alex"
    # The actor renamed; a later allocation arrives with a new base. The
    # persisted assignment must be returned unchanged — handle AND stored
    # base — not re-derived.
    second = _alloc(store, [_cand(ALEX, "alexandra", 1.0)])
    assert second[0].handle == "alex"
    assert second[0].normalized_base == "alex"
    rows = _rows(store, "guild")
    assert len(rows) == 1 and rows[0]["handle"] == "alex"


def test_freed_looking_names_are_never_reused_within_a_lifecycle(store):
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0)])
    # Even after the first holder renamed, a same-named newcomer suffixes:
    # the bare handle stays pinned to its original actor forever within this
    # audience lifecycle.
    out = _alloc(store, [_cand(BLAKE, "alex", 2.0)])
    assert out[0].actor_id == BLAKE and out[0].handle == "alex.2"


def test_reserved_identity_is_never_allocated_to_a_human(store):
    _conversation(store, "guild")
    out = _alloc(store, [
        _cand(ALEX, "assistant", 1.0),
        _cand(BLAKE, "Assistant", 2.0),
    ])
    handles = {a.handle for a in out}
    assert "assistant" not in handles
    assert handles == {"user", "user.2"}
    for row in _rows(store, "guild"):
        assert row["handle"] not in RESERVED_SPEAKER_HANDLES


def test_empty_actor_ids_are_skipped_not_allocated(store):
    _conversation(store, "guild")
    out = _alloc(store, [_cand("", "ghost", 1.0), _cand(ALEX, "alex", 2.0)])
    assert [a.handle for a in out] == ["alex"]
    assert len(_rows(store, "guild")) == 1


def test_fetch_covers_only_the_supplied_actor_set(store):
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0), _cand(BLAKE, "blake", 2.0)])
    got = store.get_speaker_handles("t1", "guild", [ALEX])
    assert [a.actor_id for a in got] == [ALEX]
    # Fetch is never an enumeration surface: no actor set, no rows.
    assert store.get_speaker_handles("t1", "guild", []) == []
    assert store.get_speaker_handles("t1", "guild", [CASEY]) == []
    # And never cross-tenant / cross-audience.
    assert store.get_speaker_handles("t2", "guild", [ALEX]) == []
    assert store.get_speaker_handles("t1", "dm", [ALEX]) == []


# ---------------------------------------------------------------------------
# Lifecycle fences
# ---------------------------------------------------------------------------

def test_stale_epoch_allocation_is_rejected(store):
    _conversation(store, "guild", epoch=2)
    with pytest.raises(LifecycleEpochMismatch):
        _alloc(store, [_cand(ALEX, "alex", 1.0)], epoch=1)
    assert _rows(store, "guild") == []


def test_missing_audience_fails_closed(store):
    with pytest.raises(KeyError):
        _alloc(store, [_cand(ALEX, "alex", 1.0)], audience="nowhere",
               owner="nowhere")


def test_cross_tenant_audience_fails_closed(store):
    _conversation(store, "guild", tenant="other-tenant")
    with pytest.raises(ValueError):
        _alloc(store, [_cand(ALEX, "alex", 1.0)])
    assert _rows(store, "guild") == []


def test_audience_that_is_neither_owner_nor_alias_fails_closed(store):
    _conversation(store, "guild")
    _conversation(store, "dm")
    # dm is a live conversation but has no alias pointing at guild, so it
    # cannot be used as an audience namespace under that owner.
    with pytest.raises(ValueError):
        _alloc(store, [_cand(ALEX, "alex", 1.0)], audience="dm",
               owner="guild")
    assert _rows(store, "dm") == []


def test_deleted_audience_fences_and_resurrect_starts_fresh(store):
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0)])
    store.mark_conversation_deleted("guild")
    # Delete removed the audience's assignments transactionally.
    assert _rows(store, "guild") == []
    # A stale allocator that captured epoch 1 cannot recreate them: the
    # phase re-proof fails while deleted...
    with pytest.raises(LifecycleEpochMismatch):
        _alloc(store, [_cand(ALEX, "alex", 1.0)], epoch=1)
    new_epoch = store.increment_lifecycle_epoch_on_resurrect("guild")
    assert new_epoch == 2
    # ...and the epoch re-proof fails after resurrect.
    with pytest.raises(LifecycleEpochMismatch):
        _alloc(store, [_cand(ALEX, "alex", 1.0)], epoch=1)
    assert _rows(store, "guild") == []
    # A fresh lifecycle is a fresh namespace: the handle string may be
    # minted again (for any actor) because non-repointing is scoped to one
    # audience-conversation lifecycle.
    out = _alloc(store, [_cand(BLAKE, "alex", 1.0)], epoch=2)
    assert out[0].handle == "alex"
    assert out[0].lifecycle_epoch == 2


def test_delete_speaker_handles_for_audience_is_tenant_scoped(store):
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0)])
    assert store.delete_speaker_handles_for_audience("t2", "guild") == 0
    assert len(_rows(store, "guild")) == 1
    assert store.delete_speaker_handles_for_audience("t1", "guild") == 1
    assert _rows(store, "guild") == []


def test_fence_errors_never_carry_actor_ids(store):
    _conversation(store, "guild", epoch=2)
    with pytest.raises(LifecycleEpochMismatch) as exc_info:
        _alloc(store, [_cand(ALEX, "alex", 1.0)], epoch=1)
    assert ALEX not in str(exc_info.value)


def test_assignment_repr_hides_the_actor_id(store):
    _conversation(store, "guild")
    out = _alloc(store, [_cand(ALEX, "alex", 1.0)])
    assert ALEX not in repr(out)
    assert ALEX not in repr(out[0])


# ---------------------------------------------------------------------------
# Two-allocator concurrency
# ---------------------------------------------------------------------------

def test_concurrent_collision_never_duplicates_or_repoints(store):
    _conversation(store, "guild")
    shared = [_cand(f"actor:discord:u{i}", "alex", float(i)) for i in range(6)]
    # Each worker sees an overlapping-but-different eligible set, all
    # colliding on one normalized base.
    sets = [shared[:4], shared[2:]]
    results: list[dict] = [{}, {}]
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def run(idx):
        try:
            barrier.wait(timeout=10)
            out = _alloc(store, sets[idx])
            results[idx] = {a.actor_id: a.handle for a in out}
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    rows = _rows(store, "guild")
    # No duplicated actor, no duplicated handle.
    actors = [r["actor_id"] for r in rows]
    handles = [r["handle"] for r in rows]
    assert len(actors) == len(set(actors)) == 6
    assert len(handles) == len(set(handles)) == 6
    # Both allocators agree on every actor they both saw: no repointing.
    for actor in set(results[0]) & set(results[1]):
        assert results[0][actor] == results[1][actor]
    # And the winners are exactly the durable rows.
    durable = {r["actor_id"]: r["handle"] for r in rows}
    for res in results:
        for actor, handle in res.items():
            assert durable[actor] == handle


# ---------------------------------------------------------------------------
# VCMERGE separation
# ---------------------------------------------------------------------------

def _reserve_merge(store, *, tenant="t1", source="dm", target="guild"):
    merge_id = str(uuid.uuid4())
    result = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id=tenant,
        source_conversation_id=source, target_conversation_id=target,
        source_label_at_merge="lbl",
    )
    assert result.status == "reserved"
    return merge_id


def test_vcmerge_preserves_separate_audience_namespaces(store):
    _conversation(store, "dm")
    _conversation(store, "guild")
    # The same handle string legitimately names DIFFERENT actors in the two
    # audiences. The merge must not move, coalesce, or rekey either side.
    _alloc(store, [_cand(ALEX, "alex", 1.0)], audience="dm", owner="dm")
    _alloc(store, [_cand(BLAKE, "alex", 1.0)], audience="guild",
           owner="guild")

    merge_id = _reserve_merge(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="t1",
        source_conversation_id="dm", target_conversation_id="guild",
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )

    dm_rows = _rows(store, "dm")
    guild_rows = _rows(store, "guild")
    assert [(r["actor_id"], r["handle"]) for r in dm_rows] == [(ALEX, "alex")]
    assert [(r["actor_id"], r["handle"]) for r in guild_rows] == [
        (BLAKE, "alex"),
    ]


def test_post_merge_alias_audience_still_allocates_without_repointing(store):
    _conversation(store, "dm")
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0)], audience="dm", owner="dm")

    merge_id = _reserve_merge(store)
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id="t1",
        source_conversation_id="dm", target_conversation_id="guild",
        expected_target_lifecycle_epoch=1, source_label_at_merge="lbl",
    )

    # The retained alias route stays a live audience namespace: a NEW actor
    # arriving through it gets a handle keyed to the dm audience with the
    # merged owner, and the pre-merge assignment is untouched.
    out = _alloc(store, [
        _cand(ALEX, "alexandra", 1.0),
        _cand(BLAKE, "alex", 2.0),
    ], audience="dm", owner="guild")
    by_actor = {a.actor_id: a.handle for a in out}
    assert by_actor == {ALEX: "alex", BLAKE: "alex.2"}
    # A wrong owner claim through the alias still fails closed.
    with pytest.raises(ValueError):
        _alloc(store, [_cand(CASEY, "casey", 3.0)], audience="dm",
               owner="somewhere-else")


def test_hard_delete_removes_the_audience_namespace(store):
    _conversation(store, "guild")
    _alloc(store, [_cand(ALEX, "alex", 1.0)])
    store.delete_conversation("guild")
    conn = store._get_conn()
    left = conn.execute(
        "SELECT COUNT(*) FROM speaker_handles "
        "WHERE audience_conversation_id = 'guild'"
    ).fetchone()[0]
    assert left == 0


# ---------------------------------------------------------------------------
# Unsupported / composite degradation
# ---------------------------------------------------------------------------

def _bare_delegate():
    return SimpleNamespace()


def test_composite_without_handle_backend_degrades_asymmetrically():
    composite = CompositeStore(
        segments=_bare_delegate(), facts=_bare_delegate(),
        fact_links=_bare_delegate(), state=_bare_delegate(),
        search=_bare_delegate(),
    )
    # Reads fail open: no handle is simply no annotation.
    assert composite.supports_speaker_handles() is False
    assert composite.get_speaker_handles("t1", "guild", [ALEX]) == []
    assert composite.delete_speaker_handles_for_audience("t1", "guild") == 0
    # Allocation fails closed: it must never mint process-local handles.
    with pytest.raises(NotImplementedError):
        composite.allocate_speaker_handles(
            "t1", "guild", [_cand(ALEX, "alex", 1.0)],
            owner_conversation_id="guild", expected_lifecycle_epoch=1,
        )


def test_composite_forwards_handles_to_the_segment_delegate(tmp_path):
    sql = SQLiteStore(db_path=str(tmp_path / "composite.db"))
    composite = CompositeStore(
        segments=sql, facts=sql, fact_links=sql, state=sql, search=sql,
    )
    _conversation(sql, "guild")
    assert composite.supports_speaker_handles() is True
    out = composite.allocate_speaker_handles(
        "t1", "guild", [_cand(ALEX, "alex", 1.0)],
        owner_conversation_id="guild", expected_lifecycle_epoch=1,
    )
    assert [a.handle for a in out] == ["alex"]
    got = composite.get_speaker_handles("t1", "guild", [ALEX])
    assert [a.handle for a in got] == ["alex"]
    assert composite.delete_speaker_handles_for_audience("t1", "guild") == 1
