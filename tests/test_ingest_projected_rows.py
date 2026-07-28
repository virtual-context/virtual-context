"""Reconciliation loads a projection, and it must behave like the full row.

Merging an inbound payload against stored history keys on hashes, sort
keys and provenance columns. It never reads a stored row's text, but the
loader fetched it anyway, and the content columns are both the widest
part of a row and the part stored out of line.

The projection drops them. These tests pin the three ways that can go
wrong silently:

* **The role gate.** ``ingest_single`` decides whether a stored row is
  the user half of a turn, and therefore whether it may take speaker
  attribution, by asking whether it carries user text. Dropping the
  column without replacing that check makes the answer always "no" and
  attribution quietly stops persisting. The replacement flag must agree
  with ``str.strip()`` on the same inputs, whitespace included.
* **Type leakage.** A projected row is not writable back to storage. If
  one reaches a write path it must raise, not blank real content, and it
  must never escape to a caller.
* **The ordinal.** ``exact_resend`` used to return stored rows, which
  carry a real ``turn_number``. Returning prepared rows instead is what
  keeps the result homogeneous, but prepared rows default to the ``-1``
  sentinel that downstream stamping deliberately skips, so the ordinal
  has to be carried across explicitly.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

from virtual_context.core.canonical_turns import STRIP_WHITESPACE
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.proxy.formats import detect_format
from virtual_context.types import CanonicalTurnReconcileRow, CanonicalTurnRow

_CONTENT_COLUMNS = (
    "user_content",
    "assistant_content",
    "user_raw_content",
    "assistant_raw_content",
    "normalized_user_text",
    "normalized_assistant_text",
)


def _fmt():
    return detect_format({"messages": []})


def _reconciler(store) -> IngestReconciler:
    from virtual_context.config import VirtualContextConfig
    from virtual_context.types import StorageConfig, TagGeneratorConfig

    config = VirtualContextConfig(
        conversation_id="c",
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    return IngestReconciler(store=store, semantic=semantic)


def _store(tmp_path: Path, name: str = "s.db") -> SQLiteStore:
    store = SQLiteStore(tmp_path / name)
    store.upsert_conversation(tenant_id="t", conversation_id="c")
    return store


def _pairs(n: int) -> dict:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"user {i}"})
        msgs.append({"role": "assistant", "content": f"assistant {i}"})
    return {"messages": msgs}


# ---------------------------------------------------------------------------
# The projection carries what reconciliation reads and nothing more.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-048")
def test_projection_omits_content_and_matches_row_order(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    epoch = store.get_lifecycle_epoch("c")
    rec.ingest_batch("c", body=_pairs(4), fmt=_fmt(), expected_lifecycle_epoch=epoch)

    full = store.get_all_canonical_turns("c")
    projected = store.get_canonical_turn_reconcile_rows("c")

    assert [r.canonical_turn_id for r in projected] == [r.canonical_turn_id for r in full]
    assert [r.turn_hash for r in projected] == [r.turn_hash for r in full]
    assert [r.sort_key for r in projected] == [r.sort_key for r in full]
    assert [r.turn_number for r in projected] == [r.turn_number for r in full]

    for column in _CONTENT_COLUMNS:
        assert not hasattr(projected[0], column), (
            f"{column} must not survive into the projection"
        )


@pytest.mark.regression("BUG-048")
def test_projection_preserves_every_field_the_enrichment_merge_reads(tmp_path: Path):
    """Whatever _preserve_existing_enrichment reads must be projected.

    A field it reads but the projection drops would raise; a field the
    projection zeroes would silently blank stored provenance on the next
    resend.
    """
    store = _store(tmp_path)
    # Every merge-read column carries a DISTINCT non-empty value, so a
    # projection that blanks one is caught. Comparing defaults would make
    # the assertion trivially true.
    store.save_canonical_turn(
        "c", 0, "user text", "assistant text",
        canonical_turn_id="id-1", sort_key=1000.0, turn_hash="h1",
        session_date="2026-07-27", sender="Alice",
        turn_group_number=3,
        origin_channel_id="chan-id", origin_channel_label="chan-label",
        sender_actor_id="actor:telegram:1",
        source_message_id="msg-7", reply_target_message_id="msg-3",
        reply_subject_actor_id="actor:telegram:2",
        reply_subject_label="Bob", reply_target_body="quoted body",
        reply_attribution_version=2,
        audience_conversation_id="aud-1", audience_attribution_version=1,
    )

    full = store.get_all_canonical_turns("c")[0]
    projected = store.get_canonical_turn_reconcile_rows("c")[0]

    read_by_merge = (
        "session_date", "turn_group_number", "sender",
        "origin_channel_id", "origin_channel_label", "sender_actor_id",
        "canonical_turn_id", "source_message_id", "reply_target_message_id",
        "reply_subject_actor_id", "reply_subject_label", "reply_target_body",
        "audience_conversation_id", "reply_attribution_version",
        "audience_attribution_version",
    )
    for name in read_by_merge:
        assert hasattr(projected, name), f"{name} is read by the merge but not projected"
        stored = getattr(full, name)
        assert stored not in ("", 0, -1), (
            f"fixture must give {name} a distinguishing value"
        )
        assert getattr(projected, name) == stored, name


# ---------------------------------------------------------------------------
# The role gate. The flag must answer exactly what .strip() answers.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-048")
def test_strip_whitespace_constant_matches_python():
    """The trim set must be exactly what ``str.strip()`` removes.

    Regenerated from Python rather than compared against a copy, so the
    constant cannot drift from the behavior it stands in for. If a future
    Python widens its whitespace definition, this fails instead of
    quietly reintroducing the mismatch.
    """
    regenerated = {
        cp for cp in range(0x110000) if chr(cp).strip() == ""
    }
    assert {ord(ch) for ch in STRIP_WHITESPACE} == regenerated
    # And it must not be written as a dialect escape sequence, which is how
    # the vertical tab was lost on one backend and the letter "v" gained.
    assert "\\" not in STRIP_WHITESPACE


@pytest.mark.regression("BUG-048")
@pytest.mark.parametrize("codepoint", sorted(ord(c) for c in STRIP_WHITESPACE))
def test_presence_flag_false_for_every_whitespace_character(
    tmp_path: Path, codepoint: int,
):
    """Every character Python strips must read as "no content" in SQL.

    A disagreement here is a false POSITIVE: a row holding only whitespace
    reports as carrying user text, which lets a row that is not the user
    half of a turn take durable speaker attribution. Hand-written per
    dialect this was wrong for 23 Unicode characters on both backends,
    plus the vertical tab on one of them.
    """
    store = _store(tmp_path)
    store.save_canonical_turn(
        "c", 0, chr(codepoint) * 3, "assistant text",
        canonical_turn_id="id-1", sort_key=1000.0, turn_hash="h1",
    )
    projected = store.get_canonical_turn_reconcile_rows("c")[0]
    assert projected.has_user_content is False, (
        f"U+{codepoint:04X} reads as content in SQL but Python strips it"
    )


@pytest.mark.regression("BUG-048")
@pytest.mark.parametrize(
    "user_text",
    [
        "real content",
        "",
        " ",
        "   \t  ",
        "\n",
        "\r\n",
        "\t\n\r\x0b\x0c ",
        " leading and trailing ",
        "0",
        # A bare "v": one backend's escaped literal trimmed the letter
        # itself rather than the vertical tab it was meant to name.
        "v",
        "vvv",
        "\x0b",
        " ",
        "　",
        " ",
        " real ",
    ],
)
def test_presence_flag_agrees_with_python_strip(tmp_path: Path, user_text: str):
    """The SQL flag and ``(value or "").strip()`` must never disagree.

    The default single-argument TRIM strips spaces only, so a row holding
    just a newline would report as carrying content while Python reads it
    as empty. That disagreement points the wrong way: it lets a row that
    is not a user half take speaker attribution.
    """
    store = _store(tmp_path)
    store.save_canonical_turn(
        "c", 0, user_text, "assistant text",
        canonical_turn_id="id-1", sort_key=1000.0, turn_hash="h1",
    )
    projected = store.get_canonical_turn_reconcile_rows("c")[0]
    assert projected.has_user_content is bool(user_text.strip()), (
        f"flag disagrees with str.strip() for {user_text!r}"
    )


@pytest.mark.regression("BUG-048")
def test_attribution_still_persists_through_the_projection(tmp_path: Path):
    """The role gate still lets a late-derived sender reach storage.

    This is the failure a naive projection produces: no error, no log,
    attribution simply stops becoming durable.
    """
    store = _store(tmp_path)
    rec = _reconciler(store)
    epoch = store.get_lifecycle_epoch("c")

    # The prepare half: a lone user row persisted without a sender, exactly
    # as a payload that did not carry the envelope leaves it. A fresh
    # conversation is what makes the lone half legitimate.
    rec.ingest_batch(
        "c",
        body={"messages": [{"role": "user", "content": "hello there"}]},
        fmt=_fmt(),
        expected_lifecycle_epoch=epoch,
    )
    rows = store.get_all_canonical_turns("c")
    assert len(rows) == 1 and rows[0].sender == ""

    # The ingest half: the same user text plus its reply, now carrying the
    # sender. The tail-hash path mirrors the stored user row rather than
    # rewriting it, so the sender can only become durable through the
    # role-gated compare-and-set, which is what reads user-text presence.
    result = rec.ingest_single(
        "c", user_content="hello there", assistant_content="hi",
        user_sender="Alice", expected_lifecycle_epoch=epoch,
    )
    assert result.merge_mode == "tail_append", (
        f"fixture must exercise the tail-hash path, got {result.merge_mode}"
    )
    rows = store.get_all_canonical_turns("c")
    assert rows[0].sender == "Alice", "sender upgrade stopped persisting"


# ---------------------------------------------------------------------------
# Type leakage. No projected row escapes, and one reaching a write raises.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-048")
@pytest.mark.parametrize(
    "second_payload",
    [
        pytest.param(lambda: _pairs(3), id="exact_resend"),
        pytest.param(lambda: _pairs(4), id="append"),
        pytest.param(
            lambda: {
                "messages": [
                    {"role": "user", "content": "user 0"},
                    {"role": "assistant", "content": "assistant 0"},
                    {"role": "user", "content": "WEDGE"},
                    {"role": "assistant", "content": "WEDGED"},
                    {"role": "user", "content": "user 1"},
                    {"role": "assistant", "content": "assistant 1"},
                    {"role": "user", "content": "user 2"},
                    {"role": "assistant", "content": "assistant 2"},
                ],
            },
            id="interior_overlap",
        ),
    ],
)
def test_result_rows_never_carry_a_projected_row(tmp_path: Path, second_payload):
    """CanonicalIngestResult.rows must be one type on every path.

    Consumers read canonical_turn_id and turn_number off these rows. A
    result that is sometimes stored rows and sometimes prepared rows is a
    union callers would have to branch on, and the projected variant is
    not writable.
    """
    store = _store(tmp_path)
    rec = _reconciler(store)
    epoch = store.get_lifecycle_epoch("c")
    rec.ingest_batch("c", body=_pairs(3), fmt=_fmt(), expected_lifecycle_epoch=epoch)

    result = rec.ingest_batch(
        "c", body=second_payload(), fmt=_fmt(), expected_lifecycle_epoch=epoch,
    )
    assert result.rows, "expected the path to report rows"
    for row in result.rows:
        assert isinstance(row, CanonicalTurnRow), (
            f"{result.merge_mode} leaked {type(row).__name__} to the caller"
        )
        assert not isinstance(row, CanonicalTurnReconcileRow)


@pytest.mark.regression("BUG-048")
def test_exact_resend_rows_carry_the_stored_ordinal(tmp_path: Path):
    """Returned rows must keep a real turn_number, not the -1 sentinel.

    Downstream stamping uses turn_number as an anchor and deliberately
    skips -1, so prepared rows that never learned their position would
    silently stop anchoring rather than fail.
    """
    store = _store(tmp_path)
    rec = _reconciler(store)
    epoch = store.get_lifecycle_epoch("c")
    rec.ingest_single(
        "c", user_content="u0", assistant_content="a0",
        expected_lifecycle_epoch=epoch,
    )
    result = rec.ingest_single(
        "c", user_content="u0", assistant_content="a0",
        expected_lifecycle_epoch=epoch,
    )
    assert result.merge_mode == "exact_resend"
    assert result.rows
    # This early return is the one path that used to hand back stored rows
    # directly, so it is where a projected row would escape.
    for row in result.rows:
        assert isinstance(row, CanonicalTurnRow), (
            f"exact_resend leaked {type(row).__name__} to the caller"
        )
        assert not isinstance(row, CanonicalTurnReconcileRow)
        assert hasattr(row, "user_content"), "returned rows must be writable rows"
    assert [row.turn_number for row in result.rows] == [0, 1], (
        "exact_resend must carry stored ordinals onto the returned rows"
    )


@pytest.mark.regression("BUG-048")
def test_projected_row_cannot_be_written_back(tmp_path: Path):
    """A projected row reaching a write path raises rather than blanking.

    This is the whole reason the projection is a separate type instead of
    a CanonicalTurnRow with empty text.
    """
    store = _store(tmp_path)
    rec = _reconciler(store)
    projected = CanonicalTurnReconcileRow(
        conversation_id="c", canonical_turn_id="id-1", turn_hash="h", sort_key=1000.0,
    )
    with pytest.raises(AttributeError):
        rec._write_turn(projected, turn_number=0)


# ---------------------------------------------------------------------------
# Unsupported backends keep the full load.
# ---------------------------------------------------------------------------

@pytest.mark.regression("BUG-048")
def test_unsupported_backend_falls_back_to_the_full_load(tmp_path: Path):
    """None means "cannot project", which is not the same as "no rows".

    Reading it as an empty conversation would reconcile every payload
    against no history, which appends duplicates of everything.
    """
    inner = _store(tmp_path)
    rec_seed = _reconciler(inner)
    epoch = inner.get_lifecycle_epoch("c")
    rec_seed.ingest_batch("c", body=_pairs(3), fmt=_fmt(), expected_lifecycle_epoch=epoch)

    class _NoProjection:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def get_canonical_turn_reconcile_rows(self, conversation_id):
            return None

    rec = _reconciler(_NoProjection(inner))
    rows = rec._load_reconcile_rows("c")
    assert len(rows) == 6
    assert all(isinstance(r, CanonicalTurnRow) for r in rows)

    # And a resend against that backend still recognises the history.
    result = rec.ingest_batch(
        "c", body=_pairs(3), fmt=_fmt(), expected_lifecycle_epoch=epoch,
    )
    assert result.merge_mode == "exact_resend"
    assert result.turns_written == 0


@pytest.mark.regression("BUG-048")
def test_projection_still_triggers_the_legacy_group_backfill(tmp_path: Path):
    """Projecting rows must not stop the one-shot turn-group recompute.

    Conversations ingested before ``turn_group_number`` existed sit at -1
    on every row and fall back to content heuristics. The full-row loader
    detects that at read time and recomputes once. A projected loader that
    skipped it would leave those conversations on heuristics forever, and
    the enrichment merge would keep reading -1 and never inherit a group.
    """
    store = _store(tmp_path)
    for index in range(4):
        store.save_canonical_turn(
            "c", index,
            "u" if index % 2 == 0 else "",
            "" if index % 2 == 0 else "a",
            canonical_turn_id=f"id-{index}",
            sort_key=float((index + 1) * 1000.0),
            turn_hash=f"h{index}",
            turn_group_number=-1,
        )
    # Read the raw table, not the full loader: the full loader performs the
    # very backfill under test, so using it here would trigger the recompute
    # and the precondition would assert against an already-repaired state.
    raw = store._get_conn().execute(
        "SELECT turn_group_number FROM canonical_turns WHERE conversation_id = ?",
        ("c",),
    ).fetchall()
    assert all(row[0] < 0 for row in raw), "fixture must start out legacy"

    projected = store.get_canonical_turn_reconcile_rows("c")

    assert projected, "fixture must produce rows"
    assert any(r.turn_group_number >= 0 for r in projected), (
        "the projected loader must trigger the legacy group backfill"
    )
    # And the full-row loader agrees, so the two do not diverge.
    full = store.get_all_canonical_turns("c")
    assert [r.turn_group_number for r in projected] == [
        r.turn_group_number for r in full
    ]
