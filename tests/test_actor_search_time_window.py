"""Bounding one participant's statements by the time they were sent.

Send time comes from ``source_message_id``, which encodes it. A stored
ingest timestamp records when this system wrote the row and diverges from
send time across a backfill or a repair, so it cannot answer "what did X
say last week". The bounds are compared in SQL against the stored id, and a
row with no numeric id has no provable send time and is excluded whenever a
bound is supplied.

The window must not weaken what this method already guarantees: only user
halves, only this actor, every candidate a requester candidate, and the
audience fence the roster admits members under.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from virtual_context.core.discord_snowflake import (
    datetime_to_snowflake_floor,
    snowflake_to_datetime,
)
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import SpeakerRetrievalContext

CONV = "sk:agent:vast:discord:guild:1524917037191925871"
OTHER = "sk:agent:vast:discord:guild:9999999999999999999"
ACTOR = "actor:discord:387316537012518913"
OTHER_ACTOR = "actor:discord:167509689360187392"
CHAN_A = "1524946242499514418"
CHAN_B = "1524946242499514419"

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def _id_at(moment: datetime) -> str:
    """A plausible message id whose encoded send time is *moment*."""
    return str(datetime_to_snowflake_floor(moment) + 1)


TEN_DAYS = NOW - timedelta(days=10)
THREE_DAYS = NOW - timedelta(days=3)
ONE_DAY = NOW - timedelta(days=1)

TEN_DAY_ID = _id_at(TEN_DAYS)
THREE_DAY_ID = _id_at(THREE_DAYS)
ONE_DAY_ID = _id_at(ONE_DAY)

SEEDED_TOTAL = 5  # rows belonging to ACTOR in CONV, any window


@pytest.fixture()
def store(tmp_path):
    st = SQLiteStore(db_path=str(tmp_path / "window.db"))
    st.upsert_conversation(tenant_id="t1", conversation_id=CONV)
    st.upsert_conversation(tenant_id="t1", conversation_id=OTHER)

    def row(n, text, message_id, *, actor=ACTOR, channel=CHAN_A, conv=CONV):
        st.save_canonical_turn(
            conv, n, text, "",
            canonical_turn_id=f"ct-{n}", turn_hash=f"h-{n}",
            sort_key=float(n), sender="optics", sender_actor_id=actor,
            source_message_id=message_id,
            origin_channel_id=channel,
            audience_conversation_id=conv, audience_attribution_version=1,
        )

    row(1, "ten days ago I started tesamorelin", TEN_DAY_ID)
    row(2, "three days ago I raised the dose", THREE_DAY_ID)
    row(3, "yesterday I logged bloodwork", ONE_DAY_ID)
    row(4, "no message id on this row", "")
    row(5, "different channel entirely", _id_at(THREE_DAYS), channel=CHAN_B)
    # Another actor, and another audience: neither may ever appear.
    row(6, "someone else entirely", _id_at(THREE_DAYS), actor=OTHER_ACTOR)
    row(7, "same actor, other audience", _id_at(THREE_DAYS), conv=OTHER)
    return st


@pytest.fixture()
def ctx():
    return SpeakerRetrievalContext(
        tenant_id="t1",
        owner_conversation_id=CONV,
        audience_conversation_id=CONV,
        requester_actor_id=ACTOR,
        original_active_user_text="what did optics say",
    )


def _refs(rows) -> set[str]:
    return {r.segment_ref for r in rows}


def test_no_window_preserves_existing_behaviour(store, ctx):
    """Regression guard for every caller that passes no bounds."""
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
    )
    assert len(rows) == SEEDED_TOTAL
    # The row with no message id is included when no bound is supplied: it
    # is only unusable when a window has to be proved.
    assert "ct-4" in _refs(rows)


def test_after_excludes_older_messages(store, ctx):
    cutoff = NOW - timedelta(days=7)
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx, after=cutoff,
    )
    assert "ct-1" not in _refs(rows)
    assert {"ct-2", "ct-3"} <= _refs(rows)


def test_before_excludes_newer_messages(store, ctx):
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
        before=NOW - timedelta(days=2),
    )
    assert "ct-3" not in _refs(rows)
    assert {"ct-1", "ct-2"} <= _refs(rows)


def test_window_is_inclusive_of_both_bounds(store, ctx):
    """A message sent exactly on a bound is inside the window."""
    lo = snowflake_to_datetime(TEN_DAY_ID)
    hi = snowflake_to_datetime(ONE_DAY_ID)
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx, after=lo, before=hi,
    )
    assert {"ct-1", "ct-3"} <= _refs(rows)


def test_rows_without_a_source_message_id_are_excluded_when_windowed(
    store, ctx,
):
    for kwargs in (
        {"after": NOW - timedelta(days=30)},
        {"before": NOW},
        {"after": NOW - timedelta(days=30), "before": NOW},
    ):
        rows = store.search_canonical_turns_by_actor(
            ACTOR, 50, CONV, speaker_context=ctx, **kwargs,
        )
        assert "ct-4" not in _refs(rows), kwargs


def test_channel_filter_restricts_to_the_allowlist(store, ctx):
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx, channel_ids=[CHAN_B],
    )
    assert _refs(rows) == {"ct-5"}


def test_channel_filter_accepts_several_channels(store, ctx):
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
        channel_ids=[CHAN_A, CHAN_B],
    )
    assert "ct-5" in _refs(rows)
    assert "ct-2" in _refs(rows)


def test_window_and_channel_compose(store, ctx):
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
        after=NOW - timedelta(days=7), channel_ids=[CHAN_A],
    )
    assert _refs(rows) == {"ct-2", "ct-3"}


def test_audience_fence_still_holds_with_a_window(store):
    """A window must not become a way around the audience predicate."""
    foreign = SpeakerRetrievalContext(
        tenant_id="t1",
        owner_conversation_id=CONV,
        audience_conversation_id=OTHER,
        requester_actor_id=ACTOR,
        original_active_user_text="what did optics say",
    )
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=foreign,
        after=NOW - timedelta(days=30),
    )
    assert rows == []


def test_authorship_scoping_still_holds_with_a_window(store, ctx):
    """Only this actor's rows, and every one a requester candidate."""
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
        after=NOW - timedelta(days=30),
    )
    assert "ct-6" not in _refs(rows)
    assert rows
    for r in rows:
        assert r.provenance is not None
        assert r.provenance.source_role == "requester"
        assert r.provenance.actor_id == ACTOR
        assert r.matched_side == "user"


def test_empty_channel_list_is_not_a_filter(store, ctx):
    """``[]`` means "no allowlist supplied", not "match nothing"."""
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx, channel_ids=[],
    )
    assert len(rows) == SEEDED_TOTAL


def test_window_that_excludes_everything_returns_empty(store, ctx):
    rows = store.search_canonical_turns_by_actor(
        ACTOR, 50, CONV, speaker_context=ctx,
        after=NOW + timedelta(days=1),
    )
    assert rows == []
