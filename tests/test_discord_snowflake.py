"""A Discord message id is the authoritative time a member sent a message.

The id encodes its own creation timestamp in its high bits. Stored ingest
timestamps record when this system wrote a row, which diverges from send
time by the length of any backfill or repair, so the id is what a
time-bounded read must compare against.
"""

from __future__ import annotations

from datetime import datetime, timezone

from virtual_context.core.discord_snowflake import (
    DISCORD_EPOCH_MS,
    snowflake_to_datetime,
)


def test_known_snowflake_decodes_to_its_creation_time():
    got = snowflake_to_datetime("1533235485244133478")
    assert got is not None
    assert got.tzinfo is timezone.utc
    assert got.year == 2026


def test_non_numeric_returns_none():
    assert snowflake_to_datetime("not-a-snowflake") is None
    assert snowflake_to_datetime("") is None
    assert snowflake_to_datetime(None) is None


def test_matches_the_plugin_formula_exactly():
    mid = 1533235485244133478
    expected_ms = (mid >> 22) + 1420070400000
    got = snowflake_to_datetime(str(mid))
    assert int(got.timestamp() * 1000) == expected_ms


def test_epoch_constant_is_the_discord_epoch():
    assert DISCORD_EPOCH_MS == 1420070400000
    assert datetime.fromtimestamp(
        DISCORD_EPOCH_MS / 1000, tz=timezone.utc,
    ) == datetime(2015, 1, 1, tzinfo=timezone.utc)


def test_surrounding_whitespace_is_tolerated():
    assert snowflake_to_datetime("  1533235485244133478  ") == (
        snowflake_to_datetime("1533235485244133478")
    )


def test_signed_and_fractional_forms_are_rejected():
    """``isdigit`` is the guard; a sign or a point is not a snowflake."""
    assert snowflake_to_datetime("-1533235485244133478") is None
    assert snowflake_to_datetime("1533235485244133478.0") is None
    assert snowflake_to_datetime("1_533_235_485_244_133_478") is None


def test_zero_snowflake_is_the_discord_epoch():
    assert snowflake_to_datetime("0") == datetime.fromtimestamp(
        DISCORD_EPOCH_MS / 1000, tz=timezone.utc,
    )


def test_floor_and_ceil_bracket_the_decoded_time():
    """The bounds are the inverse of the decode, so a window is exact."""
    from virtual_context.core.discord_snowflake import (
        datetime_to_snowflake_ceil,
        datetime_to_snowflake_floor,
    )

    moment = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    lo = datetime_to_snowflake_floor(moment)
    hi = datetime_to_snowflake_ceil(moment)
    assert lo <= hi
    assert snowflake_to_datetime(str(lo)) == moment
    # Every id minted during that millisecond stays inside the window; the
    # ceil saturates the counter bits so the last one is not excluded.
    assert snowflake_to_datetime(str(hi)) == moment
    assert hi - lo == (1 << 22) - 1


def test_bounds_round_trip_a_real_message_id():
    mid = 1533235485244133478
    sent = snowflake_to_datetime(str(mid))
    from virtual_context.core.discord_snowflake import (
        datetime_to_snowflake_ceil,
        datetime_to_snowflake_floor,
    )

    assert datetime_to_snowflake_floor(sent) <= mid
    assert mid <= datetime_to_snowflake_ceil(sent)
