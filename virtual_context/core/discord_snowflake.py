"""Decode a Discord message id into its creation time.

A Discord snowflake encodes its own creation timestamp in the high bits, so
the immutable message id is itself the authoritative message time. Stored
ingest timestamps describe when this system saw a row, not when the member
sent it, and the two diverge by the length of any backfill or repair.

The decode is the inverse of the bounds helpers used by time-windowed reads:
a read compares an id against integer bounds derived from a datetime, so the
filter runs against the stored id rather than decoding every candidate row.
"""

from __future__ import annotations

from datetime import datetime, timezone

DISCORD_EPOCH_MS = 1420070400000

# A snowflake's low 22 bits are worker, process, and sequence counters; the
# timestamp occupies everything above them.
_TIMESTAMP_SHIFT = 22


def snowflake_to_datetime(message_id: str | None) -> datetime | None:
    """UTC send time for a Discord message id, or ``None`` when undecodable.

    Anything that is not a run of digits — empty, absent, signed, fractional,
    or separator-bearing — has no provable send time and returns ``None``
    rather than a guess.
    """
    raw = (message_id or "").strip()
    if not raw.isdigit():
        return None
    ms = (int(raw) >> _TIMESTAMP_SHIFT) + DISCORD_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_snowflake_floor(moment: datetime) -> int:
    """Smallest snowflake whose send time is at or after *moment*.

    Pairs with ``datetime_to_snowflake_ceil`` to turn a datetime window into
    an inclusive integer range over stored message ids.
    """
    return (int(moment.timestamp() * 1000) - DISCORD_EPOCH_MS) << _TIMESTAMP_SHIFT


def datetime_to_snowflake_ceil(moment: datetime) -> int:
    """Largest snowflake whose send time is at or before *moment*.

    The low counter bits are saturated so every id minted during *moment*'s
    millisecond stays inside the window; without that the bound would exclude
    all but the first message of its millisecond.
    """
    base = (
        int(moment.timestamp() * 1000) - DISCORD_EPOCH_MS
    ) << _TIMESTAMP_SHIFT
    return base | ((1 << _TIMESTAMP_SHIFT) - 1)
