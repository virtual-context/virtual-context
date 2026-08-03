"""Pick one time per Eastern day, without anything running.

Planning is a pure function of a calendar day and a seed. It starts no loop,
registers no timer, sleeps for nothing, and touches no state, so the times a
schedule would choose can be shown to a reviewer before any of it is
enabled — and enabling is a separate decision that belongs with live
posting.

Two properties the spec asks for and this shape gives for free. Randomness
is per day rather than one fixed hour plus a long sleep, because a fixed
wake-up with a delay is not a random time, it is a fixed time that occupies
a worker. And idempotency keys on the calendar day rather than the drawn
time, so a restart, a duplicate wake-up, or a manual re-run resolves to the
same day's single slot instead of minting a second one.

Daylight saving is handled by construction: the time is drawn in local
Eastern terms and then localised, so an 08:00-18:00 window is 08:00-18:00
for a reader in that zone on every day of the year, including the days that
are 23 or 25 hours long.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ZONE = "America/New_York"
WINDOW_START_HOUR = 8
WINDOW_END_HOUR = 18

# Ships dark. Enabling belongs with live posting and the durable migration,
# as one reviewed decision rather than three quiet ones.
SCHEDULE_ENABLED_BY_DEFAULT = False


@dataclass(frozen=True)
class ScheduleWindow:
    """One day's planned slot."""

    day: date
    at: datetime
    idempotency_key: str


def _draw(day: date, seed: str | None) -> random.Random:
    if seed is None:
        return random.Random()
    digest = hashlib.blake2b(
        f"{seed}:{day.isoformat()}".encode(), digest_size=8,
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def plan_day(day: date, seed: str | None = None) -> ScheduleWindow:
    """The slot for *day*. Pure: no state, no side effect, nothing started."""
    rng = _draw(day, seed)
    span_seconds = (WINDOW_END_HOUR - WINDOW_START_HOUR) * 3600
    offset = rng.randrange(span_seconds)
    local = datetime.combine(
        day, time(WINDOW_START_HOUR),
    ).replace(tzinfo=ZoneInfo(ZONE)) + timedelta(seconds=offset)
    # Re-localise after the arithmetic so a DST boundary inside the window
    # cannot push the result into the previous or next civil hour.
    local = local.replace(tzinfo=None).replace(tzinfo=ZoneInfo(ZONE))
    return ScheduleWindow(
        day=day,
        at=local,
        # Keyed on the day, never the draw: a re-plan must resolve to the
        # same slot rather than mint a second one.
        idempotency_key=f"engagement:{day.isoformat()}",
    )


def may_run_now(
    plan: ScheduleWindow,
    *,
    now: datetime,
    already_posted: bool,
) -> bool:
    """Whether the day's post may run at *now*.

    A missed slot may still run later the same day while the window is open,
    because a late question is better than an absent one. It may never run
    after the window closes merely to catch up, and never twice.
    """
    if already_posted:
        return False
    local = now.astimezone(ZoneInfo(ZONE))
    if local.date() != plan.day:
        return False
    if not (WINDOW_START_HOUR <= local.hour < WINDOW_END_HOUR):
        return False
    return local >= plan.at.astimezone(ZoneInfo(ZONE))


def preview_schedule(
    start: date, *, days: int = 7, seed: str | None = None,
) -> list[str]:
    """Render the times a schedule WOULD pick. Executes nothing."""
    lines: list[str] = []
    for offset in range(max(0, days)):
        day = start + timedelta(days=offset)
        plan = plan_day(day, seed=seed)
        local = plan.at.astimezone(ZoneInfo(ZONE))
        lines.append(
            f"{day.isoformat()} {local:%H:%M} {local:%Z} "
            f"(key {plan.idempotency_key})"
        )
    return lines
