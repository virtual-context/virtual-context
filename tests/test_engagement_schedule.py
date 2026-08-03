"""Randomized daily scheduling, computable without anything running.

The schedule is a pure function of a date and a seed, so the times it would
pick can be shown to a reviewer before anything is installed. Nothing here
starts a loop, registers a timer, or sleeps; enabling is a separate decision
that belongs with live posting.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from virtual_context.core.engagement import (
    SCHEDULE_ENABLED_BY_DEFAULT,
    ScheduleWindow,
    may_run_now,
    plan_day,
    preview_schedule,
)
from virtual_context.core.engagement.schedule import (
    WINDOW_END_HOUR,
    WINDOW_START_HOUR,
    ZONE,
)

EASTERN = ZoneInfo("America/New_York")


class TestShipsDisabled:
    def test_the_shipped_default_is_off(self):
        assert SCHEDULE_ENABLED_BY_DEFAULT is False

    def test_planning_is_pure_and_starts_nothing(self):
        """No side effect: planning the same day twice is identical."""
        first = plan_day(date(2026, 8, 3), seed="s")
        second = plan_day(date(2026, 8, 3), seed="s")
        assert first == second


class TestWindow:
    @pytest.mark.parametrize("offset", range(0, 40, 3))
    def test_every_planned_time_is_inside_the_shipped_window(self, offset):
        day = date(2026, 8, 3) + timedelta(days=offset)
        planned = plan_day(day, seed=f"seed-{offset}")
        local = planned.at.astimezone(ZoneInfo(ZONE))
        assert local.date() == day
        assert WINDOW_START_HOUR <= local.hour < WINDOW_END_HOUR, local

    def test_the_window_is_eight_to_eighteen_eastern(self):
        assert (WINDOW_START_HOUR, WINDOW_END_HOUR) == (8, 18)
        assert ZONE == "America/New_York"


class TestRandomisation:
    def test_a_seed_makes_it_reproducible(self):
        assert plan_day(date(2026, 8, 3), seed="a") == plan_day(
            date(2026, 8, 3), seed="a",
        )

    def test_different_days_get_different_times(self):
        times = {
            plan_day(date(2026, 8, 3) + timedelta(days=d), seed="x").at
            for d in range(14)
        }
        assert len(times) >= 12, "the schedule is barely varying"

    def test_different_seeds_give_different_times(self):
        a = plan_day(date(2026, 8, 3), seed="a").at
        b = plan_day(date(2026, 8, 3), seed="b").at
        assert a != b

    def test_unseeded_planning_still_lands_in_the_window(self):
        local = plan_day(date(2026, 8, 3)).at.astimezone(ZoneInfo(ZONE))
        assert WINDOW_START_HOUR <= local.hour < WINDOW_END_HOUR


class TestDaylightSaving:
    def test_the_spring_forward_day_is_handled(self):
        """2026-03-08: 02:00 does not exist in Eastern."""
        local = plan_day(date(2026, 3, 8), seed="dst").at.astimezone(EASTERN)
        assert local.date() == date(2026, 3, 8)
        assert WINDOW_START_HOUR <= local.hour < WINDOW_END_HOUR
        assert local.utcoffset() == timedelta(hours=-4)

    def test_the_fall_back_day_is_handled(self):
        """2026-11-01: 01:00 occurs twice, and the window is already EST.

        The ambiguous hour is 01:00, an hour the window never reaches, so a
        slot drawn between 08:00 and 18:00 on this day is unambiguously
        standard time. My first version of this test asserted -4 and was
        simply wrong about the clock.
        """
        local = plan_day(date(2026, 11, 1), seed="dst").at.astimezone(EASTERN)
        assert local.date() == date(2026, 11, 1)
        assert WINDOW_START_HOUR <= local.hour < WINDOW_END_HOUR
        assert local.utcoffset() == timedelta(hours=-5)

    def test_a_winter_day_is_standard_time(self):
        local = plan_day(date(2026, 12, 15), seed="dst").at.astimezone(EASTERN)
        assert local.utcoffset() == timedelta(hours=-5)


class TestOnePerDay:
    def test_one_plan_per_eastern_calendar_day(self):
        plans = [
            plan_day(date(2026, 8, 3) + timedelta(days=d), seed="x")
            for d in range(10)
        ]
        assert len({p.day for p in plans}) == 10
        assert len({p.idempotency_key for p in plans}) == 10

    def test_the_idempotency_key_is_stable_for_a_day(self):
        a = plan_day(date(2026, 8, 3), seed="x").idempotency_key
        b = plan_day(date(2026, 8, 3), seed="y").idempotency_key
        assert a == b, "the key identifies the day, not the draw"


class TestPreviewWithoutRunning:
    def test_a_reviewer_can_see_the_times_without_executing_anything(self):
        preview = preview_schedule(date(2026, 8, 3), days=7, seed="demo")
        assert len(preview) == 7
        for line in preview:
            assert "2026-" in line
            hour = int(line.split()[1].split(":")[0])
            assert WINDOW_START_HOUR <= hour < WINDOW_END_HOUR

    def test_the_preview_is_text_and_starts_nothing(self):
        assert all(isinstance(x, str) for x in preview_schedule(
            date(2026, 8, 3), days=3, seed="demo",
        ))


class TestMissedWindow:
    def test_a_missed_slot_may_still_run_inside_the_window(self):
        plan = plan_day(date(2026, 8, 3), seed="x")
        late = plan.at + timedelta(minutes=30)
        if late.astimezone(ZoneInfo(ZONE)).hour < WINDOW_END_HOUR:
            assert may_run_now(plan, now=late, already_posted=False) is True

    def test_it_never_runs_after_the_window_closes(self):
        plan = plan_day(date(2026, 8, 3), seed="x")
        closed = datetime.combine(
            date(2026, 8, 3), time(WINDOW_END_HOUR, 1),
        ).replace(tzinfo=ZoneInfo(ZONE))
        assert may_run_now(plan, now=closed, already_posted=False) is False

    def test_it_never_runs_twice_in_a_day(self):
        plan = plan_day(date(2026, 8, 3), seed="x")
        assert may_run_now(plan, now=plan.at, already_posted=True) is False
