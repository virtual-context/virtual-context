# Bug log

## BUG-068 — Actor card style observations become repeated speech habits

- **Reported:** 2026-09-05.
- **Cause:** The influence-only wrapper rendered entries without explaining how each kind should affect a response. A style observation could become a repeated address term instead of guiding register naturally.
- **Fix:** Add constant orientation before the JSON entries: follow communication preferences, let interaction style shape register without recurring verbal tics, and use goals/history for relevance and depth. Entry scalars, angle-bracket escaping, selection and stored kinds are unchanged.
- **Regression:** `tests/test_actor_card_assembly.py::test_actor_card_orientation_precedes_entries` and `::test_actor_card_render_golden`; both failed before the fix. Existing escaping and budgeting checks retain their guarantees with the added wrapper cost.
- **Validation:** Both new regressions failed before the fix; six focused rendering, escaping and budget checks passed afterward. The full suite passed 6,038 tests with 506 skips and exposed two fixture assumptions: a character-count card budget and an epoch-race simulation using a nested transaction. Both fixtures were corrected; their targeted checks and the three rendering regressions passed (5 tests).
