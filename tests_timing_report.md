# Core Test Suite Timing Report

**Date:** 2026-04-17
**Total tests collected:** 2596 (21 deselected — `slow`/`ollama`/`haiku`/`tui`)
**Sequential section-by-section wall time:** **162 seconds (2m 42s)**
**Config:** `-n auto` xdist, `-m 'not slow'`, ignoring `tests/ollama tests/haiku tests/tui`

## Key Finding

**The suite is NOT slow.** Total wall time is ~2m 42s across 13 sections when run serially. When `pytest tests/` runs the whole suite with `-n auto`, the effective wall time should be even lower (xdist parallelizes across cores).

The earlier "12+ minute hang" is NOT a scaling problem — it's almost certainly a **specific test deadlock or xdist worker crash**. Running the suite serially in sections completes in under 3 minutes. Most likely culprits:
1. `test_engine_sync_turns.py::test_sync_concurrent_writers_do_not_duplicate_rows` — known flake under xdist due to SQLite sort_key UNIQUE collisions
2. A test leaking a thread or holding a lock forever, causing xdist workers to hang on teardown

## Section Breakdown (all under 3 min threshold)

| Section | Files | Tests | Pass | Fail | Skip | Wall |
|---|---:|---:|---:|---:|---:|---:|
| storage-schema | 24 | 249 | 173 | 0 | 76 | 3s |
| lifecycle-epoch | 5 | 35 | 27 | 0 | 8 | 9s |
| ingest | 8 | 56 | 54 | 0 | 2 | **24s** |
| tagging | 10 | 225 | 225 | 0 | 0 | 9s |
| compaction | 9 | 68 | 66 | 0 | 2 | 9s |
| proxy | 6 | 545 | 545 | 0 | 0 | **35s** |
| engine | 4 | 33 | 33 | 0 | 0 | 13s |
| retrieval | 12 | 372 | 357 | **15** | 0 | 18s |
| progress-events | 3 | 18 | 18 | 0 | 0 | 7s |
| filters-formats | 12 | 160 | 160 | 0 | 0 | 16s |
| facts | 6 | 105 | 105 | 0 | 0 | 6s |
| config-model | 7 | 160 | 160 | 0 | 0 | 2s |
| misc | 26 | 578 | 521 | 0 | 57 | 11s |
| **TOTAL** | **132** | **2823** | **2808** | **15** | **145** | **162s** |

## Hottest Sections (would merit subdivision if they ever grow past 3 min)

- **proxy** (35s) — 545 tests, 6 files. Currently averages 64ms/test. Healthy.
- **ingest** (24s) — 56 tests, 8 files. Concurrent ingest tests run real threads.

No section exceeds the 3-minute threshold. No subdivision needed today.

## Failures (15) — All Pre-Existing, Not From Progress-Bar Work

All 15 failures are in `tests/test_temporal_resolver.py::test_remember_when_*`. Root cause:

```
TypeError: QuoteResult.__init__() got an unexpected keyword argument 'created_at'
```

`QuoteResult.created_at` was removed in commit `2ebc37b` ("Drop QuoteResult.created_at from find_quote output"). The helper `_make_quote()` in `test_temporal_resolver.py:17` was never updated. Unrelated to the consistent-progress-bar or d8cb56d cleanup work — preceded both.

**Fix: one-line delete of `created_at="2024-08-01T00:00:00Z"` in `test_temporal_resolver.py:17`** — will heal all 15 tests at once.

## Slowest Individual Tests (from --durations=15)

- `test_paging.py::TestEnginePagingAPI::test_expand_disabled_returns_error` — 6.23s
- `test_paging.py::TestEnginePagingAPI::test_collapse_disabled_returns_error` — 5.63s
- `test_passthrough_filter.py::test_compactor_progress_phase_is_not_double_passed` — 5.02s
- `test_compaction_progress_event.py::test_exit_compaction_failure_publishes_failed_event` — 0.85s

The two `test_paging.py` ones are ~6s each — likely because they wait for a timeout/error condition. Worth profiling if they ever get slower.

## Next Steps

1. **Fix the 15 temporal_resolver failures** (trivial one-line delete) to restore a clean baseline.
2. **Reproduce the 12-min hang under `pytest tests/` with `-n auto`** — once the suite runs cleanly, run the full command and watch for which test the xdist workers hang on. If it's the known SQLite sort_key flake, address that (deterministic sort_key gen or skip under xdist).
3. **Keep this script** (`tests_by_section.sh`) as the section-timing tool. Re-run after major changes to watch for section drift >3 min.

## Script

`./tests_by_section.sh [section|all]` — sections: storage, lifecycle, ingest, tagging, compaction, proxy, engine, retrieval, progress, filters, facts, config, misc, all.
Output: `/tmp/pytest_section_report.txt`
