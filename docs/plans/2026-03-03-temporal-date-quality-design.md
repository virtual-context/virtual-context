# Temporal Date Quality & Fact Anchoring

**Date:** 2026-03-03
**Branch:** feat/enriched-fact-extraction

## Problem

For temporal ordering questions (e.g., "what order did I take these three trips?"), the reader needs to know when each event happened. Two failures were identified:

1. **Wrong `when_date` extraction**: The compactor LLM conflates "recently" with the session date. When a user says "I recently returned from Yosemite" in a session dated 2023/04/20, the LLM stamps `when="2023/04/20"` — the session date — even though the trip happened *before* that session.

2. **No session-level temporal anchoring on facts**: Even when `when_date` is correct (or correctly empty), the reader has no way to know *which session* a fact was discussed in. The `Fact.session_date` field exists in the dataclass but is never populated, so facts carry no temporal context at all.

**Motivating question:** `gpt4_7f6b06db` — "What is the order of the three trips I took in the past three months, from earliest to latest?" Gold: Muir Woods → Big Sur → Yosemite.

After analysis of the actual store.db, the three trips appear in sessions dated:
- Muir Woods: only in 2023/03/10 segment
- Big Sur: first "recently returned" in 2023/04/20 sessions
- Yosemite: most recently discussed in 2023/05/15 sessions

With `session_date` on facts, the reader can infer the correct ordering from session proxies.

## Design

### Change 1: V4 compactor prompt (session_date injection)

**File:** `virtual_context/core/compactor.py`

Modify `DEFAULT_SUMMARY_PROMPT` to:
- Accept `{session_date}` as a template variable
- Inject `SESSION DATE: {session_date}` at the top of the prompt
- Replace the existing `"when"` field instruction with V4 formulation:

```
- "when": the date this event occurred.
  DATE RULES — read carefully:
  "today" / "this morning" / "just now" = {session_date} (use the session date).
  "recently" / "last week" / "a while ago" = date UNKNOWN — the event happened
  BEFORE today but we don't know when. Use "".
  If no temporal language at all, use "".
```

In `_compact_one`, pass `session_date=segment.session_date or ""` into `.format()`.

If `session_date` is empty (e.g., test segments), the instruction degrades gracefully: "today" = "" which is still a correct fallback.

### Change 2: Populate `fact.session_date` at compaction

**File:** `virtual_context/core/compactor.py`, in `_compact_one` when building each `Fact`

Add `session_date=segment.session_date or ""` to the `Fact(...)` constructor call.

`Fact.session_date` already exists as a dataclass field with default `""`. The SQLite schema does not have a `session_date` column on the `facts` table — this needs to be added as a migration (or new column with default).

### Change 3: Show temporal context in `_format_facts()`

**File:** `virtual_context/core/assembler.py`

Append temporal context to each fact line in `_format_facts()`:
- If `fact.when_date`: append `[when: {when_date}]`
- Else if `fact.session_date`: append `[session: {session_date}]`
- Otherwise: no suffix (unchanged)

Example output:
```
<facts>
- user | hiked | Dipsea Trail at Muir Woods — User hiked Dipsea Trail [session: 2023/03/10 (Fri) 23:32]
- user | visited | Big Sur and Monterey — User completed road trip to Big Sur [session: 2023/04/20 (Thu) 04:17]
- user | returned | from solo camping trip to Yosemite — User completed solo camping trip to Yosemite [session: 2023/05/15 (Mon) 09:54]
</facts>
```

### Change 4: SQLite schema — add `session_date` to facts table

**File:** `virtual_context/storage/sqlite.py`

Add `session_date TEXT NOT NULL DEFAULT ""` column to the `facts` table DDL and to the INSERT/SELECT statements.

## What This Fixes

- **V4 prompt**: Correctly leaves `when_date=""` for "recently" language instead of wrongly stamping the session date. Correctly sets `when_date=session_date` for "today"/"this morning" language.
- **session_date on facts**: Gives the reader session-level temporal anchoring for all facts, even when explicit dates are absent. For `gpt4_7f6b06db`, this provides session proxies of March 10 / April 20 / May 15 → correct ordering.
- **`_format_facts()` temporal output**: Reader can see temporal context without expanding tags or calling `vc_remember_when`.

## What This Does Not Fix

- Muir Woods has no explicit "today" marker → `when_date` stays `""` even with V4. Correct, since the trip date is genuinely unknown from the text; session_date is the fallback.
- Multiple Yosemite facts across sessions will show different session_dates. Requires reader inference to pick the "most recent discussion = most recent trip" pattern.
- Cross-segment date propagation (a fact in one segment knowing the date established in another segment) is not addressed. Supersession handles contradictions but not date enrichment.

## Files Changed

1. `virtual_context/core/compactor.py` — prompt + fact.session_date
2. `virtual_context/core/assembler.py` — `_format_facts()` temporal suffix
3. `virtual_context/storage/sqlite.py` — schema migration for `session_date` on facts
