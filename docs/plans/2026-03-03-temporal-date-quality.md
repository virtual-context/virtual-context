# Temporal Date Quality & Fact Anchoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix fact date extraction ("recently" ≠ session date) and give the reader session-level temporal anchoring on all facts so temporal ordering questions can be answered from the facts block alone.

**Architecture:** Three coordinated changes — (1) V4 compactor prompt injects `SESSION DATE` so the LLM correctly distinguishes "today" from "recently"; (2) `fact.session_date` is persisted to the DB so every fact carries its origin session; (3) `_format_facts()` renders `[when: date]` or `[session: date]` on each line so the reader has temporal context without expanding tags.

**Tech Stack:** Python, SQLite, pytest. All changes are in `virtual_context/`. Tests in `tests/`.

---

### Task 1: Add `session_date` column to facts table (schema + migration)

**Files:**
- Modify: `virtual_context/storage/sqlite.py`

**Context:** The `facts` table has no `session_date` column. The `Fact` dataclass has a `session_date: str = ""` field (types.py:60) but it's never saved. The migration pattern used here matches the existing `fact_type` migration at sqlite.py:322-326.

**Step 1: Write the failing test**

In `tests/test_fact_enrichment.py`, add a new test class at the bottom:

```python
class TestFactSessionDate:
    def test_fact_session_date_default(self):
        f = Fact(subject="user", verb="hiked", object="Muir Woods")
        assert f.session_date == ""

    def test_store_and_retrieve_session_date(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(
            subject="user", verb="hiked", object="Muir Woods",
            session_date="2023/03/10 (Fri) 23:32",
        )
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].session_date == "2023/03/10 (Fri) 23:32"

    def test_session_date_defaults_to_empty_in_db(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(subject="user", verb="visited", object="Paris")
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert results[0].session_date == ""
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/yursilkidwai/projects/virtual-context
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate -v
```

Expected: FAIL — `store_facts` doesn't save `session_date`, `_row_to_fact` doesn't read it.

**Step 3: Add `session_date` to facts DDL in SCHEMA_SQL**

In `virtual_context/storage/sqlite.py`, in `SCHEMA_SQL`, the facts table definition (around line 287) currently ends with `superseded_by TEXT`. Add `session_date` before it:

```python
# Change this block (around line 287-305):
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    verb TEXT NOT NULL DEFAULT '',
    object TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    what TEXT NOT NULL DEFAULT '',
    who TEXT NOT NULL DEFAULT '',
    when_date TEXT NOT NULL DEFAULT '',
    "where" TEXT NOT NULL DEFAULT '',
    why TEXT NOT NULL DEFAULT '',
    fact_type TEXT NOT NULL DEFAULT 'personal',
    tags_json TEXT NOT NULL DEFAULT '[]',
    segment_ref TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    turn_numbers_json TEXT NOT NULL DEFAULT '[]',
    mentioned_at TEXT NOT NULL DEFAULT '',
    superseded_by TEXT
);

# To (add session_date before superseded_by):
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    verb TEXT NOT NULL DEFAULT '',
    object TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    what TEXT NOT NULL DEFAULT '',
    who TEXT NOT NULL DEFAULT '',
    when_date TEXT NOT NULL DEFAULT '',
    "where" TEXT NOT NULL DEFAULT '',
    why TEXT NOT NULL DEFAULT '',
    fact_type TEXT NOT NULL DEFAULT 'personal',
    tags_json TEXT NOT NULL DEFAULT '[]',
    segment_ref TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    turn_numbers_json TEXT NOT NULL DEFAULT '[]',
    mentioned_at TEXT NOT NULL DEFAULT '',
    session_date TEXT NOT NULL DEFAULT '',
    superseded_by TEXT
);
```

**Step 4: Add migration after the existing `fact_type` migration (sqlite.py ~line 325)**

Right after:
```python
conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type)")
```

Add:
```python
try:
    conn.execute("SELECT session_date FROM facts LIMIT 1")
except Exception:
    conn.execute("ALTER TABLE facts ADD COLUMN session_date TEXT NOT NULL DEFAULT ''")
```

**Step 5: Update `store_facts` INSERT to include `session_date` (sqlite.py ~line 1060)**

Change:
```python
conn.execute(
    """INSERT OR REPLACE INTO facts
    (id, subject, verb, object, status, what, who, when_date,
     "where", why, fact_type, tags_json, segment_ref, session_id,
     turn_numbers_json, mentioned_at, superseded_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        fact.id, fact.subject, fact.verb, fact.object,
        fact.status, fact.what, fact.who, fact.when_date,
        fact.where, fact.why, fact.fact_type, json.dumps(fact.tags),
        fact.segment_ref, fact.session_id, json.dumps(fact.turn_numbers),
        _dt_to_str(fact.mentioned_at), fact.superseded_by,
    ),
)
```

To:
```python
conn.execute(
    """INSERT OR REPLACE INTO facts
    (id, subject, verb, object, status, what, who, when_date,
     "where", why, fact_type, tags_json, segment_ref, session_id,
     turn_numbers_json, mentioned_at, session_date, superseded_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        fact.id, fact.subject, fact.verb, fact.object,
        fact.status, fact.what, fact.who, fact.when_date,
        fact.where, fact.why, fact.fact_type, json.dumps(fact.tags),
        fact.segment_ref, fact.session_id, json.dumps(fact.turn_numbers),
        _dt_to_str(fact.mentioned_at), fact.session_date or "",
        fact.superseded_by,
    ),
)
```

**Step 6: Update `_row_to_fact` to read `session_date` (sqlite.py ~line 1099)**

Change:
```python
def _row_to_fact(self, row: sqlite3.Row) -> Fact:
    return Fact(
        ...
        mentioned_at=_str_to_dt(row["mentioned_at"]) if row["mentioned_at"] else datetime.now(timezone.utc),
        superseded_by=row["superseded_by"],
    )
```

To (add `session_date` before `superseded_by`):
```python
def _row_to_fact(self, row: sqlite3.Row) -> Fact:
    return Fact(
        ...
        mentioned_at=_str_to_dt(row["mentioned_at"]) if row["mentioned_at"] else datetime.now(timezone.utc),
        session_date=row["session_date"] if "session_date" in row.keys() else "",
        superseded_by=row["superseded_by"],
    )
```

**Step 7: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate -v
```

Expected: PASS (all 3 tests).

**Step 8: Run full test suite to check for regressions**

```bash
.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' --ignore=tests/test_document_ingest.py -q
```

Expected: ~same count as before, no new failures.

**Step 9: Commit**

```bash
git add virtual_context/storage/sqlite.py tests/test_fact_enrichment.py
git commit -m "feat(storage): add session_date column to facts table with migration"
```

---

### Task 2: Populate `fact.session_date` in the compactor

**Files:**
- Modify: `virtual_context/core/compactor.py:342-354`

**Context:** In `_compact_one`, facts are built from parsed LLM output at lines 342-354. Each fact is missing `session_date`. `segment.session_date` is available at the top of `_compact_one` (it's used at line 314 for `SegmentMetadata`).

**Step 1: Write the failing test**

In `tests/test_fact_enrichment.py`, add to `TestFactSessionDate`:

```python
def test_compactor_sets_session_date_on_facts(self, tmp_path):
    """Compactor should stamp each extracted fact with the segment's session_date."""
    from unittest.mock import MagicMock
    from virtual_context.core.compactor import DomainCompactor
    from virtual_context.types import CompactorConfig, TaggedSegment, Message
    import json

    # Minimal LLM that returns a fact
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "summary": "User hiked Muir Woods.",
        "entities": [], "key_decisions": [], "action_items": [],
        "date_references": [], "refined_tags": ["hiking"],
        "related_tags": [],
        "facts": [{
            "subject": "user", "verb": "hiked", "object": "Muir Woods",
            "status": "completed", "fact_type": "personal",
            "what": "User hiked Muir Woods.", "who": "", "when": "", "where": "", "why": "",
        }],
    })

    segment = TaggedSegment(
        id="seg-001",
        primary_tag="hiking",
        tags=["hiking"],
        messages=[Message(role="user", content="I hiked Muir Woods today.")],
        session_date="2023/03/10 (Fri) 23:32",
    )

    compactor = DomainCompactor(
        llm_provider=llm,
        config=CompactorConfig(),
    )
    result = compactor.compact([segment])
    assert len(result[0].facts) == 1
    assert result[0].facts[0].session_date == "2023/03/10 (Fri) 23:32"
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate::test_compactor_sets_session_date_on_facts -v
```

Expected: FAIL — `session_date` is `""`.

**Step 3: Add `session_date` to Fact construction in `_compact_one`**

In `virtual_context/core/compactor.py`, the `facts.append(Fact(...))` block (lines 342-354). Add `session_date=segment.session_date or ""` to the call:

```python
facts.append(Fact(
    subject=_str(f.get("subject", "")),
    verb=_str(f.get("verb", f.get("role", ""))),
    object=_str(f.get("object", "")),
    status=status,
    what=_str(f.get("what", "")),
    who=_str(f.get("who", "")),
    when_date=_str(f.get("when", "")),
    where=_str(f.get("where", "")),
    why=_str(f.get("why", "")),
    fact_type=f.get("fact_type", "personal"),
    tags=refined_tags,
    session_date=segment.session_date or "",
))
```

**Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate -v
```

Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_fact_enrichment.py
git commit -m "feat(compactor): stamp fact.session_date from segment at compaction"
```

---

### Task 3: V4 compactor prompt — session_date injection + "recently vs today" rules

**Files:**
- Modify: `virtual_context/core/compactor.py:68-128, 259-267`

**Context:** `DEFAULT_SUMMARY_PROMPT` uses `.format(tags=..., target_tokens=..., conversation_text=...)`. We add `session_date` as a fourth template variable. Line 121 has the current `"when"` instruction that needs replacing with V4 rules. The `.format()` call at line 261 needs the new keyword argument.

**Step 1: Write the failing test**

In `tests/test_fact_enrichment.py`, add to `TestFactSessionDate`:

```python
def test_v4_prompt_recently_gives_empty_when(self, tmp_path):
    """'recently returned' should produce when='' not the session date."""
    from unittest.mock import MagicMock, call
    from virtual_context.core.compactor import DomainCompactor
    from virtual_context.types import CompactorConfig, TaggedSegment, Message
    import json

    captured_prompts = []

    llm = MagicMock()
    def capture_and_return(**kwargs):
        captured_prompts.append(kwargs.get("user", ""))
        return json.dumps({
            "summary": "User recently returned from Yosemite.",
            "entities": [], "key_decisions": [], "action_items": [],
            "date_references": [], "refined_tags": ["hiking"], "related_tags": [],
            "facts": [{
                "subject": "user", "verb": "returned from",
                "object": "solo camping trip to Yosemite",
                "status": "completed", "fact_type": "personal",
                "what": "User recently returned from Yosemite.",
                "who": "", "when": "", "where": "", "why": "",
            }],
        })
    llm.complete.side_effect = capture_and_return

    segment = TaggedSegment(
        id="seg-002", primary_tag="camping", tags=["camping"],
        messages=[Message(role="user", content="I recently returned from Yosemite.")],
        session_date="2023/04/20 (Thu) 04:17",
    )
    compactor = DomainCompactor(llm_provider=llm, config=CompactorConfig())
    result = compactor.compact([segment])

    # Session date must appear in the prompt
    assert "2023/04/20" in captured_prompts[0], "session_date not injected into prompt"
    # 'recently' → when="" (LLM returned "" and compactor should preserve it)
    assert result[0].facts[0].when_date == ""

def test_v4_prompt_today_gives_session_date_when(self, tmp_path):
    """'today' should produce when=session_date."""
    from unittest.mock import MagicMock
    from virtual_context.core.compactor import DomainCompactor
    from virtual_context.types import CompactorConfig, TaggedSegment, Message
    import json

    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "summary": "User hiked Big Sur today.",
        "entities": [], "key_decisions": [], "action_items": [],
        "date_references": [], "refined_tags": ["hiking"], "related_tags": [],
        "facts": [{
            "subject": "user", "verb": "hiked",
            "object": "Big Sur",
            "status": "completed", "fact_type": "personal",
            "what": "User hiked Big Sur today (2023/04/20).",
            "who": "", "when": "2023/04/20", "where": "", "why": "",
        }],
    })

    segment = TaggedSegment(
        id="seg-003", primary_tag="hiking", tags=["hiking"],
        messages=[Message(role="user", content="I hiked Big Sur today.")],
        session_date="2023/04/20 (Thu) 04:17",
    )
    compactor = DomainCompactor(llm_provider=llm, config=CompactorConfig())
    result = compactor.compact([segment])
    assert result[0].facts[0].when_date == "2023/04/20"
```

**Step 2: Run to verify tests fail (or pass for wrong reason)**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate::test_v4_prompt_recently_gives_empty_when tests/test_fact_enrichment.py::TestFactSessionDate::test_v4_prompt_today_gives_session_date_when -v
```

The `recently` test will FAIL because session_date is not in the prompt yet.

**Step 3: Modify `DEFAULT_SUMMARY_PROMPT`**

In `virtual_context/core/compactor.py`, change the prompt string:

At the very top of `DEFAULT_SUMMARY_PROMPT` (before "Summarize the following"), add:
```
SESSION DATE: {session_date}

```

Change line 121 (`"when"` field):
```
- "when": date if mentioned (ISO format or free-form, empty string if n/a)
```
To:
```
- "when": the date this event occurred.
  DATE RULES — read carefully:
  "today" / "this morning" / "just now" = {session_date} (use the session date above).
  "recently" / "last week" / "a while ago" / "just got back" = date UNKNOWN,
  event happened BEFORE today but we don't know when — use "".
  If no temporal language at all, use "".
```

**Step 4: Pass `session_date` into `.format()` at the call site (~line 261)**

Change:
```python
prompt = DEFAULT_SUMMARY_PROMPT.format(
    tags=tags_str,
    target_tokens=target_tokens,
    conversation_text=conversation_text,
)
```

To:
```python
prompt = DEFAULT_SUMMARY_PROMPT.format(
    tags=tags_str,
    target_tokens=target_tokens,
    conversation_text=conversation_text,
    session_date=segment.session_date or "(unknown)",
)
```

**Step 5: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSessionDate -v
```

Expected: all tests PASS.

**Step 6: Run full suite**

```bash
.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' --ignore=tests/test_document_ingest.py -q
```

Expected: no regressions.

**Step 7: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_fact_enrichment.py
git commit -m "feat(compactor): V4 prompt — inject session_date, fix 'recently vs today' date extraction"
```

---

### Task 4: Show temporal context in `_format_facts()`

**Files:**
- Modify: `virtual_context/core/assembler.py:191-211`

**Context:** `_format_facts()` currently outputs `- user | hiked | Muir Woods — User hiked...` with no temporal info. We append `[when: date]` if `fact.when_date` is set, else `[session: date]` if `fact.session_date` is set. This gives the reader temporal anchoring without changing the budget calculation significantly.

**Step 1: Write the failing test**

In `tests/test_fact_enrichment.py`, add a new class:

```python
class TestFormatFacts:
    def _make_assembler(self):
        from virtual_context.core.assembler import ContextAssembler
        from virtual_context.types import AssemblerConfig
        return ContextAssembler(config=AssemblerConfig())

    def test_format_facts_shows_when_date(self):
        assembler = self._make_assembler()
        f = Fact(
            subject="user", verb="hiked", object="Big Sur",
            what="User hiked Big Sur.",
            when_date="2023/04/20", session_date="2023/04/20 (Thu) 04:17",
        )
        result = assembler._format_facts([f], max_tokens=500)
        assert "[when: 2023/04/20]" in result
        assert "[session:" not in result  # when_date takes precedence

    def test_format_facts_shows_session_date_when_no_when(self):
        assembler = self._make_assembler()
        f = Fact(
            subject="user", verb="hiked", object="Muir Woods",
            what="User hiked Muir Woods.",
            when_date="", session_date="2023/03/10 (Fri) 23:32",
        )
        result = assembler._format_facts([f], max_tokens=500)
        assert "[session: 2023/03/10 (Fri) 23:32]" in result
        assert "[when:" not in result

    def test_format_facts_no_suffix_when_no_dates(self):
        assembler = self._make_assembler()
        f = Fact(
            subject="user", verb="prefers", object="dark theme",
            what="User prefers dark theme.",
            when_date="", session_date="",
        )
        result = assembler._format_facts([f], max_tokens=500)
        assert "[when:" not in result
        assert "[session:" not in result
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFormatFacts -v
```

Expected: FAIL — no `[when:]` or `[session:]` in current output.

**Step 3: Modify `_format_facts()` in assembler**

In `virtual_context/core/assembler.py`, change the fact line builder (lines 201-203):

```python
# Current:
line = f"- {fact.subject} | {fact.verb} | {fact.object}"
if fact.what:
    line += f" — {fact.what}"

# New:
line = f"- {fact.subject} | {fact.verb} | {fact.object}"
if fact.what:
    line += f" — {fact.what}"
if fact.when_date:
    line += f" [when: {fact.when_date}]"
elif fact.session_date:
    line += f" [session: {fact.session_date}]"
```

**Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFormatFacts -v
```

Expected: all 3 tests PASS.

**Step 5: Run full suite**

```bash
.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' --ignore=tests/test_document_ingest.py -q
```

Expected: no regressions.

**Step 6: Commit**

```bash
git add virtual_context/core/assembler.py tests/test_fact_enrichment.py
git commit -m "feat(assembler): show [when:] or [session:] temporal context in facts block"
```

---

### Task 5: Verify end-to-end with existing integration tests

**Files:**
- Read: `tests/test_fact_enrichment.py` (existing integration tests)

**Step 1: Run all fact-related tests**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py -v
```

Expected: all pass.

**Step 2: Run full suite one more time**

```bash
.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' --ignore=tests/test_document_ingest.py -q
```

Expected: ~1000 tests, no failures, under 30 seconds.

**Step 3: Final commit if any test files were touched**

```bash
git status
# Only commit if there are uncommitted changes
git add -p
git commit -m "test: verify temporal date quality end-to-end"
```
