# Enriched Fact Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve fact/memory extraction quality from ~75% to >85% LongMemEval by enriching extraction prompts, adding fact types, wiring supersession into the conversation pipeline, and adding post-compaction dedup.

**Architecture:** Enriched tagger + compactor prompts produce richer facts with `fact_type` classification. After compaction stores facts, a new supersession pass detects contradictions/updates. A periodic dedup pass merges duplicates. Both use separated LLM prompts (validated in v4 POC).

**Tech Stack:** Python dataclasses, SQLite (ALTER TABLE migration), LLM prompts (Qwen3-30B default via OpenRouter), pytest

---

### Task 1: Add `fact_type` and `what` to FactSignal

**Files:**
- Modify: `virtual_context/types.py:25-33`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

```python
# tests/test_fact_enrichment.py
"""Tests for enriched fact extraction fields (fact_type, what)."""

from virtual_context.types import FactSignal, Fact


class TestFactSignalEnrichment:
    def test_fact_signal_has_fact_type_default(self):
        fs = FactSignal(subject="user", verb="runs", object="5K")
        assert fs.fact_type == "personal"

    def test_fact_signal_accepts_fact_type(self):
        fs = FactSignal(subject="user", verb="runs", object="5K", fact_type="experience")
        assert fs.fact_type == "experience"

    def test_fact_signal_has_what_default(self):
        fs = FactSignal(subject="user", verb="runs", object="5K")
        assert fs.what == ""

    def test_fact_signal_accepts_what(self):
        fs = FactSignal(subject="user", verb="runs", object="5K",
                        what="User runs a 5K charity race every spring.")
        assert fs.what == "User runs a 5K charity race every spring."
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/yursilkidwai/projects/virtual-context && .venv/bin/pytest tests/test_fact_enrichment.py::TestFactSignalEnrichment -v`
Expected: FAIL — `TypeError: FactSignal.__init__() got an unexpected keyword argument 'fact_type'`

**Step 3: Write minimal implementation**

In `virtual_context/types.py`, add two fields to `FactSignal` (after line 32):

```python
@dataclass
class FactSignal:
    """Lightweight fact signal extracted per-turn by the tagger.
    Cheap to produce, may be noisy/incomplete. Consolidated at compaction."""
    subject: str = ""
    verb: str = ""       # free-form action verb
    object: str = ""
    status: str = ""     # TemporalStatus value
    fact_type: str = "personal"  # personal|experience|world
    what: str = ""               # full-sentence memory with ALL specifics
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestFactSignalEnrichment -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add virtual_context/types.py tests/test_fact_enrichment.py
git commit -m "feat: add fact_type and what fields to FactSignal"
```

---

### Task 2: Add `fact_type` to Fact dataclass

**Files:**
- Modify: `virtual_context/types.py:35-59`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestFactEnrichment:
    def test_fact_has_fact_type_default(self):
        f = Fact(subject="user", verb="runs", object="5K")
        assert f.fact_type == "personal"

    def test_fact_accepts_fact_type(self):
        f = Fact(subject="user", verb="runs", object="5K", fact_type="world")
        assert f.fact_type == "world"

    def test_fact_type_values(self):
        for ft in ("personal", "experience", "world"):
            f = Fact(fact_type=ft)
            assert f.fact_type == ft
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestFactEnrichment -v`
Expected: FAIL — `TypeError: Fact.__init__() got an unexpected keyword argument 'fact_type'`

**Step 3: Write minimal implementation**

In `virtual_context/types.py`, add `fact_type` to `Fact` (after the `why` field, before `tags`):

```python
    why: str = ""
    fact_type: str = "personal"  # personal|experience|world
    # Provenance
    tags: list[str] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add virtual_context/types.py tests/test_fact_enrichment.py
git commit -m "feat: add fact_type field to Fact dataclass"
```

---

### Task 3: SQLite migration — add `fact_type` column

**Files:**
- Modify: `virtual_context/storage/sqlite.py:287-304` (CREATE TABLE), `virtual_context/storage/sqlite.py:1043-1089` (store_facts), `virtual_context/storage/sqlite.py:1091-1110` (_row_to_fact)
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
import tempfile
from pathlib import Path
from virtual_context.storage.sqlite import SQLiteStore


class TestSQLiteFactType:
    def test_store_and_query_fact_type(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(
            subject="user", verb="runs", object="5K",
            fact_type="experience", what="User runs 5K races.",
        )
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].fact_type == "experience"

    def test_fact_type_defaults_to_personal_in_db(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        f = Fact(subject="user", verb="cooks", object="pasta")
        store.store_facts([f])
        results = store.query_facts(subject="user")
        assert results[0].fact_type == "personal"

    def test_query_facts_with_fact_type_filter(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.store_facts([
            Fact(subject="user", verb="runs", object="5K", fact_type="personal"),
            Fact(subject="user", verb="learned", object="interval training", fact_type="experience"),
            Fact(subject="Emily", verb="lives in", object="Portland", fact_type="world"),
        ])
        personal = store.query_facts(subject="user", fact_type="personal")
        assert len(personal) == 1
        assert personal[0].verb == "runs"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestSQLiteFactType -v`
Expected: FAIL — `fact_type` column doesn't exist / `query_facts` doesn't accept `fact_type`

**Step 3: Write minimal implementation**

3a. Update CREATE TABLE in `sqlite.py:287-304` — add `fact_type TEXT NOT NULL DEFAULT 'personal'` after `superseded_by`:

```python
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
```

Add index: `CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type);`

3b. Add migration for existing DBs. In the `_migrate()` method (or equivalent init), add:

```python
# Migration: add fact_type column if missing
try:
    conn.execute("SELECT fact_type FROM facts LIMIT 1")
except Exception:
    conn.execute("ALTER TABLE facts ADD COLUMN fact_type TEXT NOT NULL DEFAULT 'personal'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type)")
```

3c. Update `store_facts` INSERT (line 1053-1075) — add `fact_type` to the column list and values:

```python
conn.execute(
    """INSERT OR REPLACE INTO facts
    (id, subject, verb, object, status, what, who, when_date,
     "where", why, fact_type, tags_json, segment_ref, session_id,
     turn_numbers_json, mentioned_at, superseded_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        fact.id,
        fact.subject,
        fact.verb,
        fact.object,
        fact.status,
        fact.what,
        fact.who,
        fact.when_date,
        fact.where,
        fact.why,
        fact.fact_type,
        json.dumps(fact.tags),
        fact.segment_ref,
        fact.session_id,
        json.dumps(fact.turn_numbers),
        _dt_to_str(fact.mentioned_at),
        fact.superseded_by,
    ),
)
```

3d. Update `_row_to_fact` (line 1091-1110) — add `fact_type`:

```python
def _row_to_fact(self, row: sqlite3.Row) -> Fact:
    return Fact(
        ...
        why=row["why"],
        fact_type=row["fact_type"] if "fact_type" in row.keys() else "personal",
        tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
        ...
    )
```

3e. Update `query_facts` signature (line 1132-1188) — add `fact_type` parameter:

```python
def query_facts(
    self,
    *,
    subject: str | None = None,
    verb: str | None = None,
    verbs: list[str] | None = None,
    object_contains: str | None = None,
    status: str | None = None,
    fact_type: str | None = None,  # NEW
    tags: list[str] | None = None,
    limit: int = 50,
) -> list[Fact]:
```

Add filtering logic:

```python
if fact_type is not None:
    conditions.append("f.fact_type = ?")
    params.append(fact_type)
```

3f. Update `ContextStore` ABC in `virtual_context/core/store.py:137-149` — add `fact_type` parameter:

```python
def query_facts(
    self,
    *,
    subject: str | None = None,
    verb: str | None = None,
    verbs: list[str] | None = None,
    object_contains: str | None = None,
    status: str | None = None,
    fact_type: str | None = None,
    tags: list[str] | None = None,
    limit: int = 50,
) -> list[Fact]:
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py -v`
Expected: PASS (10 tests)

**Step 5: Run full test suite to check for regressions**

Run: `.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -x -q`
Expected: ~960 tests pass

**Step 6: Commit**

```bash
git add virtual_context/storage/sqlite.py virtual_context/core/store.py tests/test_fact_enrichment.py
git commit -m "feat: add fact_type column to SQLite facts table with migration"
```

---

### Task 4: Enrich tagger prompt — fact extraction instructions

**Files:**
- Modify: `virtual_context/core/tag_generator.py:88-94` (detailed prompt), `virtual_context/core/tag_generator.py:112-113` (compact prompt)
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestTaggerPromptEnrichment:
    def test_detailed_prompt_has_fact_type(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_DETAILED
        assert "fact_type" in TAG_GENERATOR_PROMPT_DETAILED
        assert "personal|experience|world" in TAG_GENERATOR_PROMPT_DETAILED

    def test_detailed_prompt_has_what_field(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_DETAILED
        assert '"what"' in TAG_GENERATOR_PROMPT_DETAILED

    def test_detailed_prompt_suppresses_meta_verbs(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_DETAILED
        assert "asks about" in TAG_GENERATOR_PROMPT_DETAILED or "conversational act" in TAG_GENERATOR_PROMPT_DETAILED

    def test_compact_prompt_has_fact_type(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_COMPACT
        assert "fact_type" in TAG_GENERATOR_PROMPT_COMPACT

    def test_compact_prompt_has_what_field(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_COMPACT
        assert '"what"' in TAG_GENERATOR_PROMPT_COMPACT
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestTaggerPromptEnrichment -v`
Expected: FAIL — current prompts don't contain `fact_type` or `what`

**Step 3: Write minimal implementation**

Replace lines 88-94 (the fact extraction + JSON schema lines) in `TAG_GENERATOR_PROMPT_DETAILED`:

```python
- Extract facts about the user's life, experiences, preferences, plans, and world.
  For each fact, classify:
  - "fact_type": "personal" (user's life/identity/preferences), "experience" (assistant-provided info the user engaged with), or "world" (facts about other people, places, things in the user's world)
  - "subject": who (usually "user"; proper names for others)
  - "verb": the exact action verb from the conversation (e.g. "led", "ordered", "prefers", "lives in")
  - "object": what (specific noun phrase — preserve ALL numbers, names, dates, amounts)
  - "status": one of: active, completed, planned, abandoned, recurring
  - "what": one full sentence capturing the complete fact with ALL specifics preserved.
    WRONG: "User has a personal best time." RIGHT: "User has a personal best 5K time of 27:12."
    WRONG: "User paid a parking ticket." RIGHT: "User paid a $40 parking ticket."
  Extract the FACT behind the question, not the conversational act.
  WRONG: "user asks about Cairo restaurants" RIGHT: "user wants to try authentic Egyptian food in Cairo"
  DO NOT extract: mere asks, mentions, discusses, requests for information.
  Only extract facts with genuine substance. Skip greetings and filler.
  When a pronoun refers to a named person mentioned earlier, resolve it: "Emily (user's college roommate)".
- Return JSON only: {{"tags": ["tag1", "tag2"], "primary": "tag1", "temporal": false, "related_tags": ["alt1", "alt2"], "facts": [{{"subject": "user", "verb": "...", "object": "...", "status": "...", "fact_type": "personal", "what": "..."}}]}}
```

Replace lines 112-115 in `TAG_GENERATOR_PROMPT_COMPACT` similarly:

```python
- Extract facts: {{"subject": "user", "verb": "exact action verb", "object": "noun phrase with ALL specifics", "status": "active|completed|planned|abandoned|recurring", "fact_type": "personal|experience|world", "what": "full sentence with ALL specifics"}}
  Use the real verb (e.g. "led", "ordered", "prefers"). Extract the fact behind the question, not the conversational act.
  DO NOT extract mere asks/mentions/discusses. Preserve ALL numbers, names, dates, amounts.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestTaggerPromptEnrichment -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/tag_generator.py tests/test_fact_enrichment.py
git commit -m "feat: enrich tagger prompt with fact_type, what, meta-verb suppression"
```

---

### Task 5: Update tagger FactSignal parsing

**Files:**
- Modify: `virtual_context/core/tag_generator.py:396-407`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.types import TagGeneratorConfig


class TestTaggerParsing:
    def test_parses_fact_type_from_response(self):
        llm = type('MockLLM', (), {
            'complete': lambda self, **kw: '{"tags": ["running"], "primary": "running", "temporal": false, "related_tags": [], "facts": [{"subject": "user", "verb": "runs", "object": "5K", "status": "active", "fact_type": "experience", "what": "User runs 5K races."}]}'
        })()
        gen = LLMTagGenerator(llm, TagGeneratorConfig(type="llm"))
        result = gen.generate_tags("I run 5K races")
        assert len(result.fact_signals) == 1
        assert result.fact_signals[0].fact_type == "experience"
        assert result.fact_signals[0].what == "User runs 5K races."

    def test_fact_type_defaults_to_personal(self):
        llm = type('MockLLM', (), {
            'complete': lambda self, **kw: '{"tags": ["cooking"], "primary": "cooking", "temporal": false, "related_tags": [], "facts": [{"subject": "user", "verb": "prefers", "object": "French cuisine", "status": "active"}]}'
        })()
        gen = LLMTagGenerator(llm, TagGeneratorConfig(type="llm"))
        result = gen.generate_tags("I prefer French cuisine")
        assert result.fact_signals[0].fact_type == "personal"
        assert result.fact_signals[0].what == ""
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestTaggerParsing -v`
Expected: FAIL — `AttributeError: 'FactSignal' object has no attribute 'fact_type'` (if Task 1 not done yet) or FactSignal is created without `fact_type`/`what` fields

**Step 3: Write minimal implementation**

Update `_parse_response()` in `tag_generator.py:402-407`:

```python
fact_signals.append(FactSignal(
    subject=f.get("subject", ""),
    verb=f.get("verb", f.get("role", "")),
    object=f.get("object", ""),
    status=f.get("status", "active"),
    fact_type=f.get("fact_type", "personal"),
    what=f.get("what", ""),
))
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestTaggerParsing -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/tag_generator.py tests/test_fact_enrichment.py
git commit -m "feat: parse fact_type and what from tagger LLM response"
```

---

### Task 6: Enrich compactor prompt

**Files:**
- Modify: `virtual_context/core/compactor.py:110-121`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestCompactorPromptEnrichment:
    def test_compactor_prompt_has_fact_type(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        assert "fact_type" in DEFAULT_SUMMARY_PROMPT

    def test_compactor_prompt_has_specifics_instruction(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        assert "ALL specifics" in DEFAULT_SUMMARY_PROMPT or "all specifics" in DEFAULT_SUMMARY_PROMPT

    def test_compactor_prompt_has_dedup_instruction(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        # Within-segment dedup: if two signals describe the same event, emit one fact
        assert "same event" in DEFAULT_SUMMARY_PROMPT or "duplicate" in DEFAULT_SUMMARY_PROMPT.lower()

    def test_compactor_prompt_requires_all_dimensions(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        for dim in ("what", "who", "when", "where", "why"):
            assert f'"{dim}"' in DEFAULT_SUMMARY_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorPromptEnrichment -v`
Expected: FAIL — `fact_type` not in prompt, no dedup instruction

**Step 3: Write minimal implementation**

Replace lines 110-121 in `virtual_context/core/compactor.py` (`DEFAULT_SUMMARY_PROMPT` fact section):

```python
Also extract facts from the conversation. For each fact:
- "subject": who (usually "user"; proper names for others)
- "verb": the exact action verb (e.g. "led", "built", "prefers", "lives in", "ordered")
- "object": what (specific noun phrase — preserve ALL numbers, names, dates, amounts exactly)
- "status": one of: active, completed, planned, abandoned, recurring
- "fact_type": classify as "personal" (user's life, identity, preferences, plans),
  "experience" (assistant-provided info the user engaged with), or
  "world" (facts about other people, places, things in the user's world)
- "what": one full sentence capturing the complete fact with ALL specifics preserved.
  WRONG: "User has a personal best time." RIGHT: "User has a personal best 5K time of 27:12."
- "who": people involved (populate when present, empty string if n/a)
- "when": date if mentioned (ISO format or free-form, empty string if n/a)
- "where": location (populate when present, empty string if n/a)
- "why": context or significance (populate when present, empty string if n/a)
Extract the FACT behind the question, not the conversational act.
WRONG: "user asks about Cairo restaurants" RIGHT: "user wants to try authentic Egyptian food in Cairo"
If two signals describe the same event, emit one fact with the richest details.
Include "facts" in the JSON response.
Only extract facts with genuine substance. Skip greetings and filler.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorPromptEnrichment -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_fact_enrichment.py
git commit -m "feat: enrich compactor prompt with fact_type, all dimensions, within-segment dedup"
```

---

### Task 7: Update compactor fact parsing — add `fact_type`

**Files:**
- Modify: `virtual_context/core/compactor.py:316-357`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
from tests.conftest import MockLLMProvider
from virtual_context.core.compactor import DomainCompactor
from virtual_context.types import CompactorConfig, Message, TaggedSegment
from datetime import datetime, timedelta, timezone


class TestCompactorFactParsing:
    def test_compactor_parses_fact_type(self):
        response = (
            '{"summary": "Test", "entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], "refined_tags": ["running"], '
            '"facts": [{"subject": "user", "verb": "runs", "object": "5K", '
            '"status": "active", "fact_type": "experience", '
            '"what": "User runs 5K races.", "who": "", "when": "", "where": "", "why": ""}]}'
        )
        llm = MockLLMProvider(response=response)
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(summary_ratio=0.15, min_summary_tokens=50, max_summary_tokens=500),
            model_name="test-model",
        )
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        seg = TaggedSegment(
            primary_tag="running", tags=["running"],
            messages=[
                Message(role="user", content="I run 5K races", timestamp=ts),
                Message(role="assistant", content="Great!", timestamp=ts + timedelta(seconds=30)),
            ],
            token_count=20, start_timestamp=ts, end_timestamp=ts + timedelta(seconds=30), turn_count=1,
        )
        results = compactor.compact([seg])
        assert len(results[0].facts) == 1
        assert results[0].facts[0].fact_type == "experience"

    def test_compactor_defaults_fact_type_to_personal(self):
        response = (
            '{"summary": "Test", "entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], "refined_tags": ["cooking"], '
            '"facts": [{"subject": "user", "verb": "prefers", "object": "French cuisine", '
            '"status": "active", "what": "User prefers French cuisine.", "who": "", "when": "", "where": "", "why": ""}]}'
        )
        llm = MockLLMProvider(response=response)
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(summary_ratio=0.15, min_summary_tokens=50, max_summary_tokens=500),
            model_name="test-model",
        )
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        seg = TaggedSegment(
            primary_tag="cooking", tags=["cooking"],
            messages=[
                Message(role="user", content="I prefer French cuisine", timestamp=ts),
                Message(role="assistant", content="Noted!", timestamp=ts + timedelta(seconds=30)),
            ],
            token_count=20, start_timestamp=ts, end_timestamp=ts + timedelta(seconds=30), turn_count=1,
        )
        results = compactor.compact([seg])
        assert results[0].facts[0].fact_type == "personal"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorFactParsing -v`
Expected: FAIL — `fact_type` not parsed from response

**Step 3: Write minimal implementation**

In `compactor.py`, update the fact parsing loop (around line 332):

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
))
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorFactParsing -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_fact_enrichment.py
git commit -m "feat: parse fact_type from compactor LLM response"
```

---

### Task 8: Update compactor signal hints to include `fact_type` and `what`

**Files:**
- Modify: `virtual_context/core/compactor.py:226-236`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestCompactorSignalHints:
    def test_signal_hints_include_fact_type_and_what(self):
        response = (
            '{"summary": "Test", "entities": [], "key_decisions": [], '
            '"action_items": [], "date_references": [], "refined_tags": ["running"], "facts": []}'
        )
        llm = MockLLMProvider(response=response)
        compactor = DomainCompactor(
            llm_provider=llm,
            config=CompactorConfig(summary_ratio=0.15, min_summary_tokens=50, max_summary_tokens=500),
            model_name="test-model",
        )
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        seg = TaggedSegment(
            primary_tag="running", tags=["running"],
            messages=[
                Message(role="user", content="I run 5K", timestamp=ts),
                Message(role="assistant", content="Great!", timestamp=ts + timedelta(seconds=30)),
            ],
            token_count=20, start_timestamp=ts, end_timestamp=ts + timedelta(seconds=30), turn_count=1,
        )
        signals = [FactSignal(
            subject="user", verb="runs", object="5K races",
            status="active", fact_type="personal",
            what="User runs 5K charity races every spring.",
        )]
        compactor.compact([seg], fact_signals_by_segment={seg.id: signals})
        # Check the LLM prompt includes fact_type and what
        prompt_sent = llm.calls[0]["user"]
        assert "personal" in prompt_sent
        assert "User runs 5K charity races every spring." in prompt_sent
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorSignalHints -v`
Expected: FAIL — signal hints currently only show `subject verb object (status)`, no `fact_type` or `what`

**Step 3: Write minimal implementation**

In `compactor.py`, update the signal hints section (around line 229-235):

```python
if fact_signals:
    hint_lines = []
    for s in fact_signals:
        if s.subject and s.object:
            line = f"- [{s.fact_type}] {s.subject} {s.verb} {s.object} ({s.status})"
            if s.what:
                line += f" — {s.what}"
            hint_lines.append(line)
    if hint_lines:
        signals_text = (
            "\n\nPer-turn fact signals (verify and consolidate with full context):\n"
            + "\n".join(hint_lines)
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestCompactorSignalHints -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/core/compactor.py tests/test_fact_enrichment.py
git commit -m "feat: include fact_type and what in compactor signal hints"
```

---

### Task 9: Add `set_fact_superseded` to ContextStore ABC and SQLiteStore

**Files:**
- Modify: `virtual_context/core/store.py` (after line 161)
- Modify: `virtual_context/storage/sqlite.py` (add method)
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestSetFactSuperseded:
    def test_set_fact_superseded(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has PB", object="27:12", status="completed")
        new = Fact(subject="user", verb="has PB", object="25:50", status="completed")
        store.store_facts([old, new])
        store.set_fact_superseded(old.id, new.id)
        # Old fact should be superseded
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].id == new.id

    def test_set_fact_superseded_updates_field(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has PB", object="27:12")
        new = Fact(subject="user", verb="has PB", object="25:50")
        store.store_facts([old, new])
        store.set_fact_superseded(old.id, new.id)
        # Raw query to check superseded_by field
        conn = store._get_conn()
        row = conn.execute("SELECT superseded_by FROM facts WHERE id = ?", (old.id,)).fetchone()
        assert row["superseded_by"] == new.id
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestSetFactSuperseded -v`
Expected: FAIL — `AttributeError: 'SQLiteStore' object has no attribute 'set_fact_superseded'`

**Step 3: Write minimal implementation**

3a. Add to `ContextStore` ABC in `store.py` (after the `search_facts` method):

```python
def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
    """Mark old_fact_id as superseded by new_fact_id."""
    pass
```

3b. Add to `SQLiteStore` in `sqlite.py`:

```python
def set_fact_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
    """Mark old_fact_id as superseded by new_fact_id."""
    conn = self._get_conn()
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE id = ?",
        (new_fact_id, old_fact_id),
    )
    conn.commit()
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestSetFactSuperseded -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/store.py virtual_context/storage/sqlite.py tests/test_fact_enrichment.py
git commit -m "feat: add set_fact_superseded to ContextStore ABC and SQLiteStore"
```

---

### Task 10: Wire supersession checker into engine.py

**Files:**
- Modify: `virtual_context/engine.py:320-337` (after storing facts in compaction loop)
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
from unittest.mock import MagicMock, patch


class TestEngineSupersessionWiring:
    def test_engine_calls_supersession_after_compaction(self, tmp_path):
        """Verify that the engine wires supersession into the compaction pipeline."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.ingest.supersession import FactSupersessionChecker

        # Check that the engine has a reference to a supersession checker
        # and that compact_manual triggers it
        config_dict = {
            "context_window": 10000,
            "store": {"type": "sqlite", "path": str(tmp_path / "test.db")},
            "tag_generator": {"type": "keyword", "keyword_fallback": {"tag_keywords": {"test": ["test"]}}},
        }
        from virtual_context.config import load_config
        config = load_config(config_dict=config_dict)
        engine = VirtualContextEngine(config=config)

        # Verify the attribute exists (may be None if no provider configured)
        assert hasattr(engine, '_supersession_checker')
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestEngineSupersessionWiring -v`
Expected: FAIL — `AssertionError: hasattr returned False`

**Step 3: Write minimal implementation**

3a. In `engine.py`, in `__init__` (after `self._split_processed_tags` around line 109), add:

```python
self._supersession_checker = None  # Initialized when provider is available
```

3b. In the compaction result storage loop (around line 321-329 in `_run_compaction`), after `self._store.store_facts(result.facts)`, add:

```python
# D1: Run supersession check on new facts
if result.facts and self._supersession_checker:
    try:
        superseded = self._supersession_checker.check_and_supersede(result.facts)
        if superseded:
            logger.info("  Superseded %d facts for segment %s", superseded, result.primary_tag)
    except Exception as e:
        logger.warning("Supersession check failed: %s", e)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestEngineSupersessionWiring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add virtual_context/engine.py tests/test_fact_enrichment.py
git commit -m "feat: wire supersession checker into engine compaction pipeline"
```

---

### Task 11: Update supersession prompt — add duplicate detection

**Files:**
- Modify: `virtual_context/ingest/supersession.py:71-90`
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
from virtual_context.ingest.supersession import FactSupersessionChecker


class TestSupersessionPrompt:
    def test_prompt_asks_about_duplicates(self):
        """Supersession prompt should detect both contradictions AND duplicates."""
        llm = MockLLMProvider(response="[]")
        from virtual_context.types import SupersessionConfig
        store = SQLiteStore(str(Path(tempfile.mkdtemp()) / "test.db"))
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new_fact = Fact(subject="user", verb="has PB", object="25:50")
        candidates = [Fact(subject="user", verb="has PB", object="27:12")]
        checker._check_batch(new_fact, candidates)
        prompt = llm.calls[0]["user"]
        # Should mention both contradicted and duplicated
        assert "DUPLICATE" in prompt or "duplicate" in prompt

    def test_prompt_includes_what_field(self):
        """Supersession prompt should include the what field for richer comparison."""
        llm = MockLLMProvider(response="[]")
        from virtual_context.types import SupersessionConfig
        store = SQLiteStore(str(Path(tempfile.mkdtemp()) / "test.db"))
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new_fact = Fact(subject="user", verb="has PB", object="25:50",
                        what="User has a personal best 5K time of 25:50.")
        candidates = [Fact(subject="user", verb="has PB", object="27:12",
                           what="User set a personal best time of 27:12.")]
        checker._check_batch(new_fact, candidates)
        prompt = llm.calls[0]["user"]
        assert "25:50" in prompt
        assert "27:12" in prompt
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestSupersessionPrompt -v`
Expected: FAIL — prompt doesn't mention "DUPLICATE", doesn't include `what`

**Step 3: Write minimal implementation**

Update `_build_prompt` in `supersession.py:71-90`:

```python
def _build_prompt(self, new_fact: Fact, candidates: list[Fact]) -> str:
    """Build prompt asking which candidates are superseded or duplicated."""
    lines = [
        "A new fact has been extracted from a conversation:",
        f"  Subject: {new_fact.subject}",
        f"  Verb: {new_fact.verb}",
        f"  Object: {new_fact.object}",
        f"  Status: {new_fact.status}",
    ]
    if new_fact.what:
        lines.append(f"  What: {new_fact.what}")
    lines.append("")
    lines.append("Existing facts with the same subject:")
    for i, c in enumerate(candidates):
        line = f"  [{i}] {c.verb} {c.object} (status: {c.status})"
        if c.what:
            line += f" — {c.what}"
        lines.append(line)
    lines.append("")
    lines.append(
        "Which existing facts (by index) are CONTRADICTED, SUPERSEDED, or "
        "DUPLICATED by the new fact? A fact is duplicated if it describes the "
        "same underlying event/state with different wording. When duplicates "
        "are found, mark the LESS detailed version for removal. "
        "Reply with a JSON array of indices, e.g. [0, 2]. "
        "Reply [] if none are superseded or duplicated."
    )
    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestSupersessionPrompt -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add virtual_context/ingest/supersession.py tests/test_fact_enrichment.py
git commit -m "feat: expand supersession prompt to detect duplicates, include what field"
```

---

### Task 12: Update `vc_query_facts` tool — add `fact_type` to response and filter

**Files:**
- Modify: `virtual_context/core/tool_loop.py:160-192` (tool schema), `virtual_context/core/tool_loop.py:635-663` (execution)
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the failing test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestToolLoopFactType:
    def test_tool_schema_has_fact_type_param(self):
        from virtual_context.core.tool_loop import TOOLS
        query_tool = next(t for t in TOOLS if t["name"] == "vc_query_facts")
        props = query_tool["input_schema"]["properties"]
        assert "fact_type" in props
        assert "enum" in props["fact_type"]
        assert "personal" in props["fact_type"]["enum"]

    def test_tool_response_includes_fact_type(self):
        """Verify fact_type is included in the response payload."""
        # This tests the response format construction
        f = Fact(subject="user", verb="runs", object="5K", fact_type="experience")
        # Simulate the response dict construction from tool_loop.py
        response_fact = {
            "subject": f.subject,
            "verb": f.verb,
            "object": f.object,
            "status": f.status,
            "fact_type": f.fact_type,
            "what": f.what,
            "who": f.who,
            "when": f.when_date,
            "where": f.where,
            "why": f.why,
        }
        assert response_fact["fact_type"] == "experience"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestToolLoopFactType -v`
Expected: FAIL — `fact_type` not in tool schema

**Step 3: Write minimal implementation**

3a. Update the `vc_query_facts` tool schema in `tool_loop.py` (around line 170-191), add to `properties`:

```python
"fact_type": {
    "type": "string",
    "enum": ["personal", "experience", "world"],
    "description": "Filter by fact type. Omit to get all types except 'experience' (personal recall default).",
},
```

3b. Update the execution block (around line 635-663) to pass `fact_type` and include it in the response:

In the call to `engine.query_facts()`:

```python
meta = engine.query_facts(
    subject=tool_input.get("subject"),
    verb=tool_input.get("verb"),
    object_contains=tool_input.get("object_contains"),
    status=tool_input.get("status"),
    fact_type=tool_input.get("fact_type"),
    _return_meta=True,
    _intent_context=intent_context,
)
```

In the response dict construction, add `fact_type`:

```python
{
    "subject": f.subject,
    "verb": f.verb,
    "object": f.object,
    "status": f.status,
    "fact_type": f.fact_type,
    "what": f.what,
    "who": f.who,
    "when": f.when_date,
    "where": f.where,
    "why": f.why,
    "session_id": f.session_id,
    "tags": f.tags,
}
```

3c. Update `engine.query_facts()` (around line 1650) to accept and pass `fact_type`:

```python
def query_facts(self, **kwargs) -> list | dict:
    # ... existing code ...
    # Pass fact_type through to store
    # (kwargs already carries it since we use **kwargs)
```

Verify this just works since `engine.query_facts` uses `**kwargs` pass-through.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestToolLoopFactType -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add virtual_context/core/tool_loop.py tests/test_fact_enrichment.py
git commit -m "feat: add fact_type to vc_query_facts tool schema and response"
```

---

### Task 13: Run full test suite — regression check

**Files:**
- No new files

**Step 1: Run full test suite**

Run: `.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -x -q`
Expected: ~960 tests pass, no regressions

**Step 2: Fix any regressions**

If any tests fail, investigate and fix. Common issues:
- Tests that construct `Fact` or `FactSignal` might need `fact_type` added if they check field ordering
- Tests that mock `query_facts` may need to accept the new `fact_type` parameter
- SQLite tests that check column counts may need updating

**Step 3: Commit fixes if any**

```bash
git add -A
git commit -m "fix: resolve regressions from fact_type addition"
```

---

### Task 14: Integration test — end-to-end fact enrichment

**Files:**
- Test: `tests/test_fact_enrichment.py`

**Step 1: Write the integration test**

Append to `tests/test_fact_enrichment.py`:

```python
class TestFactEnrichmentIntegration:
    def test_enriched_fact_roundtrip(self, tmp_path):
        """Store enriched fact → query it back → verify all fields."""
        store = SQLiteStore(str(tmp_path / "test.db"))
        fact = Fact(
            subject="user",
            verb="has",
            object="personal best 5K time of 25:50",
            status="completed",
            fact_type="personal",
            what="User has a personal best 5K time of 25:50.",
            who="user",
            when_date="2026-01-15",
            where="Central Park",
            why="Training for a charity run",
        )
        store.store_facts([fact])
        results = store.query_facts(subject="user", fact_type="personal")
        assert len(results) == 1
        r = results[0]
        assert r.fact_type == "personal"
        assert r.what == "User has a personal best 5K time of 25:50."
        assert r.where == "Central Park"
        assert r.why == "Training for a charity run"
        assert "25:50" in r.object

    def test_supersession_filters_old_facts(self, tmp_path):
        """Old fact superseded → query only returns new fact."""
        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has PB", object="27:12",
                   fact_type="personal", what="User has a PB of 27:12.")
        new = Fact(subject="user", verb="has PB", object="25:50",
                   fact_type="personal", what="User has a PB of 25:50.")
        store.store_facts([old, new])
        store.set_fact_superseded(old.id, new.id)
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert "25:50" in results[0].object

    def test_fact_type_filter_excludes_experience(self, tmp_path):
        """Experience facts should be filterable separately."""
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.store_facts([
            Fact(subject="user", verb="runs", object="5K", fact_type="personal"),
            Fact(subject="user", verb="learned about", object="interval training", fact_type="experience"),
        ])
        personal_only = store.query_facts(subject="user", fact_type="personal")
        all_facts = store.query_facts(subject="user")
        assert len(personal_only) == 1
        assert len(all_facts) == 2
```

**Step 2: Run integration tests**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py::TestFactEnrichmentIntegration -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/test_fact_enrichment.py
git commit -m "test: add integration tests for enriched fact extraction roundtrip"
```

---

### Task 15: Run full test suite — final validation

**Step 1: Run full test suite**

Run: `.venv/bin/pytest --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -v 2>&1 | tail -20`
Expected: All tests pass, count should be ~960+

**Step 2: Verify no remaining issues**

Run: `.venv/bin/pytest tests/test_fact_enrichment.py -v`
Expected: All ~25+ enrichment tests pass

---

## Deferred Tasks (Phase 2 — requires live LLM)

These tasks need a live Qwen3-30B or equivalent LLM to test properly. They should be done in a separate session with API keys available.

### Deferred Task A: Make supersession/dedup model configurable

Add `SupersessionConfig` fields for provider/model and wire into `engine.py` initialization. This enables using Qwen3-30B for supersession while keeping Haiku for tagging.

**Files:** `virtual_context/types.py`, `virtual_context/config.py`, `virtual_context/engine.py`

### Deferred Task B: Add periodic dedup pass

New method `engine.deduplicate_facts(subject="user")` that queries all non-superseded facts for a subject, sends to LLM with dedup-only prompt (from v4 POC Pass 2), marks duplicates as superseded.

**Files:** `virtual_context/engine.py`, `virtual_context/ingest/supersession.py`

### Deferred Task C: Rerun LongMemEval 21-question suite

Validate the enrichment against the benchmark. Target: >85% accuracy.

**Files:** `benchmarks/longmemeval/`

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Add `fact_type` + `what` to FactSignal | types.py |
| 2 | Add `fact_type` to Fact | types.py |
| 3 | SQLite migration for `fact_type` | sqlite.py, store.py |
| 4 | Enrich tagger prompt | tag_generator.py |
| 5 | Update tagger parsing | tag_generator.py |
| 6 | Enrich compactor prompt | compactor.py |
| 7 | Update compactor fact parsing | compactor.py |
| 8 | Update compactor signal hints | compactor.py |
| 9 | Add `set_fact_superseded` | store.py, sqlite.py |
| 10 | Wire supersession into engine | engine.py |
| 11 | Update supersession prompt | supersession.py |
| 12 | Update `vc_query_facts` tool | tool_loop.py |
| 13 | Full test suite regression check | — |
| 14 | Integration tests | test_fact_enrichment.py |
| 15 | Final validation | — |
