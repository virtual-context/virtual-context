# Supersession Object-Similarity Candidate Selection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the supersession checker find cross-session duplicates about the same event by adding object-keyword-based candidate lookup alongside the existing tag-based lookup.

**Architecture:** Currently `check_and_supersede` builds candidates using `subject + tags`. Facts about the same real-world event (e.g. a Yosemite trip) often appear across sessions with different tags, so they never get compared. The fix adds a second `query_facts` call keyed on the most distinctive word extracted from the new fact's `object` field (e.g. "Yosemite"), then merges both candidate lists before sending to the LLM. The LLM prompt and supersession logic are unchanged.

**Tech Stack:** Python, SQLite (`object LIKE ?` query already supported), existing `FactSupersessionChecker` in `virtual_context/ingest/supersession.py`.

---

### Task 1: `_extract_object_keyword` — extract most distinctive word from a fact object

**Files:**
- Modify: `virtual_context/ingest/supersession.py`
- Test: `tests/test_fact_enrichment.py` (add to existing supersession test classes)

**Background:** The object field contains strings like "solo camping trip to Yosemite National Park" or "visited Big Sur and Monterey". We need the most distinctive word (proper noun preferred, length as tiebreak) to use as a `LIKE` search key. Generic trip words ("trip", "solo", "back") must be filtered out.

**Step 1: Write the failing test**

Add to `tests/test_fact_enrichment.py`:

```python
class TestExtractObjectKeyword:
    def test_extracts_proper_noun_from_destination(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("solo camping trip to Yosemite National Park") == "Yosemite"

    def test_extracts_from_big_sur(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        result = _extract_object_keyword("Big Sur and Monterey")
        assert result in ("Monterey", "Sur")

    def test_extracts_from_muir_woods(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        result = _extract_object_keyword("Dipsea Trail at Muir Woods")
        assert result in ("Dipsea", "Trail", "Woods")

    def test_returns_none_for_generic_object(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("back") is None
        assert _extract_object_keyword("from solo trip") is None

    def test_returns_none_for_short_words_only(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("went to gym") is None

    def test_visited_yosemite(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("visited Yosemite") == "Yosemite"
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestExtractObjectKeyword -v
```
Expected: FAIL with `ImportError: cannot import name '_extract_object_keyword'`

**Step 3: Implement `_extract_object_keyword`**

Add to `virtual_context/ingest/supersession.py` after the imports, before the `_MERGE_SYSTEM` constant:

```python
_STOPWORDS = frozenset({
    "from", "with", "that", "about", "into", "over", "have", "been", "will",
    "this", "their", "there", "where", "which", "would", "could", "should",
    "after", "before", "during", "while", "other", "another", "these", "those",
    "trip", "solo", "recent", "just", "back", "today", "recently", "returned",
    "camping", "hiking", "visited", "started", "began",
})


def _extract_object_keyword(object_str: str) -> str | None:
    """Extract the most distinctive word from a fact's object string.

    Used to find cross-session duplicate facts via object_contains lookup.
    Prefers proper nouns (initial capital); falls back to longest word >= 5 chars.
    Returns None if no distinctive word is found.
    """
    words = re.findall(r"[A-Za-z']+", object_str)
    candidates = [w for w in words if len(w) >= 5 and w.lower() not in _STOPWORDS]
    if not candidates:
        return None
    proper = [w for w in candidates if w[0].isupper()]
    pool = proper if proper else candidates
    return max(pool, key=len)
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestExtractObjectKeyword -v
```
Expected: 6 PASS

**Step 5: Commit**

```bash
git add virtual_context/ingest/supersession.py tests/test_fact_enrichment.py
git commit -m "feat: add _extract_object_keyword for supersession candidate expansion"
```

---

### Task 2: Add object-similarity candidate lookup in `check_and_supersede`

**Files:**
- Modify: `virtual_context/ingest/supersession.py:64-97`
- Test: `tests/test_fact_enrichment.py`

**Background:** `check_and_supersede` currently builds candidates with `query_facts(subject=..., tags=..., limit=batch_size)`. We add a second lookup keyed on the object keyword, then merge (dedup by `fact.id`) before passing to the LLM. The existing `_check_batch` / prompt / merge logic is completely unchanged.

**Step 1: Write the failing test**

Add to `tests/test_fact_enrichment.py`:

```python
class TestSupersessionObjectSimilarity:
    """Cross-session duplicates are found via object keyword, not just tags."""

    def test_cross_session_duplicate_is_superseded(self, tmp_path):
        """Fact from different session/tags IS found when object keyword matches."""
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))

        # Old fact: different tags (session 1, tags=['backpack'])
        old = Fact(
            subject="user", verb="returned",
            object="from solo camping trip to Yosemite National Park",
            status="completed", tags=["backpack"],
            what="User recently returned from solo camping trip to Yosemite.",
        )
        store.store_facts([old], segment_ref="seg-old", tags=["backpack"])

        # New fact: different tags (session 2, tags=['bear-safety'])
        new_fact = Fact(
            subject="user", verb="started",
            object="solo camping trip to Yosemite National Park",
            status="completed", tags=["bear-safety"],
            what="User started solo camping trip to Yosemite National Park.",
        )
        store.store_facts([new_fact], segment_ref="seg-new", tags=["bear-safety"])

        # LLM says index 0 is superseded
        llm = MockLLMProvider(response="[0]")
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store,
            config=SupersessionConfig(enabled=True, batch_size=20),
        )
        count = checker.check_and_supersede([new_fact])

        assert count == 1
        # Old fact is now marked superseded
        remaining = store.query_facts(subject="user")
        assert all(f.id != old.id for f in remaining)

    def test_no_object_keyword_skips_object_search(self, tmp_path):
        """Fact with generic object (no keyword) uses only tag-based candidates."""
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        new_fact = Fact(
            subject="user", verb="went", object="back",
            status="completed", tags=["misc"],
        )
        store.store_facts([new_fact], segment_ref="seg-1", tags=["misc"])

        llm = MockLLMProvider(response="[]")
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store,
            config=SupersessionConfig(enabled=True, batch_size=20),
        )
        # Should not crash; 0 superseded since no candidates
        count = checker.check_and_supersede([new_fact])
        assert count == 0
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestSupersessionObjectSimilarity -v
```
Expected: `test_cross_session_duplicate_is_superseded` FAIL (old fact not found, count == 0)

**Step 3: Implement object-similarity candidate merge**

Replace the candidate-building block in `check_and_supersede` (`virtual_context/ingest/supersession.py:79-86`):

```python
# Current code to REPLACE:
candidates = self.store.query_facts(
    subject=fact.subject,
    tags=fact.tags if fact.tags else None,
    limit=self.config.batch_size,
)
candidates = [c for c in candidates if c.id != fact.id]
```

With:

```python
# Tag-based candidates (existing behaviour)
candidates = self.store.query_facts(
    subject=fact.subject,
    tags=fact.tags if fact.tags else None,
    limit=self.config.batch_size,
)
# Object-similarity candidates — catches cross-session duplicates
# whose tags don't overlap with the new fact's tags
keyword = _extract_object_keyword(fact.object)
if keyword:
    obj_candidates = self.store.query_facts(
        subject=fact.subject,
        object_contains=keyword,
        limit=self.config.batch_size,
    )
    seen_ids = {c.id for c in candidates}
    for c in obj_candidates:
        if c.id not in seen_ids:
            candidates.append(c)
            seen_ids.add(c.id)
candidates = [c for c in candidates if c.id != fact.id]
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestSupersessionObjectSimilarity -v
```
Expected: 2 PASS

**Step 5: Run full suite to check for regressions**

```bash
.venv/bin/pytest tests/ --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -x -q
```
Expected: all pass, no regressions

**Step 6: Commit**

```bash
git add virtual_context/ingest/supersession.py tests/test_fact_enrichment.py
git commit -m "feat: expand supersession candidates via object-keyword lookup for cross-session dedup"
```

---

### Task 3: Verify on the gpt4_7f6b06db benchmark case

**Files:**
- Read-only: `benchmarks/longmemeval/cache_qwen3_30b_a3b/gpt4_7f6b06db/store.db`

**Background:** The store for question gpt4_7f6b06db has 6 "recently returned from Yosemite" facts across sessions March 10 and April 20 that were never consolidated. After the fix, re-running compaction on a fresh store should produce a clean single Yosemite fact per distinct event.

This task is a manual smoke-test — no code changes.

**Step 1: Verify keyword extraction on the known duplicate facts**

```bash
.venv/bin/python -c "
from virtual_context.ingest.supersession import _extract_object_keyword
cases = [
    'from solo camping trip to Yosemite National Park',
    'solo camping trip to Yosemite National Park',
    'visited Yosemite',
    'recently returned from solo camping trip to Yosemite',
    'Big Sur and Monterey',
    'visited Big Sur and Monterey',
    'Dipsea Trail at Muir Woods',
]
for c in cases:
    print(repr(c), '->', _extract_object_keyword(c))
"
```

Expected: all Yosemite variants → `'Yosemite'`, Big Sur variants → `'Monterey'` or `'Sur'`, Muir Woods → `'Dipsea'` or `'Woods'`

**Step 2: Confirm the object_contains query finds cross-session candidates**

```bash
.venv/bin/python -c "
from virtual_context.storage.sqlite import SQLiteStore
store = SQLiteStore('benchmarks/longmemeval/cache_qwen3_30b_a3b/gpt4_7f6b06db/store.db')
candidates = store.query_facts(subject='user', object_contains='Yosemite', limit=20)
for f in candidates:
    print(f.verb, '|', f.object, '| session:', f.session_date[:10] if f.session_date else '-')
"
```

Expected: 6-8 Yosemite facts spanning sessions 2023/03/10, 2023/04/20, 2023/05/15 — all now reachable as candidates.

**Step 3: Commit**

No code changes — just confirm the query works.

```bash
git commit --allow-empty -m "chore: verify object_contains query finds cross-session Yosemite candidates"
```
