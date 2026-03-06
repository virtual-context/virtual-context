"""Tests for semantic verb expansion in query_facts."""

from datetime import datetime, timezone

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact


@pytest.fixture
def store(tmp_sqlite_db):
    s = SQLiteStore(db_path=tmp_sqlite_db)
    yield s
    s.close()


def _make_fact(verb: str, obj: str, **kwargs) -> Fact:
    defaults = dict(
        subject="user",
        verb=verb,
        object=obj,
        status="active",
        tags=["projects"],
        segment_ref="seg-1",
        conversation_id="session-1",
        turn_numbers=[1],
        mentioned_at=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return Fact(**defaults)


class TestGetUniqueFactVerbs:
    def test_returns_distinct_verbs(self, store):
        facts = [
            _make_fact("led", "project Alpha"),
            _make_fact("leads", "project Beta"),
            _make_fact("built", "a CLI tool"),
        ]
        store.store_facts(facts)
        verbs = store.get_unique_fact_verbs()
        assert set(verbs) == {"led", "leads", "built"}

    def test_excludes_empty_verbs(self, store):
        facts = [
            _make_fact("", "something"),
            _make_fact("built", "a tool"),
        ]
        store.store_facts(facts)
        verbs = store.get_unique_fact_verbs()
        assert verbs == ["built"]

    def test_excludes_superseded_facts(self, store):
        facts = [
            _make_fact("led", "old project", superseded_by="newer-fact-id"),
            _make_fact("leads", "current project"),
        ]
        store.store_facts(facts)
        verbs = store.get_unique_fact_verbs()
        assert verbs == ["leads"]

    def test_empty_store(self, store):
        assert store.get_unique_fact_verbs() == []


class TestQueryFactsWithVerbs:
    def test_verbs_disjunctive_match(self, store):
        facts = [
            _make_fact("led", "project Alpha"),
            _make_fact("leads", "project Beta"),
            _make_fact("built", "a CLI tool"),
        ]
        store.store_facts(facts)
        results = store.query_facts(verbs=["led", "leads"])
        assert len(results) == 2
        result_verbs = {f.verb for f in results}
        assert result_verbs == {"led", "leads"}

    def test_verbs_takes_precedence_over_verb(self, store):
        facts = [
            _make_fact("led", "project Alpha"),
            _make_fact("leads", "project Beta"),
            _make_fact("built", "a CLI tool"),
        ]
        store.store_facts(facts)
        # verbs param should be used, verb param ignored
        results = store.query_facts(verb="built", verbs=["led", "leads"])
        assert len(results) == 2
        result_verbs = {f.verb for f in results}
        assert result_verbs == {"led", "leads"}

    def test_single_verb_fallback(self, store):
        facts = [
            _make_fact("led", "project Alpha"),
            _make_fact("leads", "project Beta"),
        ]
        store.store_facts(facts)
        # Without verbs param, uses LIKE match on single verb
        results = store.query_facts(verb="led")
        assert len(results) == 1
        assert results[0].verb == "led"

    def test_verbs_with_subject_filter(self, store):
        facts = [
            _make_fact("led", "project Alpha", subject="user"),
            _make_fact("leads", "project Beta", subject="user"),
            _make_fact("led", "other project", subject="colleague"),
        ]
        store.store_facts(facts)
        results = store.query_facts(subject="user", verbs=["led", "leads"])
        assert len(results) == 2
        assert all(f.subject == "user" for f in results)


class TestExpandVerb:
    """Test engine._expand_verb using mock embeddings."""

    def test_expand_led_finds_leads(self):
        """Mock: 'led' and 'leads' are close, 'built' is far."""
        from unittest.mock import MagicMock, patch
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = MagicMock()
        engine._store.get_unique_fact_verbs.return_value = ["led", "leads", "built", "prefers"]

        # Simulate embeddings: led=[1,0,0], leads=[0.9,0.1,0], built=[0,1,0], prefers=[0,0,1]
        def fake_embed(texts):
            vecs = {
                "led": [1.0, 0.0, 0.0],
                "leads": [0.9, 0.1, 0.0],
                "built": [0.0, 1.0, 0.0],
                "prefers": [0.0, 0.0, 1.0],
            }
            return [vecs.get(t, [0.0, 0.0, 0.0]) for t in texts]

        engine._get_embed_fn = MagicMock(return_value=fake_embed)

        # Call the real method
        result = VirtualContextEngine._expand_verb(engine, "led")
        assert "led" in result
        assert "leads" in result
        assert "built" not in result
        assert "prefers" not in result

    def test_expand_returns_none_when_no_embed_fn(self):
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._get_embed_fn = MagicMock(return_value=None)

        result = VirtualContextEngine._expand_verb(engine, "led")
        assert result is None

    def test_expand_returns_none_when_no_verbs(self):
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._get_embed_fn = MagicMock(return_value=lambda texts: [[0.0]] * len(texts))
        engine._store = MagicMock()
        engine._store.get_unique_fact_verbs.return_value = []

        result = VirtualContextEngine._expand_verb(engine, "led")
        assert result is None

    def test_expand_returns_none_when_no_similar(self):
        """If no other verbs are similar, returns None (no expansion needed)."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = MagicMock()
        engine._store.get_unique_fact_verbs.return_value = ["built", "prefers"]

        # Orthogonal embeddings — no similarity
        def fake_embed(texts):
            vecs = {
                "led": [1.0, 0.0, 0.0],
                "built": [0.0, 1.0, 0.0],
                "prefers": [0.0, 0.0, 1.0],
            }
            return [vecs.get(t, [0.0, 0.0, 0.0]) for t in texts]

        engine._get_embed_fn = MagicMock(return_value=fake_embed)

        result = VirtualContextEngine._expand_verb(engine, "led")
        assert result is None


class TestEngineQueryFactsIntegration:
    """Test that engine.query_facts calls _expand_verb and passes verbs through."""

    def test_verb_expansion_passed_to_store(self):
        from unittest.mock import MagicMock, call
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._semantic_fact_search = MagicMock(return_value=[])
        # Return results so auto-relax doesn't trigger
        store.query_facts.return_value = ["fake_fact"]

        # Mock _expand_verb to return expanded list
        engine._expand_verb = MagicMock(return_value=["led", "leads"])

        VirtualContextEngine.query_facts(engine, verb="led", subject="user")

        # Should call store with verbs instead of verb
        store.query_facts.assert_called_once_with(
            verbs=["led", "leads"], subject="user"
        )

    def test_no_expansion_keeps_verb(self):
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._semantic_fact_search = MagicMock(return_value=[])
        store.query_facts.return_value = []
        engine._expand_verb = MagicMock(return_value=None)

        VirtualContextEngine.query_facts(engine, verb="led", subject="user")

        store.query_facts.assert_called_once_with(
            verb="led", subject="user"
        )

    @pytest.mark.regression("BUG-032")
    def test_no_auto_relax_object_contains_on_zero_results(self):
        """When object_contains produces 0 results, return empty — do NOT
        retry without object_contains.  Auto-relax was removed because it
        returned facts contradicting the reader's explicit filter (BUG-032)."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)
        engine._semantic_fact_search = MagicMock(return_value=[])

        store.query_facts.return_value = []

        result = VirtualContextEngine.query_facts(
            engine, verb="led", subject="user", object_contains="project"
        )

        assert result == []
        # Store called exactly once — no retry without object_contains
        store.query_facts.assert_called_once()

    def test_no_relax_when_object_contains_has_results(self):
        """When object_contains produces results, don't retry."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)
        engine._semantic_fact_search = MagicMock(return_value=[])
        store.query_facts.return_value = ["fact1"]

        result = VirtualContextEngine.query_facts(
            engine, verb="led", subject="user", object_contains="team"
        )

        assert result == ["fact1"]
        store.query_facts.assert_called_once()

    def test_no_relax_when_no_object_contains(self):
        """When object_contains not provided, no retry even on 0 results."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)
        engine._semantic_fact_search = MagicMock(return_value=[])
        store.query_facts.return_value = []

        result = VirtualContextEngine.query_facts(
            engine, verb="led", subject="user"
        )

        assert result == []
        store.query_facts.assert_called_once()


class TestSemanticFactSearch:
    """Test engine._semantic_fact_search using mock embeddings."""

    def _make_fact(self, id, verb, what, status="active"):
        from virtual_context.types import Fact
        return Fact(id=id, subject="user", verb=verb, object="", what=what, status=status)

    def test_finds_additional_facts_by_what_field(self):
        """Semantic search finds facts SQL missed due to verb mismatch."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = MagicMock()

        # SQL found Fitbit (verb=uses), semantic should find Accu-Chek (verb=is testing)
        existing = [self._make_fact("f1", "uses", "User uses a Fitbit daily")]
        accu = self._make_fact("f2", "is testing", "User is testing an Accu-Chek glucose monitor daily")
        nebulizer = self._make_fact("f3", "is doing", "User is doing nebulizer treatments weekly")
        unrelated = self._make_fact("f4", "grows", "User grows herb plants on the balcony")

        engine._store.query_facts.return_value = [existing[0], accu, nebulizer, unrelated]

        # Mock embed_fn: query "user use" is close to health-related facts
        def fake_embed(texts):
            vecs = {
                "user use": [1.0, 0.8, 0.0],
                "User uses a Fitbit daily": [0.95, 0.85, 0.0],
                "User is testing an Accu-Chek glucose monitor daily": [0.9, 0.7, 0.05],
                "User is doing nebulizer treatments weekly": [0.85, 0.6, 0.1],
                "User grows herb plants on the balcony": [0.1, 0.1, 0.9],
            }
            return [vecs.get(t, [0.0, 0.0, 0.0]) for t in texts]

        engine._get_embed_fn = MagicMock(return_value=fake_embed)

        result = VirtualContextEngine._semantic_fact_search(
            engine, existing=existing, subject="user", verb="use"
        )

        result_ids = {f.id for f in result}
        assert "f2" in result_ids  # Accu-Chek
        assert "f3" in result_ids  # nebulizer
        assert "f1" not in result_ids  # already in existing
        assert "f4" not in result_ids  # unrelated (low similarity)

    def test_returns_empty_without_embed_fn(self):
        """Gracefully returns [] when embeddings are unavailable."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._get_embed_fn = MagicMock(return_value=None)

        result = VirtualContextEngine._semantic_fact_search(
            engine, existing=[], subject="user", verb="use"
        )
        assert result == []

    def test_returns_empty_without_verb_or_object(self):
        """Requires at least verb or object_contains for a meaningful query."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._get_embed_fn = MagicMock(return_value=lambda t: [[0.0]] * len(t))

        result = VirtualContextEngine._semantic_fact_search(
            engine, existing=[], subject="user"
        )
        assert result == []

    def test_deduplicates_existing_facts(self):
        """Facts already in existing are not returned again."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = MagicMock()

        existing = [self._make_fact("f1", "uses", "User uses a Fitbit daily")]
        # Store returns the same fact that's already in existing
        engine._store.query_facts.return_value = [existing[0]]
        engine._get_embed_fn = MagicMock(return_value=lambda t: [[1.0, 0.0]] * len(t))

        result = VirtualContextEngine._semantic_fact_search(
            engine, existing=existing, subject="user", verb="use"
        )
        assert result == []

    def test_skips_facts_without_what_field(self):
        """Facts with empty 'what' are excluded from semantic comparison."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = MagicMock()

        no_what = self._make_fact("f2", "is testing", "")
        engine._store.query_facts.return_value = [no_what]
        engine._get_embed_fn = MagicMock(return_value=lambda t: [[1.0, 0.0]] * len(t))

        result = VirtualContextEngine._semantic_fact_search(
            engine, existing=[], subject="user", verb="use"
        )
        assert result == []


class TestQueryFactsSemanticIntegration:
    """Test that query_facts integrates semantic search correctly."""

    def _make_fact(self, id, verb, what, status="active"):
        from virtual_context.types import Fact
        return Fact(id=id, subject="user", verb=verb, object="", what=what, status=status)

    def test_semantic_results_merged_into_output(self):
        """Semantic matches are appended to SQL results."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        sql_fact = self._make_fact("f1", "uses", "User uses a Fitbit")
        sem_fact = self._make_fact("f2", "is testing", "User is testing an Accu-Chek")
        store.query_facts.return_value = [sql_fact]
        engine._semantic_fact_search = MagicMock(return_value=[sem_fact])

        result = VirtualContextEngine.query_facts(
            engine, verb="use", subject="user"
        )

        assert len(result) == 2
        assert result[0].id == "f1"
        assert result[1].id == "f2"

    def test_semantic_prevents_noisy_auto_relax(self):
        """When semantic search finds results, auto-relax doesn't trigger."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        sem_fact = self._make_fact("f2", "is testing", "User is testing an Accu-Chek health monitor")

        # SQL returns 0, but semantic finds 1
        store.query_facts.return_value = []
        engine._semantic_fact_search = MagicMock(return_value=[sem_fact])

        result = VirtualContextEngine.query_facts(
            engine, verb="use", subject="user", object_contains="health"
        )

        assert len(result) == 1
        assert result[0].id == "f2"
        # Store was only called once (the initial SQL query), NOT again for auto-relax
        store.query_facts.assert_called_once()

    def test_semantic_note_in_meta(self):
        """_return_meta includes semantic_note when semantic search adds facts."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        sem_fact = self._make_fact("f2", "is testing", "User is testing an Accu-Chek")
        store.query_facts.return_value = []
        engine._semantic_fact_search = MagicMock(return_value=[sem_fact])

        result = VirtualContextEngine.query_facts(
            engine, verb="use", subject="user", _return_meta=True
        )

        assert result["semantic_note"] is not None
        assert "1 fact(s)" in result["semantic_note"]

    def test_semantic_respects_status_filter(self):
        """When status filter is used, only matching semantic results are included."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        active_fact = self._make_fact("f2", "is testing", "User is testing Accu-Chek", status="active")
        completed_fact = self._make_fact("f3", "used", "User used a different monitor", status="completed")
        store.query_facts.return_value = []
        engine._semantic_fact_search = MagicMock(return_value=[active_fact, completed_fact])

        result = VirtualContextEngine.query_facts(
            engine, verb="use", subject="user", status="active"
        )

        # Only the active fact should be in results
        assert len(result) == 1
        assert result[0].id == "f2"

    def test_total_all_statuses_includes_semantic_matches(self):
        """Semantic matches are merged into the total_all_statuses calculation."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        sql_fact = self._make_fact("f1", "uses", "User uses Fitbit", status="active")
        sem_active = self._make_fact("f2", "is testing", "User is testing Accu-Chek", status="active")
        sem_completed = self._make_fact("f3", "used", "User used old monitor", status="completed")

        # SQL query with status="active" returns f1
        # Unfiltered SQL query (no status) also returns just f1
        store.query_facts.side_effect = [[sql_fact], [sql_fact]]
        engine._semantic_fact_search = MagicMock(return_value=[sem_active, sem_completed])

        result = VirtualContextEngine.query_facts(
            engine, verb="use", subject="user", status="active", _return_meta=True
        )

        # Main facts: f1 (SQL) + f2 (semantic, active) = 2
        assert len(result["facts"]) == 2
        # total_all_statuses: f1 (SQL unfiltered) + f2 + f3 (semantic) = 3
        assert result["total_all_statuses"] == 3
        assert result["all_statuses"]["active"] == 2
        assert result["all_statuses"]["completed"] == 1

    @pytest.mark.regression("BUG-032")
    def test_semantic_search_respects_object_contains_filter(self):
        """Semantic search must post-filter against the reader's explicit
        object_contains constraint.  Regression: 6d550036 — reader asked for
        query_facts(verb='led', object_contains='project', status='active')
        but semantic search returned 'User leads a team of five engineers'
        because it ignores structured SQL filters, causing over-count."""
        from unittest.mock import MagicMock
        from virtual_context.engine import VirtualContextEngine

        store = MagicMock()
        engine = MagicMock(spec=VirtualContextEngine)
        engine._store = store
        engine._expand_verb = MagicMock(return_value=None)

        # Fact that matches semantically but NOT object_contains="project"
        team_fact = self._make_fact("f-team", "leads", "User leads a team of five engineers")
        # Fact that matches both semantically AND object_contains="project"
        project_fact = self._make_fact("f-proj", "leads", "User leads the migration project")

        store.query_facts.return_value = []
        engine._semantic_fact_search = MagicMock(return_value=[team_fact, project_fact])

        result = VirtualContextEngine.query_facts(
            engine, verb="led", subject="user", object_contains="project"
        )

        # Only the project fact should survive post-filtering
        assert len(result) == 1
        assert result[0].id == "f-proj"
