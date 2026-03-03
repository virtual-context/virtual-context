"""Tests for enriched fact extraction fields (fact_type, what)."""

from virtual_context.types import FactSignal, Fact
from virtual_context.storage.sqlite import SQLiteStore


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
        assert "conversational act" in TAG_GENERATOR_PROMPT_DETAILED or "asks about" in TAG_GENERATOR_PROMPT_DETAILED

    def test_compact_prompt_has_fact_type(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_COMPACT
        assert "fact_type" in TAG_GENERATOR_PROMPT_COMPACT

    def test_compact_prompt_has_what_field(self):
        from virtual_context.core.tag_generator import TAG_GENERATOR_PROMPT_COMPACT
        assert '"what"' in TAG_GENERATOR_PROMPT_COMPACT


class TestTaggerParsing:
    def test_parses_fact_type_from_response(self):
        from virtual_context.core.tag_generator import LLMTagGenerator
        from virtual_context.types import TagGeneratorConfig
        llm = type('MockLLM', (), {
            'complete': lambda self, **kw: '{"tags": ["running"], "primary": "running", "temporal": false, "related_tags": [], "facts": [{"subject": "user", "verb": "runs", "object": "5K", "status": "active", "fact_type": "experience", "what": "User runs 5K races."}]}'
        })()
        gen = LLMTagGenerator(llm, TagGeneratorConfig(type="llm"))
        result = gen.generate_tags("I run 5K races")
        assert len(result.fact_signals) == 1
        assert result.fact_signals[0].fact_type == "experience"
        assert result.fact_signals[0].what == "User runs 5K races."

    def test_fact_type_defaults_to_personal(self):
        from virtual_context.core.tag_generator import LLMTagGenerator
        from virtual_context.types import TagGeneratorConfig
        llm = type('MockLLM', (), {
            'complete': lambda self, **kw: '{"tags": ["cooking"], "primary": "cooking", "temporal": false, "related_tags": [], "facts": [{"subject": "user", "verb": "prefers", "object": "French cuisine", "status": "active"}]}'
        })()
        gen = LLMTagGenerator(llm, TagGeneratorConfig(type="llm"))
        result = gen.generate_tags("I prefer French cuisine")
        assert result.fact_signals[0].fact_type == "personal"
        assert result.fact_signals[0].what == ""


class TestCompactorPromptEnrichment:
    def test_compactor_prompt_has_fact_type(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        assert "fact_type" in DEFAULT_SUMMARY_PROMPT

    def test_compactor_prompt_has_specifics_instruction(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        assert "ALL specifics" in DEFAULT_SUMMARY_PROMPT

    def test_compactor_prompt_has_dedup_instruction(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        assert "same event" in DEFAULT_SUMMARY_PROMPT or "duplicate" in DEFAULT_SUMMARY_PROMPT.lower()

    def test_compactor_prompt_requires_all_dimensions(self):
        from virtual_context.core.compactor import DEFAULT_SUMMARY_PROMPT
        for dim in ("what", "who", "when", "where", "why"):
            assert f'"{dim}"' in DEFAULT_SUMMARY_PROMPT


class TestCompactorFactParsing:
    def test_compactor_parses_fact_type(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, Message, TaggedSegment
        from datetime import datetime, timedelta, timezone

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
        from tests.conftest import MockLLMProvider
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, Message, TaggedSegment
        from datetime import datetime, timedelta, timezone

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


class TestCompactorSignalHints:
    def test_signal_hints_include_fact_type_and_what(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, Message, TaggedSegment, FactSignal
        from datetime import datetime, timedelta, timezone

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
        prompt_sent = llm.calls[0]["user"]
        assert "[personal]" in prompt_sent
        assert "User runs 5K charity races every spring." in prompt_sent


class TestEngineSupersessionWiring:
    def test_engine_has_supersession_checker_attribute(self, tmp_path):
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config
        config = load_config(config_dict={
            "context_window": 10000,
            "store": {"type": "sqlite", "path": str(tmp_path / "test.db")},
            "tag_generator": {"type": "keyword", "keyword_fallback": {"tag_keywords": {"test": ["test"]}}},
        })
        engine = VirtualContextEngine(config=config)
        assert hasattr(engine, '_supersession_checker')


class TestSetFactSuperseded:
    def test_set_fact_superseded(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact
        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has PB", object="27:12", status="completed")
        new = Fact(subject="user", verb="has PB", object="25:50", status="completed")
        store.store_facts([old, new])
        store.set_fact_superseded(old.id, new.id)
        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].id == new.id

    def test_set_fact_superseded_updates_field(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact
        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has PB", object="27:12")
        new = Fact(subject="user", verb="has PB", object="25:50")
        store.store_facts([old, new])
        store.set_fact_superseded(old.id, new.id)
        conn = store._get_conn()
        row = conn.execute("SELECT superseded_by FROM facts WHERE id = ?", (old.id,)).fetchone()
        assert row["superseded_by"] == new.id


class TestToolLoopFactType:
    def test_tool_schema_has_fact_type_param(self):
        from virtual_context.core.tool_loop import vc_tool_definitions
        tools = vc_tool_definitions()
        query_tool = next(t for t in tools if t["name"] == "vc_query_facts")
        props = query_tool["input_schema"]["properties"]
        assert "fact_type" in props
        assert "enum" in props["fact_type"]
        assert set(props["fact_type"]["enum"]) == {"personal", "experience", "world"}


class TestCurationConfig:
    def test_defaults(self):
        from virtual_context.types import CurationConfig
        c = CurationConfig()
        assert c.enabled is False
        assert c.provider == ""
        assert c.model == ""
        assert c.max_response_tokens == 2048

    def test_curation_in_vc_config(self):
        from virtual_context.types import VirtualContextConfig, CurationConfig
        cfg = VirtualContextConfig()
        assert isinstance(cfg.curation, CurationConfig)
        assert cfg.curation.enabled is False

    def test_curation_loads_from_yaml(self, tmp_path):
        from virtual_context.config import load_config
        yaml_text = """
version: "0.2"
storage_root: .vc
curation:
  enabled: true
  provider: openrouter
  model: qwen/qwen3-30b-a3b
  max_response_tokens: 4096
"""
        p = tmp_path / "vc.yaml"
        p.write_text(yaml_text)
        cfg = load_config(str(p))
        assert cfg.curation.enabled is True
        assert cfg.curation.provider == "openrouter"
        assert cfg.curation.model == "qwen/qwen3-30b-a3b"
        assert cfg.curation.max_response_tokens == 4096


class TestSupersessionPrompt:
    def test_prompt_asks_about_duplicates(self):
        import tempfile
        from pathlib import Path
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker
        llm = MockLLMProvider(response="[]")
        store = SQLiteStore(str(Path(tempfile.mkdtemp()) / "test.db"))
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new_fact = Fact(subject="user", verb="has PB", object="25:50")
        candidates = [Fact(subject="user", verb="has PB", object="27:12")]
        checker._check_batch(new_fact, candidates)
        prompt = llm.calls[0]["user"]
        assert "DUPLICATE" in prompt or "duplicate" in prompt

    def test_prompt_includes_what_field(self):
        import tempfile
        from pathlib import Path
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker
        llm = MockLLMProvider(response="[]")
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


class TestFactEnrichmentIntegration:
    def test_enriched_fact_roundtrip(self, tmp_path):
        """Store enriched fact -> query it back -> verify all fields."""
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact
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
        """Old fact superseded -> query only returns new fact."""
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact
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
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact
        store = SQLiteStore(str(tmp_path / "test.db"))
        store.store_facts([
            Fact(subject="user", verb="runs", object="5K", fact_type="personal"),
            Fact(subject="user", verb="learned about", object="interval training", fact_type="experience"),
        ])
        personal_only = store.query_facts(subject="user", fact_type="personal")
        all_facts = store.query_facts(subject="user")
        assert len(personal_only) == 1
        assert len(all_facts) == 2


class TestSupersessionTagFiltering:
    """Supersession uses tags to filter candidates, not random subject match."""

    def test_tag_filtered_candidates(self, tmp_path):
        """check_and_supersede passes fact.tags to query_facts."""
        import tempfile
        from pathlib import Path
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        # Store facts with different tags
        running_fact = Fact(
            subject="user", verb="set", object="5K PB of 27:12",
            tags=["running", "5k-run", "personal-best"],
        )
        cooking_fact = Fact(
            subject="user", verb="prefers", object="Italian cuisine",
            tags=["cooking", "food-preference"],
        )
        store.store_facts([running_fact, cooking_fact])

        # New running fact — should get running candidates, not cooking
        llm = MockLLMProvider(response="[]")
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new_fact = Fact(
            subject="user", verb="set", object="5K PB of 25:50",
            tags=["running", "5k-run", "personal-best"],
        )
        store.store_facts([new_fact])
        checker.check_and_supersede([new_fact])

        # LLM should have been called with only running-related candidates
        assert len(llm.calls) == 1
        prompt = llm.calls[0]["user"]
        assert "27:12" in prompt
        assert "Italian" not in prompt

    def test_no_tags_falls_back_to_subject_only(self, tmp_path):
        """Facts without tags still get subject-based candidates."""
        import tempfile
        from pathlib import Path
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="likes", object="coffee", tags=[])
        store.store_facts([old])

        llm = MockLLMProvider(response="[]")
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new = Fact(subject="user", verb="prefers", object="tea", tags=[])
        store.store_facts([new])
        checker.check_and_supersede([new])

        # Should still call LLM (fallback to subject-only query)
        assert len(llm.calls) == 1
        prompt = llm.calls[0]["user"]
        assert "coffee" in prompt


class TestSupersessionResponseParsing:
    """Supersession handles various LLM response formats."""

    def _make_checker(self, response, tmp_path):
        from tests.conftest import MockLLMProvider
        from virtual_context.storage.sqlite import SQLiteStore
        from virtual_context.types import SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker
        llm = MockLLMProvider(response=response)
        store = SQLiteStore(str(tmp_path / "test.db"))
        return FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )

    def test_parse_bare_array(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("[0, 2]", tmp_path)
        candidates = [
            Fact(id="a", subject="user", verb="v1", object="o1"),
            Fact(id="b", subject="user", verb="v2", object="o2"),
            Fact(id="c", subject="user", verb="v3", object="o3"),
        ]
        result = checker._parse_response("[0, 2]", candidates)
        assert result == ["a", "c"]

    def test_parse_qwen3_updated_format(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("", tmp_path)
        candidates = [
            Fact(id="a", subject="user", verb="v1", object="o1"),
            Fact(id="b", subject="user", verb="v2", object="o2"),
        ]
        result = checker._parse_response('{"contradicted": [], "updated": [0]}', candidates)
        assert result == ["a"]

    def test_parse_superseded_key(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("", tmp_path)
        candidates = [Fact(id="x", subject="user", verb="v", object="o")]
        result = checker._parse_response('{"superseded": [0]}', candidates)
        assert result == ["x"]

    def test_parse_empty_array(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("", tmp_path)
        candidates = [Fact(id="a", subject="user", verb="v", object="o")]
        result = checker._parse_response("[]", candidates)
        assert result == []

    def test_parse_with_thinking_tags(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("", tmp_path)
        candidates = [Fact(id="a", subject="user", verb="v", object="o")]
        response = '<think>Let me think...</think>{"updated": [0]}'
        result = checker._parse_response(response, candidates)
        assert result == ["a"]

    def test_parse_out_of_range_index_ignored(self, tmp_path):
        from virtual_context.types import Fact
        checker = self._make_checker("", tmp_path)
        candidates = [Fact(id="a", subject="user", verb="v", object="o")]
        result = checker._parse_response("[0, 5, 99]", candidates)
        assert result == ["a"]


class _SequentialMockLLM:
    """Mock LLM that returns responses in order."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self._responses.pop(0) if self._responses else "[]"


class TestSupersessionDetectionPrompt:
    """The improved prompt detects temporal progressions."""

    def test_prompt_includes_underlying_attribute_guidance(self, tmp_path):
        from tests.conftest import MockLLMProvider
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        llm = MockLLMProvider(response="[]")
        store = SQLiteStore(str(tmp_path / "test.db"))
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        new_fact = Fact(subject="user", verb="hoping to beat",
                        object="personal best time of 25:50")
        candidates = [Fact(subject="user", verb="achieved",
                           object="personal best 5K time of 27:12")]
        checker._check_batch(new_fact, candidates)
        prompt = llm.calls[0]["user"]
        assert "underlying attribute" in prompt
        assert "earlier value" in prompt


class TestSupersessionMerge:
    """When supersession fires, the winning fact is merged with old fact's knowledge."""

    def test_merge_updates_fact_fields_in_store(self, tmp_path):
        """After supersession, winning fact's verb/object/what are updated in DB."""
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="achieved",
                   object="personal best 5K time of 27:12",
                   status="completed",
                   what="User achieved a personal best 5K time of 27:12.",
                   tags=["5k-run"])
        new = Fact(subject="user", verb="hoping to beat",
                   object="personal best time of 25:50",
                   status="active",
                   what="User is hoping to beat personal best time of 25:50.",
                   tags=["5k-run"])
        store.store_facts([old, new])

        # Detection returns [0] (old fact superseded), merge returns updated fields
        llm = _SequentialMockLLM([
            '[0]',  # detection response
            '{"verb": "holds", "object": "personal best 5K time of 25:50", '
            '"status": "active", '
            '"what": "User improved their 5K personal best from 27:12 to 25:50."}',
        ])
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        count = checker.check_and_supersede([new])

        assert count == 1
        # Old fact should be superseded
        all_facts = store.query_facts(subject="user")
        assert len(all_facts) == 1
        merged = all_facts[0]
        assert merged.id == new.id
        assert merged.verb == "holds"
        assert "25:50" in merged.object
        assert "27:12" in merged.what
        assert "25:50" in merged.what

    def test_merge_called_with_correct_prompts(self, tmp_path):
        """Merge LLM call includes durable knowledge principle."""
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="achieved", object="27:12",
                   what="User achieved 27:12.", tags=["running"])
        new = Fact(subject="user", verb="hoping to beat", object="25:50",
                   what="User hoping to beat 25:50.", tags=["running"])
        store.store_facts([old, new])

        llm = _SequentialMockLLM([
            '[0]',
            '{"verb": "holds", "object": "25:50", "status": "active", "what": "Merged."}',
        ])
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        checker.check_and_supersede([new])

        # Should have 2 LLM calls: detection + merge
        assert len(llm.calls) == 2
        merge_user = llm.calls[1]["user"]
        assert "durable knowledge" in merge_user.lower()
        assert "27:12" in merge_user
        assert "25:50" in merge_user
        assert "SUPERSEDED" in merge_user

    def test_merge_failure_does_not_crash(self, tmp_path):
        """If merge LLM returns garbage, supersession still succeeds."""
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="has", object="27:12", tags=["run"])
        new = Fact(subject="user", verb="has", object="25:50", tags=["run"])
        store.store_facts([old, new])

        llm = _SequentialMockLLM([
            '[0]',
            'invalid json garbage',
        ])
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        count = checker.check_and_supersede([new])

        # Supersession still counted, just no merge
        assert count == 1
        results = store.query_facts(subject="user")
        assert len(results) == 1
        # Verb unchanged (merge failed gracefully)
        assert results[0].verb == "has"

    def test_merge_updates_in_memory_fact(self, tmp_path):
        """After merge, the in-memory Fact object is updated for subsequent merges."""
        from virtual_context.types import Fact, SupersessionConfig
        from virtual_context.ingest.supersession import FactSupersessionChecker

        store = SQLiteStore(str(tmp_path / "test.db"))
        old = Fact(subject="user", verb="v1", object="o1", tags=["t"])
        new = Fact(subject="user", verb="v2", object="o2", tags=["t"])
        store.store_facts([old, new])

        llm = _SequentialMockLLM([
            '[0]',
            '{"verb": "merged_v", "object": "merged_o", "status": "active", "what": "merged_w"}',
        ])
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store, config=SupersessionConfig(enabled=True),
        )
        checker.check_and_supersede([new])

        # In-memory object should be updated
        assert new.verb == "merged_v"
        assert new.object == "merged_o"
        assert new.what == "merged_w"


class TestUpdateFactFields:
    """Store.update_fact_fields persists changes and syncs FTS."""

    def test_update_persists_to_db(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        fact = Fact(subject="user", verb="old_verb", object="old_obj",
                    status="active", what="old what")
        store.store_facts([fact])

        store.update_fact_fields(fact.id, "new_verb", "new_obj", "completed", "new what")

        results = store.query_facts(subject="user")
        assert len(results) == 1
        assert results[0].verb == "new_verb"
        assert results[0].object == "new_obj"
        assert results[0].status == "completed"
        assert results[0].what == "new what"

    def test_update_syncs_fts(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "test.db"))
        fact = Fact(subject="user", verb="original", object="before",
                    what="original text")
        store.store_facts([fact])

        store.update_fact_fields(fact.id, "updated", "after", "active", "updated text")

        # FTS search should find the updated text
        results = store.search_facts("updated text")
        assert len(results) == 1
        assert results[0].id == fact.id


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

    def test_compactor_sets_session_date_on_facts(self, tmp_path):
        """Compactor should stamp each extracted fact with the segment's session_date."""
        from unittest.mock import MagicMock
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, TaggedSegment, Message
        import json

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

    def test_v4_prompt_injects_session_date(self, tmp_path):
        """session_date must appear in the compactor prompt."""
        from unittest.mock import MagicMock
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
        compactor.compact([segment])

        assert len(captured_prompts) == 1
        assert "2023/04/20" in captured_prompts[0], "session_date not injected into prompt"

    def test_v4_prompt_today_gives_session_date_when(self, tmp_path):
        """Prompt must inject session_date; LLM returning today's date is preserved in when_date."""
        from unittest.mock import MagicMock
        from virtual_context.core.compactor import DomainCompactor
        from virtual_context.types import CompactorConfig, TaggedSegment, Message
        import json

        captured_prompts = []

        llm = MagicMock()
        def capture_and_return(**kwargs):
            captured_prompts.append(kwargs.get("user", ""))
            return json.dumps({
                "summary": "User hiked Big Sur today.",
                "entities": [], "key_decisions": [], "action_items": [],
                "date_references": [], "refined_tags": ["hiking"], "related_tags": [],
                "facts": [{
                    "subject": "user", "verb": "hiked",
                    "object": "Big Sur",
                    "status": "completed", "fact_type": "personal",
                    "what": "User hiked Big Sur today.",
                    "who": "", "when": "2023/04/20", "where": "", "why": "",
                }],
            })
        llm.complete.side_effect = capture_and_return

        segment = TaggedSegment(
            id="seg-003", primary_tag="hiking", tags=["hiking"],
            messages=[Message(role="user", content="I hiked Big Sur today.")],
            session_date="2023/04/20 (Thu) 04:17",
        )
        compactor = DomainCompactor(llm_provider=llm, config=CompactorConfig())
        result = compactor.compact([segment])

        # Session date must appear in the prompt so the LLM knows what "today" means
        assert len(captured_prompts) == 1
        assert "2023/04/20" in captured_prompts[0], "session_date not injected into prompt"
        # LLM returned the session date as when_date — compactor should preserve it
        assert result[0].facts[0].when_date == "2023/04/20"


class TestExtractObjectKeyword:
    def test_extracts_proper_noun_from_destination(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("solo camping trip to Yosemite National Park") == "Yosemite"

    def test_extracts_from_big_sur(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("Big Sur and Monterey") == "Monterey"

    def test_extracts_from_muir_woods(self):
        from virtual_context.ingest.supersession import _extract_object_keyword
        assert _extract_object_keyword("Dipsea Trail at Muir Woods") == "Dipsea"

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
        store.store_facts([old])

        # New fact: different tags (session 2, tags=['bear-safety'])
        new_fact = Fact(
            subject="user", verb="started",
            object="solo camping trip to Yosemite National Park",
            status="completed", tags=["bear-safety"],
            what="User started solo camping trip to Yosemite National Park.",
        )
        store.store_facts([new_fact])

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
        store.store_facts([new_fact])

        llm = MockLLMProvider(response="[]")
        checker = FactSupersessionChecker(
            llm_provider=llm, model="test",
            store=store,
            config=SupersessionConfig(enabled=True, batch_size=20),
        )
        # Should not crash; 0 superseded since no candidates
        count = checker.check_and_supersede([new_fact])
        assert count == 0


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
