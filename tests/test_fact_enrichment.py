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
