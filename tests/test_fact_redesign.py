"""Tests for fact extraction redesign."""
from virtual_context.types import CompactorConfig


def test_compactor_config_has_code_mode():
    cc = CompactorConfig()
    assert hasattr(cc, "code_mode")
    assert cc.code_mode is True


def test_config_loader_reads_code_mode():
    from virtual_context.config import load_config
    import tempfile, os, yaml
    cfg = {"compaction": {"code_mode": False}, "context_window": 100000}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    try:
        config = load_config(tmp_path)
        assert config.compactor.code_mode is False
    finally:
        os.unlink(tmp_path)


def test_replace_facts_for_segment_default():
    from virtual_context.core.store import ContextStore

    class MinimalStore(ContextStore):
        """Minimal concrete subclass that only implements abstract methods as no-ops."""
        def store_segment(self, segment): return ""
        def get_segment(self, ref, *, conversation_id=None): return None
        def get_summary(self, ref, *, conversation_id=None): return None
        def get_summaries_by_tags(self, tags, min_overlap=1, limit=10, before=None, after=None, conversation_id=None): return []
        def search(self, query, tags=None, limit=5, conversation_id=None): return []
        def get_all_tags(self, conversation_id=None): return []
        def get_conversation_stats(self): return []
        def get_tag_aliases(self, conversation_id=None): return {}
        def set_tag_alias(self, alias, canonical, conversation_id=""): pass
        def delete_segment(self, ref): return False
        def cleanup(self, max_age=None, max_total_tokens=None): return 0
        def save_tag_summary(self, tag_summary, conversation_id=""): pass
        def get_tag_summary(self, tag, conversation_id=""): return None
        def get_all_tag_summaries(self, *, conversation_id=None): return []
        def search_full_text(self, query, limit=5, conversation_id=None): return []
        def get_segments_by_tags(self, tags, min_overlap=1, limit=20, conversation_id=None): return []

    store = MinimalStore()
    deleted, inserted = store.replace_facts_for_segment("conv1", "seg1", [])
    assert deleted == 0
    assert inserted == 0


def test_compaction_uses_replace_facts_for_segment():
    """Verify the compaction pipeline calls replace_facts_for_segment."""
    import inspect
    from virtual_context.core import compaction_pipeline
    source = inspect.getsource(compaction_pipeline)
    assert "replace_facts_for_segment" in source, \
        "compaction_pipeline must call replace_facts_for_segment"


def test_code_mode_prompt_appended():
    """When code_mode is True, the compactor prompt should include the code mode block."""
    from virtual_context.core.compactor import CODE_MODE_FACT_PROMPT
    assert "Do NOT extract intermediary" in CODE_MODE_FACT_PROMPT
    assert "conclusions, findings, discoveries" in CODE_MODE_FACT_PROMPT


def test_tag_generator_accepts_code_mode():
    from virtual_context.core.tag_generator import LLMTagGenerator
    from virtual_context.types import TagGeneratorConfig
    from unittest.mock import MagicMock
    gen = LLMTagGenerator(llm_provider=MagicMock(), config=TagGeneratorConfig(), code_mode=True)
    assert gen._code_mode is True


def test_build_tag_generator_passes_code_mode():
    from virtual_context.core.tag_generator import build_tag_generator, LLMTagGenerator
    from virtual_context.types import TagGeneratorConfig
    from unittest.mock import MagicMock
    config = TagGeneratorConfig(type="llm", provider="openrouter", model="test")
    gen = build_tag_generator(config, MagicMock(), code_mode=True)
    assert isinstance(gen, LLMTagGenerator)
    assert gen._code_mode is True
    gen2 = build_tag_generator(config, MagicMock())
    assert gen2._code_mode is False


def test_engine_passes_code_mode_to_tag_generator():
    import inspect
    from virtual_context import engine
    source = inspect.getsource(engine)
    assert "code_mode" in source, "engine.py must reference code_mode"
    assert "compactor.code_mode" in source or "config.compactor.code_mode" in source, \
        "engine.py must read code_mode from compactor config"
