# Fact Curation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable LLM-based fact curator that filters the full facts list down to query-relevant facts before injection, reducing noise from hundreds of unrelated facts to a focused subset.

**Architecture:** A new `FactCurator` class (in `virtual_context/ingest/curator.py`) receives all facts and the user's question, sends them to an LLM with the prompt "Which of these facts could be relevant to answering the question?", and returns the filtered subset. The engine calls it in `on_message_inbound` after retrieval and before assembly. Configuration mirrors `SupersessionConfig` exactly — `enabled`, `provider`, `model`, plus `max_response_tokens`.

**Tech Stack:** Python, existing `LLMProvider.complete()` protocol, existing `MockLLMProvider` from `tests/conftest.py`, SQLite store.

---

### Task 1: Add `CurationConfig` to types and config loader

**Files:**
- Modify: `virtual_context/types.py` (after `SupersessionConfig` at line 641)
- Modify: `virtual_context/config.py` (after supersession block at line 276)
- Test: `tests/test_fact_enrichment.py`

**Background:** `SupersessionConfig` at line 635–641 of `types.py` is the exact pattern to follow. `VirtualContextConfig` at line 643–664 holds all sub-configs. The config loader at `config.py:269–296` manually constructs each config from raw YAML dict — needs a parallel `curation:` block.

**Step 1: Write the failing test**

Add to `tests/test_fact_enrichment.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestCurationConfig -v
```

Expected: FAIL with `ImportError: cannot import name 'CurationConfig'`

**Step 3: Implement**

In `virtual_context/types.py`, after the `SupersessionConfig` block (after line 641):

```python
@dataclass
class CurationConfig:
    """Configuration for LLM-based fact curation."""
    enabled: bool = False
    provider: str = ""   # provider name from providers dict, or "" to use summarization provider
    model: str = ""      # model override, or "" to use summarization model
    max_response_tokens: int = 2048
```

Add `curation` to `VirtualContextConfig` (after the `supersession` field at line 662):

```python
    curation: CurationConfig = field(default_factory=CurationConfig)
```

Add `CurationConfig` to the import in `virtual_context/config.py` (line 26 already imports from types — add `CurationConfig` to that import).

In `virtual_context/config.py`, after the supersession block (after line 276):

```python
    # Curation config
    cur_raw = raw.get("curation", {})
    curation_config = CurationConfig(
        enabled=cur_raw.get("enabled", False),
        provider=cur_raw.get("provider", ""),
        model=cur_raw.get("model", ""),
        max_response_tokens=cur_raw.get("max_response_tokens", 2048),
    )
```

Add `curation=curation_config,` to the `VirtualContextConfig(...)` constructor call (after `supersession=supersession_config,` at line 296).

**Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestCurationConfig -v
```

Expected: 3 PASS

**Step 5: Commit**

```bash
git add virtual_context/types.py virtual_context/config.py tests/test_fact_enrichment.py
git commit -m "feat: add CurationConfig type and config loader"
```

---

### Task 2: Implement `FactCurator` class

**Files:**
- Create: `virtual_context/ingest/curator.py`
- Test: `tests/test_fact_enrichment.py`

**Background:** The curator receives a list of `Fact` objects and a question string. It formats the facts as `[i] verb | object [date]` (same compact format as the POC), sends to LLM, parses comma-separated indices from the response, and returns the filtered list. On any failure (LLM error, parse error, empty response), it returns the original list unchanged — never silently drops all facts. `MockLLMProvider` from `tests/conftest.py` is the standard test double.

**Step 1: Write the failing tests**

Add to `tests/test_fact_enrichment.py`:

```python
class TestFactCurator:
    def test_filters_facts_by_llm_response(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.types import Fact, CurationConfig
        from virtual_context.ingest.curator import FactCurator

        facts = [
            Fact(subject="user", verb="hiked", object="Dipsea Trail at Muir Woods"),
            Fact(subject="user", verb="studied", object="automata theory chapter 4"),
            Fact(subject="user", verb="road tripped", object="with friends to Big Sur"),
        ]
        # LLM says facts 0 and 2 are relevant
        llm = MockLLMProvider(response="0, 2")
        curator = FactCurator(llm_provider=llm, model="test",
                              config=CurationConfig(enabled=True, max_response_tokens=256))
        result = curator.curate(facts, question="What trips did I take?")
        assert len(result) == 2
        assert result[0].object == "Dipsea Trail at Muir Woods"
        assert result[1].object == "with friends to Big Sur"

    def test_falls_back_to_all_facts_on_empty_response(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.types import Fact, CurationConfig
        from virtual_context.ingest.curator import FactCurator

        facts = [Fact(subject="user", verb="hiked", object="somewhere")]
        llm = MockLLMProvider(response="")
        curator = FactCurator(llm_provider=llm, model="test",
                              config=CurationConfig(enabled=True))
        result = curator.curate(facts, question="Where did I hike?")
        assert result == facts  # unchanged

    def test_falls_back_on_llm_exception(self):
        from virtual_context.types import Fact, CurationConfig, LLMProvider
        from virtual_context.ingest.curator import FactCurator

        class FailingProvider:
            def complete(self, system, user, max_tokens):
                raise RuntimeError("API down")

        facts = [Fact(subject="user", verb="visited", object="Paris")]
        curator = FactCurator(llm_provider=FailingProvider(), model="test",
                              config=CurationConfig(enabled=True))
        result = curator.curate(facts, question="Where have I traveled?")
        assert result == facts  # fallback to original

    def test_includes_when_date_in_prompt(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.types import Fact, CurationConfig
        from virtual_context.ingest.curator import FactCurator

        facts = [Fact(subject="user", verb="started", object="camping trip",
                      when_date="2023/05/15")]
        llm = MockLLMProvider(response="0")
        curator = FactCurator(llm_provider=llm, model="test",
                              config=CurationConfig(enabled=True))
        curator.curate(facts, question="When did I go camping?")
        user_prompt = llm.calls[0]["user"]
        assert "2023/05/15" in user_prompt
        assert "[when:" in user_prompt

    def test_returns_empty_list_when_no_facts(self):
        from tests.conftest import MockLLMProvider
        from virtual_context.types import CurationConfig
        from virtual_context.ingest.curator import FactCurator

        llm = MockLLMProvider(response="")
        curator = FactCurator(llm_provider=llm, model="test",
                              config=CurationConfig(enabled=True))
        result = curator.curate([], question="Any question")
        assert result == []
        assert llm.calls == []  # no LLM call for empty input
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactCurator -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'virtual_context.ingest.curator'`

**Step 3: Implement `FactCurator`**

Create `virtual_context/ingest/curator.py`:

```python
"""Fact curator: LLM-based relevance filter for the facts block."""

from __future__ import annotations

import logging
import re

from ..types import CurationConfig, Fact, LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a memory relevance assistant. "
    "Given a user question and a list of personal facts, identify which facts "
    "could possibly be relevant to answering the question. "
    "Respond only with the fact numbers, comma-separated (e.g. '0, 3, 7'). "
    "No explanation. If none are relevant, respond with an empty string."
)


class FactCurator:
    """Filter a facts list down to query-relevant facts using an LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        config: CurationConfig,
    ) -> None:
        self.llm = llm_provider
        self.model = model
        self.config = config

    def curate(self, facts: list[Fact], question: str) -> list[Fact]:
        """Return the subset of facts relevant to the question.

        Falls back to the original list on any error.
        """
        if not facts:
            return facts

        facts_text = self._format_facts(facts)
        user_prompt = (
            f'User question: "{question}"\n\n'
            f"Which of these facts could be relevant to answering the question? "
            f"Output only the fact numbers, comma-separated.\n\n"
            f"Facts:\n{facts_text}"
        )

        try:
            response = self.llm.complete(
                system=_SYSTEM,
                user=user_prompt,
                max_tokens=self.config.max_response_tokens,
            )
        except Exception as e:
            logger.warning("Fact curation LLM call failed: %s — returning all facts", e)
            return facts

        selected = self._parse_response(response, len(facts))
        if not selected:
            logger.debug("Fact curation returned no indices — returning all facts")
            return facts

        logger.info("Fact curation: %d → %d facts", len(facts), len(selected))
        return [facts[i] for i in selected]

    def _format_facts(self, facts: list[Fact]) -> str:
        lines = []
        for i, f in enumerate(facts):
            line = f"[{i}] {f.verb} | {f.object}"
            if f.when_date:
                line += f" [when: {f.when_date}]"
            elif f.session_date:
                line += f" [session: {f.session_date}]"
            lines.append(line)
        return "\n".join(lines)

    def _parse_response(self, response: str, total: int) -> list[int]:
        """Extract valid fact indices from LLM response."""
        # Strip thinking tags (e.g. Qwen3)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        nums = re.findall(r"\d+", response)
        return [int(n) for n in nums if 0 <= int(n) < total]
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestFactCurator -v
```

Expected: 5 PASS

**Step 5: Commit**

```bash
git add virtual_context/ingest/curator.py tests/test_fact_enrichment.py
git commit -m "feat: add FactCurator class with LLM-based fact relevance filtering"
```

---

### Task 3: Wire `FactCurator` into the engine

**Files:**
- Modify: `virtual_context/engine.py` (around lines 94–111 for init, and 559–570 for on_message_inbound)
- Test: `tests/test_fact_enrichment.py`

**Background:** The engine already has `_supersession_checker` wired in with `_init_supersession_checker()` at line 426. Follow the exact same pattern for `_fact_curator`. In `on_message_inbound`, curation happens after `self._retriever.retrieve(...)` returns `retrieval_result` (line 525) and before `self._assembler.assemble(...)` (line 561). Mutate `retrieval_result.facts` in place.

**Step 1: Write the failing test**

Add to `tests/test_fact_enrichment.py`:

```python
class TestEngineCuration:
    def test_curator_filters_facts_before_assembly(self, tmp_path):
        """When curation is enabled, engine passes curated facts to assembler."""
        from tests.conftest import MockLLMProvider, MockTagGenerator
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.types import VirtualContextConfig, CurationConfig, Message

        cfg = VirtualContextConfig(
            storage_root=str(tmp_path / ".vc"),
            curation=CurationConfig(enabled=True),
        )
        engine = VirtualContextEngine(config=cfg)

        # Inject a curator with a mock that returns index 0 only
        from virtual_context.ingest.curator import FactCurator
        from virtual_context.types import Fact
        mock_llm = MockLLMProvider(response="0")
        engine._fact_curator = FactCurator(
            llm_provider=mock_llm,
            model="test",
            config=cfg.curation,
        )

        # Store two facts
        from virtual_context.types import Fact
        facts = [
            Fact(subject="user", verb="hiked", object="Dipsea Trail"),
            Fact(subject="user", verb="studied", object="automata theory"),
        ]
        engine._store.store_facts(facts, segment_ref="seg-1", tags=["hiking"])

        assembled = engine.on_message_inbound(
            "Where did I hike?",
            conversation_history=[],
        )
        # Curator was called (mock recorded a call)
        assert mock_llm.calls, "Curator LLM was not called"
        # Only 1 fact survived curation — assembler saw only the hiking fact
        assert assembled.facts_text is not None
        assert "Dipsea Trail" in assembled.facts_text
        assert "automata theory" not in assembled.facts_text

    def test_curator_disabled_passes_all_facts(self, tmp_path):
        """When curation is disabled, all facts reach the assembler."""
        from tests.conftest import MockLLMProvider
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.types import VirtualContextConfig, CurationConfig, Fact

        cfg = VirtualContextConfig(
            storage_root=str(tmp_path / ".vc"),
            curation=CurationConfig(enabled=False),
        )
        engine = VirtualContextEngine(config=cfg)
        assert engine._fact_curator is None

        facts = [
            Fact(subject="user", verb="hiked", object="trail"),
            Fact(subject="user", verb="studied", object="math"),
        ]
        engine._store.store_facts(facts, segment_ref="seg-1", tags=["misc"])
        assembled = engine.on_message_inbound("test", conversation_history=[])
        # Both facts present (curation never ran)
        assert "trail" in (assembled.facts_text or "")
        assert "math" in (assembled.facts_text or "")
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestEngineCuration -v
```

Expected: FAIL — `AttributeError: 'VirtualContextEngine' object has no attribute '_fact_curator'`

**Step 3: Implement engine wiring**

In `virtual_context/engine.py`:

After line 110 (`self._supersession_checker = None`), add:

```python
        self._fact_curator = None
        self._init_fact_curator()
```

Add `_init_fact_curator` method after `_init_supersession_checker` (after line 445):

```python
    def _init_fact_curator(self) -> None:
        """Initialize the fact curator from config if enabled."""
        cc = self.config.curation
        if not cc.enabled:
            return
        provider_name = cc.provider or self.config.summarization.provider
        model = cc.model or self.config.summarization.model
        provider_config = self.config.providers.get(provider_name, {})
        llm = self._build_provider(provider_name, provider_config)
        if not llm:
            logger.warning("Curation enabled but provider '%s' could not be built", provider_name)
            return
        from .ingest.curator import FactCurator
        self._fact_curator = FactCurator(
            llm_provider=llm,
            model=model,
            config=cc,
        )
        logger.info("Fact curator initialized (provider=%s, model=%s)", provider_name, model)
```

In `on_message_inbound`, after `retrieval_result = self._retriever.retrieve(...)` (after line 531) and before `context_hint = self._build_context_hint(...)` (line 535), add:

```python
        # D2: Curate facts down to query-relevant subset before assembly
        if self._fact_curator and retrieval_result.facts:
            retrieval_result.facts = self._fact_curator.curate(
                retrieval_result.facts,
                question=message,
            )
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_fact_enrichment.py::TestEngineCuration -v
```

Expected: 2 PASS

**Step 5: Run full suite**

```bash
.venv/bin/pytest tests/ --ignore=tests/ollama --ignore=tests/haiku --ignore=tests/test_tui.py -m 'not slow' -x -q
```

Expected: all pass, no regressions

**Step 6: Commit**

```bash
git add virtual_context/engine.py tests/test_fact_enrichment.py
git commit -m "feat: wire FactCurator into engine on_message_inbound pipeline"
```
