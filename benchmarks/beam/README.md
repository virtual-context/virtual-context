# BEAM Benchmark Runner

Evaluates Virtual Context against the [BEAM](https://github.com/mohammadtavakoli78/BEAM) long-term memory benchmark.

BEAM contains 100 synthetic multi-turn conversations (128K-10M tokens) with 2,000 probing questions across 10 memory ability categories: abstention, contradiction_resolution, event_ordering, information_extraction, instruction_following, knowledge_update, multi_session_reasoning, preference_following, summarization, temporal_reasoning.

## Prerequisites

1. Clone the BEAM repo:
   ```
   git clone https://github.com/mohammadtavakoli78/BEAM ~/projects/BEAM
   ```

2. Set API keys in `benchmarks/beam/.env` (gitignored):
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENROUTER_API_KEY=sk-or-...
   ```

3. Activate the project venv:
   ```
   source .venv/bin/activate
   ```

## Quick start

```bash
# Run 1 conversation (100K), with judging, verbose logging
python -m benchmarks.beam --end 1 --judge -v -o results.json

# Run a specific conversation by ID
python -m benchmarks.beam --conversations 100K_3 --judge -o results.json

# Filter to specific question categories
python -m benchmarks.beam --end 1 --categories knowledge_update temporal_reasoning --judge -o results.json

# Run a single question by ID
python -m benchmarks.beam --end 1 --question-id 100K_1_abstention_0 --judge -o results.json

# Ingest only (no question answering) — useful for validating ingestion
python -m benchmarks.beam --end 1 --ingest-only -v
```

## Two-phase architecture

The runner follows the same pattern as LongMemEval and LocOMo:

1. **Ingestion** (`ingest_conversation`): Loads the conversation, feeds it through the VC engine (tagging, summarization, compaction, fact extraction), and caches the result to SQLite. Ingestion is expensive (minutes per conversation) and is only done once per conversation. The cache lives at `benchmarks/beam/cache/{chat_size}/{conv_id}/`.

2. **Query** (`query_question`): For each probing question, calls `engine.on_message_inbound()` to assemble relevant context (tag sections, facts, conversation history), then sends it to the reader model. This phase reuses the cached ingestion — you only pay reader costs on reruns.

On subsequent runs, if the cache exists, ingestion is skipped entirely and the engine is restored from the cached SQLite store.

## CLI flags

### Data selection

| Flag | Default | Description |
|------|---------|-------------|
| `--beam-root PATH` | `~/projects/BEAM` | Path to cloned BEAM repo |
| `--chat-size SIZE` | `100K` | `100K`, `500K`, `1M`, or `10M` |
| `--start N` | `0` | First conversation index |
| `--end N` | all | Last conversation index (exclusive) |
| `--conversations ID [...]` | all | Run only these conversation IDs |
| `--categories CAT [...]` | all | Filter probing question categories |
| `--question-id ID [...]` | all | Run only these question IDs (e.g. `100K_1_abstention_0`) |

### VC engine configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--context-window N` | `65536` | Context window in tokens. Controls how much of the compacted conversation the assembler can use. |
| `--reader-model MODEL` | `claude-sonnet-4-20250514` | Model for answering probing questions |
| `--reader-provider NAME` | `anthropic` | `anthropic`, `openai`, or `openrouter` |
| `--tagger-model MODEL` | `xiaomi/mimo-v2-flash` | Model for tagging and summarization during ingestion |
| `--tagger-provider NAME` | `openrouter` | Provider for the tagger model |
| `--summarizer-provider NAME` | (same as tagger) | Override provider for summarizer only |
| `--summarizer-model MODEL` | (same as tagger) | Override model for summarizer only |
| `--tagger-mode MODE` | `split` | `split` or `unified` |
| `--fact-provider NAME` | (same as tagger) | Override provider for fact extraction |
| `--fact-model MODEL` | (same as tagger) | Override model for fact extraction |

### Cache control

These flags control what gets recomputed vs reused from cache.

| Flag | Effect on cache | Cost |
|------|----------------|------|
| *(no flag)* | Reuse everything. Skip ingestion if cache exists, only pay for reader queries. | Cheapest |
| `--recompact` | Keep cached tags and facts, re-run compaction only. Useful if you changed compaction settings. | Medium |
| `--clear-cache` | **Delete the entire cache directory** and re-ingest from scratch. Removes all stored segments, tags, facts, compaction state, and payload logs. You will pay full ingestion cost again. | Most expensive |
| `--ingest-only` | Run ingestion and compaction only, skip question answering. Useful for validating ingestion before spending on reader queries. | Ingestion cost only |
| `--cache-dir PATH` | Override the default cache location (`benchmarks/beam/cache/`). | N/A |

### Judging

| Flag | Default | Description |
|------|---------|-------------|
| `--judge` | off | Enable LLM-as-judge scoring after each question |
| `--judge-model MODEL` | `claude-sonnet-4-20250514` | Model for judging |
| `--judge-provider NAME` | `anthropic` | Provider for the judge model |

The judge uses BEAM's official 0/0.5/1 scoring scale with semantic tolerance rules (paraphrase acceptance, case/punctuation ignoring, equivalent number formats). Event ordering uses Kendall tau-b rank correlation.

### Autopsy reports

| Flag | Default | Description |
|------|---------|-------------|
| `--no-autopsy-report` | off | Skip autopsy report generation |
| `--autopsy-output-prefix P` | (derived from -o) | Output prefix for autopsy files (without extension) |

Autopsy reports are generated automatically after each run (unless `--no-autopsy-report`). They produce `.autopsy.json` and `.autopsy.md` sidecars with per-question analysis: rubric breakdown, retrieval stats, tool loop analysis, store diagnostics, and reader payload summary.

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--budget N` | `50.0` | Max spend in USD. Stops processing when exceeded. |
| `-o, --output PATH` | none | Write results JSON to this path. Results are saved incrementally after each conversation. |
| `-v, --verbose` | off | Enable DEBUG-level logging |

## Cache directory structure

```
benchmarks/beam/cache/
  100K/
    100K_1/
      store.db           # SQLite — segments, tags, facts, compaction state
      telemetry.json     # Ingestion timing and stats
      payloads/          # Per-question reader payloads and engine logs
        100K_1_knowledge_update_1_*.payload.json
        100K_1_knowledge_update_1_*.meta.json
        100K_1_knowledge_update_1_*.engine.log
    100K_2/
      ...
  500K/
    ...
```

## Scoring

The judge follows BEAM's official methodology:

- **Most categories**: 0/0.5/1 scale per rubric criterion, averaged across criteria
- **Event ordering**: Kendall tau-b rank correlation (0-1), not rubric-based
- **Semantic tolerance**: Paraphrases accepted, case/punctuation ignored, equivalent number formats matched
- **Abstention**: Scored on whether the model correctly identifies unanswerable questions

Overall score is the mean across all judged questions.
