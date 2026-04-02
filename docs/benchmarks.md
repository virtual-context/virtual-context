# Benchmarks

Virtual-context is evaluated against four established memory benchmarks and an internal stress test suite. All benchmarks run against the full pipeline: tagging, compaction, retrieval, assembly, and LLM response.

## Benchmark Suites

### LocOMo (Long Conversation Memory)

Tests memory accuracy over extended multi-turn conversations. Questions are categorized by type:

| Question Type | Description |
|--------------|-------------|
| Single-hop | Direct recall of a stated fact |
| Multi-hop | Combining facts from different conversation turns |
| Open-ended | Questions requiring synthesis across topics |
| Temporal | Questions about when events occurred |
| Adversarial | Questions designed to confuse retrieval (similar topics, contradictions) |

**Results**: 95% overall accuracy vs. 33% for full-history baselines. The largest gains are on temporal and multi-hop questions, where raw history dumps bury the relevant facts in noise.

### LongMemEval

Evaluates long-term memory fidelity after compaction. Tests whether facts survive multiple compaction cycles without degradation.

The benchmark runs a conversation through hundreds of turns, triggering multiple compaction events, then queries for facts stated early in the conversation. This tests the full compaction -> storage -> retrieval -> assembly pipeline.

### MRCR (Multi-Round Conversational Retrieval)

Tests retrieval precision across topic switches. The conversation covers multiple distinct topics, then questions target specific topics to measure whether retrieval surfaces the right segments without cross-contamination.

This is where the context bleed gate and active tag skipping are tested: the system must retrieve "the database migration discussion" without also pulling in "the API design discussion" that happened in adjacent turns.

### AMB (Agent Memory Benchmark)

Tests memory in agentic contexts where the model uses tools, executes code, and maintains state across complex multi-step tasks. This is the most realistic benchmark: conversations include `tool_use`/`tool_result` pairs, chain collapses, and interleaved planning discussions.

AMB tests whether chain collapse preserves recoverable information, whether fact extraction captures decisions made during tool use, and whether retrieval handles the mixed content types in agentic conversations.

## Benchmark Infrastructure

Benchmarks live in `benchmarks/` and are structured as:

```
benchmarks/
  locomo/
    baseline.py       # LLM provider calls with retry logic
    runner.py          # Orchestrates test execution
    evaluator.py       # Scores responses against ground truth
    datasets/          # Question sets
  longmemeval/
    ...
  mrcr/
    ...
  amb/
    ...
```

Each benchmark runner:
1. Loads a dataset of conversations and questions
2. Replays the conversation through the virtual-context engine
3. Asks each question via the retrieval pipeline
4. Scores the response against ground truth
5. Reports accuracy by question type

Benchmarks use the engine directly (not through the proxy) for reproducibility and speed.

## Stress Tests

The stress test suite validates pipeline behavior under adversarial conditions. It uses a set of prompt files that exercise edge cases:

| Test Category | What It Tests |
|--------------|--------------|
| Topic cycling | Rapid switches between 10+ topics, verifying retrieval stability |
| Compaction cascade | 200+ turns forcing multiple compaction events, checking for content loss |
| Tag explosion | Conversations that generate 100+ unique tags, testing index performance |
| Concurrent access | Multiple simultaneous requests against the same session |
| Large payloads | Messages with embedded images, code blocks, and tool results exceeding 50K tokens per turn |
| Empty turns | Messages with no semantic content, testing graceful degradation |
| Contradiction storms | Sequences of contradictory facts, testing supersession |

Stress tests are runnable via the dashboard's Replay feature: point it at a prompt file and watch metrics update live.

## Running Benchmarks

```bash
# Run a specific benchmark
python -m benchmarks.locomo.runner --config virtual-context.yaml

# Run with a specific provider
python -m benchmarks.locomo.runner --provider anthropic --model claude-sonnet-4-20250514

# Run stress tests via the proxy dashboard
# 1. Start the proxy
virtual-context proxy --upstream https://api.anthropic.com

# 2. Open http://localhost:8100/dashboard
# 3. Use the Replay panel with a stress test file
```

## Interpreting Results

**Accuracy by question type** is the primary metric. Overall accuracy can mask weaknesses: a system might score 90% overall but 40% on temporal questions.

**Tokens freed** measures compaction efficiency. Higher is better, but not at the cost of accuracy. The goal is maximum compression with minimum information loss.

**Retrieval precision** measures what fraction of retrieved segments were actually relevant to the question. Low precision means the system is wasting context budget on irrelevant content.

**Compression ratio** is the ratio of summary tokens to original tokens. Typical values are 0.15-0.25 (4x-7x compression). Below 0.10 risks losing detail; above 0.30 suggests summaries are too verbose.

## Regression Tests

The test suite includes regression markers tied to specific bugs (BUG-001 through BUG-031+). Each regression test reproduces the exact scenario that triggered a bug and verifies the fix. These run as part of the standard `pytest` suite:

```bash
pytest tests/ -k "BUG"
```

Key regression areas:
- **BUG-012**: Turn number indexing breaks when preamble messages shift indices
- **BUG-018**: Tag summary rebuild drops segments during concurrent compaction
- **BUG-024**: Chain restore produces orphaned tool_use without matching tool_result
- **BUG-027**: Fact supersession fails when old and new facts use different verb forms
