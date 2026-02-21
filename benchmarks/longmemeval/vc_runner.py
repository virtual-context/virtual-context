"""VC pipeline: ingest → compact → retrieve → query for a single LongMemEval question."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from virtual_context.config import load_config
from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message

from .cost import BudgetTracker
from .dataset import LongMemEvalQuestion

logger = logging.getLogger(__name__)

API_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    # Gemini URL is model-dependent, handled by adapter
}
API_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
CACHE_DIR = Path(__file__).parent / "cache"


def _cache_dir_for(question_id: str) -> Path:
    """Stable cache directory for a question's ingested + compacted state."""
    return CACHE_DIR / question_id


def clear_cache(question_ids: list[str] | None = None) -> int:
    """Remove cached VC state for given questions (or all if None).

    Returns number of caches removed.
    """
    if not CACHE_DIR.exists():
        return 0

    if question_ids is None:
        # Clear all
        count = sum(1 for p in CACHE_DIR.iterdir() if p.is_dir())
        shutil.rmtree(CACHE_DIR)
        return count

    count = 0
    for qid in question_ids:
        d = _cache_dir_for(qid)
        if d.exists():
            shutil.rmtree(d)
            count += 1
    return count


def _build_vc_config(
    context_window: int = 65536,
    storage_dir: str | None = None,
    session_id: str = "",
    summarizer_model: str | None = None,
) -> dict:
    """Build a VC config dict for benchmark use."""
    cfg: dict = {
        "version": "0.2",
        "storage_root": storage_dir or "",
        "context_window": context_window,
        "token_counter": "estimate",
        "paging": {
            "enabled": True,
            "auto_promote": True,
            "auto_evict": True,
        },
        "tag_generator": {
            "type": "llm",
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "max_tags": 10,
            "min_tags": 5,
            "prompt_mode": "detailed",
            "broad_heuristic_enabled": False,
            "temporal_heuristic_enabled": False,
            "context_bleed_threshold": 0,  # disable embedding gate for batch processing
        },
        "compaction": {
            "soft_threshold": 0.70,
            "hard_threshold": 0.85,
            "protected_recent_turns": 4,
            "overflow_buffer": 1.2,
            "summary_ratio": 0.15,
            "min_summary_tokens": 100,
            "max_summary_tokens": 800,
            "max_concurrent_summaries": 4,
        },
        "summarization": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "temperature": 0.3,
        },
        "providers": {
            "anthropic": {
                "type": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-haiku-4-5-20251001",
            },
        },
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": f"{storage_dir}/store.db"},
            "filesystem": {"root": f"{storage_dir}/store"},
        },
        "assembly": {
            "core_context_max_tokens": 2000,
            "tag_context_max_tokens": 50000,
            "context_hint_enabled": True,
            "context_hint_max_tokens": 2000,
            "core_files": [],
        },
        "retrieval": {
            "skip_active_tags": True,
            "active_tag_lookback": 4,
            "strategy_config": {
                "default": {
                    "min_overlap": 1,
                    "max_results": 10,
                    "max_budget_fraction": 0.25,
                    "include_related": True,
                },
            },
        },
        "cost_tracking": {
            "enabled": True,
            "pricing": {
                "anthropic": {
                    "input_per_1k": 0.00025,
                    "output_per_1k": 0.00125,
                },
            },
        },
    }
    # Optional: override summarizer with a non-Anthropic model (e.g. gpt-4o-mini)
    if summarizer_model:
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        cfg["providers"]["openai"] = {
            "type": "generic_openai",
            "base_url": "https://api.openai.com/v1",
            "model": summarizer_model,
            "api_key": openai_key,
        }
        cfg["summarization"]["provider"] = "openai"
        cfg["summarization"]["model"] = summarizer_model
        logger.info("Summarizer override: %s via OpenAI provider", summarizer_model)

    if session_id:
        cfg["session_id"] = session_id
    return cfg


def _dump_payload_log(
    *,
    question_id: str,
    cache_dir: Path,
    question_text: str,
    gold_answer: str,
    question_date: str,
    tags_matched: list[str],
    tokens_injected: int,
    budget_breakdown: dict,
    tool_calls: list[dict],
    continuation_count: int,
    stop_reason: str,
    raw_requests: list[dict],
    raw_responses: list[dict],
    hypothesis: str,
    reader_input: int,
    reader_output: int,
    haiku_input: int,
    haiku_output: int,
    haiku_calls: int,
    timings: dict,
    cached: bool,
) -> None:
    """Dump the full HTTP request/response payloads for post-run analysis.

    Writes to ``cache_dir/payload_log.json`` — the raw Anthropic API
    conversation: every request body sent, every response body received,
    in order. This is what you'd see in a network inspector.
    """
    # Build the HTTP conversation: interleaved request/response pairs
    http_conversation = []
    for i in range(max(len(raw_requests), len(raw_responses))):
        if i < len(raw_requests):
            http_conversation.append({
                "step": i + 1,
                "direction": "REQUEST",
                "body": raw_requests[i],
            })
        if i < len(raw_responses):
            http_conversation.append({
                "step": i + 1,
                "direction": "RESPONSE",
                "body": raw_responses[i],
            })

    payload = {
        "_description": "Raw HTTP payloads for LongMemEval question. "
                        "http_conversation contains the actual request/response "
                        "bodies sent to/received from the Anthropic API, in order.",
        "question_id": question_id,
        "question": question_text,
        "gold_answer": gold_answer,
        "question_date": question_date,
        "hypothesis": hypothesis,
        "correct": None,  # filled in by judge later
        "cached": cached,
        "summary": {
            "tags_matched": tags_matched,
            "tokens_injected": tokens_injected,
            "budget_breakdown": budget_breakdown,
            "tool_calls_count": len(tool_calls),
            "continuation_count": continuation_count,
            "stop_reason": stop_reason,
            "tools_used": [tc["tool"] for tc in tool_calls],
            "reader_input_tokens": reader_input,
            "reader_output_tokens": reader_output,
            "haiku_input_tokens": haiku_input,
            "haiku_output_tokens": haiku_output,
            "haiku_calls": haiku_calls,
            "timings": timings,
        },
        "tool_calls": tool_calls,
        "http_conversation": http_conversation,
    }

    out_path = cache_dir / "payload_log.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("VC [%s]: payload log saved to %s (%d request/response pairs)",
                     question_id, out_path, len(raw_requests))
    except Exception as e:
        logger.warning("VC [%s]: failed to save payload log: %s", question_id, e)


def _sessions_to_messages(q: LongMemEvalQuestion) -> list[Message]:
    """Convert LongMemEval haystack sessions into a flat Message list."""
    messages: list[Message] = []
    for session, date in zip(q.haystack_sessions, q.haystack_dates):
        for i, turn in enumerate(session):
            content = turn.get("content", "")
            role = turn.get("role", "user")
            # Prepend session date to first user message for temporal grounding
            if i == 0 and role == "user":
                content = f"[Session from {date}] {content}"
            messages.append(Message(role=role, content=content))
    return messages


def _clear_compaction_state(cache_dir: str, question_id: str) -> None:
    """Clear compacted segments and tag summaries but keep TurnTagIndex.

    This allows re-running compaction with a new prompt without re-ingesting
    (re-tagging) all turns — saving significant Haiku cost and time.
    """
    import sqlite3
    db_path = Path(cache_dir) / "store.db"
    if not db_path.exists():
        logger.warning("No DB found at %s — nothing to clear", db_path)
        return

    conn = sqlite3.connect(str(db_path))
    # Clear compacted data
    conn.execute("DELETE FROM segment_tags")
    conn.execute("DELETE FROM segments")
    conn.execute("DELETE FROM tag_summaries")
    # Clear FTS indexes
    try:
        conn.execute("DELETE FROM segments_fts")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("DELETE FROM segments_fts_full")
    except sqlite3.OperationalError:
        pass
    # Clear chunk embeddings
    try:
        conn.execute("DELETE FROM chunk_embeddings")
    except sqlite3.OperationalError:
        pass

    # Reset compacted_through to 0 but keep turn_tag_entries intact
    conn.execute("UPDATE engine_state SET compacted_through = 0")

    conn.commit()
    conn.close()
    logger.info("VC [%s]: cleared compaction state (segments + tag_summaries), kept TurnTagIndex", question_id)


def run_vc(
    question: LongMemEvalQuestion,
    budget: BudgetTracker,
    context_window: int = 65536,
    reader_model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
    verbose: bool = False,
    fresh: bool = False,
    recompact: bool = False,
    summarizer_model: str | None = None,
    reader_provider: str = "anthropic",
) -> dict:
    """Run the VC pipeline for a single question.

    Steps: convert sessions → ingest_history → compact_manual loop → on_message_inbound → query reader

    When a cached ingestion+compaction exists for this question_id, steps 2-4
    are skipped entirely (~10 min saved). Use fresh=True to force re-ingestion.

    Parameters
    ----------
    reader_provider : str
        LLM provider for the reader model: ``"anthropic"``, ``"openai"``, or ``"gemini"``.

    Returns dict with: hypothesis, input_tokens, output_tokens, cost,
                       tags_matched, tokens_injected, compaction_events, timings.
    """
    # Resolve API key: explicit > env var for provider > ANTHROPIC_API_KEY fallback
    key_env = API_KEY_ENVS.get(reader_provider, "ANTHROPIC_API_KEY")
    api_key = api_key or os.environ.get(key_env, "")
    # Anthropic key is always needed for ingestion/compaction (Haiku)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(f"{key_env} not set")
    if not anthropic_key and reader_provider != "anthropic":
        raise ValueError("ANTHROPIC_API_KEY not set (needed for ingestion/compaction)")

    timings: dict[str, float] = {}

    # 1. Build engine — use stable cache dir for resumable state
    cache_dir = _cache_dir_for(question.question_id)
    if fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("VC [%s]: cleared cache (--fresh)", question.question_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = str(cache_dir)

    if recompact and not fresh:
        _clear_compaction_state(storage_dir, question.question_id)

    cfg_dict = _build_vc_config(
        context_window=context_window,
        storage_dir=storage_dir,
        session_id=f"bench-{question.question_id}",
        summarizer_model=summarizer_model,
    )
    config = load_config(config_dict=cfg_dict)
    engine = VirtualContextEngine(config=config)

    # 2. Convert sessions to messages
    messages = _sessions_to_messages(question)
    n_pairs = len(messages) // 2

    # 3. Check if we can resume from cached state
    #    Three modes:
    #    - fully cached: compacted_through > 0 → skip ingestion + compaction
    #    - recompact: compacted_through == 0 but TurnTagIndex has entries → skip ingestion, re-compact
    #    - fresh: nothing cached → ingest + compact
    n_index_entries = len(engine._turn_tag_index.entries)
    fully_cached = engine._compacted_through > 0
    tags_only = not fully_cached and n_index_entries > 0

    if fully_cached:
        logger.info(
            "VC [%s]: CACHE HIT — %d turns indexed, compacted_through=%d. Skipping ingestion + compaction.",
            question.question_id, n_index_entries, engine._compacted_through,
        )
        compaction_events = -1  # sentinel: cached
        timings["ingest_s"] = 0.0
        timings["compact_s"] = 0.0
        turns_ingested = n_index_entries
    elif tags_only:
        logger.info(
            "VC [%s]: RECOMPACT — %d turns indexed, compacted_through=0. Skipping ingestion, re-running compaction.",
            question.question_id, n_index_entries,
        )
        timings["ingest_s"] = 0.0
        turns_ingested = n_index_entries
    else:
        logger.info(
            "VC [%s]: %d messages (%d pairs), ~%d tokens",
            question.question_id, len(messages), n_pairs, question.haystack_tokens_est,
        )

        # 3a. Ingest history (tags all pairs)
        t0 = time.time()
        _ingest_start = time.time()

        def _progress(done: int, total: int, entry: object) -> None:
            if done % 20 == 0 or done == total:
                elapsed = time.time() - _ingest_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info("  Ingest: %d/%d (%.1f turns/s, ETA %.0fs)", done, total, rate, eta)

        turns_ingested = engine.ingest_history(messages, progress_callback=_progress)
        timings["ingest_s"] = round(time.time() - t0, 1)
        logger.info("VC [%s]: ingested %d turns in %.1fs", question.question_id, turns_ingested, timings["ingest_s"])

    # 3b. Compact in a loop until nothing left to compact
    #     Runs for both fresh ingestion and recompact (tags_only) modes
    if not fully_cached:
        t0 = time.time()
        compaction_events = 0
        total_tokens_freed = 0
        total_segments = 0
        est_rounds = max(1, question.haystack_tokens_est // (context_window // 2))
        logger.info("VC [%s]: starting compaction (est ~%d rounds for %dK tokens into %dK window)...",
                    question.question_id, est_rounds, question.haystack_tokens_est // 1000, context_window // 1000)
        while True:
            t_round = time.time()
            report = engine.compact_manual(messages)
            if report is None:
                break
            compaction_events += 1
            total_tokens_freed += report.tokens_freed
            total_segments += report.segments_compacted
            round_s = time.time() - t_round
            logger.info(
                "  Compaction #%d (%.1fs): %d segments, %d tokens freed, %d tag summaries, tags: %s",
                compaction_events, round_s, report.segments_compacted, report.tokens_freed,
                report.tag_summaries_built, report.tags[:5],
            )
        timings["compact_s"] = round(time.time() - t0, 1)
        logger.info("VC [%s]: %d compaction events, %d segments, %d tokens freed in %.1fs",
                    question.question_id, compaction_events, total_segments, total_tokens_freed, timings["compact_s"])

        # 3c. Persist engine state for future cache hits
        engine._save_state(messages)
        logger.info("VC [%s]: saved state to %s", question.question_id, storage_dir)

    cached = fully_cached  # backward compat for downstream references

    # Snapshot engine cost tracker BEFORE reader call — captures actual Haiku tokens
    # from ingestion + compaction (0 for cached questions, which is correct)
    pre_reader_cost = engine.get_cost_report()
    haiku_input = pre_reader_cost.total_input_tokens
    haiku_output = pre_reader_cost.total_output_tokens
    haiku_calls = (pre_reader_cost.total_tag_generations
                   + pre_reader_cost.total_compactions
                   + pre_reader_cost.total_retrievals)

    # Persist/restore haiku cost data so cached reruns don't lose it
    cost_snapshot_path = cache_dir / "cost_snapshot.json"
    if haiku_input > 0 or haiku_output > 0:
        # Fresh or recompact run — save costs
        cost_snapshot_path.write_text(json.dumps({
            "haiku_input_tokens": haiku_input,
            "haiku_output_tokens": haiku_output,
            "haiku_calls": haiku_calls,
        }))
    elif cost_snapshot_path.exists():
        # Cache hit — restore costs from snapshot
        snap = json.loads(cost_snapshot_path.read_text())
        haiku_input = snap.get("haiku_input_tokens", 0)
        haiku_output = snap.get("haiku_output_tokens", 0)
        haiku_calls = snap.get("haiku_calls", 0)
        logger.info("VC [%s]: restored haiku costs from snapshot: %d in / %d out (%d calls)",
                    question.question_id, haiku_input, haiku_output, haiku_calls)

    # 4. Retrieve context for the question
    t0 = time.time()
    question_prompt = f"Current Date: {question.question_date}\nQuestion: {question.question}"
    assembled = engine.on_message_inbound(question_prompt, messages)
    timings["retrieve_s"] = round(time.time() - t0, 1)

    tags_matched = assembled.matched_tags
    prepend_text = assembled.prepend_text
    tokens_injected = assembled.total_tokens

    prepend_tokens = len(prepend_text) // 4 if prepend_text else 0
    logger.info(
        "VC [%s]: assembled total=%d tokens, prepend_text=%d chars (~%d tokens), tags: %s",
        question.question_id, tokens_injected, len(prepend_text), prepend_tokens, tags_matched[:10],
    )
    logger.info(
        "VC [%s]: budget_breakdown: %s",
        question.question_id, assembled.budget_breakdown,
    )
    if verbose and prepend_text:
        logger.info("VC [%s]: prepend_text preview:\n%s", question.question_id, prepend_text[:1000])

    # Record actual Haiku cost from VC internal operations (tagging + compaction)
    # For cached runs haiku_input/output will be 0 (correct — already paid)
    if haiku_input > 0 or haiku_output > 0:
        budget.record(
            label=f"vc_haiku:{question.question_id}",
            model="claude-haiku-4-5-20251001",
            input_tokens=haiku_input,
            output_tokens=haiku_output,
        )
        logger.info(
            "VC [%s]: haiku actuals: %d in / %d out (%d calls)",
            question.question_id, haiku_input, haiku_output, haiku_calls,
        )

    # 5. Send retrieved context + question to reader model (with tool loop)
    t0 = time.time()

    # System prompt: VC tool instructions only (context hint).
    # The benchmark defines NO system prompt — we only add tool guidance.
    context_hint = assembled.context_hint
    system_prompt = context_hint if context_hint else ""

    # User prompt: LongMemEval's exact template with VC summaries replacing
    # the raw session history.  This keeps the benchmark framing faithful.
    vc_summaries = "\n\n".join(assembled.tag_sections.values())
    user_prompt = (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history.\n\n\n"
        "History Chats:\n\n"
        f"{vc_summaries}\n\n"
        f"Current Date: {question.question_date}\n"
        f"Question: {question.question}\n"
        f"Answer:"
    )

    reader_api_url = API_URLS.get(reader_provider, "")
    loop_result = engine.query_with_tools(
        messages=[{"role": "user", "content": user_prompt}],
        model=reader_model,
        system=system_prompt,
        max_tokens=1024,
        api_key=api_key,
        api_url=reader_api_url,
        temperature=0.0,
        force_tools=True,
        provider=reader_provider,
    )

    timings["query_s"] = round(time.time() - t0, 1)

    hypothesis = loop_result.text
    reader_input = loop_result.input_tokens
    reader_output = loop_result.output_tokens

    # Extract tool call details for analysis (full results, not truncated)
    tool_calls_log = [
        {
            "tool": tc.tool_name,
            "input": tc.tool_input,
            "result": tc.result_json,
            "duration_ms": tc.duration_ms,
        }
        for tc in loop_result.tool_calls
    ]
    if tool_calls_log:
        logger.info(
            "VC [%s]: tool loop: %d calls, %d continuations — %s",
            question.question_id, len(tool_calls_log), loop_result.continuation_count,
            [tc["tool"] for tc in tool_calls_log],
        )
        for tc in tool_calls_log:
            logger.info(
                "  %s(%s) → %s (%.0fms)",
                tc["tool"], tc["input"], tc["result"][:100], tc["duration_ms"],
            )
    else:
        logger.info("VC [%s]: no tool calls in reader response", question.question_id)

    # Dump full payload log — raw HTTP request/response bodies
    _dump_payload_log(
        question_id=question.question_id,
        cache_dir=cache_dir,
        question_text=question.question,
        gold_answer=question.answer,
        question_date=question.question_date,
        tags_matched=tags_matched,
        tokens_injected=tokens_injected,
        budget_breakdown=assembled.budget_breakdown,
        tool_calls=tool_calls_log,
        continuation_count=loop_result.continuation_count,
        stop_reason=loop_result.stop_reason,
        raw_requests=loop_result.raw_requests,
        raw_responses=loop_result.raw_responses,
        hypothesis=hypothesis,
        reader_input=reader_input,
        reader_output=reader_output,
        haiku_input=haiku_input,
        haiku_output=haiku_output,
        haiku_calls=haiku_calls,
        timings=timings,
        cached=cached,
    )

    budget.record(
        label=f"vc_reader:{question.question_id}",
        model=reader_model,
        input_tokens=reader_input,
        output_tokens=reader_output,
    )

    total_cost = sum(
        e.cost_usd for e in budget.entries
        if question.question_id in e.label and ("vc_haiku" in e.label or "vc_reader" in e.label)
    )

    logger.info(
        "VC [%s]: reader %d in / %d out, haiku %d in / %d out (%d calls), "
        "total VC cost $%.4f, %.1fs%s",
        question.question_id, reader_input, reader_output,
        haiku_input, haiku_output, haiku_calls,
        total_cost, timings["query_s"],
        " (CACHED)" if cached else "",
    )

    return {
        "hypothesis": hypothesis,
        "input_tokens": reader_input,
        "output_tokens": reader_output,
        "haiku_input_tokens": haiku_input,
        "haiku_output_tokens": haiku_output,
        "haiku_calls": haiku_calls,
        "cost": round(total_cost, 6),
        "tags_matched": tags_matched,
        "tokens_injected": tokens_injected,
        "compaction_events": compaction_events,
        "turns_ingested": turns_ingested,
        "timings": timings,
        "cached": cached,
        "tool_calls": tool_calls_log,
        "continuation_count": loop_result.continuation_count,
        "stop_reason": loop_result.stop_reason,
    }
